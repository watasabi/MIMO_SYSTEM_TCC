"""Output decomposition (à trous, EWT-c, seasonal) vs. the baseline.

Plan and rules: `docs/changes/2026-10-22-decomposicao-saidas.md`.
Every forecast goes through `evaluation.forecast_at_origin` (the
production contract), at **every** hour of val and test. Decisions are
made on val; test is reported for the final table.

Uso (da raiz do repo):
    S=notebooks/training/scripts/run_decomposition.py
    uv run --extra ml python $S run [variante ...]   # default: todas
    uv run --extra ml python $S report
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch

from mimo_sys.training.data import (
    INPUT_COLS,
    OUTPUT_COLS,
    PROJECT_ROOT,
    TARGET_IDX,
    load_splits,
)
from mimo_sys.training.evaluation import (
    diebold_mariano,
    forecast_origins,
    paired_block_bootstrap,
    r2_per_output,
)
from mimo_sys.training.mass_balance_cascade import (
    fit_mass_balance,
    mass_balance_cascade_forecast,
)
from mimo_sys.training.train import (
    TrainConfig,
    _windows,  # noqa: PLC2701
    add_derived_features,
    known_columns,
    target_layout,
    train,
)

OUT = PROJECT_ROOT / "outputs"
RESULTS = OUT / "decomposition_results.jsonl"
PREDS = OUT / "decomposition_preds"
REPORT = OUT / "decomposition_report.json"
SEEDS = (42, 43, 44)
VAZAO, NIVEL = 0, 1
PUMPS = ("freq_b1", "freq_b2", "freq_b3")
# Acceptance (fixed before running): pumps must gain more than the
# seed spread of the reference model, full must not lose beyond noise.
SEED_SPREAD = 0.012
FULL_TOLERANCE = -0.005

# Doc 8's reference model: SpectralTFNet + residual, 168h in, 12h out.
BASE = dict(
    model="spectral_tf_net",
    input_window=168,
    output_window=12,
    batch_size=128,
    future_exog=True,
    mass_balance_feature=True,
    revin=None,
    uncertainty_weighted=True,
    cascade_weight=1.0,
    message_l1_weight=1e-3,
    # Doc 8's runs predate SpectralTFNet's backcast term; the default
    # 1.0 costs ~0.01 (full) / ~0.04 (pumps), see `_bw1` runs.
    backcast_weight=0.0,
    max_epochs=150,
    patience=25,
    accelerator="auto",
    experiment="mimo_sys_decomposition",
    save=False,
    model_kwargs=dict(n_bands=8, n_regimes=1, residual_hidden=16),
)
SCENARIOS = {
    "full": dict(future_known=tuple(INPUT_COLS)),
    "pumps": dict(future_known=PUMPS),
}
ATROUS = dict(target_decomposition="atrous", decomposition_levels=4)
VARIANTS: dict[str, dict] = {
    "C0_baseline": {},
    "C1_atrous_parent_l1": {
        **ATROUS,
        "component_scale": "parent",
        "component_loss_weight": 1.0,
    },
    "C2_atrous_unit_l0": {**ATROUS, "component_loss_weight": 0.0},
    "C3_atrous_unit_l05": {**ATROUS, "component_loss_weight": 0.5},
    "C4a_atrous3_unit_l0": {
        **ATROUS,
        "decomposition_levels": 3,
        "component_loss_weight": 0.0,
    },
    "C4b_atrous3_unit_l05": {
        **ATROUS,
        "decomposition_levels": 3,
        "component_loss_weight": 0.5,
    },
    "C2e3_ewt3_unit_l0": {
        "target_decomposition": "ewt",
        "decomposition_levels": 3,
        "component_loss_weight": 0.0,
    },
    "C3e3_ewt3_unit_l05": {
        "target_decomposition": "ewt",
        "decomposition_levels": 3,
        "component_loss_weight": 0.5,
    },
    "C2e5_ewt5_unit_l0": {
        "target_decomposition": "ewt",
        "decomposition_levels": 5,
        "component_loss_weight": 0.0,
    },
    "C3e5_ewt5_unit_l05": {
        "target_decomposition": "ewt",
        "decomposition_levels": 5,
        "component_loss_weight": 0.5,
    },
    "C5_seasonal24": {"target_decomposition": "seasonal24"},
    "C6_atrous_input_only": {**ATROUS, "components_as_targets": False},
}


def config(variant: str, scenario: str, seed: int) -> TrainConfig:
    return TrainConfig(
        run_name=f"{variant}_{scenario}_s{seed}",
        seed=seed,
        **{**BASE, **SCENARIOS[scenario], **VARIANTS[variant]},
    )


def cascade_nivel(
    cfg: TrainConfig,
    seg: np.ndarray,
    train_raw: np.ndarray,
    origins: np.ndarray,
    vazao: np.ndarray,
) -> np.ndarray:
    """`nivel_res` from the mass-balance cascade fed by the forecast flow.

    Uses only the row before each origin and the inflow known at the
    origin (or the same hours one day earlier when it is not known).
    """
    inflow = INPUT_COLS.index("vazao_entrada")
    vz, nv = TARGET_IDX[VAZAO], TARGET_IDX[NIVEL]
    a, b = fit_mass_balance(
        train_raw[:, nv], train_raw[:, inflow], train_raw[:, vz]
    )
    h = cfg.output_window
    shift = 0 if "vazao_entrada" in cfg.future_known else 24
    return mass_balance_cascade_forecast(
        seg[origins - 1, nv],
        np.stack([seg[o - shift : o - shift + h, inflow] for o in origins]),
        vazao,
        a,
        b,
        last_imbalance=seg[origins - 1, inflow] - seg[origins - 1, vz],
    )


def evaluate(cfg: TrainConfig, result) -> dict[str, np.ndarray]:
    """Forecasts at every hour of val and test, production contract."""
    splits = load_splits(cfg.data_prefix)
    derived = add_derived_features(cfg, splits)
    _, _, test_seg, _ = _windows(cfg, splits)
    raw = np.concatenate([derived["train"], derived["val"], derived["test"]])
    train_end = len(derived["train"])
    val_end = train_end + len(derived["val"])
    segs = {
        "val": raw[train_end - cfg.input_window : val_end],
        "test": test_seg,
    }
    model = result.module.model.cpu().eval()
    out = {}
    h = cfg.output_window
    for name, seg in segs.items():
        origins = np.arange(cfg.input_window, len(seg) - h + 1)
        pred = forecast_origins(
            model,
            seg,
            result.preprocessor,
            origins,
            output_window=h,
            future_exog=cfg.future_exog,
            future_known_idx=known_columns(cfg),
            input_window=cfg.input_window,
            layout=target_layout(cfg),
        )
        out[f"{name}_origins"] = origins
        out[f"{name}_true"] = np.stack(
            [seg[o : o + h, TARGET_IDX] for o in origins]
        )
        out[f"{name}_pred"] = pred
        out[f"{name}_nivel_cascata"] = cascade_nivel(
            cfg, seg, derived["train"], origins, pred[..., VAZAO]
        )
    return out


def run(variant: str, scenario: str, seed: int) -> None:
    cfg = config(variant, scenario, seed)
    done = set()
    if RESULTS.exists():
        done = {json.loads(line)["run"] for line in RESULTS.open()}
    if cfg.run_name in done:
        print(f"skip {cfg.run_name}", flush=True)
        return
    torch.set_float32_matmul_precision("high")
    result = train(cfg)
    arrays = evaluate(cfg, result)
    PREDS.mkdir(parents=True, exist_ok=True)
    np.savez(PREDS / f"{cfg.run_name}.npz", **arrays)
    row = {
        "run": cfg.run_name,
        "variant": variant,
        "scenario": scenario,
        "seed": seed,
        "best_val_mse": result.best_val_mse,
        "epochs": result.module.current_epoch,
    }
    for split in ("val", "test"):
        yt, yp = arrays[f"{split}_true"], arrays[f"{split}_pred"]
        cas = yp.copy()
        cas[..., NIVEL] = arrays[f"{split}_nivel_cascata"]
        row[f"{split}_r2_direto"] = r2_per_output(yt, yp).tolist()
        row[f"{split}_r2_cascata"] = r2_per_output(yt, cas).tolist()
        # Doc 8's headline protocol: one origin every 12 h.
        sub = slice(0, None, cfg.output_window)
        row[f"{split}_r2_cascata_12h"] = r2_per_output(
            yt[sub], cas[sub]
        ).tolist()
    RESULTS.parent.mkdir(exist_ok=True)
    with RESULTS.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(
        f"{cfg.run_name}: val {np.round(row['val_r2_cascata'], 3)} "
        f"test {np.round(row['test_r2_cascata'], 3)}",
        flush=True,
    )


def _load(variant: str, scenario: str, split: str) -> dict[str, np.ndarray]:
    """Seed-stacked predictions, nivel direct and through the cascade."""
    runs = [np.load(PREDS / f"{variant}_{scenario}_s{s}.npz") for s in SEEDS]
    direct = np.stack([r[f"{split}_pred"] for r in runs])
    cascade = direct.copy()
    cascade[..., NIVEL] = np.stack([r[f"{split}_nivel_cascata"] for r in runs])
    return {
        "true": runs[0][f"{split}_true"],
        "direto": direct,
        "cascata": cascade,
    }


def _nivel_source(variant: str, scenario: str) -> str:
    """Direct head or cascade for `nivel_res`, chosen on val only."""
    val = _load(variant, scenario, "val")
    score = {
        src: np.mean([r2_per_output(val["true"], p)[NIVEL] for p in val[src]])
        for src in ("direto", "cascata")
    }
    return max(score, key=score.get)


def report() -> None:  # noqa: PLR0914 - one table, linear
    rows = []
    for scenario in SCENARIOS:
        base_src = _nivel_source("C0_baseline", scenario)
        for variant in VARIANTS:
            if not all(
                (PREDS / f"{variant}_{scenario}_s{s}.npz").exists()
                for s in SEEDS
            ):
                continue
            src = _nivel_source(variant, scenario)
            entry = {"variant": variant, "scenario": scenario, "nivel": src}
            for split in ("val", "test"):
                cand = _load(variant, scenario, split)
                base = _load("C0_baseline", scenario, split)
                per_seed = np.array(
                    [r2_per_output(cand["true"], p) for p in cand[src]]
                )
                boot = paired_block_bootstrap(
                    cand["true"], base[base_src], cand[src]
                )
                stat, p = diebold_mariano(
                    cand["true"], base[base_src], cand[src]
                )
                entry[split] = {
                    "r2": per_seed.mean(axis=0).round(4).tolist(),
                    "r2_medio": round(float(per_seed.mean()), 4),
                    "r2_medio_seed_std": round(
                        float(per_seed.mean(axis=1).std()), 4
                    ),
                    "delta_medio": round(boot["delta_mean"], 4),
                    "ic95": [
                        round(boot["ci_low_mean"], 4),
                        round(boot["ci_high_mean"], 4),
                    ],
                    "dm_stat": round(stat, 2),
                    "dm_p": float(f"{p:.3g}"),
                }
            rows.append(entry)
    decide(rows)
    REPORT.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    header = (
        f"{'variante':24s} {'cen':5s} {'nível':7s} | val médio (Δ, IC95) "
        f"| test médio (Δ, IC95) | test {'/'.join(OUTPUT_COLS)} | aceita"
    )
    print(header)
    for r in rows:
        v, t = r["val"], r["test"]
        print(
            f"{r['variant']:24s} {r['scenario']:5s} {r['nivel']:7s} | "
            f"{v['r2_medio']:.3f} ({v['delta_medio']:+.3f}, "
            f"{v['ic95'][0]:+.3f}..{v['ic95'][1]:+.3f}) | "
            f"{t['r2_medio']:.3f} ({t['delta_medio']:+.3f}, "
            f"{t['ic95'][0]:+.3f}..{t['ic95'][1]:+.3f}) | "
            f"{'/'.join(f'{x:.3f}' for x in t['r2'])} | "
            f"{r.get('aceita', '-')}"
        )


def decide(rows: list[dict]) -> None:
    """Pre-registered rule, on val: pumps gains, full does not lose."""
    by = {(r["variant"], r["scenario"]): r for r in rows}
    for (variant, scenario), r in by.items():
        if scenario != "pumps" or variant == "C0_baseline":
            continue
        v = r["val"]
        full = by.get((variant, "full"))
        r["aceita"] = bool(
            v["delta_medio"] > SEED_SPREAD
            and v["ic95"][0] > 0
            and full is not None
            and full["val"]["delta_medio"] >= FULL_TOLERANCE
        )


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report()
    else:
        chosen = sys.argv[2:] or list(VARIANTS)
        for variant in chosen:
            for scenario in SCENARIOS:
                for seed in SEEDS:
                    run(variant, scenario, seed)
