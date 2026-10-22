"""Level/pressure improvements that keep the graph extractable.

Plan: `docs/changes/2026-10-23-melhorias-nivel-pressao.md`. Steps:
clean training data (I1), the graph forecasting the unknown inflow
(I4, `pumps` only), then post hoc the per-step direct/cascade level
blend and the seed ensemble, either averaging forecasts or a single
model with the seeds' transfer functions averaged (`H` soup).
Production contract and statistics as in `run_decomposition.py`;
decisions on val (blend weights cross-fitted across val halves), test
reported once for the final table.

Uso (da raiz do repo):
    S=notebooks/training/scripts/run_improvements.py
    uv run --extra ml python $S run [variante ...]
    uv run --extra ml python $S report
"""

from __future__ import annotations

import copy
import json
import sys

import numpy as np
import torch
from run_decomposition import BASE, SCENARIOS
from torch import nn

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
from mimo_sys.training.features import bad_sample_mask
from mimo_sys.training.mass_balance_cascade import (
    apply_level_blend,
    fit_level_blend,
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

OUT = PROJECT_ROOT / "outputs" / "improvements"
RESULTS = OUT / "results.jsonl"
SEEDS = (42, 43, 44, 45, 46)
STAT_SEEDS = SEEDS[:3]
VAZAO, NIVEL = 0, 1
SEED_SPREAD = 0.012
FULL_TOLERANCE = -0.005

VARIANTS: dict[str, dict] = {
    "I0_ref": {},
    "I1_clean": {"clean_train": True},
    "I4_clean_inflow": {"clean_train": True, "predict_inflow": True},
    # Step 5: explicit pump-curve polynomials (Hammerstein) instead of,
    # or on top of, the MLP residual that has no equation.
    "I5_clean_hamm3": {
        "clean_train": True,
        "model_kwargs": dict(
            n_bands=8,
            n_regimes=1,
            residual_hidden=16,
            input_nonlinearity_degree=3,
        ),
    },
    "I6_clean_linear": {
        "clean_train": True,
        "model_kwargs": dict(n_bands=8, n_regimes=1, residual_hidden=0),
    },
    # Pressure setpoint (causal 7-day same-hour median) as a known
    # future node; then one response bank per regime picked at every
    # step by it (switched ARX); then the explicit polynomial residual.
    "R1_setpoint": {"setpoint_feature": True},
    "R2_setpoint_regime3": {
        "setpoint_feature": True,
        "model_kwargs": dict(
            n_bands=8,
            n_regimes=3,
            residual_hidden=16,
            regime_per_step=True,
            gate_idx=(8,),  # 7 raw nodes + mass balance -> setpoint is 8
        ),
    },
    "P1_cheb_residual": {
        "model_kwargs": dict(
            n_bands=8,
            n_regimes=1,
            residual_hidden=1,
            residual_kind="chebyshev",
            residual_lags=24,
            residual_degree=3,
        ),
    },
    "R3_setpoint_cheb": {
        "setpoint_feature": True,
        "model_kwargs": dict(
            n_bands=8,
            n_regimes=1,
            residual_hidden=1,
            residual_kind="chebyshev",
            residual_lags=24,
            residual_degree=3,
        ),
    },
    "I6_clean_linear_hamm3": {
        "clean_train": True,
        "model_kwargs": dict(
            n_bands=8,
            n_regimes=1,
            residual_hidden=0,
            input_nonlinearity_degree=3,
        ),
    },
}
# `predict_inflow` only exists where the future inflow is unknown.
ONLY = {"I4_clean_inflow": ("pumps",)}


def config(variant: str, scenario: str, seed: int) -> TrainConfig:
    return TrainConfig(
        run_name=f"{variant}_{scenario}_s{seed}",
        seed=seed,
        **{
            **BASE,
            "experiment": "mimo_sys_improvements",
            **SCENARIOS[scenario],
            **VARIANTS[variant],
        },
    )


def cascade_nivel(  # noqa: PLR0913
    cfg: TrainConfig,
    seg: np.ndarray,
    train_raw: np.ndarray,
    origins: np.ndarray,
    vazao: np.ndarray,
    inflow_pred: np.ndarray | None,
) -> np.ndarray:
    """Mass-balance cascade from the row before each origin.

    Inflow: known (full), the graph's own forecast (`predict_inflow`)
    or the same hours one day earlier. The gray-box skips the
    training rows flagged as sensor failures when `clean_train`.
    """
    inflow = INPUT_COLS.index("vazao_entrada")
    vz, nv = TARGET_IDX[VAZAO], TARGET_IDX[NIVEL]
    a, b = fit_mass_balance(
        train_raw[:, nv],
        train_raw[:, inflow],
        train_raw[:, vz],
        exclude=bad_sample_mask(load_splits(cfg.data_prefix)["train"])
        if cfg.clean_train
        else None,
    )
    h = cfg.output_window
    if inflow_pred is not None:
        future_inflow = inflow_pred
    else:
        shift = 0 if "vazao_entrada" in cfg.future_known else 24
        future_inflow = np.stack(
            [seg[o - shift : o - shift + h, inflow] for o in origins]
        )
    return mass_balance_cascade_forecast(
        seg[origins - 1, nv],
        future_inflow,
        vazao,
        a,
        b,
        last_imbalance=seg[origins - 1, inflow] - seg[origins - 1, vz],
    )


def _segments(cfg: TrainConfig) -> tuple[dict, np.ndarray]:
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
    return segs, derived["train"]


def evaluate(
    cfg: TrainConfig, model: nn.Module, pp: object
) -> dict[str, np.ndarray]:
    """Every-hour forecasts of val and test, production contract."""
    segs, train_raw = _segments(cfg)
    layout = target_layout(cfg)
    h = cfg.output_window
    out = {}
    for name, seg in segs.items():
        origins = np.arange(cfg.input_window, len(seg) - h + 1)
        pred = forecast_origins(
            model,
            seg,
            pp,
            origins,
            output_window=h,
            future_exog=cfg.future_exog,
            future_known_idx=known_columns(cfg),
            input_window=cfg.input_window,
            layout=layout,
            return_aux=bool(layout.aux_cols),
        )
        inflow_pred = None
        if layout.aux_cols:
            pred, aux = pred
            inflow_pred = aux[..., 0]
            out[f"{name}_inflow_pred"] = inflow_pred
        out[f"{name}_true"] = np.stack(
            [seg[o : o + h, TARGET_IDX] for o in origins]
        )
        out[f"{name}_inflow_true"] = np.stack(
            [
                seg[o : o + h, INPUT_COLS.index("vazao_entrada")]
                for o in origins
            ]
        )
        out[f"{name}_pred"] = pred
        out[f"{name}_nivel_cascata"] = cascade_nivel(
            cfg, seg, train_raw, origins, pred[..., VAZAO], inflow_pred
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
    model = result.module.model.cpu().eval()
    arrays = evaluate(cfg, model, result.preprocessor)
    (OUT / "preds").mkdir(parents=True, exist_ok=True)
    (OUT / "weights").mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "preds" / f"{cfg.run_name}.npz", **arrays)
    torch.save(model.state_dict(), OUT / "weights" / f"{cfg.run_name}.pt")
    torch.save(result.preprocessor, OUT / "weights" / f"{cfg.run_name}_pp.pt")
    row = {
        "run": cfg.run_name,
        "variant": variant,
        "scenario": scenario,
        "seed": seed,
        "epochs": result.module.current_epoch,
        "test_r2_direto": r2_per_output(
            arrays["test_true"], arrays["test_pred"]
        ).tolist(),
    }
    if "test_inflow_pred" in arrays:
        row["test_r2_inflow"] = float(
            r2_per_output(
                arrays["test_inflow_true"][..., None],
                arrays["test_inflow_pred"][..., None],
            )[0]
        )
    OUT.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"{cfg.run_name}: {row}", flush=True)


class SoupSpectral(nn.Module):
    """One SpectralTFNet with the seeds' `H_ij(f)` and biases averaged.

    The linear part stays a single transfer-function matrix (one
    equation); the nonlinear residual is the mean of the seeds'
    residuals, as in the forecast-averaging ensemble.
    """

    def __init__(self, models: list[nn.Module]) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)
        self.linear = copy.deepcopy(models[0])
        with torch.no_grad():
            for name in ("response", "bias", "hammerstein_coeffs"):
                if getattr(models[0], name, None) is None:
                    continue
                mean = torch.stack([getattr(m, name) for m in models]).mean(0)
                getattr(self.linear, name).copy_(mean)
        self.linear.residual = None
        self.linear.residual_coeffs = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        residuals = []
        for m in self.models:
            m(x)
            if m._last_residual is not None:
                residuals.append(m._last_residual)
        if residuals:
            out += torch.stack(residuals).mean(0)
        return out


def load_weights(run_name: str) -> dict[str, torch.Tensor]:
    """Saved `state_dict`, minus buffers rebuilt by the constructor."""
    state = torch.load(OUT / "weights" / f"{run_name}.pt")
    state.pop("gate_idx", None)
    return state


def _load(variant: str, scenario: str, seeds: tuple[int, ...]) -> dict:
    runs = [
        np.load(OUT / "preds" / f"{variant}_{scenario}_s{s}.npz")
        for s in seeds
    ]
    return {k: np.stack([r[k] for r in runs]) for k in runs[0].files}


def _soup_preds(variant: str, scenario: str) -> dict[str, np.ndarray]:
    """Forecasts of the `H` soup over all seeds (cached)."""
    path = OUT / "preds" / f"{variant}_{scenario}_soup.npz"
    if path.exists():
        return dict(np.load(path))
    from mimo_sys.training.train import build_model  # noqa: PLC0415

    cfg = config(variant, scenario, SEEDS[0])
    layout = target_layout(cfg)
    models = []
    for s in SEEDS:
        c = config(variant, scenario, s)
        n_nodes = _windows(c, load_splits(c.data_prefix))[1]["train"][0].shape[
            2
        ]
        kwargs = dict(c.model_kwargs)
        kwargs.setdefault("known_future_idx", tuple(known_columns(c)))
        m = build_model(
            c.model,
            c.input_window + c.output_window,
            c.output_window,
            n_nodes,
            target_idx=list(layout.model_cols),
            **kwargs,
        )
        m.load_state_dict(load_weights(c.run_name))
        models.append(m.eval())
    pp = torch.load(
        OUT / "weights" / f"{cfg.run_name}_pp.pt", weights_only=False
    )
    arrays = evaluate(cfg, SoupSpectral(models).eval(), pp)
    np.savez(path, **arrays)
    return arrays


def _levels(d: dict, weights: np.ndarray | None) -> np.ndarray:
    """Forecasts with `nivel_res` from the blend (or the direct head)."""
    pred = d["pred"].copy()
    if weights is not None:
        pred[..., NIVEL] = apply_level_blend(
            weights, d["pred"][..., NIVEL], d["nivel_cascata"]
        )
    return pred


def _blend_weights(val: dict) -> np.ndarray:
    return fit_level_blend(
        val["true"][..., NIVEL], val["pred"][..., NIVEL], val["nivel_cascata"]
    )


def _crossfit_val(val: dict) -> np.ndarray:
    """Val forecasts with blend weights fitted on the other val half."""
    n = len(val["true"])
    halves = (slice(0, n // 2), slice(n // 2, n))
    pred = val["pred"].copy()
    for fit_on, apply_on in (halves, halves[::-1]):
        part = {k: v[fit_on] for k, v in val.items()}
        w = _blend_weights(part)
        pred[apply_on, :, NIVEL] = apply_level_blend(
            w, val["pred"][apply_on, :, NIVEL], val["nivel_cascata"][apply_on]
        )
    return pred


def _split(arrays: dict, split: str) -> dict:
    keys = ("true", "pred", "nivel_cascata")
    return {k: arrays[f"{split}_{k}"] for k in keys}


def candidates(scenario: str) -> dict[str, dict[str, np.ndarray]]:
    """Every candidate's forecasts (3 stat seeds stacked), val and test."""
    out = {}
    for variant in VARIANTS:
        if scenario not in ONLY.get(variant, SCENARIOS):
            continue
        if not all(
            (OUT / "preds" / f"{variant}_{scenario}_s{s}.npz").exists()
            for s in SEEDS
        ):
            continue
        stacked = _load(variant, scenario, STAT_SEEDS)
        per_seed = [
            {k: v[i] for k, v in stacked.items()}
            for i in range(len(STAT_SEEDS))
        ]
        for mode in ("", "+cascata", "+blend"):
            name = variant + mode
            val_preds, test_preds = [], []
            for arrays in per_seed:
                val, test = _split(arrays, "val"), _split(arrays, "test")
                if mode == "+blend":
                    val_preds.append(_crossfit_val(val))
                    test_preds.append(_levels(test, _blend_weights(val)))
                elif mode == "+cascata":
                    ones = np.ones(val["pred"].shape[1])
                    val_preds.append(_levels(val, ones))
                    test_preds.append(_levels(test, ones))
                else:
                    val_preds.append(val["pred"])
                    test_preds.append(test["pred"])
            out[name] = {
                "val": np.stack(val_preds),
                "test": np.stack(test_preds),
                "val_true": per_seed[0]["val_true"],
                "test_true": per_seed[0]["test_true"],
            }
        # Ensembles over all 5 seeds: forecast mean and `H` soup. With
        # several regimes the banks are not aligned across seeds (label
        # switching), so averaging `H` is meaningless there.
        every = _load(variant, scenario, SEEDS)
        mean = {k: v.mean(axis=0) for k, v in every.items()}
        ensembles = [("ens_media", mean)]
        n_regimes = config(variant, scenario, SEEDS[0]).model_kwargs.get(
            "n_regimes", 1
        )
        if n_regimes == 1:
            ensembles.append(("ens_soupH", _soup_preds(variant, scenario)))
        for tag, arrays in ensembles:
            val, test = _split(arrays, "val"), _split(arrays, "test")
            w = _blend_weights(val)
            # Broadcast to the stat seeds so pairing stays seed-wise.
            rep = len(STAT_SEEDS)
            out[f"{variant}+blend+{tag}"] = {
                "val": np.repeat(_crossfit_val(val)[None], rep, axis=0),
                "test": np.repeat(_levels(test, w)[None], rep, axis=0),
                "val_true": val["true"],
                "test_true": test["true"],
            }
    return out


def report() -> None:
    rows = []
    for scenario in SCENARIOS:
        cands = candidates(scenario)
        # Reference: I0 with the level source picked on val (cascade
        # with known inflow, direct head without), as in
        # `run_decomposition.py`.
        ref_name = max(
            ("I0_ref", "I0_ref+cascata"),
            key=lambda n: np.mean(
                [
                    r2_per_output(cands[n]["val_true"], p).mean()
                    for p in cands[n]["val"]
                ]
            ),
        )
        ref = cands[ref_name]
        print(f"referência {scenario}: {ref_name}")
        for name, c in cands.items():
            entry = {"candidato": name, "cenario": scenario}
            for split in ("val", "test"):
                truth = c[f"{split}_true"]
                per_seed = np.array(
                    [r2_per_output(truth, p) for p in c[split]]
                )
                boot = paired_block_bootstrap(truth, ref[split], c[split])
                stat, p = diebold_mariano(truth, ref[split], c[split])
                entry[split] = {
                    "r2": per_seed.mean(0).round(4).tolist(),
                    "r2_medio": round(float(per_seed.mean()), 4),
                    "delta_medio": round(boot["delta_mean"], 4),
                    "ic95": [
                        round(boot["ci_low_mean"], 4),
                        round(boot["ci_high_mean"], 4),
                    ],
                    "dm_p": float(f"{p:.3g}"),
                }
            rows.append(entry)
    (OUT / "report.json").write_text(json.dumps(rows, indent=2))
    print(
        f"{'candidato':34s} {'cen':5s} | val médio (Δ, IC95) | "
        f"test médio (Δ, IC95) | test {'/'.join(OUTPUT_COLS)}"
    )
    for r in rows:
        v, t = r["val"], r["test"]
        print(
            f"{r['candidato']:34s} {r['cenario']:5s} | {v['r2_medio']:.3f} "
            f"({v['delta_medio']:+.3f}, "
            f"{v['ic95'][0]:+.3f}..{v['ic95'][1]:+.3f})"
            f" | {t['r2_medio']:.3f} ({t['delta_medio']:+.3f}, "
            f"{t['ic95'][0]:+.3f}..{t['ic95'][1]:+.3f}) | "
            f"{'/'.join(f'{x:.3f}' for x in t['r2'])}"
        )


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report()
    else:
        chosen = sys.argv[2:] or list(VARIANTS)
        for variant in chosen:
            for scenario in ONLY.get(variant, SCENARIOS):
                for seed in SEEDS:
                    run(variant, scenario, seed)
