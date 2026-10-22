"""Treino e avaliação do SpectralTFNet (grade, ablações, horizontes).

Uso (da raiz do repo):
    S=notebooks/training/scripts/run_spectral_tf_net.py
    uv run --extra ml python $S grid [batch ...]
    uv run --extra ml python $S ablation <window> <batch>
    uv run --extra ml python $S horizons <window> <batch>
    uv run --extra ml python $S no_future_inflow [seed ...]
    uv run --extra ml python $S paper
    uv run --extra ml python $S paper_split
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch

from mimo_sys.training import (
    TrainConfig,
    add_derived_features,
    fit_mass_balance,
    mass_balance_cascade_forecast,
    train,
)
from mimo_sys.training.data import (
    INPUT_COLS,
    PROJECT_ROOT,
    TARGET_IDX,
    load_splits,
)
from mimo_sys.training.evaluation import chained_forecast, metrics_table
from mimo_sys.training.train import _windows  # noqa: PLC2701

RESULTS = PROJECT_ROOT / "outputs" / "spectral_tf_net_results.jsonl"
PREDS = PROJECT_ROOT / "outputs" / "spectral_tf_net_preds"
OUTPUT_WINDOW = 12
WINDOWS = (24, 48, 96, 168, 336)
BATCHES = (32, 128, 512)
VAZAO, NIVEL = 0, 1

COMMON = dict(
    model="spectral_tf_net",
    output_window=OUTPUT_WINDOW,
    future_exog=True,
    mass_balance_feature=True,
    revin=None,
    uncertainty_weighted=True,
    cascade_weight=1.0,
    message_l1_weight=1e-3,
    max_epochs=150,
    patience=25,
    accelerator="cpu",
    experiment="mimo_sys_spectral_tf_net",
    save=False,
)


def cascade_level(
    cfg: TrainConfig, y_pred: np.ndarray, start: int | None = None
) -> np.ndarray:
    """`nivel_res` from the physics cascade fed by the predicted flow."""
    splits = add_derived_features(cfg, load_splits(cfg.data_prefix))
    raw = np.concatenate([splits["train"], splits["val"], splits["test"]])
    val_end = len(splits["train"]) + len(splits["val"])
    seg = raw[val_end - cfg.input_window :]
    inflow = INPUT_COLS.index("vazao_entrada")
    vz, nv = TARGET_IDX[VAZAO], TARGET_IDX[NIVEL]
    tr = splits["train"]
    a, b = fit_mass_balance(tr[:, nv], tr[:, inflow], tr[:, vz])
    h = cfg.output_window
    first = cfg.input_window if start is None else start
    anchors = np.arange(first, len(seg) - h + 1, h)[: len(y_pred) // h]
    return mass_balance_cascade_forecast(
        seg[anchors - 1, nv],
        # Known future inflow, or the same hours of the previous day
        # (seasonal naive) when `vazao_entrada` is not a known input.
        np.stack(
            [
                seg[t : t + h, inflow]
                if "vazao_entrada" in cfg.future_known
                else seg[t - 24 : t - 24 + h, inflow]
                for t in anchors
            ]
        ),
        y_pred[:, VAZAO].reshape(len(anchors), h),
        a,
        b,
        last_imbalance=seg[anchors - 1, inflow] - seg[anchors - 1, vz],
    ).reshape(-1)


def per_step_r2(
    y_true: np.ndarray, y_pred: np.ndarray, horizon: int
) -> dict[str, list[float]]:
    """R² per output at each horizon step, pooled over windows."""
    yt = y_true.reshape(-1, horizon, 3)
    yp = y_pred.reshape(-1, horizon, 3)
    ss_res = ((yt - yp) ** 2).sum(axis=0)
    ss_tot = ((yt - yt.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1 - ss_res / ss_tot  # (horizon, 3)
    return {
        out: r2[:, k].round(4).tolist()
        for k, out in enumerate(("vazao", "nivel_cascata", "pressao"))
    }


def per_step_all_offsets(
    cfg: TrainConfig, model: torch.nn.Module, preprocessor: object
) -> dict[str, list[float]]:
    """Per-step R² and RMSE with windows starting at every offset.

    The headline evaluation starts a window every `output_window`
    hours, so step `h` always falls on the same hour of the day and
    the per-step curve mixes "horizon" with "hour of day" (pump
    switching hours). Here windows start at all `output_window`
    offsets, so every step covers every hour. R² uses the total test
    variance (not the per-step one); `nivel` goes through the cascade.
    """
    splits = add_derived_features(cfg, load_splits(cfg.data_prefix))
    raw = np.concatenate([splits["train"], splits["val"], splits["test"]])
    val_end = len(splits["train"]) + len(splits["val"])
    seg = raw[val_end - cfg.input_window :]
    h = cfg.output_window
    model = model.cpu().eval()
    sq, y_all = [], []
    for off in range(h):
        start = cfg.input_window + off
        y_pred = chained_forecast(
            model,
            seg,
            preprocessor,
            start=start,
            stop=len(seg),
            device="cpu",
            input_window=cfg.input_window,
            output_window=h,
            future_exog=cfg.future_exog,
            future_known_idx=[INPUT_COLS.index(c) for c in cfg.future_known],
        )
        y_true = seg[start : start + len(y_pred)][:, TARGET_IDX]
        final = y_pred.copy()
        final[:, NIVEL] = cascade_level(cfg, y_pred, start=start)
        sq.append(((y_true - final) ** 2).reshape(-1, h, 3))
        y_all.append(y_true)
    mse = np.concatenate(sq).mean(axis=0)  # (h, 3)
    var = np.concatenate(y_all).var(axis=0)
    names = ("vazao", "nivel_cascata", "pressao")
    return {
        **{
            f"r2_{o}": (1 - mse[:, k] / var[k]).round(4).tolist()
            for k, o in enumerate(names)
        },
        **{
            f"rmse_{o}": np.sqrt(mse[:, k]).round(4).tolist()
            for k, o in enumerate(names)
        },
    }


def run(name: str, **overrides: object) -> None:
    """Train one config and append its metrics to `RESULTS`."""
    done = set()
    if RESULTS.exists():
        done = {json.loads(line)["run"] for line in RESULTS.open()}
    if name in done:
        print(f"skip {name}", flush=True)
        return
    cfg = TrainConfig(run_name=name, **{**COMMON, **overrides})
    result = train(cfg)
    y_true, y_pred = result.y_true, result.y_pred
    final = y_pred.copy()
    final[:, NIVEL] = cascade_level(cfg, y_pred)
    r2_raw = metrics_table(y_true, y_pred)["r2"]
    r2_final = metrics_table(y_true, final)["r2"]
    h = cfg.output_window
    err = (y_pred[:, VAZAO] - y_true[:, VAZAO]).reshape(-1, h)
    bias_share = (err.mean(1) ** 2).mean() / (err**2).mean()
    core = result.module.model
    while hasattr(core, "model"):
        core = core.model
    # Nonlinear share over the whole test split, not one window.
    _, windows, _, _ = _windows(cfg, load_splits(cfg.data_prefix))
    with torch.no_grad():
        result.module.model.cpu().eval()(
            torch.as_tensor(windows["test"][0], dtype=torch.float32)
        )
    PREDS.mkdir(parents=True, exist_ok=True)
    np.savez(PREDS / f"{name}.npz", y_true=y_true, y_pred=final)
    row = {
        "run": name,
        "input_window": cfg.input_window,
        "batch_size": cfg.batch_size,
        **{k: v for k, v in overrides.items() if k != "model_kwargs"},
        "model_kwargs": cfg.model_kwargs,
        "r2_vazao": float(r2_raw.iloc[VAZAO]),
        "r2_nivel_direto": float(r2_raw.iloc[NIVEL]),
        "r2_nivel_cascata": float(r2_final.iloc[NIVEL]),
        "r2_pressao": float(r2_raw.iloc[2]),
        "r2_medio_cascata": float(r2_final.mean()),
        "vies_janela_vazao": float(bias_share),
        "nonlinearity_share": core.nonlinearity_share(),
        # R² at each horizon step (nível through the cascade).
        "r2_por_passo": per_step_r2(y_true, final, h),
        "por_passo_todos_offsets": per_step_all_offsets(
            cfg, result.module.model, result.preprocessor
        ),
        "band_strength": core.band_strength().tolist(),
    }
    RESULTS.parent.mkdir(exist_ok=True)
    with RESULTS.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(
        f"{name}: médio {row['r2_medio_cascata']:.3f} "
        f"(vazão {row['r2_vazao']:.3f}, nível {row['r2_nivel_cascata']:.3f}"
        f", pressão {row['r2_pressao']:.3f}, viés {bias_share:.2f})",
        flush=True,
    )


def grid(batches: tuple[int, ...] = BATCHES) -> None:
    """Window x batch, 1 seed, regime gate on, purely linear."""
    for window in WINDOWS:
        for batch in batches:
            run(
                f"stf_w{window}_b{batch}",
                input_window=window,
                batch_size=batch,
                model_kwargs=dict(n_bands=8, n_regimes=3),
            )


def ablation(window: int, batch: int) -> None:
    """Around the best grid cell: what each piece buys, 3 seeds each."""
    base = dict(
        input_window=window,
        batch_size=batch,
        model_kwargs=dict(n_bands=8, n_regimes=3),
    )
    variants = {
        "full": {},
        "no_future": dict(
            future_exog=False,
            model_kwargs=dict(n_bands=8, n_regimes=3, future_in_input=False),
        ),
        "no_regime": dict(model_kwargs=dict(n_bands=8, n_regimes=1)),
        "no_cascade_loss": dict(cascade_weight=0.0),
        "residual": dict(
            model_kwargs=dict(n_bands=8, n_regimes=3, residual_hidden=16)
        ),
    }
    for seed in (42, 43, 44):
        for tag, extra in variants.items():
            run(
                f"stf_abl_{tag}_w{window}_b{batch}_s{seed}",
                seed=seed,
                **{**base, **extra},
            )


def horizons(window: int, batch: int) -> None:
    """Train for H=6 and H=12, linear and with residual, 3 seeds.

    Every run also reports h=1 (one-step-ahead) from its per-step R².
    """
    for seed in (42, 43, 44):
        for h in (6, 12):
            for tag, residual in (("lin", 0), ("res", 16)):
                run(
                    f"stf_h{h}_{tag}_w{window}_b{batch}_s{seed}",
                    seed=seed,
                    input_window=window,
                    batch_size=batch,
                    output_window=h,
                    model_kwargs=dict(
                        n_bands=8, n_regimes=1, residual_hidden=residual
                    ),
                )


def paper(data_prefix: str = "mimo_full_sample", tag_prefix: str = "") -> None:
    """The reference paper's protocol: order 3, one step, `u[k-1]` only.

    Ferrari et al. (2025), Eq. 15: ARMAX with lags k-1..k-3 for every
    variable, one-step-ahead prediction. No known future inputs here
    (the paper never sees `u[k]`).
    """
    for seed in (42, 43, 44):
        for tag, residual in (("lin", 0), ("res", 16)):
            run(
                f"stf_paper{tag_prefix}_w3_h1_{tag}_s{seed}",
                seed=seed,
                data_prefix=data_prefix,
                input_window=3,
                output_window=1,
                batch_size=32,
                future_exog=False,
                model_kwargs=dict(
                    n_bands=4,
                    n_regimes=1,
                    residual_hidden=residual,
                    future_in_input=False,
                ),
            )


if __name__ == "__main__":
    if sys.argv[1] == "grid":
        grid(tuple(int(b) for b in sys.argv[2:]) or BATCHES)
    elif sys.argv[1] == "horizons":
        horizons(int(sys.argv[2]), int(sys.argv[3]))
    elif sys.argv[1] == "perstep":
        # Re-run the four headline configs (seed 42, deterministic) so
        # their rows carry the all-offsets per-step curve.
        for h in (6, 12):
            for tag, residual in (("lin", 0), ("res", 16)):
                run(
                    f"stf_perstep_h{h}_{tag}_w168_b128_s42",
                    seed=42,
                    input_window=168,
                    batch_size=128,
                    output_window=h,
                    model_kwargs=dict(
                        n_bands=8, n_regimes=1, residual_hidden=residual
                    ),
                )
    elif sys.argv[1] == "no_future_inflow":
        # Only the pump schedule is known ahead; the inflow is not, and
        # the cascade uses the inflow of the same hours one day earlier.
        for seed in [int(a) for a in sys.argv[2:]] or (42, 43, 44):
            for h, tag, residual in (
                (12, "lin", 0),
                (12, "res", 16),
                (6, "res", 16),
            ):
                run(
                    f"stf_nofutin_h{h}_{tag}_w168_b128_s{seed}",
                    seed=seed,
                    input_window=168,
                    batch_size=128,
                    output_window=h,
                    future_known=("freq_b1", "freq_b2", "freq_b3"),
                    model_kwargs=dict(
                        n_bands=8,
                        n_regimes=1,
                        residual_hidden=residual,
                        known_future_idx=(0, 1, 2),
                    ),
                )
    elif sys.argv[1] == "paper":
        paper()
    elif sys.argv[1] == "paper_split":
        # Train and test on the paper's own window (550:1150 / 1500:2000;
        # the last 15% of its training rows are the early-stopping val).
        # The 1150:1500 gap means the first test window's 3 lags come
        # from the end of the training block: 1 of 501 samples.
        paper("mimo_papper_split", "_recorte")
    else:
        ablation(int(sys.argv[2]), int(sys.argv[3]))
