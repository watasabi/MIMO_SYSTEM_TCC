"""External black-box benchmarks: the reachable ceiling, never the model.

Same splits and forecast origins as `run_decomposition.py` (every hour
of val and test, 12 h ahead). Each forecast uses only the rows before
its origin plus the horizon values of the known inputs (pumps; plus the
inflow in `full`). Saves `outputs/benchmarks/<model>_<scenario>.npz`
with `{val,test}_{true,pred,origins}`; statistics run in the main
project.

Uso (de models/benchmarks):
    uv run python run_benchmarks.py lgbm|nhits|tide [stride]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "benchmarks"
INPUTS = ["freq_b1", "freq_b2", "freq_b3", "vazao_entrada"]
OUTPUTS = ["vazao_distribuicao", "nivel_res", "pressao"]
COLS = INPUTS + OUTPUTS
H, LAGS = 12, 24
KNOWN = {"full": INPUTS, "pumps": INPUTS[:3]}
START = pd.Timestamp("2014-07-27 01:00")
# Physical limits of `mimo_sys.training.features.bad_sample_mask`.
MIN_LEVEL, MIN_PRESSURE, MAX_JUMP, SATURATION = 1.0, 5.0, 0.3, 199.0


def load() -> tuple[pd.DataFrame, int, int]:
    parts = [
        pd.read_parquet(DATA / f"mimo_full_sample_{s}.parquet")[COLS]
        for s in ("train", "val", "test")
    ]
    full = pd.concat(parts, ignore_index=True)
    full.insert(0, "ds", START + pd.to_timedelta(np.arange(len(full)), "h"))
    return full, len(parts[0]), len(parts[1])


def bad_rows(df: pd.DataFrame) -> np.ndarray:
    """Same physical rules as `mimo_sys.training.features.bad_sample_mask`."""
    bad = (
        (df.nivel_res < MIN_LEVEL)
        | (df.pressao < MIN_PRESSURE)
        | (df.vazao_distribuicao < 0)
        | (df.vazao_distribuicao >= SATURATION)
        | (df.vazao_entrada >= SATURATION + 1)
        | (df.nivel_res.diff().abs() > MAX_JUMP)
    ).to_numpy()
    idx = np.flatnonzero(bad)
    for k in range(-3, 4):
        bad[np.clip(idx + k, 0, len(bad) - 1)] = True
    return bad


def origins(n_tr: int, n_va: int, n: int, stride: int) -> dict:
    return {
        "val": np.arange(n_tr, n_tr + n_va - H + 1, stride),
        "test": np.arange(n_tr + n_va, n - H + 1, stride),
    }


def truth(values: np.ndarray, orig: np.ndarray) -> np.ndarray:
    idx = [COLS.index(c) for c in OUTPUTS]
    return np.stack([values[o : o + H][:, idx] for o in orig])


def lgbm(full: pd.DataFrame, n_tr: int, known: list[str], orig: dict) -> dict:
    """Direct multivariate LightGBM, one model per (output, step).

    Features at origin `o`: lags 1..24 of every column, the known
    inputs at `o + h` and their sum over `o .. o + h` (the integrated
    inflow matters for the level), hour of day of `o + h`. Target:
    change from the last observed value. Trained on clean training
    origins only.
    """
    from lightgbm import LGBMRegressor  # noqa: PLC0415

    v = full[COLS].to_numpy()
    k_idx = [COLS.index(c) for c in known]
    hour = full.ds.dt.hour.to_numpy()
    bad = bad_rows(full)

    def feats(o: int, h: int) -> np.ndarray:
        fut = v[o : o + h + 1, k_idx]
        ang = 2 * np.pi * hour[o + h] / 24
        return np.concatenate(
            [
                v[o - LAGS : o].ravel(),
                fut[-1],
                fut.sum(0),
                [np.sin(ang), np.cos(ang)],
            ]
        )

    train_o = [
        o for o in range(LAGS, n_tr - H + 1) if not bad[o - LAGS : o + H].any()
    ]
    preds = {s: np.zeros((len(o_), H, 3)) for s, o_ in orig.items()}
    for j, name in enumerate(OUTPUTS):
        col = COLS.index(name)
        for h in range(H):
            X = np.stack([feats(o, h) for o in train_o])
            y = np.array([v[o + h, col] - v[o - 1, col] for o in train_o])
            model = LGBMRegressor(
                n_estimators=400,
                learning_rate=0.03,
                num_leaves=15,
                min_child_samples=20,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.5,
                verbose=-1,
            ).fit(X, y)
            for split, o_ in orig.items():
                Xs = np.stack([feats(o, h) for o in o_])
                preds[split][:, h, j] = model.predict(Xs) + v[o_ - 1, col]
    return preds


def neural(
    full: pd.DataFrame, n_tr: int, known: list[str], orig: dict, kind: str
) -> dict:
    """neuralforecast NHITS/TiDE, global over the 3 outputs.

    History of every column as `hist_exog`, known inputs as
    `futr_exog`; trained once on the training rows, then rolled over
    val+test (`cross_validation(step_size=1, refit=False)`).
    """
    from neuralforecast import NeuralForecast  # noqa: PLC0415
    from neuralforecast.models import NHITS, TiDE  # noqa: PLC0415

    hist = [c for c in COLS if c not in known]
    rows = []
    for name in OUTPUTS:
        part = full[["ds", *COLS]].copy()
        part["unique_id"] = name
        part["y"] = full[name]
        rows.append(part)
    df = pd.concat(rows, ignore_index=True)
    common = dict(
        h=H,
        input_size=168,
        futr_exog_list=known,
        hist_exog_list=hist,
        scaler_type="robust",
        max_steps=1500,
        learning_rate=1e-3,
        batch_size=32,
        random_seed=42,
        enable_progress_bar=False,
        logger=False,
    )
    model = NHITS(**common) if kind == "nhits" else TiDE(**common)
    nf = NeuralForecast(models=[model], freq="h")
    n = len(full)
    cv = nf.cross_validation(
        df=df, test_size=n - n_tr, step_size=1, refit=False, n_windows=None
    )
    col = type(model).__name__
    if col not in cv.columns:  # the forecast column is named after the model
        raise KeyError(f"{col} not in {list(cv.columns)}")
    preds = {}
    for split, o_ in orig.items():
        out = np.zeros((len(o_), H, 3))
        cutoff_ds = full.ds.to_numpy()[o_ - 1]
        for j, name in enumerate(OUTPUTS):
            sub = cv[cv.unique_id == name].set_index(["cutoff", "ds"])[col]
            for i, c in enumerate(cutoff_ds):
                out[i, :, j] = sub.loc[c].to_numpy()[:H]
        preds[split] = out
    return preds


def main(kind: str, stride: int) -> None:
    full, n_tr, n_va = load()
    orig = origins(n_tr, n_va, len(full), stride)
    values = full[COLS].to_numpy()
    OUT.mkdir(parents=True, exist_ok=True)
    for scenario, known in KNOWN.items():
        path = OUT / f"{kind}_{scenario}.npz"
        if path.exists():
            print("skip", path.name)
            continue
        if kind == "lgbm":
            preds = lgbm(full, n_tr, known, orig)
        elif kind in {"nhits", "tide"}:
            preds = neural(full, n_tr, known, orig, kind)
        else:
            raise ValueError(f"unknown benchmark: {kind}")
        arrays = {}
        for split, o_ in orig.items():
            arrays[f"{split}_origins"] = o_
            arrays[f"{split}_true"] = truth(values, o_)
            arrays[f"{split}_pred"] = preds[split]
        np.savez(path, **arrays)
        print("saved", path.name, flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0], int(args[1]) if len(args) > 1 else 1)
