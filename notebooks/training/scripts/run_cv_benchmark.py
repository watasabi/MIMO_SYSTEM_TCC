"""Validação cruzada temporal de todos os modelos do artigo.

Cinco folds de janela crescente, cada um com treino, validação e teste
em ordem cronológica. Os hiperparâmetros de cada modelo, cenário e fold
são escolhidos só na validação: a busca lê um conjunto de arquivos em
que o bloco de teste é uma cópia da validação, de modo que as linhas de
teste nem são carregadas. O teste só é lido por `final`.

Plano: `docs/changes/2026-11-18-validacao-cruzada-temporal.md`.

Uso (da raiz do repo; um processo por vez, pela RAM):
    S=notebooks/training/scripts/run_cv_benchmark.py
    R="uv run --extra ml --extra tune --extra boost python $S"
    $R folds
    $R tune <modelo> <cenario> <fold> [tentativas]
    $R final <modelo> <cenario> <fold>
    $R all [tentativas]
    $R report
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess  # noqa: S404
import sys
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path

import mlflow
import numpy as np
import optuna
import pandas as pd
import torch
from catboost import CatBoostRegressor
from optuna_integration import PyTorchLightningPruningCallback
from run_decomposition import BASE, SCENARIOS
from run_improvements import (
    _blend_weights,  # noqa: PLC2701
    _crossfit_val,  # noqa: PLC2701
    _levels,  # noqa: PLC2701
    _split,  # noqa: PLC2701
    evaluate,
)
from xgboost import XGBRegressor

from mimo_sys.training.benchmarks import (
    TreeForecaster,
    armax_forecast,
    fit_armax,
    forecast_metrics,
    seasonal_scale,
    temporal_folds,
    tree_features,
    valid_origins,
)
from mimo_sys.training.data import (
    DATA_PROCESSED,
    FEATURE_COLS,
    INPUT_IDX,
    PROJECT_ROOT,
    TARGET_IDX,
    load_splits,
)
from mimo_sys.training.evaluation import r2_per_output
from mimo_sys.training.features import bad_sample_mask
from mimo_sys.training.train import (
    TrainConfig,
    _windows,  # noqa: PLC2701
    known_columns,
    train,
)
from mimo_sys.training.tuning import MODEL_SPACES

# `MIMO_CV_OUT` separa testes curtos da rodada oficial.
OUT = Path(os.environ.get("MIMO_CV_OUT", PROJECT_ROOT / "outputs" / "cv"))
STORAGE = f"sqlite:///{OUT / 'optuna.db'}"
EXPERIMENT = "mimo_sys_cv"
H = 12
N_FOLDS = 5
N_TRIALS = 30
FINAL_SEEDS = (42, 43, 44)
TUNE_SEED = 42
BOOST_ROUNDS = 600
BOOST_PATIENCE = 30
BOOST_THREADS = 6
BOOST_BINS = 64
MAX_PROCESS_MB = 3500
MIN_AVAILABLE_MB = 400
MIN_AVAILABLE_PLUS_SWAP_MB = 1000
# Falha do sensor de vazão de entrada que `bad_sample_mask` não marca: a
# entrada medida fica baixa enquanto o nível sobe.
UNFLAGGED_FAILURE = (1445, 1488)
WINDOWS = [24, 48, 96, 168]
NEURAL = [
    "gotas",
    "mlp",
    "lstm",
    "gru",
    "mpnn",
    "sgn",
    "stemgnn",
    "seq2seq_latent_gnn",
    "fourier_gnn",
]
MODELS = [
    "gotas",
    "mlp",
    "lstm",
    "gru",
    "xgboost",
    "catboost",
    "mpnn",
    "sgn",
    "stemgnn",
    "seq2seq_latent_gnn",
    "fourier_gnn",
    "armax",
]
LABELS = {
    "gotas": "GOTAS",
    "mlp": "MLP",
    "lstm": "LSTM",
    "gru": "GRU",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "mpnn": "MPNN",
    "sgn": "Symbolic Graph Network",
    "stemgnn": "StemGNN",
    "seq2seq_latent_gnn": "Seq2SeqLatentGNN",
    "fourier_gnn": "FourierGNN",
    "armax": "ARMAX",
    "persistencia": "Persistência",
    "sazonal": "Sazonal ingênuo",
}


def series() -> pd.DataFrame:
    """A série completa, na ordem cronológica."""
    return pd.concat(
        [
            pd.read_parquet(
                DATA_PROCESSED / f"mimo_full_sample_{name}.parquet"
            )
            for name in ("train", "val", "test")
        ],
        ignore_index=True,
    )[FEATURE_COLS]


DATA = series()
RAW = DATA.to_numpy(dtype=float)
FOLDS = temporal_folds(len(RAW))
BAD = bad_sample_mask(RAW)
BAD[UNFLAGGED_FAILURE[0] : UNFLAGGED_FAILURE[1] + 1] = True


def write_folds() -> None:
    """Grava os arquivos de cada fold, com e sem o teste real."""
    for k, fold in enumerate(FOLDS, start=1):
        parts = {
            "train": DATA.iloc[: fold.train_end],
            "val": DATA.iloc[fold.train_end : fold.val_end],
            "test": DATA.iloc[fold.val_end : fold.test_end],
        }
        for name, part in parts.items():
            part.to_parquet(DATA_PROCESSED / f"mimo_cv5_f{k}_{name}.parquet")
            tune_part = parts["val"] if name == "test" else part
            tune_part.to_parquet(
                DATA_PROCESSED / f"mimo_cv5_f{k}_tune_{name}.parquet"
            )
        print(
            k,
            fold,
            "falhas no teste:",
            BAD[fold.val_end : fold.test_end].sum(),
        )


def memory_guard() -> None:
    """Encerra este processo antes de a falta de memória travar a máquina."""

    def watch() -> None:
        page_mb = os.sysconf("SC_PAGE_SIZE") / 1024**2
        while True:
            info = {
                line.split(":")[0]: int(line.split()[1]) / 1024
                for line in Path("/proc/meminfo")
                .read_text(encoding="ascii")
                .splitlines()
            }
            rss = int(
                Path("/proc/self/statm").read_text(encoding="ascii").split()[1]
            )
            process = rss * page_mb
            available = info["MemAvailable"]
            if (
                process > MAX_PROCESS_MB
                or available < MIN_AVAILABLE_MB
                or available + info["SwapFree"] < MIN_AVAILABLE_PLUS_SWAP_MB
            ):
                print(
                    f"MEMÓRIA: processo {process:.0f} MB, "
                    f"disponível {available:.0f} MB, "
                    f"swap livre {info['SwapFree']:.0f} MB; encerrando",
                    flush=True,
                )
                os._exit(137)
            time.sleep(1)

    threading.Thread(target=watch, daemon=True).start()


def isolate_mlflow() -> None:
    """MLflow temporário: milhares de treinos não vão para o repositório."""
    tmp = Path(tempfile.mkdtemp(prefix="mimo_cv_mlflow_"))
    atexit.register(shutil.rmtree, tmp, ignore_errors=True)
    mlflow.set_tracking_uri(f"sqlite:///{tmp / 'mlflow.db'}")
    mlflow.create_experiment(EXPERIMENT, artifact_location=tmp.as_uri())


def base_config(  # noqa: PLR0913
    trial: optuna.Trial, scenario: str, fold: int, seed: int, tuning: bool
) -> TrainConfig:
    prefix = f"mimo_cv5_f{fold}" + ("_tune" if tuning else "")
    return TrainConfig(
        **{
            **BASE,
            **SCENARIOS[scenario],
            "experiment": EXPERIMENT,
            "data_prefix": prefix,
            "seed": seed,
            "run_name": f"f{fold}_{scenario}_s{seed}",
            "input_window": trial.suggest_categorical("input_window", WINDOWS),
            "message_l1_weight": 0.0,
        }
    )


def neural_config(
    trial: optuna.Trial, model: str, base: TrainConfig
) -> TrainConfig:
    """Espaço de busca das redes: arquitetura mais o treino."""
    l1 = 0.0
    if model == "gotas":
        # Um único regime: mantém a equação legível no modelo final.
        kwargs = {
            "n_bands": trial.suggest_int("n_bands", 4, 16),
            "n_regimes": 1,
            "residual_hidden": trial.suggest_categorical(
                "residual_hidden", [0, 8, 16, 32]
            ),
        }
        l1 = trial.suggest_float("message_l1_weight", 1e-4, 1e-2, log=True)
    else:
        kwargs = MODEL_SPACES[model](trial)
    return replace(
        base,
        model=model,
        model_kwargs=kwargs,
        message_l1_weight=l1,
        lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
        cascade_weight=trial.suggest_float(
            "cascade_weight", 0.1, 3.0, log=True
        ),
    )


def fit_booster(
    trial: optuna.Trial, model: str, cfg: TrainConfig
) -> tuple[TreeForecaster, object]:
    """XGBoost ou CatBoost com saída única de 3 x 12 valores."""
    pp, windows, _, _ = _windows(cfg, load_splits(cfg.data_prefix))
    past, known = cfg.input_window, known_columns(cfg)

    def arrays(split: str) -> tuple[np.ndarray, np.ndarray]:
        x, y = windows[split]
        delta = y - x[:, past - 1, TARGET_IDX][:, None]
        return tree_features(x, past, known), delta.reshape(len(y), -1)

    x_train, y_train = arrays("train")
    x_val, y_val = arrays("val")
    rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    depth = trial.suggest_int("depth", 3, 6)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    columns = trial.suggest_float("colsample", 0.3, 1.0)
    l2 = trial.suggest_float("l2", 1e-3, 10.0, log=True)
    if model == "xgboost":
        booster = XGBRegressor(
            n_estimators=BOOST_ROUNDS,
            early_stopping_rounds=BOOST_PATIENCE,
            multi_strategy="multi_output_tree",
            tree_method="hist",
            max_bin=BOOST_BINS,
            learning_rate=rate,
            max_depth=depth,
            subsample=subsample,
            colsample_bytree=columns,
            min_child_weight=trial.suggest_float(
                "min_child_weight", 1.0, 20.0, log=True
            ),
            reg_lambda=l2,
            n_jobs=BOOST_THREADS,
            random_state=cfg.seed,
        )
        booster.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    else:
        booster = CatBoostRegressor(
            iterations=BOOST_ROUNDS,
            od_type="Iter",
            od_wait=BOOST_PATIENCE,
            loss_function="MultiRMSE",
            bootstrap_type="Bernoulli",
            border_count=BOOST_BINS,
            learning_rate=rate,
            depth=depth,
            subsample=subsample,
            rsm=columns,
            l2_leaf_reg=l2,
            thread_count=BOOST_THREADS,
            random_seed=cfg.seed,
            allow_writing_files=False,
            verbose=False,
        )
        booster.fit(x_train, y_train, eval_set=(x_val, y_val))
    return TreeForecaster(booster.predict, past, H, known), pp


def armax_arrays(
    trial: optuna.Trial,
    scenario: str,
    fold: int,
    splits: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """ARMAX reestimado no treino do fold; ordens vindas da busca."""
    order = tuple(
        trial.suggest_int(name, 1, 5) for name in ("n_a", "n_b", "n_c")
    )
    f = FOLDS[fold - 1]
    last = f.test_end if "test" in splits else f.val_end
    y, u = RAW[:last][:, TARGET_IDX], RAW[:last][:, INPUT_IDX]
    model = fit_armax(
        y[: f.train_end], u[: f.train_end], order, BAD[: f.train_end]
    )
    starts = {"val": f.train_end, "test": f.val_end}
    out = {}
    for split in splits:
        origins = starts[split] + np.arange(220 - H + 1)
        pred = armax_forecast(
            model, y, u, origins, H, inflow_known=scenario == "full"
        )
        out[f"{split}_true"] = np.stack([y[o : o + H] for o in origins])
        out[f"{split}_pred"] = pred
        out[f"{split}_nivel_cascata"] = pred[..., 1]
    return out


def fit_predict(  # noqa: PLR0913
    trial: optuna.Trial,
    model: str,
    scenario: str,
    fold: int,
    seed: int,
    splits: tuple[str, ...],
    callbacks: list | None = None,
) -> dict[str, np.ndarray]:
    """Treina um candidato e devolve as previsões de `splits`."""
    if model == "armax":
        return armax_arrays(trial, scenario, fold, splits)
    base = base_config(trial, scenario, fold, seed, "test" not in splits)
    if model in NEURAL:
        cfg = neural_config(trial, model, base)
        result = train(cfg, extra_callbacks=callbacks)
        net, pp = result.module.model.cpu().eval(), result.preprocessor
    else:
        cfg = replace(base, cascade_weight=0.0)
        net, pp = fit_booster(trial, model, cfg)
    return evaluate(cfg, net, pp, splits)


def usable(split: dict, start: int) -> dict:
    """Só as origens cujo horizonte não toca falha de sensor."""
    keep = valid_origins(BAD, start, len(split["true"]), H)
    return {k: v[keep] for k, v in split.items()}


def study_of(model: str, scenario: str, fold: int) -> optuna.Study:
    return optuna.create_study(
        study_name=f"{model}_{scenario}_f{fold}",
        storage=STORAGE,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=TUNE_SEED),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=10
        ),
    )


def tune(model: str, scenario: str, fold: int, n_trials: int) -> None:
    """Busca retomável; o objetivo é o R² médio de 12 h na validação."""
    start = FOLDS[fold - 1].train_end
    study = study_of(model, scenario, fold)

    def objective(trial: optuna.Trial) -> float:
        callbacks = (
            [PyTorchLightningPruningCallback(trial, monitor="val_r2")]
            if model in NEURAL
            else None
        )
        arrays = fit_predict(
            trial, model, scenario, fold, TUNE_SEED, ("val",), callbacks
        )
        val = usable(_split(arrays, "val"), start)
        return float(r2_per_output(val["true"], _crossfit_val(val)).mean())

    finished = (
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL,
    )
    done = len(study.get_trials(deepcopy=False, states=finished))
    study.optimize(
        objective,
        n_trials=max(0, n_trials - done),
        catch=(RuntimeError, ValueError),
        gc_after_trial=True,
    )
    print(study.study_name, study.best_value, study.best_params, flush=True)


def pred_path(model: str, scenario: str, fold: int, seed: int) -> Path:
    return OUT / "preds" / f"{model}_{scenario}_f{fold}_s{seed}.npz"


def final(model: str, scenario: str, fold: int) -> None:
    """Melhor configuração do fold, avaliada no teste com 3 sementes."""
    best = study_of(model, scenario, fold).best_params
    (OUT / "preds").mkdir(parents=True, exist_ok=True)
    seeds = FINAL_SEEDS[:1] if model == "armax" else FINAL_SEEDS
    for seed in seeds:
        path = pred_path(model, scenario, fold, seed)
        if path.exists():
            continue
        trial = optuna.trial.FixedTrial(best)
        arrays = fit_predict(
            trial, model, scenario, fold, seed, ("val", "test")
        )
        np.savez(path, **arrays)
        print("salvo", path.name, flush=True)


def run_all(n_trials: int) -> None:
    """Toda a rodada, um processo por etapa, retomável."""
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        for scenario in SCENARIOS:
            for fold in range(1, N_FOLDS + 1):
                for step in ("tune", "final"):
                    cmd = [
                        sys.executable,
                        __file__,
                        step,
                        model,
                        scenario,
                        str(fold),
                    ]
                    if step == "tune":
                        cmd.append(str(n_trials))
                    name = f"{step}_{model}_{scenario}_f{fold}"
                    began = time.time()
                    with (logs / f"{name}.log").open("a") as log:
                        code = subprocess.run(  # noqa: S603
                            cmd,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=False,
                        ).returncode
                    print(
                        f"{name}: código {code}, "
                        f"{(time.time() - began) / 60:.1f} min",
                        flush=True,
                    )


def scores(true: np.ndarray, pred: np.ndarray, fold: int) -> dict:
    """Métricas de um teste em 1, 6 e 12 passos e R² por passo."""
    f = FOLDS[fold - 1]
    n = len(true)
    scale = seasonal_scale(RAW[: f.train_end][:, TARGET_IDX])
    keep = valid_origins(BAD, f.val_end, n, 1)
    out = {"1": forecast_metrics(true[keep, 0], pred[keep, 0], scale)}
    for h in (6, 12):
        idx = np.arange(0, n, h)
        idx = idx[valid_origins(BAD, f.val_end, n, h)[idx]]
        out[str(h)] = forecast_metrics(
            true[idx, :h].reshape(-1, 3), pred[idx, :h].reshape(-1, 3), scale
        )
    keep = valid_origins(BAD, f.val_end, n, H)
    t, p = true[keep], pred[keep]
    var = t.reshape(-1, 3).var(axis=0)
    out["por_passo"] = {"r2": (1 - ((t - p) ** 2).mean(axis=0) / var).mean(1)}
    return out


def reference_forecasts(fold: int) -> dict[str, np.ndarray]:
    f = FOLDS[fold - 1]
    y = RAW[:, TARGET_IDX]
    rows = f.val_end + np.arange(220 - H + 1)
    return {
        "true": np.stack([y[r : r + H] for r in rows]),
        "persistencia": np.stack(
            [np.repeat(y[r - 1][None], H, 0) for r in rows]
        ),
        "sazonal": np.stack([y[r - 24 : r - 24 + H] for r in rows]),
    }


def model_scores(model: str, scenario: str, fold: int) -> dict | None:
    """Métricas do fold na média das sementes disponíveis."""
    start = FOLDS[fold - 1].train_end
    runs = []
    for seed in FINAL_SEEDS:
        path = pred_path(model, scenario, fold, seed)
        if not path.exists():
            continue
        arrays = dict(np.load(path))
        weights = _blend_weights(usable(_split(arrays, "val"), start))
        test = _split(arrays, "test")
        runs.append(scores(test["true"], _levels(test, weights), fold))
    if not runs:
        return None
    return {
        proto: {
            name: np.mean([r[proto][name] for r in runs], axis=0)
            for name in runs[0][proto]
        }
        for proto in runs[0]
    }


def summarize(per_fold: list[dict]) -> dict:
    """Média e desvio entre folds, por saída e na média das saídas."""
    out = {}
    for proto in per_fold[0]:
        out[proto] = {}
        for name in per_fold[0][proto]:
            values = np.stack([m[proto][name] for m in per_fold])
            out[proto][name] = {
                "media": values.mean(axis=0).round(5).tolist(),
                "desvio": values.std(axis=0).round(5).tolist(),
                "media_saidas": round(float(values.mean()), 5),
                "desvio_saidas": round(float(values.mean(1).std()), 5),
            }
    out["folds"] = len(per_fold)
    return out


def report() -> None:
    """Resume os testes e imprime a tabela de R² e erro em 12 h."""
    summary: dict = {}
    for scenario in SCENARIOS:
        refs = [reference_forecasts(k) for k in range(1, N_FOLDS + 1)]
        table: dict = {}
        for model in ("persistencia", "sazonal"):
            table[model] = summarize(
                [
                    scores(r["true"], r[model], k)
                    for k, r in enumerate(refs, start=1)
                ]
            )
        for model in MODELS:
            found = [
                m
                for k in range(1, N_FOLDS + 1)
                if (m := model_scores(model, scenario, k)) is not None
            ]
            if len(found) == N_FOLDS:
                table[model] = summarize(found)
        summary[scenario] = table
        print(f"\n## {scenario} (12 h, média ± desvio entre folds)")
        ranked = sorted(
            table,
            key=lambda m: -table[m]["12"]["r2"]["media_saidas"],
        )
        for model in ranked:
            m = table[model]["12"]
            print(
                f"{LABELS[model]:24s} "
                f"R² {m['r2']['media_saidas']:.3f} "
                f"± {m['r2']['desvio_saidas']:.3f} | "
                f"MASE {m['mase']['media_saidas']:.2f} | "
                f"KGE {m['kge']['media_saidas']:.3f}"
            )
    best = {
        s.study_name: {"valor": s.best_trial.value, **s.best_trial.params}
        for s in (
            optuna.load_study(study_name=name, storage=STORAGE)
            for name in optuna.get_all_study_names(STORAGE)
        )
        if s.get_trials(
            deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)
        )
    }
    summary["hiperparametros"] = best
    (OUT / "report.json").write_text(json.dumps(summary, indent=1))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    command, *args = sys.argv[1:]
    if command == "folds":
        write_folds()
    elif command == "all":
        run_all(int(args[0]) if args else N_TRIALS)
    elif command == "report":
        report()
    else:
        os.nice(10)
        memory_guard()
        torch.set_float32_matmul_precision("high")
        isolate_mlflow()
        model, scenario, fold = args[0], args[1], int(args[2])
        if command == "tune":
            trials = int(args[3]) if len(args) > 3 else N_TRIALS  # noqa: PLR2004
            tune(model, scenario, fold, trials)
        else:
            final(model, scenario, fold)


if __name__ == "__main__":
    main()
