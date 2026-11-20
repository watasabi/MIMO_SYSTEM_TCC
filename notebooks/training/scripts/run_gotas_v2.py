"""Triagem dos ajustes do GOTAS v2, só na validação.

Plano: `docs/18_plano-gotas-w.md`, Etapas 2 e 3. Cada variante parte da
melhor configuração da busca de cada fold e cenário e é treinada com 3
sementes. O teste nem é carregado: os dados usados são os `_tune`, em
que o bloco de teste é uma cópia da validação.

Duas medidas por treino, nas mesmas origens:
- validação normal: R² médio das 3 saídas em 12 h, todas as origens
  válidas, com a combinação do nível ajustada em metades cruzadas (o
  objetivo da busca);
- validação sob estresse: a mesma, com falhas simuladas nas entradas da
  validação (zero, valor preso e picos de saturação, os tipos de
  `docs/1_analise-dados-reservatorio.md` §2), comparada com os valores
  reais. As colunas derivadas são recalculadas sobre as linhas com
  falha, como aconteceria em operação.

Uso (da raiz do repo):
    S=notebooks/training/scripts/run_gotas_v2.py
    R="uv run --extra ml --extra tune python $S"
    $R screen <base> <variante> [cenario] [fold]
    $R report
    $R final <nome> <base> <variante> <cenario>   # abre o teste
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace

import numpy as np
import optuna
import run_improvements
import torch
from run_cv_all import FINAL_SEEDS, N_FOLDS, OUT, pred_path
from run_cv_benchmark import (
    FOLDS,
    SCENARIOS,
    base_config,
    isolate_mlflow,
    memory_guard,
    neural_config,
    usable,
)
from run_gotas_v2_status import report
from run_improvements import _crossfit_val, _split  # noqa: PLC2701

from mimo_sys.training.augmentations import Batch, Compose
from mimo_sys.training.data import FEATURE_COLS
from mimo_sys.training.evaluation import r2_per_output
from mimo_sys.training.train import train

# O pacote `mimo_sys.training` exporta a função `train` com o nome do
# módulo; o módulo em si fica em `sys.modules`.
train_module = sys.modules["mimo_sys.training.train"]

RESULTS = OUT / "gotas_v2_screen.jsonl"
STORAGE = f"sqlite:///{OUT / 'optuna.db'}"
SATURATION = 199.6
FAILURE_P = 0.3
MAX_FAILURE_ROWS = 24
# Variante -> mudanças na configuração do modelo ou do treino.
VARIANTS: dict[str, dict] = {
    "base": {},
    "centro_mediana": {"model_kwargs": {"centre": "median"}},
    "centro_ultimo": {"model_kwargs": {"centre": "last"}},
    "limite_detalhes": {"model_kwargs": {"detail_limit": 3.0}},
    "balanco_168": {"cfg": {"mass_balance_window": 168}},
    "falhas_simuladas": {"failure_augment": True},
    # Sem o nó do balanço acumulado: a soma desde o início da série fica
    # fora da faixa do treino depois de cada falha de sensor (diagnóstico
    # com ridge nas entradas do XGBoost). A cascata física não depende
    # dele: usa `fit_mass_balance` sobre os dados brutos.
    "sem_balanco": {"cfg": {"mass_balance_feature": False}},
    "sem_balanco_falhas": {
        "cfg": {"mass_balance_feature": False},
        "failure_augment": True,
    },
    # Resíduo polinomial explícito no lugar do MLP: o modelo inteiro vira
    # equação (critério de explicabilidade, `docs/19`).
    "chebyshev": {"model_kwargs": {"residual_kind": "chebyshev"}},
    # Repetições das variantes do GOTAS 3, só para gravar a parte
    # explicável (`explicavel_r2`), medida depois dessas triagens.
    "base_medido": {},
    "sem_balanco_falhas_medido": {
        "cfg": {"mass_balance_feature": False},
        "failure_augment": True,
    },
}


class FailureInjection:
    """Falhas de sensor simuladas no passado de janelas de treino.

    Em cada janela, com probabilidade `FAILURE_P`, um nó sorteado recebe
    um trecho de até `MAX_FAILURE_ROWS` linhas zerado (0 é o mínimo do
    treino no MinMax), preso no primeiro valor ou no máximo do treino
    (1, um pico de saturação). O alvo não muda: o modelo aprende a
    prever o valor real apesar da falha na entrada.
    """

    def __init__(self, past_len: int) -> None:
        self.past_len = past_len

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Batch:
        x = x.clone()
        b, _, n = x.shape
        hit = torch.rand(b) < FAILURE_P
        for i in torch.nonzero(hit).flatten().tolist():
            width = int(torch.randint(1, MAX_FAILURE_ROWS + 1, (1,)))
            width = min(width, self.past_len - 1)
            start = int(torch.randint(0, self.past_len - width, (1,)))
            node = int(torch.randint(0, n, (1,)))
            kind = int(torch.randint(0, 3, (1,)))
            rows = slice(start, start + width)
            if kind == 0:
                x[i, rows, node] = 0.0
            elif kind == 1:
                x[i, rows, node] = x[i, start, node].item()
            else:
                x[i, rows, node] = 1.0
        return x, y


def corrupt(val: np.ndarray, fold: int) -> np.ndarray:
    """Falhas simuladas nas linhas da validação, fixas por fold.

    Uma queda a zero de 12 h na vazão de entrada, um valor preso de 24 h
    numa saída e três picos de saturação na vazão de distribuição, em
    posições sorteadas com semente fixa.
    """
    rng = np.random.default_rng(1000 + fold)
    out = val.copy()
    n = len(out)
    inflow = FEATURE_COLS.index("vazao_entrada")
    outputs = [
        FEATURE_COLS.index(c)
        for c in ("vazao_distribuicao", "nivel_res", "pressao")
    ]
    start = rng.integers(0, n - 12)
    out[start : start + 12, inflow] = 0.0
    col = int(rng.choice(outputs))
    start = rng.integers(0, n - 24)
    out[start : start + 24, col] = out[start, col]
    for row in rng.choice(n, 3, replace=False):
        out[row, outputs[0]] = SATURATION
    return out


def val_score(arrays: dict, fold: int) -> float:
    """R² médio das 3 saídas na validação, como no objetivo da busca."""
    val = usable(_split(arrays, "val"), FOLDS[fold - 1].train_end)
    return float(r2_per_output(val["true"], _crossfit_val(val)).mean())


def stressed(cfg: object, model: object, pp: object, fold: int) -> dict:
    """`evaluate` com a validação corrompida antes das colunas derivadas."""
    original = run_improvements.load_splits

    def load_corrupted(prefix: str) -> dict:
        splits = dict(original(prefix))
        splits["val"] = corrupt(splits["val"], fold)
        return splits

    run_improvements.load_splits = load_corrupted
    try:
        return run_improvements.evaluate(cfg, model, pp, ("val",))
    finally:
        run_improvements.load_splits = original


def train_variant(  # noqa: PLR0913
    base: str,
    variant: str,
    scenario: str,
    fold: int,
    seed: int,
    tuning: bool,
) -> tuple[object, object]:
    """Melhor configuração da busca do fold, com a mudança da variante.

    `tuning=True` lê os dados `_tune` (o teste nem é carregado); `False`
    lê o fold de verdade, para o treino final avaliado no teste.
    """
    study = optuna.load_study(
        study_name=f"{base}_{scenario}_f{fold}", storage=STORAGE
    )
    change = VARIANTS[variant]
    trial = optuna.trial.FixedTrial(study.best_params)
    cfg = neural_config(
        trial, base, base_config(trial, scenario, fold, seed, tuning)
    )
    cfg = replace(
        cfg,
        model_kwargs={**cfg.model_kwargs, **change.get("model_kwargs", {})},
        **change.get("cfg", {}),
    )
    original_augmentations = train_module._augmentations  # noqa: SLF001
    if change.get("failure_augment"):
        past = cfg.input_window
        train_module._augmentations = lambda _cfg: Compose(  # noqa: SLF001
            [FailureInjection(past)], p=1.0
        )
    try:
        result = train(cfg)
    finally:
        train_module._augmentations = original_augmentations  # noqa: SLF001
    return cfg, result


def final_one(  # noqa: PLR0913
    name: str, base: str, variant: str, scenario: str, fold: int
) -> None:
    """Treino final com o fold real: grava val e teste como `name`.

    As previsões vão para o mesmo lugar e formato das de
    `run_cv_benchmark.py final`, para entrar nos mesmos relatórios.
    """
    for seed in FINAL_SEEDS:
        path = pred_path(name, scenario, fold, seed)
        if path.exists():
            continue
        cfg, result = train_variant(base, variant, scenario, fold, seed, False)
        model = result.module.model.cpu().eval()
        arrays = run_improvements.evaluate(
            cfg, model, result.preprocessor, ("val", "test")
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)
        print("salvo", path.name, flush=True)


def screen_one(base: str, variant: str, scenario: str, fold: int) -> None:
    """Treina a variante com 3 sementes e grava as duas medidas."""
    done = set()
    if RESULTS.exists():
        done = {
            (r["base"], r["variante"], r["cenario"], r["fold"], r["semente"])
            for r in map(json.loads, RESULTS.open())
        }
    for seed in FINAL_SEEDS:
        if (base, variant, scenario, fold, seed) in done:
            continue
        cfg, result = train_variant(base, variant, scenario, fold, seed, True)
        model = result.module.model.cpu().eval()
        clean = run_improvements.evaluate(
            cfg, model, result.preprocessor, ("val",)
        )
        noisy = stressed(cfg, model, result.preprocessor, fold)
        # Os valores reais vêm da validação limpa.
        noisy["val_true"] = clean["val_true"]
        # Parte explicável: o modelo sem o resíduo não linear, que é o
        # grafo (GOTAS) ou a equação exata (GOTAS-AE, `equation`).
        explainable = clean
        if getattr(model, "residual", None) is not None:
            residual, model.residual = model.residual, None
            explainable = run_improvements.evaluate(
                cfg, model, result.preprocessor, ("val",)
            )
            model.residual = residual
        row = {
            "base": base,
            "variante": variant,
            "cenario": scenario,
            "fold": fold,
            "semente": seed,
            "val_r2": val_score(clean, fold),
            "estresse_r2": val_score(noisy, fold),
            "explicavel_r2": val_score(explainable, fold),
        }
        with RESULTS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(row, flush=True)


def main() -> None:
    command = sys.argv[1]
    if command == "report":
        report()
        return
    memory_guard()
    isolate_mlflow()
    if command == "final":
        # final <nome> <base> <variante> <cenario>
        name, base, variant, scenario = sys.argv[2:6]
        for fold in range(1, N_FOLDS + 1):
            final_one(name, base, variant, scenario, fold)
        return
    base, variant = sys.argv[2], sys.argv[3]
    scenarios = [sys.argv[4]] if len(sys.argv) > 4 else list(SCENARIOS)  # noqa: PLR2004
    folds = (
        [int(sys.argv[5])] if len(sys.argv) > 5 else range(1, N_FOLDS + 1)  # noqa: PLR2004
    )
    for scenario in scenarios:
        for fold in folds:
            screen_one(base, variant, scenario, fold)


if __name__ == "__main__":
    main()
