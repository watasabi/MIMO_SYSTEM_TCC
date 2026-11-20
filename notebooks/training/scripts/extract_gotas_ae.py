"""Equação e subgrafo do GOTAS-AE, com a fidelidade, só na validação.

Treina o GOTAS-AE com a melhor configuração da busca do fold (dados
`_tune`: o teste nem é carregado) e extrai:

- a equação: com a atenção fixada na média da validação e sem o resíduo
  não linear, o modelo é uma equação afim exata por aresta e defasagem
  (`GOTASAutoencoder.equation`);
- o subgrafo: a força de cada aresta nó -> saída nessa equação, com as
  arestas abaixo de 10% da mais forte de cada saída podadas.

A fidelidade diz se a equação explica o modelo: R² entre a previsão da
equação e a do modelo completo, e o R² de cada um contra o valor real.

Uso (da raiz do repo):
    .venv/bin/python notebooks/training/scripts/extract_gotas_ae.py \
        <cenario> <fold> [variante] [modelo: gotas_ae | gotas_aex]
"""

from __future__ import annotations

import sys

import numpy as np
import run_improvements
import torch
from run_cv_all import OUT
from run_cv_benchmark import isolate_mlflow, memory_guard
from run_gotas_v2 import train_variant, val_score

from mimo_sys.training.data import FEATURE_COLS

KEEP = 0.1


def r2(true: np.ndarray, pred: np.ndarray) -> float:
    """R² médio das saídas, sobre todas as origens e passos."""
    true = true.reshape(-1, true.shape[-1])
    pred = pred.reshape(true.shape)
    num = ((true - pred) ** 2).sum(0)
    den = ((true - true.mean(0)) ** 2).sum(0)
    return float((1 - num / den).mean())


def main() -> None:  # noqa: PLR0914 - linear report script
    scenario, fold = sys.argv[1], int(sys.argv[2])
    variant = sys.argv[3] if len(sys.argv) > 3 else "base"  # noqa: PLR2004
    base = sys.argv[4] if len(sys.argv) > 4 else "gotas_ae"  # noqa: PLR2004
    memory_guard()
    isolate_mlflow()
    cfg, result = train_variant(
        base, variant, scenario, fold, 42, tuning=True
    )
    model = result.module.model.cpu().eval()
    pp = result.preprocessor

    model.attention_log = []
    full = run_improvements.evaluate(cfg, model, pp, ("val",))
    total = sum(t for t, _ in model.attention_log)
    mean_attention = total / sum(n for _, n in model.attention_log)
    model.attention_log = None

    offset, h, g = model.equation(mean_attention)
    residual = model.residual
    model.residual, model.fixed_attention = None, mean_attention
    eq = run_improvements.evaluate(cfg, model, pp, ("val",))
    # Separa as duas aproximações: só sem o resíduo (atenção de cada
    # origem: a equação LPV, exata) e só com a atenção fixa.
    model.fixed_attention = None
    lpv = run_improvements.evaluate(cfg, model, pp, ("val",))
    model.residual, model.fixed_attention = residual, mean_attention
    fixed = run_improvements.evaluate(cfg, model, pp, ("val",))
    model.fixed_attention = None

    print(f"\n{base} {variant} {scenario} fold {fold}: {cfg.model_kwargs}")
    print(f"  validação, modelo completo: R² {val_score(full, fold):.3f}")
    print(f"  sem o resíduo (equação LPV): R² {val_score(lpv, fold):.3f}")
    print(f"  só a atenção fixa:          R² {val_score(fixed, fold):.3f}")
    print(f"  equação (fixa, sem resíduo): R² {val_score(eq, fold):.3f}")
    fidelity = r2(full["val_pred"], eq["val_pred"])
    print(f"  fidelidade (equação vs modelo): R² {fidelity:.3f}")

    n_nodes = h.shape[-1]
    names = FEATURE_COLS + ["balanco_acumulado"][: n_nodes - len(FEATURE_COLS)]
    targets = [names[i] for i in model.target_idx.tolist()]
    strength, kept = model.subgraph(h, g, model.known_idx, KEEP)
    print(f"\n  Subgrafo (força da aresta; * = mantida, >= {KEEP:.0%}):")
    print("  " + " " * 20 + "".join(f"{n[:12]:>14s}" for n in names))
    for i, target in enumerate(targets):
        cells = "".join(
            f"{strength[i, j]:>13.3f}{'*' if kept[i, j] else ' '}"
            for j in range(n_nodes)
        )
        print(f"  {target:20s}{cells}")

    # Defasagens que mais pesam em cada aresta mantida, passo 1 e 12.
    p = h.shape[2]
    print("\n  Maiores ganhos por aresta (defasagem em horas antes de t):")
    for i, target in enumerate(targets):
        for j in torch.nonzero(kept[i]).flatten().tolist():
            for k in (0, h.shape[0] - 1):
                gains = h[k, i, :, j]
                top = torch.topk(gains.abs(), 3).indices.tolist()
                terms = ", ".join(f"{gains[t]:+.3f}·t-{p - t}" for t in top)
                print(f"    {names[j]} -> {target}, passo {k + 1}: {terms}")

    path = OUT / f"{base}_equacao_{scenario}_f{fold}_{variant}.npz"
    np.savez(
        path,
        offset=offset.numpy(),
        h=h.numpy(),
        g=g.numpy(),
        attention=mean_attention.numpy(),
        strength=strength.numpy(),
        kept=kept.numpy(),
        names=np.array(names),
        targets=np.array(targets),
        fidelity=fidelity,
        # Faixa do MinMax do treino, para levar os ganhos a unidades físicas.
        scale_min=np.asarray(pp.scaler.data_min_),
        scale_max=np.asarray(pp.scaler.data_max_),
        residual_coeffs=(
            model.residual_coeffs.detach().numpy()
            if model.residual_coeffs is not None
            else np.zeros(0)
        ),
    )
    print(f"\n  salvo: {path}")


if __name__ == "__main__":
    main()
