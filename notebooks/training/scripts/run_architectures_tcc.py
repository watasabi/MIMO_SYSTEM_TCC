"""As outras arquiteturas no protocolo do artigo (o do GOTAS).

Mesmo contrato de produção, split, janela de 168 h, horizonte de 12 h,
entradas futuras conhecidas, loss, cenários (`full`/`pumps`), 5 seeds e
combinação do nível de `run_improvements.py`; só a arquitetura muda. O
`message_l1` (lasso por grupo do grafo do GOTAS) fica desligado nelas.
Hiperparâmetros de cada arquitetura: os de `run_collection_v2.py` e
`run_fourier_gnn_grid.py`.

O relatório usa os mesmos protocolos de `report_lean_model.py`: 1 passo
com valores medidos, simulação livre de 6 e 12 passos (origens que não
se sobrepõem) e R² por passo.

Uso (da raiz do repo; um processo por treino, pela RAM):
    S=notebooks/training/scripts/run_architectures_tcc.py
    uv run --extra ml python $S run <variante> <cenario> <seed>
    uv run --extra ml python $S report
"""

from __future__ import annotations

import json
import sys

import numpy as np
import run_improvements
from report_spectral_tf_net import (
    OUR_TEST_START,
    Y,
    all_metrics,
    armax_free_run,
    armax_one_step,
)
from run_improvements import (
    OUT,
    SEEDS,
    _blend_weights,  # noqa: PLC2701
    _levels,  # noqa: PLC2701
    _split,  # noqa: PLC2701
    run,
)

from mimo_sys.training.data import PROJECT_ROOT

SUMMARY = PROJECT_ROOT / "outputs" / "architectures_tcc.json"
H = 12
N_ORIGINS = 321
ARCHS: dict[str, dict] = {
    "A_mpnn": {
        "model": "mpnn",
        "model_kwargs": {"hidden_size": 64, "message_dim": 8, "n_layers": 1},
    },
    "A_sgn": {
        "model": "sgn",
        "model_kwargs": {"hidden_size": 64, "message_dim": 8, "n_layers": 1},
    },
    "A_stemgnn": {"model": "stemgnn", "model_kwargs": {"hidden_size": 64}},
    "A_seq2seq_latent_gnn": {
        "model": "seq2seq_latent_gnn",
        "model_kwargs": {"hidden_size": 64},
    },
    "A_fourier_gnn": {
        "model": "fourier_gnn",
        "model_kwargs": {"embed_dim": 32, "n_layers": 3, "ffn_hidden": 64},
    },
}
LABELS = {
    "A_mpnn": "MPNN",
    "A_sgn": "SymbolicGraphNetwork",
    "A_stemgnn": "StemGNN",
    "A_seq2seq_latent_gnn": "Seq2SeqLatentGNN",
    "A_fourier_gnn": "FourierGNN",
    "I0_ref": "GOTAS",
}
for _arch in ARCHS.values():
    _arch["message_l1_weight"] = 0.0
run_improvements.VARIANTS.update(ARCHS)


def seed_preds(variant: str, scenario: str, seed: int) -> dict:
    """Test forecasts with the level blend fitted on val (as the GOTAS)."""
    d = dict(np.load(OUT / "preds" / f"{variant}_{scenario}_s{seed}.npz"))
    val, test = _split(d, "val"), _split(d, "test")
    return {
        "true": test["true"],
        "final": _levels(test, _blend_weights(val)),
        "direto": test["pred"],
    }


def protocols(truth: np.ndarray, pred: np.ndarray) -> dict:
    """R² per output in each protocol of the article, and per step."""
    out = {"1_passo": all_metrics(truth[:, 0], pred[:, 0])["r2"]}
    for h in (6, 12):
        idx = np.arange(N_ORIGINS)[::h]
        out[f"{h}_passos"] = all_metrics(
            truth[idx, :h].reshape(-1, 3), pred[idx, :h].reshape(-1, 3)
        )["r2"]
    var = truth.reshape(-1, 3).var(0)
    out["por_passo"] = (1 - ((truth - pred) ** 2).mean(0) / var).mean(1)
    return out


def report() -> None:  # noqa: PLR0914 - linear on purpose
    rows = OUR_TEST_START + np.arange(N_ORIGINS)
    y = Y.T
    truth = np.stack([y[r : r + H] for r in rows])
    refs = {
        "Persistência": np.stack(
            [np.repeat(y[r - 1][None], H, 0) for r in rows]
        ),
        "Sazonal ingênuo": np.stack([y[r - 24 : r - 24 + H] for r in rows]),
        "ARMAX MOGWO": np.stack([armax_free_run(r, H) for r in rows]),
    }
    armax_1 = armax_one_step(rows[0], rows[-1] + 1)
    out: dict = {}
    for scenario in ("full", "pumps"):
        res: dict = {}
        for name, p in refs.items():
            res[name] = protocols(truth, p)
        res["ARMAX MOGWO"]["1_passo"] = all_metrics(truth[:, 0], armax_1)["r2"]
        for variant in [*ARCHS, "I0_ref"]:
            files = [
                OUT / "preds" / f"{variant}_{scenario}_s{s}.npz" for s in SEEDS
            ]
            done = [s for s, f in zip(SEEDS, files, strict=True) if f.exists()]
            if not done:
                continue
            runs = [seed_preds(variant, scenario, s) for s in done]
            assert all(np.allclose(r["true"], truth) for r in runs)
            for mode in ("final", "direto"):
                per_seed = [protocols(truth, r[mode]) for r in runs]
                entry = {
                    k: np.mean([m[k] for m in per_seed], 0)
                    for k in per_seed[0]
                }
                entry["desvio_12"] = float(
                    np.std([m["12_passos"].mean() for m in per_seed])
                )
                entry["seeds"] = done
                suffix = "" if mode == "final" else " (nível direto)"
                res[LABELS[variant] + suffix] = entry
        out[scenario] = res
    SUMMARY.write_text(json.dumps(out, indent=1, default=_json))
    print_tables(out)


def _json(o: object) -> object:
    if isinstance(o, np.ndarray):
        return o.round(4).tolist()
    if isinstance(o, np.floating):
        return round(float(o), 4)
    raise TypeError(type(o))


def print_tables(out: dict) -> None:
    for scenario, res in out.items():
        print(f"\n## {scenario}")
        print("| Modelo | 1 passo | 6 passos | 12 passos (V/N/P) | 12 médio |")
        for name, m in res.items():
            if name.endswith("(nível direto)"):
                continue
            r12 = "/".join(f"{v:.3f}" for v in m["12_passos"])
            sd = m.get("desvio_12")
            sd_txt = f" ± {sd:.3f}" if sd is not None else ""
            print(
                f"| {name} | {m['1_passo'].mean():.3f} | "
                f"{m['6_passos'].mean():.3f} | {r12} | "
                f"{m['12_passos'].mean():.3f}{sd_txt} |"
            )


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report()
    else:
        run(sys.argv[2], sys.argv[3], int(sys.argv[4]))
