"""As outras arquiteturas no protocolo do artigo (o do GOTAS).

Mesmo contrato de produção, split, janela de 168 h, horizonte de 12 h,
entradas futuras conhecidas, loss, cenários (`full`/`pumps`), 5 seeds e
combinação do nível de `run_improvements.py`; só a arquitetura muda. O
`message_l1` (lasso por grupo do grafo do GOTAS) fica desligado nelas.
Hiperparâmetros de cada arquitetura: os de `run_collection_v2.py` e
`run_fourier_gnn_grid.py`.

Ajuste justo: além da base, cada arquitetura tem a melhor variante de
treino da coleção (`_best`), a janela de 24 h do estudo original
(`_w24`) e as duas juntas (`_w24_best`). `select` escolhe, por
arquitetura e cenário, a configuração de maior R² na validação (nível
combinado em cross-fit, 3 seeds); o teste não entra na escolha. O
relatório usa a escolhida, com as 5 seeds.

O relatório usa os mesmos protocolos de `report_lean_model.py`: 1 passo
com valores medidos, simulação livre de 6 e 12 passos (origens que não
se sobrepõem) e R² por passo.

Uso (da raiz do repo; um processo por treino, pela RAM):
    S=notebooks/training/scripts/run_architectures_tcc.py
    uv run --extra ml python $S run <variante> <cenario> <seed>
    uv run --extra ml python $S select
    uv run --extra ml python $S report [all]
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
    STAT_SEEDS,
    _blend_weights,  # noqa: PLC2701
    _crossfit_val,  # noqa: PLC2701
    _levels,  # noqa: PLC2701
    _split,  # noqa: PLC2701
    run,
)

from mimo_sys.training.data import PROJECT_ROOT
from mimo_sys.training.evaluation import r2_per_output

SUMMARY = PROJECT_ROOT / "outputs" / "architectures_tcc.json"
SUMMARY_ALL = PROJECT_ROOT / "outputs" / "architectures_tcc_all.json"
SELECTION = PROJECT_ROOT / "outputs" / "architectures_tcc_selection.json"
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
# Melhor variante de treino de cada arquitetura (coleção v2; FourierGNN:
# cosseno com clipping, o estabilizador da sua grade).
BEST = {
    "A_mpnn": {"scheduler": "cosine_warmup", "gradient_clip_val": 1.0},
    "A_sgn": {"augment": True},
    "A_stemgnn": {"loss": "huber"},
    "A_seq2seq_latent_gnn": {
        "revin": "standard",
        "augment": True,
        "spectral_weight": 0.1,
        "ema_decay": 0.999,
    },
    "A_fourier_gnn": {"scheduler": "cosine_warmup", "gradient_clip_val": 1.0},
}
GRID: dict[str, dict] = {}
for _name, _arch in ARCHS.items():
    _arch["message_l1_weight"] = 0.0
    GRID[_name] = _arch
    GRID[f"{_name}_best"] = {**_arch, **BEST[_name]}
    GRID[f"{_name}_w24"] = {**_arch, "input_window": 24}
    GRID[f"{_name}_w24_best"] = {**_arch, **BEST[_name], "input_window": 24}
run_improvements.VARIANTS.update(GRID)


def select() -> None:
    """Best config per architecture and scenario, on val only."""
    out: dict = {}
    for scenario in ("full", "pumps"):
        out[scenario] = {}
        for arch in ARCHS:
            scores = {}
            for variant in (v for v in GRID if v.startswith(arch)):
                files = [
                    OUT / "preds" / f"{variant}_{scenario}_s{s}.npz"
                    for s in STAT_SEEDS
                ]
                if not all(f.exists() for f in files):
                    continue
                r2 = []
                for f in files:
                    val = _split(dict(np.load(f)), "val")
                    r2.append(
                        r2_per_output(val["true"], _crossfit_val(val)).mean()
                    )
                scores[variant] = round(float(np.mean(r2)), 4)
            out[scenario][arch] = {
                "escolhida": max(scores, key=scores.get),
                "val_r2": scores,
            }
            print(scenario, arch, scores)
    SELECTION.write_text(json.dumps(out, indent=1))


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


def report(all_configs: bool = False) -> None:  # noqa: PLR0914
    """Test table; with `all_configs`, every config of every architecture."""
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
        chosen = (
            json.loads(SELECTION.read_text())[scenario]
            if SELECTION.exists()
            else {}
        )
        pairs = [
            (arch, variant)
            for arch in [*ARCHS, "I0_ref"]
            for variant in (
                [v for v in GRID if v.startswith(arch)] or [arch]
                if all_configs
                else [chosen.get(arch, {}).get("escolhida", arch)]
            )
        ]
        for arch, variant in pairs:
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
                entry["config"] = variant
                sel = chosen.get(arch, {})
                entry["escolhida"] = sel.get("escolhida") == variant
                entry["val_r2"] = sel.get("val_r2", {}).get(variant)
                label = LABELS[arch]
                if all_configs and arch in ARCHS:
                    label += variant.removeprefix(arch) or " (base)"
                suffix = "" if mode == "final" else " (nível direto)"
                res[label + suffix] = entry
        out[scenario] = res
    path = SUMMARY_ALL if all_configs else SUMMARY
    path.write_text(json.dumps(out, indent=1, default=_json))
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
            cfg = m.get("config", "")
            print(
                f"| {name} [{cfg}] | {m['1_passo'].mean():.3f} | "
                f"{m['6_passos'].mean():.3f} | {r12} | "
                f"{m['12_passos'].mean():.3f}{sd_txt} |"
            )


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report(all_configs=sys.argv[2:] == ["all"])
    elif sys.argv[1] == "select":
        select()
    else:
        run(sys.argv[2], sys.argv[3], int(sys.argv[4]))
