"""Modelo final: GOTAS do fold 5 + nível combinado por passo.

Um modelo só; as métricas são a média de 5 seeds e as figuras usam a
seed 42. Comparação com o ARMAX MOGWO do artigo
nos protocolos do doc 8, todos no nosso teste e nas mesmas origens:

- 1 passo com valores medidos (série-paralelo, o protocolo do artigo);
- free-run de 6 passos (origem a cada 6 h) e de 12 passos (a cada 12 h);
- R² a cada passo do horizonte, com origem em todas as horas.

Uso (da raiz do repo, depois de treinar a variante `F5_final`):
    uv run --extra ml python \
        notebooks/training/scripts/report_lean_model.py [variante]
"""

from __future__ import annotations

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from figure_style import COLORS, OUTPUT_LABELS, WIDTH, apply_style, save
from report_spectral_tf_net import (
    OUR_TEST_START,
    Y,
    all_metrics,
    armax_free_run,
    armax_one_step,
)
from run_improvements import (
    FINAL_VARIANT,
    OUT,
    SEEDS,
    _blend_weights,  # noqa: PLC2701
    _levels,  # noqa: PLC2701
    _split,  # noqa: PLC2701
    config,
    load_weights,
)

from mimo_sys.training.data import (
    FEATURE_COLS,
    PROJECT_ROOT,
    TARGET_IDX,
    load_splits,
)
from mimo_sys.training.train import (
    _windows,  # noqa: PLC2701
    build_model,
    known_columns,
    target_layout,
)

VARIANT = sys.argv[1] if len(sys.argv) > 1 else FINAL_VARIANT
# O modelo final é avaliado no teste do fold 5.
TEST_START, N_ORIGINS = (
    (1988, 209) if VARIANT == FINAL_VARIANT else (OUR_TEST_START, 321)
)
SUMMARY = PROJECT_ROOT / "outputs" / f"lean_model_{VARIANT}.json"
H = 12
NIVEL = 1
FIG_SEED = 42
y = Y.T  # (n, 3)


def seed_preds(scenario: str, seed: int) -> dict[str, np.ndarray]:
    """Test forecasts of one model: doc 8 level and blended level."""
    d = dict(np.load(OUT / "preds" / f"{VARIANT}_{scenario}_s{seed}.npz"))
    val, test = _split(d, "val"), _split(d, "test")
    w = _blend_weights(val)
    # Doc 8: nível pela cascata no full e direto no pumps.
    doc8 = np.ones(H) if scenario == "full" else np.zeros(H)
    return {
        "true": test["true"],
        "final": _levels(test, w),
        "doc8": _levels(test, doc8),
        "w": w,
    }


def mean_metrics(pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """`all_metrics` averaged over seeds, plus the seeds' R² spread."""
    ms = [all_metrics(t, p) for t, p in pairs]
    out = {k: np.mean([m[k] for m in ms], 0) for k in ms[0]}
    out["r2_mean_std"] = float(np.std([m["r2"].mean() for m in ms]))
    return out


def per_step(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    """R² mean over outputs at each step, total variance of the test."""
    var = t.reshape(-1, 3).var(0)
    return (1 - ((t - p) ** 2).mean(0) / var).mean(1)


def main() -> None:  # noqa: PLR0914, PLR0915 - linear on purpose
    origins = np.arange(N_ORIGINS)
    rows = TEST_START + origins
    truth = np.stack([y[r : r + H] for r in rows])
    persist = np.stack([np.repeat(y[r - 1][None], H, 0) for r in rows])
    seasonal = np.stack([y[r - 24 : r - 24 + H] for r in rows])
    armax = np.stack([armax_free_run(r, H) for r in rows])
    armax_1 = armax_one_step(rows[0], rows[-1] + 1)

    out: dict = {"variante": VARIANT}
    for scenario in ("full", "pumps"):
        runs = [seed_preds(scenario, s) for s in SEEDS]
        assert np.allclose(runs[0]["true"], truth)
        res: dict = {}

        # Protocolo 1: um passo à frente com os valores medidos.
        res["1_passo"] = {
            "persistencia": all_metrics(truth[:, 0], persist[:, 0]),
            "armax": all_metrics(truth[:, 0], armax_1),
            "doc8": mean_metrics(
                [(truth[:, 0], r["doc8"][:, 0]) for r in runs]
            ),
            "final": mean_metrics(
                [(truth[:, 0], r["final"][:, 0]) for r in runs]
            ),
        }
        # Protocolos 6 e 12: free-run com janelas que não se sobrepõem.
        for h in (6, 12):
            idx = origins[::h]

            def flat(a: np.ndarray, idx: np.ndarray = idx, h: int = h):
                return a[idx, :h].reshape(-1, 3)

            res[f"{h}_passos"] = {
                "persistencia": all_metrics(flat(truth), flat(persist)),
                "sazonal_24h": all_metrics(flat(truth), flat(seasonal)),
                "armax": all_metrics(flat(truth), flat(armax)),
                "doc8": mean_metrics(
                    [(flat(truth), flat(r["doc8"])) for r in runs]
                ),
                "final": mean_metrics(
                    [(flat(truth), flat(r["final"])) for r in runs]
                ),
            }
        res["todas_origens_12"] = mean_metrics(
            [(truth.reshape(-1, 3), r["final"].reshape(-1, 3)) for r in runs]
        )
        res["por_passo"] = {
            "persistencia": per_step(truth, persist),
            "armax": per_step(truth, armax),
            "doc8": np.mean([per_step(truth, r["doc8"]) for r in runs], 0),
            "final": np.mean([per_step(truth, r["final"]) for r in runs], 0),
        }
        res["pesos_blend"] = np.mean([r["w"] for r in runs], 0)
        out[scenario] = res

        if scenario == "full":
            fig_run = runs[SEEDS.index(FIG_SEED)]
            series_figure(truth, fig_run["final"])

    out["extracao"] = extraction()
    SUMMARY.write_text(json.dumps(out, indent=1, default=_json))
    print_tables(out)


def _json(o: object) -> object:
    if isinstance(o, np.ndarray):
        return o.round(6).tolist()
    if isinstance(o, np.floating):
        return round(float(o), 6)
    raise TypeError(type(o))


def series_figure(t: np.ndarray, p: np.ndarray) -> None:
    apply_style()
    yt, yp = t[::H].reshape(-1, 3), p[::H].reshape(-1, 3)
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 5.0), sharex=True)
    for k, ax in enumerate(axes):
        for start in range(0, len(yt), H):
            ax.axvline(start, color="0.88", lw=0.5, zorder=0)
        ax.plot(yt[:, k], color=COLORS[0], lw=0.9)
        ax.plot(yp[:, k], color=COLORS[2], lw=0.9)
        ax.set_ylabel(OUTPUT_LABELS[k])
    axes[-1].set_xlabel("Hora do teste")
    fig.align_ylabels(axes)
    fig.tight_layout()
    save(fig, "stf_modelo_serie_full.pdf")


def extraction() -> dict:
    """Mass-balance edges of each seed's linear part (physical units)."""
    out = {}
    for scenario in ("full", "pumps"):
        cfg = config(VARIANT, scenario, SEEDS[0])
        layout = target_layout(cfg)
        windows = _windows(cfg, load_splits(cfg.data_prefix))[1]
        n_nodes = windows["train"][0].shape[2]
        sums = []
        for s in SEEDS:
            c = config(VARIANT, scenario, s)
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
            pp = torch.load(
                OUT / "weights" / f"{c.run_name}_pp.pt", weights_only=False
            )
            span = np.asarray(pp.scaler.data_range_)
            to_phys = span[TARGET_IDX][:, None] / span[None, :n_nodes]

            def short(j: int, first: int, m: object = m, tp=to_phys) -> float:
                h = m.impulse_response(NIVEL, j).numpy()
                return float(h[first:4].sum() * tp[NIVEL, j])

            sums.append(
                [
                    short(FEATURE_COLS.index("vazao_entrada"), 0),
                    short(FEATURE_COLS.index("vazao_distribuicao"), 1),
                    short(FEATURE_COLS.index("nivel_res"), 1),
                ]
            )
        sums = np.array(sums)
        out[scenario] = {
            "media": sums.mean(0),
            "desvio": sums.std(0),
            "rotulos": [
                "entrada->nivel",
                "distribuicao->nivel",
                "nivel->nivel (1-3)",
            ],
        }
    return out


def print_tables(out: dict) -> None:
    names = {
        "persistencia": "Persistência",
        "sazonal_24h": "Sazonal 24 h",
        "armax": "ARMAX MOGWO",
        "doc8": "doc 8",
        "final": "Modelo final",
    }
    for scenario in ("full", "pumps"):
        for proto in ("1_passo", "6_passos", "12_passos"):
            print(f"\n## {scenario} · {proto}")
            for k, m in out[scenario][proto].items():
                r2, rmse, sm = m["r2"], m["rmse"], m["smape"]
                cells = " | ".join(
                    f"{r2[i]:.3f} / {rmse[i]:.3g} / {sm[i]:.1f}%"
                    for i in range(3)
                )
                sd = m.get("r2_mean_std")
                sd_txt = f" ± {sd:.3f}" if sd is not None else ""
                print(f"| {names[k]} | {cells} | {r2.mean():.3f}{sd_txt} |")
        tm = out[scenario]["todas_origens_12"]
        print(
            f"todas as origens: {tm['r2'].round(3)} -> {tm['r2'].mean():.3f}"
        )
        ps = out[scenario]["por_passo"]
        for k in ps:
            print(k, np.round(ps[k][[0, 2, 5, 11]], 3))
        print("w_h:", np.round(out[scenario]["pesos_blend"], 2))
    print("extração:", json.dumps(out["extracao"], default=_json))


if __name__ == "__main__":
    main()
