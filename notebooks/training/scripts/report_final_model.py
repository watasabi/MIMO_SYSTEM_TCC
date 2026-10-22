"""Tabelas, gráficos e extração do modelo final (doc 11).

Modelo final: SpectralTFNet de referência (doc 8), ensemble de 5 seeds
com o `H_ij(f)` médio (`run_improvements.SoupSpectral`) e nível pela
combinação por passo direto + cascata (pesos ajustados na validação).
Mesmas janelas de teste e mesmo ARMAX do `report_spectral_tf_net.py`.

Uso (da raiz do repo, depois de `run_improvements.py run`):
    uv run --extra ml python \
        notebooks/training/scripts/report_final_model.py
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from report_spectral_tf_net import (
    FIG,
    NAMES,
    all_metrics,
    armax_horizon,
    row,
)
from run_improvements import (
    OUT,
    SEEDS,
    STAT_SEEDS,
    SoupSpectral,
    _blend_weights,  # noqa: PLC2701
    _levels,  # noqa: PLC2701
    _split,  # noqa: PLC2701
    config,
    load_weights,
)

from mimo_sys.training.data import (
    FEATURE_COLS,
    OUTPUT_COLS,
    PROJECT_ROOT,
    TARGET_IDX,
    load_splits,
)
from mimo_sys.training.evaluation import amplitude_spectrum
from mimo_sys.training.mass_balance_cascade import fit_mass_balance
from mimo_sys.training.train import (
    _windows,  # noqa: PLC2701
    build_model,
    known_columns,
    target_layout,
)

H = 12
VARIANT = "I0_ref"
SENDERS = [*FEATURE_COLS, "balanco_massa"]
SUMMARY = PROJECT_ROOT / "outputs" / "final_model_report.json"
BENCH = PROJECT_ROOT / "outputs" / "benchmarks"
NIVEL = 1


def chained(arr: np.ndarray) -> np.ndarray:
    """All-origin `(n, H, 3)` -> doc 8's chained series (origin every 12 h)."""
    return arr[::H].reshape(-1, 3)


def reference(scenario: str) -> tuple[np.ndarray, list[np.ndarray]]:
    """Doc 8's model, 3 seeds: nivel via cascade (full) or direct (pumps)."""
    preds = []
    for s in STAT_SEEDS:
        d = dict(np.load(OUT / "preds" / f"{VARIANT}_{scenario}_s{s}.npz"))
        test = _split(d, "test")
        w = np.ones(H) if scenario == "full" else np.zeros(H)
        preds.append(_levels(test, w))
    return test["true"], preds


def final(scenario: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Soup forecasts with the blended level; also returns the weights."""
    d = dict(np.load(OUT / "preds" / f"{VARIANT}_{scenario}_soup.npz"))
    w = _blend_weights(_split(d, "val"))
    test = _split(d, "test")
    return test["true"], _levels(test, w), w


def lgbm(scenario: str) -> tuple[np.ndarray, np.ndarray]:  # noqa: PLR0914
    """LightGBM benchmark with the project's cascade + level blend."""
    from mimo_sys.training.mass_balance_cascade import (  # noqa: PLC0415
        apply_level_blend,
        fit_level_blend,
        mass_balance_cascade_forecast,
    )

    s = load_splits()
    full = np.concatenate([s["train"], s["val"], s["test"]])
    u4, y1, y2 = (
        FEATURE_COLS.index("vazao_entrada"),
        TARGET_IDX[0],
        TARGET_IDX[1],
    )
    a, b = fit_mass_balance(
        s["train"][:, y2], s["train"][:, u4], s["train"][:, y1]
    )
    shift = 0 if scenario == "full" else 24
    z = np.load(BENCH / f"lgbm_{scenario}.npz")
    out = {}
    for split in ("val", "test"):
        o, p = z[f"{split}_origins"], z[f"{split}_pred"]
        inflow = np.stack([full[t - shift : t - shift + H, u4] for t in o])
        cas = mass_balance_cascade_forecast(
            full[o - 1, y2],
            inflow,
            p[..., 0],
            a,
            b,
            last_imbalance=full[o - 1, u4] - full[o - 1, y1],
        )
        out[split] = (z[f"{split}_true"], p, cas)
    yv, pv, cv = out["val"]
    w = fit_level_blend(yv[..., NIVEL], pv[..., NIVEL], cv)
    yt, pt, ct = out["test"]
    pt = pt.copy()
    pt[..., NIVEL] = apply_level_blend(w, pt[..., NIVEL], ct)
    return yt, pt


def per_step(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """R² mean over outputs at each step, every origin, total variance."""
    var = y.reshape(-1, 3).var(0)
    return (1 - ((y - p) ** 2).mean(0) / var).mean(1)


def armax_per_step() -> np.ndarray:
    out = []
    for h in range(H):
        sq, ys = [], []
        for off in range(H):
            yt, yp = armax_horizon(H, off)
            yt, yp = yt.reshape(-1, H, 3), yp.reshape(-1, H, 3)
            sq.append((yt[:, h] - yp[:, h]) ** 2)
            ys.append(yt.reshape(-1, 3))
        out.append(
            np.mean(1 - np.concatenate(sq).mean(0) / np.concatenate(ys).var(0))
        )
    return np.array(out)


def load_soup(scenario: str) -> tuple[SoupSpectral, object]:
    cfg = config(VARIANT, scenario, SEEDS[0])
    layout = target_layout(cfg)
    n_nodes = _windows(cfg, load_splits())[1]["train"][0].shape[2]
    models = []
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
        models.append(m.eval())
    pp = torch.load(
        OUT / "weights" / f"{cfg.run_name}_pp.pt", weights_only=False
    )
    return SoupSpectral(models).eval(), pp


def extraction(scenario: str) -> dict:  # noqa: PLR0914, PLR0915 - linear
    """Equation and graph of the soup (one transfer-function matrix)."""
    soup, pp = load_soup(scenario)
    core = soup.linear
    span = np.asarray(pp.scaler.data_range_)
    to_phys = span[TARGET_IDX][:, None] / span[None, : core.n_nodes]
    tr = load_splits()["train"]
    a, _ = fit_mass_balance(
        tr[:, TARGET_IDX[1]],
        tr[:, FEATURE_COLS.index("vazao_entrada")],
        tr[:, TARGET_IDX[0]],
    )

    def short_sum(model: object, j: int, first: int) -> float:
        h = model.impulse_response(NIVEL, j).detach().numpy()
        return float(h[first:4].sum() * to_phys[NIVEL, j])

    def mass(model: object) -> dict[str, float]:
        return {
            "vazao_entrada_para_nivel": short_sum(
                model, FEATURE_COLS.index("vazao_entrada"), 0
            ),
            "vazao_distribuicao_para_nivel": short_sum(
                model, FEATURE_COLS.index("vazao_distribuicao"), 1
            ),
            "nivel_para_nivel_lags_1_3": short_sum(
                model, FEATURE_COLS.index("nivel_res"), 1
            ),
        }

    per_seed = [mass(m) for m in soup.models]
    # Nonlinear share on the test windows: residual energy over the
    # soup output (linear part + mean residual), as nonlinearity_share.
    _, windows, _, _ = _windows(
        config(VARIANT, scenario, SEEDS[0]), load_splits()
    )
    x = torch.as_tensor(windows["test"][0], dtype=torch.float32)
    with torch.no_grad():
        soup(x)
    lin = core._last_linear.pow(2).sum()
    res = (
        torch.stack([m._last_residual for m in soup.models])
        .mean(0)
        .pow(2)
        .sum()
    )
    with torch.no_grad():
        full_out = soup(x).numpy()
        lin_out = core(x).numpy()
        shares = []
        for m in soup.models:
            m(x)
            shares.append(round(m.nonlinearity_share(), 3))
    truth = windows["test"][1][..., :3]

    def r2w(p: np.ndarray) -> list[float]:
        t, q = truth.reshape(-1, 3), p.reshape(-1, 3)
        r = 1 - ((t - q) ** 2).sum(0) / ((t - t.mean(0)) ** 2).sum(0)
        return r.round(3).tolist()

    summary = {
        "balanco_de_massa_soup": {
            k: round(v, 6) for k, v in mass(core).items()
        },
        "balanco_de_massa_por_seed": [
            {k: round(v, 6) for k, v in m.items()} for m in per_seed
        ],
        "gray_box_a": round(a, 6),
        "nonlinearity_share": round(float(res / (lin + res)), 3),
        "nonlinearity_share_por_seed": shares,
        "r2_janelas_teste_completo": r2w(full_out),
        "r2_janelas_teste_so_linear": r2w(lin_out),
        "n_params_linear": int(core.response.numel() + core.bias.numel()),
        "n_params_residual_por_seed": int(
            sum(p.numel() for p in soup.models[0].residual.parameters())
        ),
    }

    strength = core.band_strength().detach().numpy()
    n_bands = strength.shape[-1]
    fig, axes = plt.subplots(
        1, n_bands, figsize=(2.6 * n_bands, 3.2), sharey=True
    )
    vmax = strength.max()
    for m, ax in enumerate(axes):
        ax.imshow(strength[:, :, m], vmin=0, vmax=vmax, cmap="viridis")
        ax.set_title(f"banda {m}", fontsize=9)
        ax.set_xticks(range(strength.shape[1]))
        ax.set_xticklabels(
            SENDERS[: strength.shape[1]], rotation=90, fontsize=7
        )
    axes[0].set_yticks(range(3))
    axes[0].set_yticklabels(OUTPUT_COLS, fontsize=8)
    fig.suptitle(
        f"Modelo final ({scenario}): |H̄_ij| por banda (média de 5 seeds)"
    )
    fig.tight_layout()
    fig.savefig(FIG / f"stf_final_band_strength_{scenario}.png", dpi=130)
    plt.close(fig)

    edges = [
        (1, FEATURE_COLS.index("vazao_entrada")),
        (1, FEATURE_COLS.index("vazao_distribuicao")),
        (1, FEATURE_COLS.index("nivel_res")),
        (0, FEATURE_COLS.index("freq_b1")),
        (2, FEATURE_COLS.index("freq_b2")),
        (2, FEATURE_COLS.index("vazao_distribuicao")),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    lags = 48
    for ax, (i, j) in zip(axes.flat, edges, strict=True):
        for m in soup.models:
            h = m.impulse_response(i, j).detach().numpy()[: lags + 1]
            ax.plot(
                range(lags + 1), h * to_phys[i, j], color="#bbbbbb", lw=0.8
            )
        h = core.impulse_response(i, j).detach().numpy()[: lags + 1]
        ax.stem(range(lags + 1), h * to_phys[i, j], basefmt=" ")
        ax.axhline(0, color="#999999", lw=0.6)
        ax.set_title(f"{SENDERS[j]} → {OUTPUT_COLS[i]}", fontsize=9)
        ax.set_xlabel("atraso τ (h)")
    fig.suptitle(
        f"Modelo final ({scenario}): resposta ao impulso de H̄ (hastes) e "
        "das 5 seeds (cinza), unidades físicas"
    )
    fig.tight_layout()
    fig.savefig(FIG / f"stf_final_impulse_{scenario}.png", dpi=130)
    plt.close(fig)
    return summary


def main() -> None:  # noqa: PLR0914, PLR0915 - linear on purpose
    out: dict = {}
    plt.rcParams.update({"font.size": 10})
    header = (
        "| Modelo | vazão: R² / RMSE / SMAPE | nível: R² / RMSE / SMAPE | "
        "pressão: R² / RMSE / SMAPE | R² médio | R² espectral (v/n/p) | "
        "SMAPE espectral (v/n/p) |\n|---|---|---|---|---|---|---|"
    )
    steps, weights = {}, {}
    for scenario in ("full", "pumps"):
        y_all, refs = reference(scenario)
        _, fin, w = final(scenario)
        _, gbm = lgbm(scenario)
        weights[scenario] = w
        y12 = chained(y_all)
        rows = {}
        if scenario == "full":
            rows["ARMAX MOGWO (free-run)"] = all_metrics(*armax_horizon(H))
        ref_m = [all_metrics(y12, chained(p)) for p in refs]
        rows["SpectralTFNet + resíduo (doc 8, 3 seeds)"] = {
            k: np.mean([m[k] for m in ref_m], 0) for k in ref_m[0]
        }
        rows["LightGBM direto + cascata + combinação (caixa-preta)"] = (
            all_metrics(y12, chained(gbm))
        )
        rows["**Modelo final**"] = all_metrics(y12, chained(fin))
        out[scenario] = {
            k: {m: v.tolist() for m, v in r.items()} for k, r in rows.items()
        }
        print(f"\n### {scenario}, 12 passos\n\n{header}")
        for label, m in rows.items():
            print(row(label, m))
        steps[scenario] = {
            "SpectralTFNet + resíduo (doc 8)": np.mean(
                [per_step(y_all, p) for p in refs], 0
            ),
            "LightGBM (caixa-preta)": per_step(y_all, gbm),
            "modelo final": per_step(y_all, fin),
        }

        yt, yp = y12, chained(fin)
        units = ["l/s", "m", "mca"]
        fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
        for k, ax in enumerate(axes):
            ax.plot(yt[:, k], color="black", lw=1, label="medido")
            ax.plot(
                yp[:, k], color="tab:red", lw=1, label="modelo final (12 h)"
            )
            for start in range(0, len(yt), H):
                ax.axvline(start, color="#dddddd", lw=0.5, zorder=0)
            ax.set_ylabel(f"{NAMES[k]} ({units[k]})")
        axes[0].legend(fontsize=8, loc="upper right")
        axes[-1].set_xlabel(
            "hora do período de teste (linhas cinza = nova janela)"
        )
        fig.suptitle(
            f"Modelo final, cenário {scenario}: previsão encadeada de 12 h"
        )
        fig.tight_layout()
        fig.savefig(FIG / f"stf_final_serie_{scenario}.png", dpi=130)
        plt.close(fig)

        period = 1 / np.fft.rfftfreq(len(yt))[1:]
        a_t, a_p = amplitude_spectrum(yt)[1:], amplitude_spectrum(yp)[1:]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        for k, ax in enumerate(axes):
            ax.loglog(period, a_t[:, k], color="black", lw=1, label="medido")
            ax.loglog(
                period, a_p[:, k], color="tab:red", lw=1, label="previsto"
            )
            ax.axvline(24, color="gray", ls=":", lw=0.8)
            ax.axvline(12, color="gray", ls=":", lw=0.8)
            ax.set_title(NAMES[k])
            ax.set_xlabel("período (h)")
            ax.invert_xaxis()
        axes[0].set_ylabel("amplitude")
        axes[0].legend(fontsize=8)
        fig.suptitle(
            f"Espectro de amplitude, teste, {scenario} (linhas: 24 h e 12 h)"
        )
        fig.tight_layout()
        fig.savefig(FIG / f"stf_final_espectro_{scenario}.png", dpi=130)
        plt.close(fig)

    arm = armax_per_step()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, scenario in zip(axes, ("full", "pumps"), strict=True):
        styles = iter(["--o", ":s", "-o"])
        for label, r2 in steps[scenario].items():
            ax.plot(range(1, H + 1), r2, next(styles), label=label, ms=4)
        if scenario == "full":
            ax.plot(
                range(1, H + 1),
                arm,
                ":^",
                color="gray",
                label="ARMAX do paper",
            )
        ax.set_title(f"cenário {scenario}")
        ax.set_xlabel("passo do horizonte (h)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("R² médio (vazão, nível, pressão)")
    fig.suptitle(
        "Qualidade por passo do horizonte (todas as origens do teste)"
    )
    fig.tight_layout()
    fig.savefig(FIG / "stf_final_por_horizonte.png", dpi=130)
    plt.close(fig)
    out["por_passo"] = {
        s: {k: v.round(4).tolist() for k, v in d.items()}
        for s, d in steps.items()
    }
    out["por_passo"]["full"]["ARMAX"] = arm.round(4).tolist()

    fig, ax = plt.subplots(figsize=(7, 3.6))
    for scenario, w in weights.items():
        ax.plot(range(1, H + 1), w, "-o", ms=4, label=f"{scenario}")
    ax.set_xlabel("passo do horizonte (h)")
    ax.set_ylabel("peso da cascata w_h")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Nível = w_h · cascata + (1 − w_h) · direto (pesos da validação)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "stf_final_pesos_nivel.png", dpi=130)
    plt.close(fig)
    out["pesos_nivel"] = {s: w.round(3).tolist() for s, w in weights.items()}

    out["extracao"] = {s: extraction(s) for s in ("full", "pumps")}
    SUMMARY.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(
        json.dumps(
            {k: out[k] for k in ("por_passo", "pesos_nivel", "extracao")},
            indent=1,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
