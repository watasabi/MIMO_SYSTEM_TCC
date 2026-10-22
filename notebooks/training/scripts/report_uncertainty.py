"""Incerteza na equação e na previsão (doc 12).

1. Laplace sobre o `H` do modelo só linear (`I6_clean_linear`, seed 42):
   intervalo de credibilidade de cada aresta e do balanço de massa,
   comparado com a dispersão real entre as 5 seeds.
2. Intervalos de previsão do modelo final (`I0_ref` soup + combinação do
   nível): conformal por passo vs. conformal normalizado por um
   `log sigma` explícito (passo, hora, movimento das bombas, nível).
3. Cascata bayesiana (PyMC, extra `bayes`): posterior de `a`, `b` com
   verossimilhança Student-t, propagada no nível.

Uso (da raiz do repo):
    uv run --extra ml --extra bayes python \
        notebooks/training/scripts/report_uncertainty.py [parte ...]

    partes: laplace, intervals, bayes (padrão: todas)
"""

from __future__ import annotations

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from run_improvements import (
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
    INPUT_COLS,
    INPUT_IDX,
    OUTPUT_COLS,
    PROJECT_ROOT,
    TARGET_IDX,
    load_splits,
)
from mimo_sys.training.evaluation import r2_per_output
from mimo_sys.training.features import bad_sample_mask
from mimo_sys.training.mass_balance_cascade import (
    fit_mass_balance,
    mass_balance_cascade_forecast,
)
from mimo_sys.training.train import (
    _windows,  # noqa: PLC2701
    build_model,
    known_columns,
    target_layout,
)
from mimo_sys.training.uncertainty import (
    coefficient_ci,
    conformal_quantile,
    fit_log_sigma,
    impulse_response_graph,
    interval_metrics,
    laplace_linear_posterior,
)

FIG = PROJECT_ROOT / "docs" / "figures"
SUMMARY = PROJECT_ROOT / "outputs" / "uncertainty_report.json"
SENDERS = [*FEATURE_COLS, "balanco_massa"]
LINEAR = "I6_clean_linear"
FINAL = "I0_ref"
H, MAX_LAG = 12, 48
ALPHAS = (0.2, 0.1)
NV = 1
Z_DARK = 6  # |z| above which the heatmap cell is light


def _summary() -> dict:
    return json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}


def _save(key: str, value: object) -> None:
    out = _summary()
    out[key] = value
    SUMMARY.write_text(json.dumps(out, indent=2, ensure_ascii=False))


def _model(variant: str, scenario: str, seed: int) -> torch.nn.Module:
    cfg = config(variant, scenario, seed)
    n_nodes = _windows(cfg, load_splits())[1]["train"][0].shape[2]
    kwargs = dict(cfg.model_kwargs)
    kwargs.setdefault("known_future_idx", tuple(known_columns(cfg)))
    m = build_model(
        cfg.model,
        cfg.input_window + cfg.output_window,
        cfg.output_window,
        n_nodes,
        target_idx=list(target_layout(cfg).model_cols),
        **kwargs,
    )
    m.load_state_dict(load_weights(cfg.run_name))
    return m.eval()


def laplace(scenario: str) -> dict:  # noqa: PLR0914 - linear on purpose
    """Credible intervals of the linear model's coefficients."""
    cfg = config(LINEAR, scenario, SEEDS[0])
    pp, windows, _, _ = _windows(cfg, load_splits())
    x = torch.as_tensor(windows["train"][0], dtype=torch.float32)
    y = torch.as_tensor(windows["train"][1][..., :3], dtype=torch.float32)
    model = _model(LINEAR, scenario, SEEDS[0])
    _, sigma = laplace_linear_posterior(model, x, y)
    span = np.asarray(pp.scaler.data_range_)
    to_phys = span[TARGET_IDX][:, None] / span[None, : model.n_nodes]

    def static(i: int, j: int, last: int = MAX_LAG) -> callable:
        first = 0 if j in INPUT_IDX else 1  # u[k] is known, y[k] is not

        def coef(m: torch.nn.Module) -> torch.Tensor:
            h = impulse_response_graph(m, i, j)
            return h[first : last + 1].sum() * float(to_phys[i, j])

        return coef

    seeds = [_model(LINEAR, scenario, s) for s in SEEDS]
    edges = []
    for i, out in enumerate(OUTPUT_COLS):
        for j in range(model.n_nodes):
            mean, sd = coefficient_ci(model, sigma, static(i, j))
            spread = np.std(
                [float(static(i, j)(m).detach()) for m in seeds], ddof=1
            )
            edges.append(
                {
                    "saida": out,
                    "emissor": SENDERS[j],
                    "media": mean,
                    "dp_laplace": sd,
                    "z": mean / sd if sd > 0 else 0.0,
                    "dp_entre_seeds": float(spread),
                }
            )
    mass = {}
    for name, j in (
        ("vazao_entrada_para_nivel", FEATURE_COLS.index("vazao_entrada")),
        (
            "vazao_distribuicao_para_nivel",
            FEATURE_COLS.index("vazao_distribuicao"),
        ),
        ("nivel_para_nivel_lags_1_3", FEATURE_COLS.index("nivel_res")),
    ):
        mean, sd = coefficient_ci(model, sigma, static(NV, j, last=3))
        per_seed = [float(static(NV, j, last=3)(m).detach()) for m in seeds]
        mass[name] = {
            "media": mean,
            "ic95": [mean - 1.96 * sd, mean + 1.96 * sd],
            "seeds": per_seed,
        }
    tr = load_splits()["train"]
    a, _ = fit_mass_balance(
        tr[:, TARGET_IDX[1]],
        tr[:, FEATURE_COLS.index("vazao_entrada")],
        tr[:, TARGET_IDX[0]],
        exclude=bad_sample_mask(tr),
    )
    mass["gray_box_a_limpo"] = a

    z = np.array(
        [[e["z"] for e in edges if e["saida"] == o] for o in OUTPUT_COLS]
    )
    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(np.clip(np.abs(z), 0, 10), cmap="viridis", vmin=0, vmax=10)
    for (r, c), v in np.ndenumerate(z):
        ax.text(
            c,
            r,
            f"{v:+.1f}",
            ha="center",
            va="center",
            fontsize=7,
            color="white" if abs(v) < Z_DARK else "black",
        )
    ax.set_xticks(
        range(model.n_nodes),
        SENDERS[: model.n_nodes],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.set_yticks(range(3), OUTPUT_COLS, fontsize=8)
    fig.colorbar(im, ax=ax, label="|z| = |média| / dp (limitado a 10)")
    ax.set_title(
        f"Modelo linear ({scenario}): significância de cada aresta (Laplace)"
    )
    fig.tight_layout()
    fig.savefig(FIG / f"stf_laplace_z_{scenario}.png", dpi=130)
    plt.close(fig)
    return {"arestas": edges, "balanco_de_massa": mass}


def _features(origins: np.ndarray, full: np.ndarray) -> np.ndarray:
    """`(n, H, p)` features of the explicit `log sigma` (known at origin)."""
    pumps = [FEATURE_COLS.index(c) for c in INPUT_COLS[:3]]
    n = len(origins)
    step = np.broadcast_to(np.arange(1, H + 1) / H, (n, H))
    target_row = origins[:, None] + np.arange(H)
    ang = 2 * np.pi * (target_row % 24) / 24
    moved = np.stack(
        [
            np.abs(full[o : o + H, pumps] - full[o - 1, pumps]).sum(1) / 60
            for o in origins
        ]
    )
    level = np.broadcast_to(
        (full[origins - 1, TARGET_IDX[1]] - 2.4)[:, None], (n, H)
    )
    return np.stack(
        [
            np.ones((n, H)),
            step,
            step**2,
            np.sin(ang),
            np.cos(ang),
            moved,
            level,
        ],
        axis=-1,
    )


FEATURE_NAMES = [
    "const",
    "passo",
    "passo²",
    "sen(hora)",
    "cos(hora)",
    "bombas_mexem",
    "nivel_origem",
]


def intervals(scenario: str) -> dict:  # noqa: PLR0914 - linear on purpose
    """Coverage/width of conformal intervals for the final model."""
    s = load_splits()
    full = np.concatenate([s["train"], s["val"], s["test"]])
    n_tr, n_va = len(s["train"]), len(s["val"])
    d = dict(np.load(OUT / "preds" / f"{FINAL}_{scenario}_soup.npz"))
    val, test = _split(d, "val"), _split(d, "test")
    w = _blend_weights(val)
    pv, pt = _levels(val, w), _levels(test, w)
    ev = val["true"] - pv
    fv = _features(n_tr + np.arange(len(pv)), full)
    ft = _features(n_tr + n_va + np.arange(len(pt)), full)
    half = (
        len(pv) // 2
    )  # sigma fitted on the first half, quantile on the second
    out = {"r2_teste": r2_per_output(test["true"], pt).round(4).tolist()}
    betas = {}
    for alpha in ALPHAS:
        rows = {}
        for k, name in enumerate(OUTPUT_COLS):
            beta = fit_log_sigma(
                fv[:half, :, :].reshape(-1, fv.shape[-1]),
                ev[:half, :, k].ravel(),
            )
            betas[name] = dict(
                zip(FEATURE_NAMES, np.round(beta, 3).tolist(), strict=True)
            )
            sig_v = np.exp(fv[half:] @ beta)
            sig_t = np.exp(ft @ beta)
            # (a) conformal per step: constant width per (step, output).
            q_step = np.array(
                [
                    conformal_quantile(np.abs(ev[half:, h, k]), alpha)
                    for h in range(H)
                ]
            )
            a = interval_metrics(
                test["true"][..., k],
                pt[..., k] - q_step,
                pt[..., k] + q_step,
                alpha,
            )
            # (b) normalised conformal: width follows the explicit sigma.
            q_norm = np.array(
                [
                    conformal_quantile(
                        np.abs(ev[half:, h, k]) / sig_v[:, h], alpha
                    )
                    for h in range(H)
                ]
            )
            half_w = q_norm * sig_t
            b = interval_metrics(
                test["true"][..., k],
                pt[..., k] - half_w,
                pt[..., k] + half_w,
                alpha,
            )
            rows[name] = {"conformal_por_passo": a, "conformal_normalizado": b}
        out[f"nominal_{int(100 * (1 - alpha))}"] = rows
    out["log_sigma"] = betas

    # Figure: test series with the 90% normalised band, chained 12 h.
    alpha = 0.1
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for k, (ax, name) in enumerate(zip(axes, OUTPUT_COLS, strict=True)):
        beta = fit_log_sigma(
            fv[:half].reshape(-1, fv.shape[-1]), ev[:half, :, k].ravel()
        )
        sig_v, sig_t = np.exp(fv[half:] @ beta), np.exp(ft @ beta)
        q = np.array(
            [
                conformal_quantile(
                    np.abs(ev[half:, h, k]) / sig_v[:, h], alpha
                )
                for h in range(H)
            ]
        )
        yt, yp = test["true"][::H, :, k].ravel(), pt[::H, :, k].ravel()
        hw = (q * sig_t)[::H].ravel()
        x = np.arange(len(yt))
        ax.fill_between(
            x,
            yp - hw,
            yp + hw,
            color="tab:red",
            alpha=0.2,
            lw=0,
            label="intervalo de 90%",
        )
        ax.plot(x, yt, color="black", lw=1, label="medido")
        ax.plot(x, yp, color="tab:red", lw=1, label="previsto")
        ax.set_ylabel(name)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel(
        "hora do período de teste (previsão encadeada de 12 h)"
    )
    fig.suptitle(
        f"Modelo final ({scenario}) com intervalo conformal normalizado de 90%"
    )
    fig.tight_layout()
    fig.savefig(FIG / f"stf_intervalos_{scenario}.png", dpi=130)
    plt.close(fig)
    return out


def bayes() -> dict:  # noqa: PLR0914 - linear on purpose
    """Bayesian gray-box cascade: posterior of `a`, `b`, propagated."""
    import arviz as az  # noqa: PLC0415
    import pymc as pm  # noqa: PLC0415

    s = load_splits()
    tr = s["train"]
    u4, y1, y2 = (
        FEATURE_COLS.index("vazao_entrada"),
        TARGET_IDX[0],
        TARGET_IDX[1],
    )
    q = tr[:, u4] - tr[:, y1]
    dq, dy = (q[1:] + q[:-1]) / 2, np.diff(tr[:, y2])
    with pm.Model():
        a = pm.HalfNormal("a", sigma=0.01)  # inflow raises the level: a > 0
        b = pm.Normal("b", 0.0, 0.05)
        nu = pm.Exponential("nu_menos_1", 1 / 10) + 1
        scale = pm.HalfNormal("sigma", 0.1)
        # Student-t: the raw training split keeps its sensor failures,
        # the heavy tails absorb them instead of biasing a and b.
        pm.StudentT("dy", nu=nu, mu=a * dq + b, sigma=scale, observed=dy)
        trace = pm.sample(
            1500, tune=1500, chains=4, random_seed=42, progressbar=False
        )
    post = az.summary(
        trace,
        var_names=["a", "b", "sigma", "nu_menos_1"],
        ci_prob=0.95,
        ci_kind="hdi",
    )
    a_s = trace.posterior["a"].to_numpy().ravel()
    b_s = trace.posterior["b"].to_numpy().ravel()
    a_ls, b_ls = fit_mass_balance(tr[:, y2], tr[:, u4], tr[:, y1])
    a_cl, b_cl = fit_mass_balance(
        tr[:, y2], tr[:, u4], tr[:, y1], exclude=bad_sample_mask(tr)
    )

    # Propagate the posterior through the full-scenario cascade on test.
    full = np.concatenate([s["train"], s["val"], s["test"]])
    n0 = len(s["train"]) + len(s["val"])
    d = dict(np.load(OUT / "preds" / f"{FINAL}_full_soup.npz"))
    o = n0 + np.arange(len(d["test_true"]))
    inflow = np.stack([full[t : t + H, u4] for t in o])
    li = full[o - 1, u4] - full[o - 1, y1]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(a_s), 400, replace=False)
    sims = np.stack(
        [
            mass_balance_cascade_forecast(
                full[o - 1, y2],
                inflow,
                d["test_pred"][..., 0],
                a_s[k],
                b_s[k],
                li,
            )
            for k in idx
        ]
    )
    lo, hi = np.percentile(sims, [5, 95], axis=0)
    truth = d["test_true"][..., NV]
    return {
        "posterior": post.round(6).reset_index().to_dict(orient="records"),
        "a_minimos_quadrados_cru": a_ls,
        "a_minimos_quadrados_limpo": a_cl,
        "b_minimos_quadrados_cru": b_ls,
        "b_minimos_quadrados_limpo": b_cl,
        "area_implicita_m2": {
            "media": float(np.mean(3.6 / a_s)),
            "hdi95": np.percentile(3.6 / a_s, [2.5, 97.5]).round(0).tolist(),
        },
        "r2_nivel_cascata_media_posterior": float(
            r2_per_output(truth[..., None], sims.mean(0)[..., None])[0]
        ),
        "cobertura_90_so_parametros": float(
            np.mean((truth >= lo) & (truth <= hi))
        ),
        "largura_90_so_parametros_m": float((hi - lo).mean()),
    }


def main(parts: list[str]) -> None:
    if "laplace" in parts:
        _save("laplace", {s: laplace(s) for s in ("full", "pumps")})
    if "intervals" in parts:
        _save("intervalos", {s: intervals(s) for s in ("full", "pumps")})
    if "bayes" in parts:
        _save("cascata_bayesiana", bayes())
    print(
        json.dumps(
            {k: v for k, v in _summary().items() if k in parts or True},
            indent=1,
            ensure_ascii=False,
        )[:6000]
    )


if __name__ == "__main__":
    main(sys.argv[1:] or ["laplace", "intervals", "bayes"])
