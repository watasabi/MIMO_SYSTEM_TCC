"""Extrai equação e grafo do SpectralTFNet (W=168, H=12, seed 42).

Uso:
    uv run --extra ml python \
        notebooks/training/scripts/extract_spectral_tf_net.py
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import torch

from mimo_sys.architectures import SpectralTFNet
from mimo_sys.training import TrainConfig, fit_mass_balance, train
from mimo_sys.training.data import (
    FEATURE_COLS,
    INPUT_IDX,
    OUTPUT_COLS,
    PROJECT_ROOT,
    TARGET_IDX,
    load_splits,
)

FIGURES = PROJECT_ROOT / "docs" / "figures"
OUT = PROJECT_ROOT / "outputs" / "spectral_tf_net_extraction.json"
MODEL_PATH = PROJECT_ROOT / "models" / "spectral_tf_net_h12_w168_s42.pth"
MAX_LAG = 48
SENDERS = [*FEATURE_COLS, "balanco_massa"]

CFG = TrainConfig(
    model="spectral_tf_net",
    model_kwargs=dict(n_bands=8, n_regimes=1),
    input_window=168,
    output_window=12,
    batch_size=128,
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
    run_name="stf_extraction_h12_lin_w168_b128_s42",
    save=False,
    seed=42,
)


def core_of(model: torch.nn.Module) -> SpectralTFNet:
    """Peel training wrappers down to the SpectralTFNet."""
    while not isinstance(model, SpectralTFNet):
        model = model.model
    return model


def main() -> None:  # noqa: PLR0914, PLR0915 - linear on purpose
    """Train, extract, plot, dump."""
    result = train(CFG)
    core = core_of(result.module.model)
    # Model works on the preprocessor's [0, 1] scale; a coefficient
    # from sender j to target i is `range_i / range_j` in physical
    # units (centring does not change slopes).
    span = np.asarray(result.preprocessor.scaler.data_range_)
    n_send = core.n_nodes
    to_phys = span[TARGET_IDX][:, None] / span[None, :n_send]
    torch.save(core.state_dict(), MODEL_PATH)

    # Static coefficient of an edge = sum of its *causal* taps, in
    # physical units. Not H(0): the model centres every window, so the
    # DC bin never sees signal and the lasso shrinks it to ~0. Outputs
    # start at tau=1 (tau=0 is the not-yet-predicted value, always 0);
    # known inputs at tau=0 (u[k] is known).
    def causal_sum(i: int, j: int) -> float:
        h = core.impulse_response(i, j).numpy()
        first = 1 if j not in INPUT_IDX else 0
        return float(h[first : MAX_LAG + 1].sum() * to_phys[i, j])

    coef = np.array(
        [
            [causal_sum(i, j) for j in range(n_send)]
            for i in range(len(OUTPUT_COLS))
        ]
    )
    coefs = {
        out: {SENDERS[j]: round(float(coef[i, j]), 6) for j in range(n_send)}
        for i, out in enumerate(OUTPUT_COLS)
    }
    # Mass balance, read straight off the level's edges (lags 0-3):
    # the gray-box says +a for the inflow and -a for the outflow.
    nv = OUTPUT_COLS.index("nivel_res")
    tr = load_splits()["train"]
    a, _ = fit_mass_balance(
        tr[:, TARGET_IDX[1]],
        tr[:, FEATURE_COLS.index("vazao_entrada")],
        tr[:, TARGET_IDX[0]],
    )

    def short_sum(j: int, first: int) -> float:
        h = core.impulse_response(nv, j).numpy()
        return float(h[first:4].sum() * to_phys[nv, j])

    inflow = short_sum(FEATURE_COLS.index("vazao_entrada"), 0)
    outflow = short_sum(FEATURE_COLS.index("vazao_distribuicao"), 1)
    # Steady state of the one-step ARX for flow and pressure:
    # y = A y + B u  ->  y = (I - A)^-1 B u. The level is an integrator
    # (its self-coefficient ~ 1), so it is left out of the inverse.
    static_out = [0, 2]
    a_mat = coef[np.ix_(static_out, [TARGET_IDX[k] for k in static_out])]
    b_mat = coef[np.ix_(static_out, INPUT_IDX)]
    gain = np.linalg.solve(np.eye(len(static_out)) - a_mat, b_mat)
    summary = {
        "chained_r2_direto": result.metrics["r2"].round(4).to_dict(),
        "causal_coefficients_physical": coefs,
        "balanco_de_massa_lags_0_3": {
            "vazao_entrada_para_nivel": round(inflow, 6),
            "vazao_distribuicao_para_nivel": round(outflow, 6),
            "nivel_para_nivel_lags_1_3": round(
                short_sum(FEATURE_COLS.index("nivel_res"), 1), 4
            ),
            "gray_box_a": round(a, 6),
            "sinais_opostos": bool(np.sign(inflow) != np.sign(outflow)),
        },
        "steady_state_gain": {
            OUTPUT_COLS[k]: {
                FEATURE_COLS[j]: round(float(gain[r, c]), 4)
                for c, j in enumerate(INPUT_IDX)
            }
            for r, k in enumerate(static_out)
        },
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # Route B: |H_ij| per band.
    strength = core.band_strength().numpy()  # (i, j, m)
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
    axes[0].set_yticks(range(len(OUTPUT_COLS)))
    axes[0].set_yticklabels(OUTPUT_COLS, fontsize=8)
    fig.suptitle("SpectralTFNet: |H_ij| por banda (0 = DC, log-espaçadas)")
    fig.tight_layout()
    fig.savefig(FIGURES / "spectral_tf_net_band_strength.png", dpi=130)

    # Route A: impulse responses of the physically expected edges.
    edges = [
        (1, FEATURE_COLS.index("vazao_entrada")),
        (1, FEATURE_COLS.index("vazao_distribuicao")),
        (1, FEATURE_COLS.index("nivel_res")),
        (0, FEATURE_COLS.index("freq_b1")),
        (2, FEATURE_COLS.index("freq_b3")),
        (2, FEATURE_COLS.index("vazao_distribuicao")),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    lags = 48
    for ax, (i, j) in zip(axes.flat, edges, strict=True):
        h = core.impulse_response(i, j).numpy()[: lags + 1]
        ax.stem(range(lags + 1), h * to_phys[i, j], basefmt=" ")
        ax.axhline(0, color="#999999", lw=0.6)
        ax.set_title(f"{SENDERS[j]} → {OUTPUT_COLS[i]}", fontsize=9)
        ax.set_xlabel("atraso τ (h)")
    fig.suptitle("Resposta ao impulso h_ij(τ) = iFFT(H_ij), unidades físicas")
    fig.tight_layout()
    fig.savefig(FIGURES / "spectral_tf_net_impulse.png", dpi=130)


if __name__ == "__main__":
    main()
