"""Ensemble several trained forecasters: average or fit per-output weights.

Individual architectures/seeds make different errors — MPNN's and
StemGNN's/Seq2SeqLatentGNN's `nivel_res` forecasts disagree more than
their `vazao_distribuicao` ones (not surprising: `nivel_res` is the
system's only genuine state, `docs/1_analise-dados-reservatorio.md`
§4-5, and harder to get right consistently), so averaging predictions
reduces variance on the hard output even though none of the members
alone is better there. Combining different architectures/seeds of the
winning MPNN config (RevIN + per-output heads + uncertainty weighting
+ mass-balance feature) beat any single member's `nivel_res` R² by
+0.06 in experiments on this branch.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.optimize import nnls


def average_predictions(preds: list[npt.NDArray]) -> npt.NDArray:
    """Simple mean of `(n_steps, n_targets)` prediction arrays."""
    return np.mean(preds, axis=0)


def fit_ensemble_weights(
    preds: list[npt.NDArray], y_true: npt.NDArray
) -> npt.NDArray:
    """Non-negative least-squares weight per member, per output.

    Fit on a holdout the members were not evaluated against for
    selection (e.g. validation-split predictions) — fitting and
    reporting the fit on the same data this returns overstates it
    (an "oracle" ensemble), since it sees the test errors it is
    minimising.

    Args:
        preds: One `(n_steps, n_targets)` array per ensemble member.
        y_true: `(n_steps, n_targets)` ground truth for the same rows.

    Returns:
        `(n_members, n_targets)` non-negative weights, each output's
        column summing to 1 (uniform if every member underperforms a
        plain intercept for that output).
    """
    stacked = np.stack(preds, axis=-1)  # (n_steps, n_targets, n_members)
    n_targets = y_true.shape[1]
    n_members = len(preds)
    weights = np.empty((n_members, n_targets))
    for out in range(n_targets):
        w, _ = nnls(stacked[:, out, :], y_true[:, out])
        total = w.sum()
        weights[:, out] = w / total if total > 0 else 1.0 / n_members
    return weights


def weighted_predictions(
    preds: list[npt.NDArray], weights: npt.NDArray
) -> npt.NDArray:
    """Combine predictions with per-output `fit_ensemble_weights` output."""
    stacked = np.stack(preds, axis=-1)  # (n_steps, n_targets, n_members)
    return np.einsum("stm,mt->st", stacked, weights)


def select_ensemble_subset(
    val_preds: dict[str, npt.NDArray],
    y_val: npt.NDArray,
    min_size: int = 2,
    max_size: int | None = None,
) -> tuple[str, ...]:
    """Pick the subset of members whose mean-R² average is best on val.

    Brute-force over every subset size in `[min_size, max_size]` — fine
    for the handful of trained members this is meant for (dozens, not
    hundreds; cost is exponential in member count). Selecting and
    reporting on the same split overstates the ensemble (same caveat
    as `fit_ensemble_weights`): call this on validation predictions,
    then evaluate the chosen subset's *test* predictions separately.

    Args:
        val_preds: `{member_name: (n_steps, n_targets) predictions}`
            on a held-out split not used to pick members.
        y_val: `(n_steps, n_targets)` ground truth for that split.
        min_size: Smallest subset size to consider.
        max_size: Largest subset size to consider; defaults to every
            member.

    Returns:
        The winning subset's member names, in `val_preds`' iteration
        order.
    """
    import itertools  # noqa: PLC0415 - only needed here

    names = list(val_preds)
    max_size = max_size or len(names)
    best: tuple[float, tuple[str, ...]] | None = None
    for size in range(min_size, max_size + 1):
        for combo in itertools.combinations(names, size):
            avg = average_predictions([val_preds[n] for n in combo])
            err = y_val - avg
            ss_tot = ((y_val - y_val.mean(axis=0)) ** 2).sum(axis=0)
            score = float((1 - (err**2).sum(axis=0) / ss_tot).mean())
            if best is None or score > best[0]:
                best = (score, combo)
    assert best is not None
    return best[1]
