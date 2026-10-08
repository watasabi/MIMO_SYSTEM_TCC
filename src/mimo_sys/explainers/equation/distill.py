"""Symbolic distillation of `message_fn`/`update_fn` (Rota A, estágios I/II).

Operacionaliza `docs/4_rotas-de-extracao.md` §Rota A: roda PySR sobre os
pares `(inputs, outputs)` que `MPNNForecaster.collect_message_io`/
`collect_update_io` já extraem do modelo treinado. A expressão final é
`phi . Agg . psi` (fase I destila `psi`, fase II destila `phi`).

`pysr` é um extra opcional (`uv add --extra symbolic pysr`, requer
Julia); o import é lazy para que importar `mimo_sys` nunca precise dele.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pysr import PySRRegressor

__all__ = [
    "default_message_feature_names",
    "default_update_feature_names",
    "distill_message_fn",
    "distill_update_fn",
]


def _pysr_regressor(**pysr_kwargs: object) -> "PySRRegressor":
    try:
        from pysr import PySRRegressor  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "pysr e necessario para destilacao simbolica; instale o "
            "extra opcional (`uv add --extra symbolic pysr`, requer "
            "Julia local)."
        ) from exc
    return PySRRegressor(**pysr_kwargs)


def default_message_feature_names(input_window: int) -> list[str]:
    """Nomes de coluna para `collect_message_io` (entrada de `psi`).

    Ordem casa com `MPNNForecaster._pairwise_inputs`: janela do
    receptor, depois do emissor. Os nomes são agnósticos ao nó físico
    porque `message_fn` é compartilhada entre todas as arestas.
    """
    recv = [f"recv_lag{k}" for k in range(input_window)]
    send = [f"send_lag{k}" for k in range(input_window)]
    return recv + send


def default_update_feature_names(
    input_window: int, message_dim: int
) -> list[str]:
    """Nomes de coluna para `collect_update_io` (entrada de `phi`)."""
    own = [f"own_lag{k}" for k in range(input_window)]
    msg = [f"msg{k}" for k in range(message_dim)]
    return own + msg


def distill_message_fn(
    inputs: torch.Tensor,
    messages: torch.Tensor,
    *,
    input_window: int,
    feature_names: list[str] | None = None,
    **pysr_kwargs: object,
) -> "PySRRegressor":
    """Destila `psi` (estágio I) a partir de `collect_message_io`.

    Args:
        inputs: `(n_edges, 2 * input_window)`, como retornado por
            `MPNNForecaster.collect_message_io`.
        messages: `(n_edges, message_dim)`, idem.
        input_window: Tamanho da janela de lag (para nomear colunas).
        feature_names: Sobrescreve `default_message_feature_names`.
        **pysr_kwargs: Passados para `pysr.PySRRegressor` (ex.
            `niterations`, `binary_operators`).

    Returns:
        O `PySRRegressor` já ajustado (`.equations_` tem as expressões).
    """
    names = feature_names or default_message_feature_names(input_window)
    model = _pysr_regressor(**pysr_kwargs)
    model.fit(inputs.numpy(), messages.numpy(), variable_names=names)
    return model


def distill_update_fn(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    *,
    input_window: int,
    message_dim: int,
    feature_names: list[str] | None = None,
    **pysr_kwargs: object,
) -> "PySRRegressor":
    """Destila `phi` (estágio II) a partir de `collect_update_io`.

    Rodar depois de `distill_message_fn`: a mensagem agregada já está
    reduzida, então esta regressão é a barata (ver
    `docs/4_rotas-de-extracao.md`).

    Args:
        inputs: `(n, input_window + message_dim)`, como retornado por
            `MPNNForecaster.collect_update_io`.
        outputs: `(n, out_dim)`, idem.
        input_window: Tamanho da janela de lag (para nomear colunas).
        message_dim: Largura do bottleneck de mensagem (idem).
        feature_names: Sobrescreve `default_update_feature_names`.
        **pysr_kwargs: Passados para `pysr.PySRRegressor`.

    Returns:
        O `PySRRegressor` já ajustado.
    """
    names = feature_names or default_update_feature_names(
        input_window, message_dim
    )
    model = _pysr_regressor(**pysr_kwargs)
    model.fit(inputs.numpy(), outputs.numpy(), variable_names=names)
    return model
