"""Subgrafo/topologia de acoplamento (Rota B, `4_rotas-de-extracao.md`).

B1 é a leitura direta de `model.edge_strength(x)` — grátis, porque é o
mesmo mecanismo de L1 que torna a mensagem interpretável (ver
`3_investigacao-arquiteturas.md` §2). B2 é um wrapper fino e opcional
sobre PGExplainer, só para o caso em que B1 não baste; `torch_geometric`
é importado lazy, então essa dependência nunca é exigida para importar
este módulo.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import pandas as pd
import torch

if TYPE_CHECKING:
    from torch_geometric.explain import Explainer

__all__ = ["PGExplainerTopology", "edge_strength_frame"]


class _EdgeStrengthModel(Protocol):
    def edge_strength(self, x: torch.Tensor) -> torch.Tensor: ...


def edge_strength_frame(
    model: _EdgeStrengthModel,
    x: torch.Tensor,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Rotula `model.edge_strength(x)` com nomes de nó (Rota B1).

    Args:
        model: Arquitetura que expõe `edge_strength(x) -> (n, n)`
            (`MPNNForecaster`, `SymbolicGraphNetwork`).
        x: Batch de janelas de entrada, `(batch, input_window, n_nodes)`.
        feature_names: Nomes dos nós, na ordem usada pelo modelo
            (ex. `FEATURE_COLS` do notebook). Default `node0..node{n-1}`.

    Returns:
        `DataFrame` `[n_nodes, n_nodes]`; `[i, j]` é quanto o nó `j`
        empurra no nó `i`. Diagonal zero por construção do modelo.
    """
    strength = model.edge_strength(x)
    n = strength.shape[0]
    names = feature_names or [f"node{i}" for i in range(n)]
    if len(names) != n:
        raise ValueError("feature_names must have length n_nodes")
    return pd.DataFrame(strength.numpy(), index=names, columns=names)


class PGExplainerTopology:
    """Wrapper fino sobre `torch_geometric.explain.PGExplainer` (Rota B2).

    Só usar se a matriz de B1 (`edge_strength_frame`) discordar dos
    baselines clássicos (RGA, resposta ao degrau, Granger) — ver
    `4_rotas-de-extracao.md` §Rota B. Monta o grafo completo (sem
    self-loops) implícito em `MPNNForecaster`/`SymbolicGraphNetwork` e
    delega treino/explicação ao `torch_geometric`; não reimplementa
    PGExplainer.

    `model` precisa seguir a convenção node-level do `torch_geometric`
    (`forward(x, edge_index) -> (n_nodes, out_dim)`), diferente do
    `forward(x) -> (batch, window, nodes)` nativo das arquiteturas deste
    projeto — adapte com um wrapper antes de passar aqui.

    Args:
        model: Modelo node-level compatível com `torch_geometric`.
        n_nodes: Número de nós do grafo.
        epochs: Épocas de treino do explainer amortizado.
    """

    def __init__(
        self, model: torch.nn.Module, n_nodes: int, *, epochs: int = 30
    ) -> None:
        try:
            from torch_geometric.explain import (  # noqa: PLC0415
                Explainer,
                ModelConfig,
                PGExplainer,
            )
        except ImportError as exc:
            raise ImportError(
                "torch-geometric e necessario para PGExplainerTopology; "
                "instale o extra opcional "
                "(`uv add --extra explain torch-geometric`)."
            ) from exc
        self.n_nodes = n_nodes
        self.edge_index = self._full_edge_index(n_nodes)
        self.explainer: Explainer = Explainer(
            model=model,
            algorithm=PGExplainer(epochs=epochs),
            explanation_type="phenomenon",
            edge_mask_type="object",
            model_config=ModelConfig(
                mode="regression",
                task_level="node",
                return_type="raw",
            ),
        )

    @staticmethod
    def _full_edge_index(n_nodes: int) -> torch.Tensor:
        pairs = [
            (i, j) for i in range(n_nodes) for j in range(n_nodes) if i != j
        ]
        return torch.tensor(pairs, dtype=torch.long).t().contiguous()

    def fit(self, x: torch.Tensor, target: torch.Tensor) -> None:
        """Treina o explainer amortizado numa amostra `(x, target)`."""
        self.explainer.algorithm.train(
            epoch=0,
            model=self.explainer.model,
            x=x,
            edge_index=self.edge_index,
            target=target,
        )

    def explain(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Máscara de aresta aprendida, `(n_edges,)`."""
        return self.explainer(x, self.edge_index, target=target).edge_mask
