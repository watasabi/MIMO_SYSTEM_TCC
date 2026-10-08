"""Rota B — extração de subgrafo/topologia.

B1 (`edge_strength_frame`) é o entregável barato: só rotula a matriz
que o modelo já expõe de graça via `edge_strength()`. B2
(`PGExplainerTopology`) é um fallback opcional, só se B1 não bastar —
ver `docs/4_rotas-de-extracao.md` §Rota B.
"""

from mimo_sys.explainers.graph.topology import (
    PGExplainerTopology,
    edge_strength_frame,
)

__all__ = ["PGExplainerTopology", "edge_strength_frame"]
