"""Rota A — extração de equação (destilação simbólica).

Veículo: `MPNNForecaster`/`SymbolicGraphNetwork`
(`src/mimo_sys/architectures/mpnn.py`). Ver `docs/4_rotas-de-extracao.md`
§Rota A para a receita completa (estágios I/II e por que
`input_window` precisa ser pequeno).
"""

from mimo_sys.explainers.equation.distill import (
    default_message_feature_names,
    default_update_feature_names,
    distill_message_fn,
    distill_update_fn,
)

__all__ = [
    "default_message_feature_names",
    "default_update_feature_names",
    "distill_message_fn",
    "distill_update_fn",
]
