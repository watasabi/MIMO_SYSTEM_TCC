"""Architectures for MIMO system identification on the reservoir data.

All four share one contract:

    forward(x) : (batch, input_window, n_nodes)
              -> (batch, output_window, len(target_idx))

Node order follows `FEATURE_COLS` of the processing notebooks
(`u1..u4` then `y1..y3`), and `target_idx` selects the output nodes.

Which model serves which extraction route is documented in
`docs/4_rotas-de-extracao.md`; the triage that selected them is in
`docs/3_investigacao-arquiteturas.md`.
"""

from mimo_sys.architectures.mpnn import MPNNForecaster
from mimo_sys.architectures.seq2seq_latent_gnn import Seq2SeqLatentGNN
from mimo_sys.architectures.stemgnn import StemGNN
from mimo_sys.architectures.symbolic_graph_network import (
    SymbolicGraphNetwork,
)

__all__ = [
    "MPNNForecaster",
    "Seq2SeqLatentGNN",
    "StemGNN",
    "SymbolicGraphNetwork",
]
