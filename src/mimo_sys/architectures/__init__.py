"""Architectures for MIMO system identification on the reservoir data.

All five share one contract:

    forward(x) : (batch, input_window, n_nodes)
              -> (batch, output_window, len(target_idx))

Node order follows `FEATURE_COLS` of the processing notebooks
(`u1..u4` then `y1..y3`), and `target_idx` selects the output nodes.
**`n_nodes` includes the outputs** — every model sees its own past
`y1..y3` lags as part of `x`, not just the exogenous `u1..u4`. This is
`u, y(t-1, t-2, ...) -> y(t, t+1, ...)`, not a memoryless `u(t) -> y(t)`
map: the window never includes `y` at the step being predicted (no
overlap between the input window and the target horizon — see
`mimo_sys.training.data.WindowDataset`/`evaluation.chained_forecast`),
so there is no leakage, but the system's own memory of itself is very
much part of the input.

Which model serves which extraction route is documented in
`docs/4_rotas-de-extracao.md`; the triage that selected them is in
`docs/3_investigacao-arquiteturas.md`.
"""

from mimo_sys.architectures.fourier_gnn import FourierGNN
from mimo_sys.architectures.linear_edge_net import LinearEdgeNet
from mimo_sys.architectures.mpnn import MPNNForecaster
from mimo_sys.architectures.seq2seq_latent_gnn import Seq2SeqLatentGNN
from mimo_sys.architectures.spectral_tf_net import SpectralTFNet
from mimo_sys.architectures.stemgnn import StemGNN
from mimo_sys.architectures.symbolic_graph_network import (
    SymbolicGraphNetwork,
)

__all__ = [
    "FourierGNN",
    "LinearEdgeNet",
    "MPNNForecaster",
    "Seq2SeqLatentGNN",
    "SpectralTFNet",
    "StemGNN",
    "SymbolicGraphNetwork",
]
