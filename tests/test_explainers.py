"""Contract tests for the explainers that don't need optional deps.

`pysr`/`torch-geometric` are optional extras (`symbolic`/`explain`);
these tests cover the parts that work without them — feature naming,
the `edge_strength_frame` labeling, and the ImportError message when
the optional dependency is missing.
"""

import pandas as pd
import pytest
import torch

from mimo_sys.explainers.equation import (
    default_message_feature_names,
    default_update_feature_names,
    distill_message_fn,
)
from mimo_sys.explainers.graph import edge_strength_frame


def test_default_message_feature_names_order():
    names = default_message_feature_names(input_window=3)
    assert names == [
        "recv_lag0",
        "recv_lag1",
        "recv_lag2",
        "send_lag0",
        "send_lag1",
        "send_lag2",
    ]


def test_default_update_feature_names_order():
    names = default_update_feature_names(input_window=2, message_dim=2)
    assert names == ["own_lag0", "own_lag1", "msg0", "msg1"]


class _FakeEdgeStrengthModel:
    @staticmethod
    def edge_strength(x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        strength = torch.arange(n * n, dtype=torch.float32).reshape(n, n)
        return strength.fill_diagonal_(0.0)


def test_edge_strength_frame_labels_and_values():
    model = _FakeEdgeStrengthModel()
    x = torch.zeros(1, 3, 4)
    names = ["u1", "u2", "y1", "y2"]

    frame = edge_strength_frame(model, x, feature_names=names)

    assert isinstance(frame, pd.DataFrame)
    assert list(frame.index) == names
    assert list(frame.columns) == names
    assert frame.loc["u1", "u1"] == 0.0
    assert frame.loc["y2", "u1"] == model.edge_strength(x)[3, 0]


def test_edge_strength_frame_rejects_wrong_length_names():
    model = _FakeEdgeStrengthModel()
    x = torch.zeros(1, 3, 4)
    with pytest.raises(ValueError, match="feature_names"):
        edge_strength_frame(model, x, feature_names=["only_one"])


def test_distill_message_fn_without_pysr_raises_import_error():
    try:
        import pysr  # noqa: F401, PLC0415
    except ImportError:
        pass
    else:
        pytest.skip("pysr is installed; ImportError path not reachable")

    with pytest.raises(ImportError, match="pysr"):
        distill_message_fn(
            torch.zeros(2, 6), torch.zeros(2, 2), input_window=3
        )
