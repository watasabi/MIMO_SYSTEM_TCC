"""Collection Models v1 — split 70/15/15, todos os modelos.

Run pai `collection_70_15_15` com uma child run por modelo (2 ARMAX +
5 redes). Cada child registra métricas por saída e por banda de
frequência, o plot temporal, o plot espectral e — onde o eixo existir —
o mapa de attention sobre os lags. O run pai concentra os heatmaps de
resumo (R² modelo × saída e energia por banda), para não ser preciso
abrir cada child para comparar.

Lê os pesos de `models/*_70_15_15.pth`; **não treina nada**. A variante
que treina por fold é `06_collection_models_kfold.py`.

Uso:
    cd notebooks/training
    uv run --extra ml --group plot python 05_collection_models_v1.py
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from collection_common import (
    FEATURE_COLS,
    INPUT_COLS,
    INPUT_WINDOW,
    N_NODES,
    N_TARGETS,
    OUTPUT_COLS,
    OUTPUT_WINDOW,
    SEED,
    TARGET_IDX,
    band_table,
    chained_forecast,
    log_attention,
    log_figure,
    log_metrics_table,
    metrics_table,
    plot_band_heatmap,
    plot_r2_heatmap,
    plot_series,
    plot_spectrum,
)
from sklearn.preprocessing import MinMaxScaler
from torch import nn

from mimo_sys.architectures import (
    MPNNForecaster,
    Seq2SeqLatentGNN,
    StemGNN,
    SymbolicGraphNetwork,
)

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PROCESSED = Path("../../data/processed")
DATA_EXTERNAL = Path("../../data/external")
MODELS_DIR = Path("../../models")
FIGURES_DIR = Path("../../reports/figures/collection_70_15_15")

EXPERIMENT = "mimo_sys_collection_70_15_15"
PARENT_RUN = "collection_70_15_15"

# ---------------------------------------------------------------- dados

splits = {
    name: pd.read_parquet(DATA_PROCESSED / f"mimo_full_sample_{name}.parquet")
    for name in ("train", "val", "test")
}
full_df = pd.concat(splits.values(), ignore_index=True)
full_raw = full_df[FEATURE_COLS].to_numpy()
TRAIN_END = len(splits["train"])
VAL_END = TRAIN_END + len(splits["val"])
scaler = MinMaxScaler().fit(full_raw[:TRAIN_END])
scaled = scaler.transform(full_raw)

# ---------------------------------------------- ARMAX/MOGWO publicado

tabela3 = pd.read_csv(DATA_EXTERNAL / "tabela3_parametros_armax_papper.csv")
ARMAX_PARAMS = tabela3["MOGWO"].to_numpy()
U = full_df[INPUT_COLS].to_numpy().T
Y = full_df[OUTPUT_COLS].to_numpy().T


def armax_base(p, out, t, y, u):
    """Parte AR + exógena da predição da saída `out` em t (sem MA)."""
    base = out * 24
    others = [o for o in range(3) if o != out]
    a_self = p[base : base + 3]
    a_o1 = p[base + 3 : base + 6]
    a_o2 = p[base + 6 : base + 9]
    b_u = p[base + 9 : base + 21].reshape(4, 3)
    s = 0.0
    for k in range(3):
        s -= a_self[k] * y[out, t - 1 - k]
        s -= a_o1[k] * y[others[0], t - 1 - k]
        s -= a_o2[k] * y[others[1], t - 1 - k]
        s += b_u[:, k] @ u[:, t - 1 - k]
    return s


def armax_residuals(p, u, y):
    """Resíduos de 1 passo, necessários para os termos MA."""
    e = np.zeros_like(y)
    for t in range(3, y.shape[1]):
        for out in range(3):
            c = p[out * 24 + 21 : out * 24 + 24]
            pred = armax_base(p, out, t, y, u)
            pred += sum(c[k] * e[out, t - 1 - k] for k in range(3))
            e[out, t] = y[out, t] - pred
    return e


def armax_forecast(p, u, y, e, t0, horizon):
    """Free-run de `horizon` passos a partir de t0 (histórico medido)."""
    y_sim = y.copy()
    e_sim = e.copy()
    for t in range(t0, min(t0 + horizon, y.shape[1])):
        for out in range(3):
            c = p[out * 24 + 21 : out * 24 + 24]
            pred = armax_base(p, out, t, y_sim, u)
            pred += sum(c[k] * e_sim[out, t - 1 - k] for k in range(3))
            y_sim[out, t] = pred
            e_sim[out, t] = 0.0
    return y_sim[:, t0 : t0 + horizon]


def armax_predictions(start, stop):
    """Predições de 1 e de 12 passos no intervalo [start, stop)."""
    e = armax_residuals(ARMAX_PARAMS, U, Y)
    one, multi = [], []
    for t0 in range(start, stop - OUTPUT_WINDOW + 1, OUTPUT_WINDOW):
        block = np.zeros((3, OUTPUT_WINDOW))
        for i, t in enumerate(range(t0, t0 + OUTPUT_WINDOW)):
            for out in range(3):
                c = ARMAX_PARAMS[out * 24 + 21 : out * 24 + 24]
                pred = armax_base(ARMAX_PARAMS, out, t, Y, U)
                pred += sum(c[k] * e[out, t - 1 - k] for k in range(3))
                block[out, i] = pred
        one.append(block.T)
        multi.append(
            armax_forecast(ARMAX_PARAMS, U, Y, e, t0, OUTPUT_WINDOW).T
        )
    return np.concatenate(one), np.concatenate(multi)


# --------------------------------------------------- modelos treinados


def build_model(name: str) -> nn.Module:
    kwargs = dict(
        n_nodes=N_NODES,
        input_window=INPUT_WINDOW,
        output_window=OUTPUT_WINDOW,
        target_idx=TARGET_IDX,
    )
    if name == "MPNN":
        return MPNNForecaster(hidden_size=64, message_dim=8, **kwargs)
    if name == "SymbolicGraphNetwork":
        return SymbolicGraphNetwork(hidden_size=64, message_dim=8, **kwargs)
    if name == "StemGNN":
        return StemGNN(hidden_size=64, **kwargs)
    if name == "Seq2SeqLatentGNN":
        return Seq2SeqLatentGNN(hidden_size=64, **kwargs)
    raise ValueError(name)


class _Seq2SeqEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=False
        )

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell


class _Seq2SeqDotAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, hidden, encoder_outputs):
        hidden_last = hidden[-1]
        weights = torch.bmm(
            encoder_outputs.permute(1, 0, 2), hidden_last.unsqueeze(2)
        ).squeeze(2)
        weights = torch.softmax(weights, dim=1)
        context = torch.bmm(
            weights.unsqueeze(1), encoder_outputs.permute(1, 0, 2)
        ).squeeze(1)
        out = torch.tanh(self.fc(torch.cat((context, hidden_last), dim=1)))
        return out, weights


class _Seq2SeqDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=False
        )
        self.attention = _Seq2SeqDotAttention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell, encoder_outputs):
        _, (hidden, cell) = self.lstm(x, (hidden, cell))
        attn_out, weights = self.attention(hidden, encoder_outputs)
        return self.fc(attn_out), hidden, cell, weights


class Seq2SeqAttentionRef(nn.Module):
    """Mesma arquitetura de `01_seq2seq_attention.ipynb` (seq-first).

    Guarda os pesos de attention em `last_attention_map`
    `(batch, output_window, input_window)` para o heatmap de lags.
    """

    def __init__(
        self, input_size, hidden_size, output_size, output_window, num_layers=1
    ):
        super().__init__()
        self.encoder = _Seq2SeqEncoder(input_size, hidden_size, num_layers)
        self.decoder = _Seq2SeqDecoder(
            input_size, hidden_size, output_size, num_layers
        )
        self.output_window = output_window
        self.output_size = output_size
        self.last_attention_map: torch.Tensor | None = None

    def forward(self, source):
        _, batch_size, _ = source.shape
        outputs = torch.zeros(
            self.output_window,
            batch_size,
            self.output_size,
            device=source.device,
        )
        attn = []
        encoder_outputs, hidden, cell = self.encoder(source)
        decoder_input = source[-1].unsqueeze(0)
        for t in range(self.output_window):
            decoder_output, hidden, cell, weights = self.decoder(
                decoder_input, hidden, cell, encoder_outputs
            )
            outputs[t] = decoder_output
            attn.append(weights)
            next_input = source[-1].clone()
            next_input[:, : self.output_size] = decoder_output
            decoder_input = next_input.unsqueeze(0)
        self.last_attention_map = torch.stack(attn, dim=1)
        return outputs


def load_architecture(name: str) -> nn.Module:
    model = build_model(name)
    model.load_state_dict(
        torch.load(
            MODELS_DIR / f"{name.lower()}_70_15_15.pth", map_location=device
        )
    )
    return model.eval().to(device)


def load_seq2seq_ref() -> Seq2SeqAttentionRef:
    model = Seq2SeqAttentionRef(N_NODES, 64, N_TARGETS, OUTPUT_WINDOW, 2)
    model.load_state_dict(
        torch.load(
            MODELS_DIR / "seq2seq_attention_mimo_70_15_15.pth",
            map_location=device,
        )
    )
    return model.eval().to(device)


# ----------------------------------------------------------------- run

armax_1, armax_12 = armax_predictions(VAL_END, len(full_raw))
y_true = Y[:, VAL_END : VAL_END + len(armax_1)].T
idx = VAL_END + np.arange(len(y_true))
ATTENTION_T0 = VAL_END + INPUT_WINDOW

ARCH_NAMES = ["MPNN", "SymbolicGraphNetwork", "StemGNN", "Seq2SeqLatentGNN"]
members: list[tuple[str, str, object]] = [
    ("ARMAX 1 passo", "armax", armax_1),
    ("ARMAX 12 passos", "armax", armax_12),
    ("Seq2SeqAttention", "net_seq_first", load_seq2seq_ref()),
]
members += [(name, "net", load_architecture(name)) for name in ARCH_NAMES]

mlflow.set_experiment(EXPERIMENT)
summary, band_summary = {}, {}

with mlflow.start_run(run_name=PARENT_RUN) as parent:
    mlflow.log_params(
        {
            "seed": SEED,
            "split": "70/15/15",
            "input_window": INPUT_WINDOW,
            "output_window": OUTPUT_WINDOW,
            "n_membros": len(members),
            "forecast": "encadeado 12 passos, escala original",
            "treino": "nenhum (pesos de models/)",
        }
    )
    print("experimento:", EXPERIMENT)
    print("parent run:", parent.info.run_id)

    for name, kind, obj in members:
        with mlflow.start_run(run_name=name, nested=True):
            if kind == "armax":
                y_pred = obj
                mlflow.log_params({"model": name, "family": "ARMAX/MOGWO"})
            else:
                seq_first = kind == "net_seq_first"
                y_pred = chained_forecast(
                    obj,
                    scaled,
                    scaler,
                    VAL_END,
                    len(full_raw),
                    device,
                    seq_first=seq_first,
                )
                mlflow.log_params(
                    {
                        "model": name,
                        "family": "rede neural",
                        "hidden_size": 64,
                        "n_params": sum(p.numel() for p in obj.parameters()),
                    }
                )

            table = metrics_table(y_true, y_pred)
            bands = band_table(y_true, y_pred)
            summary[name] = table
            band_summary[name] = bands
            log_metrics_table(table, bands)

            log_figure(
                plot_series(y_true, y_pred, name, idx, table),
                "serie",
                FIGURES_DIR / name,
            )
            log_figure(
                plot_spectrum(y_true, y_pred, name, bands),
                "espectro",
                FIGURES_DIR / name,
            )

            if kind != "armax":
                window = torch.FloatTensor(
                    scaled[ATTENTION_T0 - INPUT_WINDOW : ATTENTION_T0]
                )
                window = (
                    window.unsqueeze(1)
                    if kind == "net_seq_first"
                    else window.unsqueeze(0)
                ).to(device)
                log_attention(obj, window, name, FIGURES_DIR / name)

            print(
                f"  {name:22s} R² médio {table['r2'].mean():.3f}  "
                f"SMAPE {table['smape_%'].mean():.2f}%"
            )

    # Resumos no run pai, para comparar sem abrir cada child.
    results = pd.concat(summary, names=["modelo", "saida"])
    freq_results = pd.concat(band_summary, names=["modelo"])
    r2 = results["r2"].unstack("saida")

    log_figure(
        plot_r2_heatmap(r2, "R² por modelo e saída — split 70/15/15"),
        "resumo_r2",
        FIGURES_DIR,
    )
    for out in OUTPUT_COLS:
        log_figure(
            plot_band_heatmap(
                freq_results,
                out,
                f"Erro de energia por banda — {out} (70/15/15)",
            ),
            f"resumo_bandas__{out}",
            FIGURES_DIR,
        )
        mlflow.log_metric(f"melhor_r2__{out}", float(r2[out].max()))
        mlflow.set_tag(f"melhor_modelo__{out}", r2[out].idxmax())

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(FIGURES_DIR / "metricas.csv")
    freq_results.to_csv(FIGURES_DIR / "metricas_frequencia.csv")
    mlflow.log_artifact(str(FIGURES_DIR / "metricas.csv"))
    mlflow.log_artifact(str(FIGURES_DIR / "metricas_frequencia.csv"))

print()
print(r2.round(3))
