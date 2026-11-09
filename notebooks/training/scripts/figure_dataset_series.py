"""Figura das sete séries do reservatório, com as divisões usadas.

Gera `dados_series.pdf` para a subseção Materiais do artigo: as quatro
entradas e as três saídas no tempo, com o treino/validação/teste
(70/15/15) e o recorte do estudo de referência marcados.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path.home() / 'code/TCC_MIMO_SYS_overleaf/imagens/dados_series.pdf'

data = pd.read_parquet(ROOT / 'data/processed/mimo_data.parquet')
hours = range(len(data))

panels: list[tuple[str, str]] = [
    ('freq_b1', 'bomba 1 $u_1$ (Hz)'),
    ('freq_b2', 'bomba 2 $u_2$ (Hz)'),
    ('freq_b3', 'bomba 3 $u_3$ (Hz)'),
    ('vazao_entrada', 'vazão de\nentrada $u_4$ (l/s)'),
    ('vazao_distribuicao', 'vazão de\ndistribuição $y_1$ (l/s)'),
    ('nivel_res', 'nível $y_2$ (m)'),
    ('pressao', 'pressão $y_3$ (mca)'),
]

n_train, n_val = 1545, 331
fig, axes = plt.subplots(len(panels), 1, figsize=(10, 10), sharex=True)
for ax, (col, ylabel) in zip(axes, panels):
    ax.plot(hours, data[col], lw=0.6, color='C0')
    ax.axvspan(n_train, n_train + n_val, color='0.85', zorder=0)
    ax.axvspan(n_train + n_val, len(data), color='0.7', zorder=0)
    ax.set_ylabel(ylabel, fontsize=8)

for start, end in [(550, 1150), (1500, 2000)]:
    axes[0].annotate(
        '', xy=(end, 1.04), xytext=(start, 1.04),
        xycoords=('data', 'axes fraction'),
        arrowprops={'arrowstyle': '<->', 'color': 'k', 'lw': 0.8},
    )
axes[0].text(850, 1.08, 'estimação (referência)', ha='center',
             transform=axes[0].get_xaxis_transform(), fontsize=8)
axes[0].text(1750, 1.08, 'validação (referência)', ha='center',
             transform=axes[0].get_xaxis_transform(), fontsize=8)
axes[-1].text(n_train + n_val / 2, 0.05, 'val.', ha='center',
              transform=axes[-1].get_xaxis_transform(), fontsize=8)
axes[-1].text(n_train + n_val * 1.5, 0.05, 'teste', ha='center',
              transform=axes[-1].get_xaxis_transform(), fontsize=8)
axes[-1].set_xlabel('hora (27/07/2014 a 27/10/2014)')
fig.tight_layout()
fig.savefig(OUT)
print(OUT)
