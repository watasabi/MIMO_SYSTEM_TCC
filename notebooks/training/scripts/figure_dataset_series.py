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

panels: list[tuple[list[str], list[str], str]] = [
    (['freq_b1', 'freq_b2', 'freq_b3'],
     ['bomba 1', 'bomba 2', 'bomba 3'], 'frequência (Hz)'),
    (['vazao_entrada', 'vazao_distribuicao'],
     ['entrada $u_4$', 'distribuição $y_1$'], 'vazão (l/s)'),
    (['nivel_res'], ['nível $y_2$'], 'nível (m)'),
    (['pressao'], ['pressão $y_3$'], 'pressão (mca)'),
]

n_train, n_val = 1545, 331
fig, axes = plt.subplots(4, 1, figsize=(10, 7.5), sharex=True)
for ax, (cols, labels, ylabel) in zip(axes, panels):
    for col, label in zip(cols, labels):
        ax.plot(hours, data[col], lw=0.6, label=label)
    ax.axvspan(n_train, n_train + n_val, color='0.85', zorder=0)
    ax.axvspan(n_train + n_val, len(data), color='0.7', zorder=0)
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper left', fontsize=8, ncol=3)

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
