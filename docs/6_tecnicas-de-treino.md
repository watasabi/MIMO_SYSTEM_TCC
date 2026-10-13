# Técnicas de treino

Este documento descreve o toolkit de treino em `src/mimo_sys/training/`
e o preparo de dados reversível em
`src/mimo_sys/preprocessors/timeseries.py`. Os resultados da ablação
estão em `notebooks/training/06_tecnicas_de_treino.ipynb` (experimento
MLflow `mimo_sys_training_techniques`). O histórico da mudança está em
`docs/changes/2026-10-13-training-toolkit.md`.

## Separação: preparo de dados vs. treino

| Onde | O quê | Reconstrução |
|---|---|---|
| `TimeSeriesPreprocessor` | log, imputação, outliers, EWT/detrend, filtro, diferenciação, scaler (`standard`/`minmax`/`robust`) e janelas com `step_size` | `inverse_transform(pred, anchors, feats, reference)` desfaz tudo que é posicional ou elemento a elemento |
| `mimo_sys.training` | augmentations, losses, RevIN, EMA/SWA, pré-treino contrastivo e tuning | não altera o sinal reconstruído: as augmentations só rodam no treino, e o RevIN se desfaz dentro do `forward` |

Regra prática: se a etapa muda o sinal que precisa voltar para a escala
original, ela pertence ao preprocessor.

### Reconstrução

- `anchor_indices(n)` devolve o índice do primeiro passo previsto de
  cada janela (`i * step_size + window_size`).
- `feats` permite inverter só as colunas-alvo, porque os modelos
  devolvem 3 saídas, não 7.
- `reference` é o segmento bruto que gerou as janelas. Com ele, a
  diferenciação e a tendência são desfeitas com os valores do próprio
  segmento, e o round trip fica exato (testado em
  `tests/test_preprocessors.py`). Sem ele, vale o comportamento
  anterior: estado do treino com a tendência extrapolada linearmente.
- **Atenção:** o EWT/detrend em `transform` é **não causal**, porque usa
  o segmento inteiro, incluindo pontos futuros à janela. Isso já valia
  antes desta mudança e afeta as variantes com EWT, não o baseline
  MinMax.
- **Correção:** o `transform` passou a aplicar o filtro **antes** da
  normalização, na mesma ordem do `fit_transform`. Antes a ordem era
  invertida, então val/teste eram processados diferente do treino.
  Variantes V1/V2 do notebook 03 mudam se forem rodadas de novo.

## O que veio de onde

- **Nixtla (`neuralforecast`), reimplementado sem depender do pacote:**
  - tipos de scaler (`scaler_type`) e janelas com `step_size`, no
    preprocessor;
  - RevIN/TemporalNorm (`training/scalers.py`);
  - MASE com escala do naive sazonal de 24h e Huber
    (`training/losses.py`);
  - espaços de busca por modelo no estilo `Auto*` (`training/tuning.py`).
- **`vision-toolbelt-liga`:**
  - NT-Xent e projection head (`training/contrastive.py`);
  - mixup, adaptado para regressão em batch;
  - padrão `freeze_backbone` (`EncoderFreeze`);
  - `MLflowModelCheckpoint`.

  O toolbelt não tem SimCLR, EMA, warmup nem Optuna; essas partes
  foram escritas aqui.
- **Próprio do problema:** `SpectralLoss`. A análise de
  `docs/changes/2026-10-09-analise-dominio-frequencia.md` mostrou que as
  redes perdem 30–94% da energia acima da banda diária, e essa loss
  penaliza essa perda diretamente no espectro (`log1p` das amplitudes).

## Técnicas

### Augmentations (`training/augmentations.py`)

São aplicadas só no `training_step`, com cada uma sorteada com
probabilidade `p` (`Compose`).

- **Conjuntas (x e y):**
  - `Scaling` e `MagnitudeWarp` escalam o desvio em torno da média da
    janela com o mesmo fator ou curva na entrada e no alvo, preservando
    a relação entrada→saída;
  - `Mixup` faz combinação convexa de pares, com x e y misturados pelo
    mesmo λ.
- **Só entrada:**
  - `Jitter`;
  - `TimeWarp`, que mantém o último passo fixo para não descolar do
    alvo;
  - `SpanMask`, que nunca mascara o último passo;
  - `ChannelMask`, só nas entradas `u1..u4` (queda de sensor);
  - `FreqMask`, que zera uma faixa do espectro sem tocar no DC.

### Regularização e otimização

- `Forecaster` (`training/forecaster.py`):
  - AdamW com `weight_decay`;
  - schedulers `plateau`, `cosine_warmup` e `onecycle`;
  - `gradient_clip_val` via `TrainConfig`;
  - `message_l1_weight` (L1 das mensagens do MPNN/SGN, como em
    Cranmer et al.);
  - `backcast_weight` (backcast do StemGNN, sem o qual o ramo de
    backcast não recebe gradiente);
  - teacher forcing do `Seq2SeqLatentGNN`.
- `EMA` (`training/callbacks.py`): valida com os pesos médios, salva
  os pesos médios no checkpoint e os mantém no modelo ao fim do treino.
  SWA usa o `StochasticWeightAveraging` do Lightning; com SWA, o teste
  usa os pesos finais e não o melhor checkpoint.
- **Seleção de modelo** (checkpoint, early stopping e plateau) usa
  `val_mse` no espaço do preprocessor, não `val_loss`, para que runs
  com losses diferentes sejam comparáveis.

### Pré-treino contrastivo (`training/contrastive.py`)

O pré-treino segue o SimCLR:

1. cada janela de treino gera duas vistas aumentadas;
2. as vistas passam por `model.encode(x)`, média no eixo 1 e projeção;
3. a NT-Xent aproxima as duas vistas da mesma janela e afasta as
   demais janelas do batch.

Só o encoder é treinado, no próprio objeto do modelo, e a projection
head é descartada. Depois, `EncoderFreeze` congela o encoder nas
primeiras épocas (linear probe) e o libera para o fine-tune.

| Arquitetura | `encode` |
|---|---|
| `Seq2SeqLatentGNN` | `fourier → gcn → LSTM encoder`, `(B, L, hidden)` |
| `MPNNForecaster` / `SymbolicGraphNetwork` | features que entram na última camada, `(B, nós, dim)`. Exige `n_layers ≥ 2`. |
| `StemGNN` | não suportado: os blocos produzem a previsão direto, sem um corte limpo, e o modelo já tem backcast auto-supervisionado |

### Tuning (`training/tuning.py`)

O tuning usa o extra `tune`:

```bash
uv run --extra ml --extra tune python -c \
  "from mimo_sys.training.tuning import tune; tune('mpnn', 30)"
```

- **Busca:** TPE (seed fixa) com `MedianPruner` sobre `val_mse`. Cada
  trial é um run MLflow aninhado sob `tune_<modelo>`.
- **Espaço:** os kwargs da arquitetura mais as técnicas acima (loss,
  RevIN, augment, peso espectral, scheduler, EMA).
- **Retreinar o vencedor:**
  `suggest_config(optuna.trial.FixedTrial(study.best_params), modelo,
  base)` e depois `train(...)`.

## Uso

```python
from mimo_sys.training import TrainConfig, train

result = train(TrainConfig(
    model="seq2seq_latent_gnn",
    revin="standard",
    augment=True,
    spectral_weight=0.1,
    ema_decay=0.999,
))
result.metrics       # R²/SMAPE por saída, forecast encadeado, escala original
result.bands         # energia por banda
```

- `train` salva `models/<run_name>.pth` e
  `models/<run_name>_preprocessor.joblib`, este último necessário para
  reconstruir. Use `save=False` em experimentos.
- Treine **um modelo por processo**, como faz o notebook 06.
