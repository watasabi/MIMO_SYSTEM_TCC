# Split do artigo vs. dataset completo

Resumo da exploração em
`notebooks/eda/02_split_vs_dataset_completo.ipynb`. Nota de mudança
associada: `docs/changes/2026-10-02-full-dataset-splits.md`. Contexto
do dataset: `docs/1_analise-dados-reservatorio.md`.

## 1. Pergunta

O artigo ajusta o ARMAX/MOGWO em uma janela pequena da série. O
recorte é representativo do regime operacional completo? Quanto o
modelo publicado degrada fora dessa janela?

## 2. Recorte usado no artigo

`SANEPAR_COMPLETO.txt` tem **2.208 amostras** (as 2.216 do
`mimo_data.xlsx` menos as 8 linhas `"Bad"`). O artigo usa só:

| Papel | Linhas (MATLAB, 1-indexado) | Slice Python | Amostras |
|---|---|---|---|
| Estimação (treino) | `550:1150` | `549:1150` | 601 |
| Validação (teste) | `1500:2000` | `1499:2000` | 501 |
| **Total usado** | | | **1.102 (49,9%)** |
| Fora do recorte | | | 1.106 |

O trecho `1150:1500` entre as duas janelas, e as pontas da série,
nunca entram no ajuste nem na validação publicados.

## 3. Estatísticas do recorte vs. restante

Estatísticas descritivas (completo, recorte, fora do recorte) e
histogramas sobrepostos por saída estão no notebook. Em linhas gerais,
médias do recorte ficam próximas às do completo (ex.: vazão de
distribuição 68,5 vs. 67,6 l/s; nível 2,40 vs. 2,40 m; pressão 22,3
vs. 22,2 mca), mas o recorte tem **menos dispersão no nível**
(desvio 0,25 vs. 0,30 m; mínimo 1,60 vs. 0,007 m). Ou seja, o recorte
exclui os episódios de falha do transmissor de nível descritos em
`1_analise-dados-reservatorio.md` e parte dos regimes extremos.

## 4. ARMAX/MOGWO (Tabela 3) no dataset completo

Mesma `armax_predict` (predição one-step, equação 15) da reprodução
em `notebooks/modeling/01_mogwo_armax_reproducao.ipynb`, com os 72
parâmetros da coluna MOGWO da Tabela 3
(`data/external/tabela3_parametros_armax_papper.csv`), aplicada às
amostras do recorte (sanity check) e às 2.208 completas.

| Saída | R² recorte | R² completo | MSE recorte | MSE completo | SMAPE% recorte | SMAPE% completo |
|---|---|---|---|---|---|---|
| y1 vazão distribuição | 0,938 | 0,879 | 41,15 | 80,06 | 7,90 | 8,52 |
| y2 nível | 0,954 | 0,862 | 0,0029 | 0,0120 | 1,54 | 2,34 |
| y3 pressão | 0,848 | 0,800 | 2,563 | 3,614 | 5,04 | 5,88 |

MAE: recorte 4,45 / 0,036 / 1,00; completo 4,79 / 0,047 / 1,11.

Notas:

- Os valores no recorte batem com a Tabela 2 do artigo, o que valida
  a reimplementação.
- Cuidado ao comparar: MSE do recorte e do completo são calculados
  sobre conjuntos de tamanhos diferentes, e o R² do "recorte" mistura
  treino e teste do artigo (as duas janelas concatenadas, com a
  predição one-step reiniciada no início da concatenação).

## 5. Conclusões

1. O artigo usa metade dos dados disponíveis e a metade ignorada
   contém regimes mais difíceis (principalmente no nível).
2. O modelo publicado **degrada de forma moderada, sem colapsar**:
   R² cai de ~0,94/0,95/0,85 para ~0,88/0,86/0,80. A maior perda
   relativa é em `y2` (nível), onde o MSE quadruplica.
3. **Hipótese, não verificada:** parte da degradação pode vir de
   outliers/falhas de sensor no trecho excluído (o nível mínimo do
   completo é 0,007 m), e não só de mudança de regime. Para testar,
   recalcular as métricas excluindo essas amostras (ver
   `1_analise-dados-reservatorio.md`, seção 2).
4. Para comparar modelos novos (ex.: Seq2Seq) de forma justa, é
   melhor avaliar também na série completa, e não só no recorte do
   artigo.

## 6. Splits gerados sobre o dataset completo

`notebooks/processing/03_full_dataset_sample.py` não embaralha
(preserva ordem temporal) e escreve em `data/processed/`:

| Esquema | Descrição | Arquivos |
|---|---|---|
| `TimeSeriesSplit` + holdout | 5 folds sobre as primeiras 2.008 amostras; 200 últimas como teste fixo, fora dos folds | `mimo_full_kfold_assignments.parquet` (fold/role/sample_idx), `mimo_full_kfold_test.parquet` |
| Cronológico 70/15/15 | treino 1.545 / validação 331 / teste 332 | `mimo_full_sample_{train,val,test}.parquet` |

Use o holdout do esquema 1 só na avaliação final; para seleção de
hiperparâmetros use os folds ou o conjunto de validação do esquema 2.

## 7. Seq2Seq vs. ARMAX/MOGWO (70/15/15 e k-fold)

`notebooks/training/02_seq2seq_attention_splits_vs_armax.ipynb` treina
o Seq2Seq com attention de `01_seq2seq_attention.ipynb`
(`INPUT_WINDOW=24`, `OUTPUT_WINDOW=12`, LSTM 2x64, `SEED=42`, early
stopping com validação cronológica) nos dois esquemas da seção 6 e
compara com o ARMAX/MOGWO da Tabela 3, **sem reajuste**. A avaliação
usa forecast encadeado de janelas não sobrepostas de 12 passos, com as
24 amostras anteriores como contexto, e as mesmas amostras para os
três modelos.

No k-fold, cada fold usa os últimos 15% do seu treino como validação
de early stopping (treinos efetivos de 288 a 1.423 amostras); a
validação do fold e o holdout de 200 amostras só avaliam. O holdout
é o mesmo em todos os folds, então o desvio do ARMAX nele é zero.

O ARMAX entra em dois modos, porque o artigo reporta 1 passo e o
Seq2Seq prevê 12:

- **1 passo**: usa o `y` medido nos lags. Não é previsão de 12 passos;
  serve de teto.
- **12 passos**: free-run a partir do último `y` e resíduo medidos,
  realimentando a própria predição. Mesmo horizonte do Seq2Seq, mas o
  ARMAX enxerga `u` futuro e o Seq2Seq não.

R² (k-fold: média ± desvio entre os 5 folds):

| Esquema | Saída | ARMAX 1 passo | ARMAX 12 passos | Seq2Seq 12 passos |
|---|---|---|---|---|
| 70/15/15 teste (332) | vazão | 0,915 | 0,570 | **0,877** |
| | nível | 0,932 | 0,293 | **0,435** |
| | pressão | 0,671 | 0,148 | **0,653** |
| k-fold, validação do fold | vazão | 0,879 ± 0,143 | 0,589 ± 0,054 | **0,734 ± 0,159** |
| | nível | 0,929 ± 0,033 | -0,167 ± 1,033 | **0,193 ± 0,244** |
| | pressão | 0,819 ± 0,066 | 0,372 ± 0,102 | **0,604 ± 0,190** |
| k-fold, holdout (200) | vazão | 0,899 | 0,611 | **0,746 ± 0,251** |
| | nível | 0,936 | **0,337** | 0,075 ± 0,231 |
| | pressão | 0,592 | 0,107 | **0,515 ± 0,207** |

MSE e MAE completos estão no notebook. Exemplos (70/15/15 teste, MSE
vazão / nível / pressão): ARMAX 1 passo 53,6 / 0,0034 / 4,72; ARMAX
12 passos 270,3 / 0,0353 / 12,23; Seq2Seq 77,0 / 0,0282 / 4,98.

Leitura:

1. No horizonte de 12 passos o Seq2Seq supera o ARMAX publicado em 8
   das 9 combinações esquema/saída. A exceção é o nível no holdout do
   k-fold (0,075 vs. 0,337).
2. O ARMAX de 1 passo é o melhor em quase tudo, mas não é comparável a
   um forecast de 12 passos. A comparação da nota
   `2026-09-28-seq2seq-attention-mimo.md` (MSE 33,5 vs. 85,9) mistura
   os dois horizontes.
3. A variância do Seq2Seq é grande: nos folds, o desvio do R² fica
   entre 0,16 e 0,25, e o MSE de vazão no holdout tem média 155,5 e
   desvio 153,8. Os folds iniciais treinam com só 288 a 856 amostras,
   o que explica parte disso. O resultado de 70/15/15, com um único
   split e um seed, parece otimista frente ao k-fold; use o k-fold
   como medida de incerteza.
4. O nível (`y2`) é o ponto fraco dos dois modelos em 12 passos. Ele se
   comporta como integrador de vazão e acumula erro. O ARMAX free-run
   chega a R² negativo em alguns folds (desvio de 1,03).
5. Limites desta comparação: um seed por treino e sem tuning de
   hiperparâmetros. O ARMAX tem parâmetros fitados na janela `549:1150`
   do artigo, que **cai dentro das validações dos folds 0 a 3** do
   k-fold. Nessas validações o ARMAX já viu os dados, o que o favorece.
   No 70/15/15 (teste a partir da amostra 1876) e no holdout (a partir
   da 2008) não há esse vazamento.

## 8. Seq2Seq com `TimeSeriesPreprocessor`

`notebooks/training/03_seq2seq_attention_preprocessor.ipynb` treina o
mesmo Seq2Seq no split 70/15/15 com variantes de
`mimo_sys.preprocessors.TimeSeriesPreprocessor` e mede o erro na
**escala original**, via `inverse_transform`, nas 297 janelas
deslizantes de teste (12 passos, passo 1). Esse protocolo difere do da
seção 7 (forecast encadeado), então os números não são comparáveis.

Variantes:

- `V0`: só `StandardScaler`.
- `V1`: remoção de outliers (IQR, limiar 1,5) + filtro Savitzky-Golay.
- `V2`: `V1` + EWT/detrend, ajustado só no treino. Val/teste passam por
  `transform`, e o `inverse_transform` extrapola a tendência.
- `V3`: `V2` ajustado na série toda. **Vaza o futuro** (a tendência do
  teste é conhecida), então serve só de limite superior.

| Variante | R² vazão | R² nível | R² pressão | MSE vazão |
|---|---|---|---|---|
| Persistência (último `y`) | -1,184 | -0,786 | -1,139 | 1368,1 |
| V0 normalização | 0,884 | **0,441** | **0,642** | 72,7 |
| V1 outliers + filtro | **0,889** | 0,357 | 0,599 | **69,3** |
| V2 EWT detrend (fit treino) | 0,779 | 0,276 | 0,432 | 138,4 |
| V3 EWT detrend (fit série toda, vaza) | 0,871 | 0,297 | 0,635 | 81,0 |

Leitura:

1. O preprocessor **não melhorou** o Seq2Seq. `V0` e `V1` ficam
   empatados dentro do que um seed e uma rodada permitem distinguir
   (`V1` ganha 0,005 de R² na vazão e perde 0,08 no nível).
2. EWT/detrend piora o resultado fora da amostra (`V2`: R² de vazão
   0,779 e de pressão 0,432). Mesmo a versão que vaza o futuro (`V3`)
   não supera `V0`. A tendência removida não parece ser o que limita o
   modelo, e somar de volta uma tendência extrapolada acrescenta erro.
3. O `inverse_transform` de `V2` extrapola a tendência a partir do fim
   do treino com uma rampa linear, a mesma para todas as janelas, o que
   é frágil para horizontes longe do treino. Para previsão
   fora da amostra, a decomposição EWT precisaria ser causal.
4. O limiar padrão do IQR (`outlier_threshold=0,05`) corta os dados para
   dentro do intervalo interquartil, o que destrói a série (por exemplo,
   a frequência da bomba 3 vira constante). Use 1,5 ou outro valor
   explícito.
5. Limites: um seed, sem tuning, e os quatro treinos pararam por early
   stopping entre as épocas 40 e 48.

Correção em `src`: `TimeSeriesPreprocessor.transform` com EWT quebrava
(`EWT1D() got an unexpected keyword argument 'boundaries'`), porque o
`ewtpy` instalado não aceita fronteiras dadas. Foi adicionada
`_ewt_with_boundaries`, que repete o espelhamento e o banco de filtros
de Meyer com as fronteiras do treino; `tests/test_preprocessors.py`
verifica que `transform` reproduz `fit_transform` nos mesmos dados.

## 10. Todas as arquiteturas vs. ARMAX/MOGWO (split 70/15/15)

`notebooks/training/04_arquiteturas_vs_armax.ipynb` treina as quatro
arquiteturas de `src/mimo_sys/architectures/` (`MPNNForecaster`,
`SymbolicGraphNetwork`, `StemGNN`, `Seq2SeqLatentGNN`; ver
[`architectures/README.md`](../src/mimo_sys/architectures/README.md))
no mesmo split 70/15/15, mesma janela (`INPUT_WINDOW=24`,
`OUTPUT_WINDOW=12`) e mesmo protocolo de avaliação da seção 7, e junta
o resultado ao `Seq2SeqAttention` (seção 7) e ao ARMAX/MOGWO. Todas as
arquiteturas usam `hidden_size=64`; `MPNN` e `SymbolicGraphNetwork` usam
`message_dim=8` (maior que o default 2, pensado para extração
simbólica — aqui o objetivo é só acurácia).

Cada arquitetura foi treinada em um **processo separado**
(`uv run --extra ml python train_single.py <nome>`, script fora do
repositório): treinar as quatro em sequência num mesmo processo
esgotava a RAM da máquina de desenvolvimento (7,7 GB). O notebook
carrega os pesos já salvos em `models/<nome>_70_15_15.pth` em vez de
retreinar.

R² no teste (12 passos, forecast encadeado):

| Modelo | R² vazão | R² nível | R² pressão | Época de parada |
|---|---|---|---|---|
| ARMAX 1 passo | 0,915 | 0,932 | 0,671 | — |
| ARMAX 12 passos | 0,570 | 0,293 | 0,148 | — |
| Seq2SeqAttention | 0,877 | 0,435 | 0,653 | 47 |
| MPNN | 0,909 | 0,248 | 0,667 | 71 |
| **SymbolicGraphNetwork** | **0,915** | 0,260 | **0,693** | 103 |
| StemGNN | 0,889 | **0,386** | 0,594 | 186 |
| Seq2SeqLatentGNN | 0,875 | 0,304 | 0,633 | 135 |

Leitura:

1. Todas as cinco redes superam o ARMAX de 12 passos nas três saídas,
   confirmando o padrão da seção 7 com mais arquiteturas.
2. Nenhuma rede chega perto do ARMAX de 1 passo — esperado, já que ele
   usa `y` medido a cada passo e as redes fazem forecast livre de 12
   passos. Não é uma comparação justa, só um teto de referência.
3. `SymbolicGraphNetwork` tem a melhor vazão e pressão entre as redes,
   e é **destilável** (mensagem de 8 dimensões, arquitetura pensada
   para extração simbólica — ver `architectures/README.md`). É a
   melhor combinação de acurácia e interpretabilidade encontrada até
   aqui.
4. `StemGNN` é o único a melhorar o nível (R² 0,386) mas o pior em
   vazão e pressão; não é destilável (opera em base espectral).
5. `Seq2SeqAttention`, a rede mais simples (sem grafo, só LSTM com
   attention), fica no meio do pacote — não é a pior nem a melhor em
   nenhuma saída.
6. Limites: um seed por arquitetura, sem tuning de hiperparâmetros
   (`hidden_size=64` fixo em todas), e `message_dim=8` em vez do
   default 2 do `MPNN`/`SymbolicGraphNetwork` não foi comparado contra
   o default nesta rodada.

## 11. Como reproduzir

```bash
cd notebooks/processing && uv run python 03_full_dataset_sample.py
cd ../training && uv run jupyter nbconvert --to notebook --execute \
    --inplace 02_seq2seq_attention_splits_vs_armax.ipynb
# Preprocessor (precisa dos extras ml e do grupo plot):
uv run --extra ml --group plot jupyter nbconvert --to notebook --execute \
    --inplace 03_seq2seq_attention_preprocessor.ipynb
# Todas as arquiteturas (treine cada uma em processo separado se a
# máquina tiver pouca RAM; ver nota na seção 10):
uv run --extra ml --group plot jupyter nbconvert --to notebook --execute \
    --inplace 04_arquiteturas_vs_armax.ipynb
# EDA: notebooks/eda/02_split_vs_dataset_completo.ipynb
```
