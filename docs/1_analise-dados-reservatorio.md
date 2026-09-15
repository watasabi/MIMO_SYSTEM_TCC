# Análise exploratória — dados do reservatório São José

Fonte: `WRSO and NSWRSO_draft_all.zip` (repositório do paper Ferrari/Leandro/Coelho 2025).
Arquivo: `mimo_data.xlsx`, aba `Planilha1`.

## 1. O dataset

**Reservatório São José** (Sanepar). 2.216 amostras horárias, **2014-07-27 01:00 → 2014-10-27 09:00** (3 meses).
O paper usou apenas 1.100 (600 estimação + 500 validação) — **há o dobro de dados disponível**.

| Col | Tag SCADA | Significado | Unid. | Média | Min | Max |
|---|---|---|---|---|---|---|
| u1 | SÃO JOSÉ CENTRO / FREQ B1 | freq. bomba 1 | Hz | 31.3 | 0 | 60 |
| u2 | SÃO JOSÉ CENTRO / FREQ B2 | freq. bomba 2 | Hz | 35.9 | 0 | 60 |
| u3 | SÃO JOSÉ CENTRO / FREQ B3 | freq. bomba 3 | Hz | 3.4 | 0 | 60 |
| u4 | FT01 / ENTR ARJ | vazão de **entrada** | l/s | 66.2 | ~0 | 200.2 |
| y1 | FT02 / RCEN | vazão de **distribuição** | l/s | 67.6 | **-20.1** | 199.6 |
| y2 | LT01 / RSE | **nível** do reservatório | m | 2.403 | **0.007** | 3.030 |
| y3 | PT02 / RCEN | **pressão** | mca | 22.17 | **0.23** | 30.29 |

Bomba 3 quase não opera (mediana 0, ativa em só ~3% das amostras).

## 2. Problemas de qualidade de dados — **o paper não fez nenhum pré-processamento**

- **8 linhas com a string literal `"Bad"`** em todas as colunas (flag de falha do SCADA). Se lidas como
  número no MATLAB viram `NaN` ou `0` silenciosamente.
- **8 amostras com `y2 < 1.0 m`** (nível operacional ≈ 2.4 m), incluindo `y2 = 0.007 m` — índices
  80 e 85–91, um bloco contíguo. É falha do transmissor de nível, não esvaziamento real.
- **`|Δy2| > 1 m/h` em 3 amostras** — fisicamente impossível para um reservatório de ~1.200 m².
- **2 amostras com vazão negativa** (`y1 < 0`) e **15 com `y3 < 5 mca`**.
- **1 gap de 2 h** na série (resto é horário contíguo).

Total: **25 amostras problemáticas de 2.216 (1,1%)**. Elas caem dentro da janela de 600 usada
para estimação no paper. **Esta é a explicação mais provável para o colapso de `y2`** nos algoritmos
mono-objetivo (R² = −15.9 para GA, −3.2 para CS), muito mais que o argumento de "diferença de
ordem de magnitude entre as saídas" dado pelos autores.

## 3. O baseline que o paper não reportou

**Persistência** — o preditor trivial `y[k] = y[k−1]`, com **zero parâmetros**:

| Saída | Persistência (0 params) | Paper, melhor modelo (≈120 params, 100 runs) | Ganho |
|---|---|---|---|
| y1 | 0.8336 | 0.9394 | +0.106 |
| y2 | 0.8233 | 0.9596 | +0.136 |
| y3 | 0.7513 | 0.8633 | +0.112 |

Na janela de 1.100 amostras usada pelo paper: y1 = 0.8521, y2 = 0.8105, y3 = 0.7938.

**Implicação forte:** um ARMAX de ~120 parâmetros, estimado com 100 execuções de metaheurística
multi-objetivo (até 2.150 s por run no NSGA-II), supera um preditor de zero parâmetros por
~0,11 de R². O R² de **predição um passo à frente** é uma métrica quase vazia para sinais
amostrados a 1 h com dinâmica lenta. **Nenhum dos seis papers de referência reporta R² de
simulação livre (free-run).** Essa é uma lacuna metodológica concreta e defensável.

## 4. A física está nos dados — mas só depois da limpeza

Balanço de massa `A·dy2/dt = u4 − y1`, discretizado com regra do trapézio:

| Condição | R² (1 passo) |
|---|---|
| dados crus, `q[k]` | 0.3606 |
| dados crus, `q[k−1]` | 0.4268 |
| dados crus, trapézio | 0.4736 |
| **limpo (−21 amostras), trapézio** | **0.8434** |
| **limpo, SEM intercepto (1 único parâmetro)** | **0.8406** |

Ajuste obtido:

```
Δy2[k] = 3.03e-3 · ½(q[k] + q[k−1]) + 5.07e-3 ,   q = u4 − y1
```

→ **área implícita do reservatório A = 3.6/a ≈ 1.190 m²** (≈ 39 m de diâmetro equivalente).
Valor fisicamente plausível e **verificável contra a planta real** — é uma validação externa
que nenhum modelo caixa-preta pode oferecer.

Correlação `corr(Δy2, ū4−ȳ1) = 0.918` após limpeza.

### Saídas algébricas

```
y3 ≈ 1.02·y2 + 0.093·y1 + 0.052·u1 + 0.054·u2 + 0.042·u3 + 9.76      R² = 0.8247  (6 params)
y1 ≈ 0.166·u1 + 0.160·u2 + 0.218·u3 + 4.76·y3 − 49.8                 R² = 0.7769  (5 params)
```

`y3 ~ y2` sozinho dá R² = 0.011 — a pressão é dominada pela vazão (perda de carga), não pela
coluna d'água estática. O termo `y1²` **não** ajuda (R² = 0.79 < 0.82), então a perda de carga
aqui é aproximadamente linear no regime observado.

**Modelo gray-box completo: 13 parâmetros** (2 + 6 + 5) contra ~120 do ARMAX.

## 5. O problema difícil: simulação livre

Split 70/30 (1.530 treino / 662 teste ≈ 33 dias). Simulação de `y2` **só a partir das entradas**,
sem realimentar a medida:

| Modelo de `y2` | R² (1 passo) | **R² free-run** | Deriva final |
|---|---|---|---|
| `a·q` (1 param) | 0.6204 | **−19.58** | −1.84 m |
| `a·q + b` (2 params) | 0.6237 | **−35.28** | +2.27 m |
| `a·q + b − λ·y2` (3 params) | 0.6272 | **+0.3563** | **+0.11 m** |

O integrador puro **diverge** — qualquer viés residual acumula ao longo de 662 passos.
O **integrador com fuga** (`λ = 0.0227`, τ ≈ 44 h) fica estável, com deriva de apenas 0,11 m
em 33 dias, mas explica só 36% da variância em free-run.

Fisicamente, o termo `−λ·y2` representa consumo/extravasamento não medido dependente do nível —
ou seja, **falta um fluxo não instrumentado no balanço**. Identificar esse termo é, por si só,
uma contribuição de engenharia.

## 6. O que isso muda no desenho do TCC

1. **A métrica vira a contribuição.** Reportar R² de simulação livre (não um passo à frente) e
   mostrar que os modelos publicados não sobrevivem a ela é um resultado forte, honesto e barato.
2. **Estrutura em espaço de estados é obrigatória.** O modelo tem que prever `Δy2`, não `y2`,
   e precisa de um termo estabilizador. Um ARMAX de ordem 3 sobre `y2` não tem como capturar isso.
3. **A GNN/rede tem que bater 13 parâmetros, não 120.** O gray-box acima é o baseline real.
   Se a rede não superar isso em free-run, o trabalho é sobre por que ela não supera.
4. **Pré-processamento não é detalhe.** Remover 1,1% das amostras dobrou o R² do balanço de massa
   (0.47 → 0.84). Vale uma seção inteira.

## Arquivos gerados

- `reservatorio_sao_jose_limpo.csv` — dados com colunas `flag_bad` e `flag_out` marcando as
  25 amostras problemáticas (nada foi removido; a decisão fica com você).

## Também no zip (ainda não explorado a fundo)

- `miso_data.xls` — versão MISO do mesmo sistema (provavelmente o estudo anterior, ref. [26]).
- `Single-objective_algorithms.zip`, `Multi_objective_algorithms.zip` — implementações MATLAB
  (GA, GWO, CS, NSGA-II, MOGWO, MOCS) incluindo os benchmarks CEC 2009/2019.
- `Single_objective_for_MISO_systems.zip`, `Multi_objective_for_MIMO_systems.zip` —
  os scripts de identificação propriamente ditos, com a função objetivo do ARMAX. **Vale ler
  para reproduzir exatamente o split 600/500 e a função de custo do paper.**
