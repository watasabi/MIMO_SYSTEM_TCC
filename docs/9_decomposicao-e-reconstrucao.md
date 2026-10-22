# Decomposição das saídas e reconstrução válida em produção

Plano e regras: `docs/changes/2026-10-22-decomposicao-saidas.md`.
Diagnóstico: `notebooks/eda/07_decomposicao_saidas.ipynb`. Código:
`training/features.py` (decomposições), `training/evaluation.py`
(`forecast_at_origin`, estatística), `notebooks/training/scripts/run_decomposition.py`.

## 1. Resumo

- **Nenhuma decomposição da saída melhora o SpectralTFNet.** Wavelet
  causal (à trous de Haar), EWT corrigido e diferença sazonal ficam
  iguais ou piores que o modelo de referência do doc 8 no teste.
  Nenhuma passa na regra definida antes de rodar.
- **Dar peso igual às bandas na loss piora muito** (−0,03 a −0,16 de R²
  médio). As bandas de 2–8h são quase ruído, e a normalização por
  componente dá a elas o mesmo peso da deriva e do ciclo diário.
- **A diferença sazonal de 24h é a única que chega perto**:
  - `pumps`: validação +0,012 (IC −0,011..+0,029), teste +0,002;
  - `full`: teste −0,022 (IC −0,040..−0,008).

  Ela é frágil a mudanças de horário da operação: no meio do teste, a
  base `y(t−24)` repete o pico de ontem na hora errada.
- **O ganho real desta rodada é de infraestrutura**: uma API de previsão
  que não consegue vazar o futuro, testes de causalidade e replay, e a
  prova de que o EWT/detrend e o Savgol do preprocessor vazavam.
- **Dois problemas de configuração foram encontrados e corrigidos:**
  - o padrão `backcast_weight=1,0` piora o SpectralTFNet (~0,01 no
    `full`, ~0,04 no `pumps`). Os experimentos usam 0, como o doc 8;
  - no tuning anterior do `pumps`, o modelo lia a vazão de entrada
    futura zerada como se fosse conhecida. Agora `known_future_idx` é
    derivado de `future_known`.

## 2. O contrato de produção

Uma previsão na origem `t0` só pode usar as linhas observadas antes de
`t0` e as entradas conhecidas do horizonte (bombas; mais a vazão de
entrada no `full`). `evaluation.forecast_at_origin` recebe exatamente
isso e nada mais. As células desconhecidas do horizonte entram como NaN.
Uma transformação que mistura linhas no tempo põe NaN na janela passada,
e a função recusa em vez de vazar.

| Teste (`tests/test_causality.py`) | O que garante |
|---|---|
| Perturbação: ruído em tudo o que é desconhecido a partir de `t0` | a previsão em `t0` não muda (diferença ≤ 1e-10) |
| Replay: colunas derivadas recalculadas só com as linhas já recebidas | mesma previsão da avaliação em lote, sem skew treino/serviço |
| Reconstrução: componentes verdadeiras → `Forecaster.reconstruct` | devolvem as saídas verdadeiras |
| EWT/detrend e Savgol centrado | mudam o passado ao mexer no futuro, e são recusados |
| Round trip com EWT usando só o passado | não fecha; o teste antigo fecha porque lê a tendência do futuro |

O `chained_forecast` passou a ser um laço sobre `forecast_at_origin`. No
baseline ele reproduz o caminho antigo com diferença 0,0, então nenhum
resultado anterior muda.

## 3. As decomposições

Todas são causais (a linha `t` só usa linhas `≤ t`) e somam o sinal com
erro ≤ 1e-13:

- **à trous de Haar** (J=3 ou 4): bandas de 2–4h, 4–8h, 8–16h, 16–32h e
  >32h, as mesmas de `evaluation.BANDS`. Sem erro de borda, mas o Haar
  separa mal as frequências.
- **EWT-c**: o EWT do projeto com três correções:
  - banco normalizado para somar 1 (o do `ewtpy` soma entre 0,5 e 1,0,
    por isso a soma errava 7–31%);
  - calculado por origem só com o passado;
  - borda direita estendida com o último dia, o que cortou o erro de
    borda de 9,7 para 3,2 na vazão.
- **sazonal 24h**: `y(t) − y(t−24)` previsto, mais `y(t−24)`, que já é
  conhecido na origem para horizontes até 24h.

Cada componente vira um nó do grafo. O modelo prevê as componentes e a
saída é a soma delas em unidades originais (com MinMax, os offsets não se
cancelam no espaço escalado). A loss é
`λ·L(soma) + (1−λ)·L(componentes)`.

## 4. Onde está o erro do baseline (Fase 1, validação)

| | ciclo diário 16–32h | deriva >32h | 2–8h |
|---|---|---|---|
| vazão: erro/sinal na banda (full → pumps) | 0,02 → 0,03 | 0,07 → **0,16** | 0,14–0,35 |
| pressão | 0,03 | 0,23–0,26 | 0,26–0,52 |
| nível direto | 0,29 → 0,42 | 0,59 → **0,70** | 0,28–0,65 |

O ciclo diário já está resolvido. O erro que sobra está na deriva de
baixa frequência, que dobra no `pumps` por falta da vazão de entrada, e
no ruído de alta frequência. Separar a saída em bandas não acrescenta
informação a nenhuma das duas, e é isso que a Fase 2 confirmou.

## 5. Resultados

R² médio das 3 saídas, média de 3 seeds (desvio entre seeds ≤ 0,004).
Previsões em **todas as horas** de validação e teste. Δ e IC 95% contra o
C0, com bootstrap pareado em blocos de 24 origens. A fonte do nível
(direto ou cascata) foi escolhida na validação: cascata no `full`,
direto no `pumps`. Todos os runs usam `backcast_weight=0` (o modelo de
referência do doc 8). A primeira rodada, com o padrão 1,0, está em
`outputs/decomposition_bw1/` e chegou às mesmas conclusões.

| Variante | full val | full teste (Δ, IC) | pumps val (Δ, IC) | pumps teste (Δ, IC) |
|---|---|---|---|---|
| C0 baseline | 0,881 | 0,831 | 0,763 | 0,708 |
| C1 à trous, escala da saída, λ=1 (controle) | 0,881 | 0,833 (+0,002) | 0,757 (−0,006) | 0,701 (−0,007) |
| C2 à trous J=4, por componente, λ=0 | 0,828 | 0,797 (**−0,034**) | 0,678 (**−0,086**) | 0,595 (**−0,113**) |
| C3 à trous J=4, por componente, λ=0,5 | 0,875 | 0,833 (+0,002) | 0,746 (−0,017) | 0,683 (−0,024) |
| C4a/C4b à trous J=3, λ=0 / 0,5 | 0,829 / 0,874 | 0,796 / 0,833 | 0,647 / 0,747 | 0,565 / 0,684 |
| EWT-c N=3, λ=0 / 0,5 | 0,801 / 0,883 | 0,762 / 0,828 | 0,625 / 0,754 | 0,561 / 0,696 |
| EWT-c N=5, λ=0 / 0,5 | 0,792 / 0,880 | 0,772 / 0,832 | 0,614 / 0,754 | 0,547 / 0,693 |
| C5 sazonal 24h | 0,874 | 0,809 (**−0,022**, −0,040..−0,008) | 0,775 (+0,012, −0,011..+0,029) | 0,710 (+0,002, −0,013..+0,014) |
| C6 à trous só na entrada | 0,883 | 0,833 (+0,002, −0,001..+0,003) | 0,760 (−0,004) | 0,706 (−0,001) |

Tabela completa, com R² por saída e Diebold–Mariano:
`outputs/decomposition_report.json` e `outputs/decomposition_report.txt`.

**Reprodução do baseline**: com `backcast_weight=0` o C0 reproduz
exatamente os runs antigos `stf_h12_res_w168_b128_s42` (0,925 / 0,635 /
0,810 / 0,765) e `stf_nofutin_h12_res_w168_b128_s42`, em CPU e em GPU.

## 6. Leitura

- **λ=0 é o problema de escala na prática.** Normalizar cada componente
  equilibra as bandas, mas a métrica é sobre a soma: o MSE da soma pesa
  cada banda pela própria variância. O peso extra nas bandas de ruído
  tira capacidade da deriva e do ciclo diário.
- **λ=0,5 e o controle C1 ficam no ruído.** Para o núcleo linear, prever
  bandas e somar é a mesma classe de modelo que prever a saída (filtro de
  filtro continua sendo um filtro). O resultado confirma a expectativa
  registrada no plano.
- **O EWT-c não supera a à trous.** A seletividade melhor das bandas não
  compensa o erro de borda que sobra.
- **O C5 é frágil a mudanças de programação.** Por saída no teste do
  `pumps`, o nível sobe de 0,426 para 0,482 e a pressão cai de 0,769 para
  0,735 (números da primeira rodada; o padrão é o mesmo). Mais de 90% da perda de pressão vem de 24 das
  321 origens, em torno da hora 170 do teste, onde o ciclo diário muda de
  fase (na linha 344, 21,0 hoje contra 26,0 ontem na mesma hora). A base
  sazonal fixa "hoje = ontem", enquanto o baseline se condiciona à
  programação real das bombas. **Não recomendo adotar o C5 como está.**

## 7. Próximos passos (não feitos)

- **Sazonal só para o nível** (é onde o C5 ganha) ou uma base sazonal
  condicionada à programação das bombas, por exemplo a mesma hora do dia
  com o mesmo padrão de bombas. Ressalva: o teste já foi consultado para o
  C5, então isso exige decidir na validação e, idealmente, uma janela de
  teste nova.
- A deriva de baixa frequência da vazão no `pumps` é falta de
  informação, não de forma do alvo. O caminho mais promissor é uma
  previsão melhor da vazão de entrada (um modelo próprio para ela, ou o
  perfil diário condicionado às bombas), e não uma decomposição.
- A Fase 4 do plano (reconciliação e Optuna sobre a vencedora) não foi
  disparada, porque nenhuma variante ganhou de forma robusta.
