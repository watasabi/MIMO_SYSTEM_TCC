# Plano de melhoria do GOTAS

Continuação de `docs/18_plano-gotas-w.md`. Este documento reúne o diagnóstico de
por que o GOTAS perde para XGBoost e CatBoost na validação cruzada, as variantes
em teste e como acompanhar o que está rodando.

## Regras

- O modelo continua uma GNN de filtros, com explicabilidade extraível: equação,
  filtro por aresta ou mapa de atenção.
- Toda escolha é feita **só na validação**. O teste é aberto uma única vez, no
  final, para a variante escolhida.
- Os concorrentes não são enfraquecidos. Se o GOTAS mudar as entradas, os
  concorrentes são refeitos com as mesmas entradas.
- Critério de aprovação (fixado antes de rodar): ganho acima de 0,012 de R² na
  métrica-alvo em algum cenário, sem perder mais de 0,005 em nenhum.

## Diagnóstico

**Na validação os modelos empatam; a diferença aparece no teste, sob falha de
sensor.** R² em 12 h, média das 3 saídas e dos 5 folds:

| Modelo | Validação `full` | Teste `full` | Validação `pumps` | Teste `pumps` |
|---|---|---|---|---|
| GOTAS | 0,868 | 0,709 | 0,731 | 0,606 |
| GOTAS-W | 0,865 | ainda não aberto | 0,736 | ainda não aberto |
| XGBoost | 0,858 | 0,832 | 0,721 | 0,703 |
| CatBoost | 0,862 | 0,824 | 0,740 | 0,721 |

- **O fold 2 decide a média.** No teste dele (horas 1328–1547, com a falha longa
  de 1445–1488), o GOTAS cai para 0,239 no `full`. As outras redes também
  desabam, e as árvores ficam perto de 0,8. A figura
  `outputs/cv/figuras/folds_teste_full.png` mostra o nível do GOTAS despencando
  durante a falha e a pressão errada logo depois.
- **Por que as árvores resistem.** Um modelo linear (ridge) com as mesmas
  entradas e o mesmo alvo do XGBoost mostra três efeitos:

  | Ingrediente | Efeito no modelo linear |
  |---|---|
  | Tirar o nó do balanço acumulado | validação +0,040 (`full`) e +0,073 (`pumps`) |
  | Prever a variação desde o último valor medido | fold 2 do `full` vai de −0,105 para 0,537 |
  | Não linearidade | o que resta: cerca de 0,05 abaixo do XGBoost no teste do `full` |

  O nó do balanço acumulado soma o desbalanço desde o início da série. Cada falha
  de sensor o desloca para sempre, e depois do bloco de falhas ele fica fora da
  faixa do treino (até −1,87 na escala em que o treino vai de 0 a 1). As árvores
  cortam valores fora da faixa, e as redes extrapolam. A cascata física do nível
  não depende desse nó: usa `fit_mass_balance` sobre os dados brutos.
- **Por fold** (teste, R² médio das 3 saídas): `outputs/cv/report_por_fold.csv`.
- **O gap está no nível direto** (validação, 19/11/2026). A cascata física do GOTAS 3
  já é boa, mas a saída direta de nível da rede desaba quando a falha está na janela
  passada (R² aproximado na validação do `full`):

  | Nível | f1 | f2 | f3 | f4 | f5 |
  |---|---|---|---|---|---|
  | GOTAS 3, direto | 0,72 | 0,81 | 0,08 | 0,84 | 0,55 |
  | GOTAS 3, cascata | 0,55 | 0,81 | 0,65 | 0,76 | 0,77 |
  | XGBoost, direto | 0,64 | 0,72 | 0,47 | 0,83 | 0,79 |
  | CatBoost, direto | 0,61 | 0,76 | 0,55 | 0,82 | 0,79 |

  O nível direto do GOTAS é uma soma linear da janela inteira: 44 h de vazão de
  entrada errada passam pelo kernel longo. A árvore só corta em limiares, e a
  resposta fica limitada. No `pumps` não há cascata, e esse é todo o nível. O
  GOTAS-AE ataca isso: o contexto da atenção é média ponderada dos estados passados
  (resposta limitada) e pode aprender a pular as horas com falha.

## Critério de explicabilidade

Todo modelo candidato precisa ter a explicabilidade extraível: subgrafo ou equação.
O GOTAS atual usa o resíduo MLP em 9 de 10 folds, e o artigo o declara (+0,036
sobre o grafo). A regra para os modelos novos é a mesma: **a parte explicável (o
modelo sem o resíduo MLP) fica no máximo cerca de 0,04 abaixo do modelo completo
na validação**. A triagem grava essa medida (`explicavel_r2`) e o relatório de
acompanhamento a mostra na coluna `explic.`. O resíduo de Chebyshev
(`residual_kind="chebyshev"`) é um polinômio NARX explícito e conta como parte
explicável.

No GOTAS-AE, `notebooks/training/scripts/extract_gotas_ae.py <cenário> <fold>` extrai:

- **a equação**: com a atenção fixada na média da validação, o modelo vira uma
  equação afim exata por aresta, defasagem e passo,
  `y_i(t+k) = âncora_i + b_ik + Σ_j Σ_τ h·s_j(t−τ) + Σ g·u(t+k)`, mais o termo de
  Chebyshev quando houver. Com a atenção de cada origem, a mesma equação vale
  exatamente, com ganhos que mudam com o ponto de operação (LPV);
- **o subgrafo**: a força de cada aresta nó → saída nessa equação, sem a âncora,
  podada em 10% da aresta mais forte de cada saída;
- **a fidelidade**: o R² da equação contra o modelo completo e contra o valor real.

Uma tentativa sorteada (`pumps` fold 3) deu 0,539 completo, 0,539 com a atenção
fixa e 0,210 sem o resíduo MLP: fixar a atenção não custou nada, mas o resíduo fazia
quase todo o trabalho, e por isso essa configuração seria reprovada. A busca passou a
escolher também entre o resíduo MLP e o de Chebyshev.

### Resultados da explicabilidade (19/11/2026)

Validação, média de 5 folds × 3 sementes; "explicável" é o modelo sem o resíduo MLP.

| Modelo | `full` completo | `full` explicável | `pumps` completo | `pumps` explicável |
|---|---|---|---|---|
| GOTAS 3 (resíduo MLP) | 0,862 | 0,603 | 0,729 | 0,367 |
| GOTAS-SS | 0,861 | 0,577 | 0,737 | 0,478 |
| GOTAS-AE (busca livre) | 0,859 | 0,574 | 0,738 | 0,481 |
| GOTAS-AE com Chebyshev | 0,854 | 0,854 | 0,722 | 0,722 |

- O +0,036 do resíduo no artigo vem de uma ablação retreinada. Tirar o MLP do GOTAS 3
  já treinado custa 0,26 (`full`) e 0,36 (`pumps`): o grafo explica só parte do que
  ele prevê.
- GOTAS-AE com Chebyshev contra o GOTAS 3, pareado: validação −0,008 / −0,007,
  estresse +0,008 / +0,020. A equação reproduz o modelo com R² de 0,998 a 1,000.
- A penalidade esparsa desliga o ramo de autoencoder e atenção (fixar a atenção
  custa no máximo 0,003, e os ganhos do passado das saídas são zero). O que sobra é
  `y_i(t+k) = média_i + b + Σ g·(u(t+k) − média) + Σ c·T_d(y_i(t−l))`: não lê o
  passado da vazão de entrada nem das outras saídas, e por isso resiste às falhas.
  No artigo, o ganho não pode ser atribuído à atenção.
- **GOTAS-AE explícito (`gotas_aex`, busca própria, só Chebyshev)**, contra o GOTAS 3,
  pareado: base, validação +0,006 / +0,005 e estresse −0,027 / +0,013; com falhas
  simuladas, validação +0,005 / +0,007 e estresse −0,018 / +0,019. A parte explicável é
  igual ao modelo, e a fidelidade da equação é 1,000 nos 10 folds e cenários. O ramo
  de autoencoder e atenção volta a ser desligado pela penalidade esparsa.
- **Teste aberto uma vez (20/11/2026)** para o GOTAS-AE explícito com falhas simuladas
  (`gotas_aex`; `run_gotas_v2.py final gotas_aex gotas_aex falhas_simuladas`). R² de
  12 h, média ± dp dos 5 folds: `full` 0,834 ± 0,066 (XGBoost 0,832, CatBoost 0,824,
  GOTAS 3 0,794); `pumps` 0,689 ± 0,131 (CatBoost 0,721, XGBoost 0,703, GOTAS 3
  0,606). Menor RMSE e MAPE nos dois cenários. Nenhuma diferença contra as árvores é
  significativa (Diebold-Mariano: p de 0,22 a 0,76). No `pumps`, perde no nível (0,336
  contra 0,444 e 0,488) e ganha na pressão. Por fold: `outputs/cv/report_por_fold.csv`.
- Busca feita: `gotas_aex` (o resíduo sempre Chebyshev, com as
  defasagens e o grau na busca), seguida da triagem e da extração (`screen7_*`,
  `extract_gotas_aex.log`).

## Variantes

| Nome | O que muda | Explicabilidade | Situação |
|---|---|---|---|
| GOTAS-W | wavelet à trous causal, um filtro por aresta e componente | um filtro por aresta (impulso, polos) | busca feita: empata na validação, menos robusto sob estresse |
| Centro na mediana ou no último valor | centro da janela | igual | triado: piora |
| Limite nos detalhes wavelet | corta picos por escala | equação com saturação | triado: +0,019 / +0,013 no estresse, abaixo do critério |
| Balanço em janela móvel de 168 h | nó de balanço sem acumular desde o início | igual | triado: −0,011 / −0,014 na validação |
| **Treino com falhas simuladas** | zero, valor preso e picos nas entradas do treino | igual | em triagem: estresse +0,126 / +0,062 até agora |
| **Sem o nó do balanço acumulado** | grafo com 7 nós | igual, cascata física mantida | na fila (lote 2 da triagem) |
| GOTAS-LPV | filtros que variam com as bombas e curvas polinomiais (KAN) nas entradas | duas equações por aresta: base e modulação | busca na fila |
| GOTAS-A | atenção por aresta sobre as defasagens, previsão direta a partir do último valor | filtro explícito `g·α` por aresta (atenção fixa) ou mapa de atenção (ligada às bombas) | busca na fila |
| GOTAS-SS | autoencoder linear de espaço de estados (Koopman) | `x' = A·x + B·u`, `y = C·x + D·u`, polos de `A`, `C(zI−A)⁻¹B + D` | busca e triagem em andamento |
| **GOTAS-AE** | wavelet à trous por nó, autoencoder linear por banda, atenção softmax no latente (seq2seq), bandas somadas, saída a partir do último valor ou da média | grafo por banda `C_m E_m`, mapa de atenção (passo × defasagem) e kernel `α·C_m E_m` que reproduz a previsão exatamente | na fila, depois do GOTAS-SS |

Todas ficam em `src/mimo_sys/architectures/` (`gotas_wavelet.py`, `gotas_lpv.py`,
`gotas_attention.py`), registradas em `ARCHITECTURES` (`training/train.py`) com os
nomes `gotas_w`, `gotas_lpv` e `gotas_a`. A triagem é
`notebooks/training/scripts/run_gotas_v2.py`.

**Validação sob estresse:** a validação de cada fold é refeita com falhas simuladas
nas entradas (uma queda a zero de 12 h na vazão de entrada, um valor preso de 24 h
numa saída e três picos de saturação na vazão de distribuição), comparando com os
valores reais. As colunas derivadas são recalculadas sobre as linhas com falha.

## Fila de execução

1. Triagem, lote 1: GOTAS, GOTAS-W e 5 variantes de robustez, 3 sementes.
2. Triagem, lote 2 (começa sozinho): sem o nó do balanço e treino com falhas
   simuladas, no GOTAS e no GOTAS-W.
3. Busca do GOTAS-LPV e do GOTAS-A (começa sozinha): 30 tentativas por fold e
   cenário, só na validação. A busca decide se o nó do balanço entra.
4. Busca do GOTAS-SS e triagem (`screen4_*`); depois a busca do GOTAS-AE e a
   triagem dele (`screen5_*`), em fila (`run_gotas_ae_tune.log`).
5. Depois: combinar as variantes aprovadas, escolher a melhor na validação, rodar
   `final` uma vez no teste e refazer os concorrentes se as entradas mudarem.

## Como acompanhar

Todos os comandos rodam da raiz do repositório (`cd ~/code/mimo_sys_v2`) e são
leves: não carregam o PyTorch, então podem rodar junto com os treinos.

```bash
# Triagem: validação, estresse e diferença pareada por variante
watch -n 60 .venv/bin/python notebooks/training/scripts/run_gotas_v2_status.py

# Buscas (Optuna) de todos os modelos, inclusive gotas_w, gotas_lpv e gotas_a
watch -n 30 .venv/bin/python notebooks/training/scripts/run_cv_status.py

# Lotes concluídos da triagem (código 0 = sem erro)
cat outputs/cv/logs/screen_all.log outputs/cv/logs/screen2_all.log

# Log da busca do GOTAS-LPV e do GOTAS-A
tail -f outputs/cv/logs/run_gotas_lpv_a_tune.log

# Processos em andamento e memória
pgrep -fa 'run_gotas_v2.py screen|run_cv_benchmark.py' | sed -E 's#.*/##'
free -m
```

Para ver erros de um lote: `outputs/cv/logs/screen_<base>_<variante>_<cenário>.log`
e `outputs/cv/logs/tune_<modelo>_<cenário>_f<fold>.log`.

## Decisões pendentes

- **Concorrentes sem o nó do balanço.** Se o GOTAS final não usar esse nó, os
  concorrentes devem ser refeitos sem ele, para todos terem as mesmas entradas
  (cerca de 8 h). As redes devem melhorar também; as árvores, pouco.
- **O que vai para o artigo.** A versão final, o histórico (v1 e variantes) e a
  declaração de que a melhoria foi feita depois da primeira rodada de validação
  cruzada (`docs/18`, Etapa 5).
