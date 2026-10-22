# Melhorias de nível e pressão dentro do escopo, e o teto de informação

Plano: `docs/changes/2026-10-23-melhorias-nivel-pressao.md`. Código:
`notebooks/training/scripts/run_improvements.py` (grafo) e
`models/benchmarks/run_benchmarks.py` (benchmark externo, projeto UV
separado). Mesmo protocolo do doc 9: previsão em todas as horas de
validação e teste via `forecast_at_origin`, 3 seeds, bootstrap pareado e
Diebold–Mariano, decisão na validação e teste usado uma vez.

## 1. Resumo

- **Modelo final:** SpectralTFNet de referência (doc 8), mais a
  **combinação por passo entre nível direto e cascata**, mais um
  **ensemble de 5 seeds com o `H` médio**. Continua sendo uma única
  função de transferência mais o resíduo, extraível como antes.

  | | full | pumps |
  |---|---|---|
  | Referência (doc 8) | 0,831 | 0,708 |
  | **Modelo final** | **0,842** | **0,718** |
  | Por saída, final (vazão / nível / pressão) | 0,925 / 0,837 / 0,763 | 0,927 / 0,455 / 0,770 |

  No `pumps` a melhora passou na regra da validação: +0,015, IC
  +0,001..+0,025. No `full` o teste confirma: +0,011, IC +0,004..+0,022.
- **Não existe salto grande nesses dados.** Os oráculos (§2) mostram
  que a pressão já está perto do que as variáveis medidas explicam, e
  que o nível no `pumps` depende de uma vazão de entrada que é pouco
  previsível.
- **Uma caixa-preta forte empata com o grafo.** O LightGBM direto
  multivariado, com a mesma cascata e a mesma combinação do nível, dá
  0,821 no `full` e 0,707 no `pumps` em todas as origens. A cada 12 h dá
  0,850 e 0,730, contra 0,836 e 0,721 do grafo. Em todos os casos a
  diferença está dentro do IC 95% (doc 11 §5.4).
- **Não ajudaram:** limpar o treino, as arestas Hammerstein, e o grafo
  prever a própria vazão de entrada. Este último prevê bem a entrada
  (R² 0,45), mas isso não chega ao nível. Detalhes em §4.

## 2. Os tetos (oráculos no teste)

| Pergunta | Resultado |
|---|---|
| Pressão explicada por bombas, vazão e nível **verdadeiros** no mesmo instante (linear / boosting) | 0,775 / 0,782 |
| Pressão com os próprios lags, 1 passo à frente | 0,856 |
| Nível pela cascata com a vazão de distribuição **verdadeira** | 0,906 (treino cru) / 0,914 (limpo) |
| A mesma cascata com a vazão de entrada sazonal (`pumps`) | −0,36 |
| Vazão de entrada 12h à frente: sazonal / LightGBM / o grafo (I4) | 0,20 / 0,33 / 0,45 |
| Correção online do fluxo não instrumentado; termo MA sobre erros passados | piora / não muda |

**Pressão:** 0,76 em 12h contra um teto estático de ~0,78. O que falta é
dinâmica que as variáveis medidas não explicam: provavelmente pressão a
montante ou manobra de válvula, o mesmo que o doc 8 §10 já suspeitava.

**Nível no `full`:** 0,837 contra um teto de ~0,91. O resto é o erro
integrado da vazão prevista.

**Nível no `pumps`:** limitado pela vazão de entrada, que é uma
perturbação externa.

## 3. O benchmark externo (só como régua, nunca no modelo)

R² médio no teste, em todas as origens.

| Modelo | full | pumps |
|---|---|---|
| LightGBM direto multivariado (lags de tudo, entradas futuras conhecidas, treino limpo) | 0,793 | 0,699 |
| … com a cascata e a combinação do nível do projeto | 0,821 | 0,707 |
| **Grafo final** | **0,842** | **0,718** |

- **neuralforecast NHITS e TiDE** rodaram, mas os resultados foram
  descartados por não serem confiáveis nesta configuração:
  - o TiDE dá previsões idênticas, bit a bit, nos dois cenários. As
    entradas futuras dos cenários são diferentes, então ele parece
    ignorá-las no `cross_validation`;
  - o NHITS diverge em algumas origens da validação.
- **TabPFN-TS ficou de fora**: exige aceitar uma licença online, o que
  foi decidido não fazer.
- **foreblocks** (MIT) foi clonado e rodado só para entender, sem virar
  dependência. Dele saíram três coisas:
  - o `auto_filter` mostrou que o ruído de medição é só 0,2–1,4% da
    variância de cada saída. O erro grande em 2–8h é sinal imprevisível,
    não ruído, e por isso o denoising dos rótulos foi descartado;
  - os detectores de outlier dele encontraram falhas que as regras da
    doc 1 não pegavam: vazão saturada em 199,6 l/s depois de paradas, e
    transientes de retomada. As regras foram ampliadas. Os detectores em
    si marcam centenas de amostras normais e não foram usados;
  - a base de Chebyshev das camadas KAN virou as arestas Hammerstein.

## 4. As etapas, uma a uma (validação decide)

Δ do R² médio contra a referência. No `full` a referência é o nível via
cascata; no `pumps`, o nível direto (escolhidos na validação).

| Etapa | full val (IC) | pumps val (IC) | Decisão |
|---|---|---|---|
| Combinação nível direto + cascata, pesos por passo | +0,007 (−0,001..+0,015) | +0,013 (−0,001..+0,024) | entra |
| + ensemble de 5 seeds, média das previsões | +0,008 (+0,001..+0,017) | +0,015 (+0,001..+0,025) | entra |
| + ensemble com o `H` médio, uma equação só | +0,009 (+0,001..+0,017) | +0,014 (+0,001..+0,024) | **escolhido**: rende o mesmo que a média e é extraível |
| Treino limpo, mais combinação | −0,009 | +0,004 | não |
| Grafo prevê a vazão de entrada, mais combinação | — | +0,007 | não |
| Hammerstein grau 3 com resíduo, mais combinação | −0,011 | +0,002 | não |
| Só linear, sem resíduo (± Hammerstein), mais combinação | −0,032 / −0,054 | −0,019 / −0,047 | não; o resíduo MLP vale ~0,03–0,05 |

Notas:

- **Pesos da combinação.** A cascata domina nos primeiros passos e
  perde peso com o horizonte. Com vazão de entrada conhecida vão de ~1
  para ~0,6; sem ela caem para ~0,1. Na validação, os pesos são
  ajustados numa metade e avaliados na outra.
- **Limpeza do treino.** A primeira versão descartava toda janela que
  tocava uma falha, o que é 49% das janelas, e piorava muito. A versão
  final descarta só as janelas com falha no alvo (6%), sem ganho
  mensurável. Valeu como verificação: os resultados anteriores não
  estavam distorcidos pelas falhas do treino.
- **Hammerstein.** A curva da bomba quase não ajuda, coerente com o
  oráculo: o mapa estático linear explica 0,775 e o boosting 0,782.
  Tirar o resíduo MLP custa caro, então os ~29% não lineares sem equação
  continuam sendo a limitação aberta do doc 8.

**Retreino em treino+validação (etapa 6 do plano): não feito.** Os pesos
da combinação precisam de previsões fora da amostra na validação, e
retreinar com ela exigiria um novo holdout, que não existe com este
volume de dados.

## 5. Arquivos

- `outputs/improvements/report.txt`, `outputs/improvements/report.json`:
  todas as variantes, com Δ, IC e DM, na validação e no teste.
- `outputs/improvements/weights/`: pesos dos 5 seeds de cada variante.
- `outputs/benchmarks/`: previsões do LightGBM.
- `outputs/decomposition_phase1.json` e
  `notebooks/eda/07_decomposicao_saidas.ipynb`: variância e erro por
  banda.
