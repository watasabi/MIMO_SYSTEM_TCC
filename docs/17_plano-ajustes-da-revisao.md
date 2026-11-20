# Plano em execução: ajustes da revisão do professor

Este é o plano em andamento. A primeira parte traz a situação em 2026-11-18, às
20h. A segunda parte traz o plano original, sem alterações.

## Situação em 2026-11-18

### Decisão pendente antes da Fase 3

A extração do modelo final, as figuras de resultado e o texto (Fases 3 a 5) dependem
de qual GOTAS será o final. Quando a validação cruzada terminar, o autor decide entre
seguir com o GOTAS atual ou fazer antes o GOTAS-W (`docs/18_plano-gotas-w.md`). Se o
GOTAS-W vier depois dessas fases, elas precisarão ser refeitas.

### Feito

| Parte | Situação |
|---|---|
| Protocolo e código (Fase 1) | Folds, máscara de falhas, MLP, LSTM, GRU, XGBoost, CatBoost, ARMAX reestimado, métricas, busca e testes prontos, em commits locais |
| Variáveis das equações e recuo | Aplicados no `main.tex`, inclusive no Apêndice A |
| Referencial e Metodologia | Seção de modelos de comparação, métricas novas, validação cruzada, `tab:folds` e `tab:busca` escritas |
| Figuras 1, 2 e 3 | Refeitas em TikZ |
| Análise exploratória | Antiga Figura 5 separada em coerência e polos, antiga Figura 6 separada em STL e variância por banda |
| Figura 4 | Mantida como a original, por decisão do autor |
| Arquitetura | Antiga Figura 7 separada em fluxo (Figura 9) e painéis (Figura 10) |

O repositório do artigo tem 4 commits locais à frente do `origin`, sem push.

### Em andamento: validação cruzada

Comando: `.venv/bin/python notebooks/training/scripts/run_cv_all.py 3 30`.

| Modelo | Tentativas | Etapas finais |
|---|---|---|
| GOTAS | 300/300 | 10/10 |
| MLP | 300/300 | 10/10 |
| LSTM | 291/300 | 9/10 |
| GRU | 202/300 | 5/10 |
| XGBoost | 300/300 | 10/10 |
| CatBoost | 300/300 | 10/10 |
| MPNN | 300/300 | 10/10 |
| Symbolic Graph Network | 300/300 | 10/10 |
| StemGNN | 174/300 | 5/10 |
| Seq2SeqLatentGNN | 59/300 | 0/10 |
| FourierGNN | 0/300 | 0/10 |
| ARMAX | 0/300 | 0/10 |

Total: 2.526 de 3.600 tentativas (70,2%). As etapas de LSTM e GRU interrompidas pela
proteção de memória são refeitas automaticamente no final.

**Resultado parcial:** XGBoost e CatBoost estão à frente do GOTAS no teste. Isso aciona
o ponto de parada da Fase 2, item 3: objetivo, resumo e conclusão só são reescritos
depois de o autor decidir com a tabela final.

### Pendente

- Terminar a rodada, rodar `run_cv_benchmark.py report` e revisar a tabela com o autor.
- Decidir qual GOTAS será o final.
- `export-final`, treino do modelo final com 5 sementes e testes dos scripts de
  relatório adaptados.
- Refazer extração, Laplace, oráculos, balanço de massa e decomposição por banda.
- Refazer ablações nos folds, protocolo de 1 passo dos autores e ARMAX MOGWO no teste
  do fold 5.
- Gerar as figuras de resultado, inclusive R² por passo.
- Atualizar resultados, tabelas (`tab:hiper`, `tab:livre`, `tab:gnns`, `tab:ablacao`,
  `tab:filtros`), resumo, objetivos, limitações e conclusão.
- Figura 10: atualizar notação e valores antigos e decidir se vai para duas páginas.
- Decidir se a menção a "70/15/15" sai da legenda da Figura 4.
- Conferir as referências novas, `docs/13`, `docs/14` e a nota de mudança, e compilar
  no Overleaf.

### Para retomar em outra sessão

- **Enquanto a rodada estiver ativa, não alterar `src/` nem os scripts de treino:** cada
  etapa abre um processo novo que importa o código atual. Evitar também treinos e testes
  pesados, porque a memória está no limite.
- Acompanhar: `watch -n 30 .venv/bin/python notebooks/training/scripts/run_cv_status.py`.
- Se a rodada parar, repetir `run_cv_all.py 3 30`: ela retoma do que já terminou.
- Ao terminar:
  `uv run --extra ml --extra tune --extra boost python notebooks/training/scripts/run_cv_benchmark.py report`.
  O `export-final` só roda depois da decisão sobre o GOTAS final.
- `report_lean_model.py`, `report_extracted_model.py`, `figure_extracted_graph.py` e
  `figure_spectral_performance.py` já foram adaptados para `F5_final`, mas **ainda não
  foram testados**.
- Não rodar `extract_spectral_tf_net.py` nem `figure_architecture_drawio.py`.
- O artigo fica em `/home/rwp/code/TCC_MIMO_SYS_overleaf`. Antes de editá-lo, ler
  `docs/14_guia-de-edicao-do-artigo.md`. A prévia local usou um `brazil.ldf` mínimo, sem
  hifenização em português, então a compilação final deve ser conferida no Overleaf.

### O que mudou em relação ao texto original

- A rodada usa 3 processos em paralelo, e não um por vez, e inclui o ARMAX: são 12
  modelos na fila.
- A Figura 4 não foi refeita, e as figuras foram renumeradas: coerência (5), polos (6),
  STL (7), variância por banda (8), fluxo da arquitetura (9) e painéis (10).
- A Figura 9 foi feita em TikZ no `main.tex`, com letra de 12 pt e as cores e letras dos
  painéis da Figura 10, porque o draw.io estava indisponível. A Figura 10 é um recorte
  da arte original.

---

# Plano original: ajustes da revisão do artigo do GOTAS

## Contexto

A revisão pediu: definir todas as variáveis das equações, comparar com MLP,
LSTM, GRU, CatBoost e XGBoost, tirar o recuo do texto depois das equações,
aumentar textos e legendas das figuras, melhorar a Figura 3, dividir as
Figuras 5, 6 e 7, usar mais métricas e adotar validação cruzada com treino,
validação (otimizando os hiperparâmetros de todos os modelos) e teste.

Decisões do autor:
- k-fold temporal com janela crescente (k1 → k2, k1+k2 → k3, ...);
- os 11 modelos entram na validação cruzada e na otimização;
- o GOTAS final é retreinado e toda a extração é refeita;
- a arquitetura continua no draw.io.

Isso reverte a decisão "sem busca de hiperparâmetros" do `docs/14`. Quase todos
os números de resultado do artigo vão mudar, então a ordem é: experimentos,
depois figuras, depois texto.

## Cobertura da revisão

| Pedido | Onde está | O que faltava e foi acrescentado |
|---|---|---|
| Definir todas as variáveis | Fase 5 | auditoria equação por equação, inclusive símbolos no meio do texto e métricas novas |
| Comparar com MLP, LSTM, GRU, CatBoost, XGBoost | Fases 1, 2 e 5 | 1 passo e R² por passo também para os modelos novos |
| Sem recuo depois da equação | Fase 5 | linha em branco antes da equação e equações do Apêndice A |
| Textos e legendas pequenos | Fase 4 | referências a painéis (a), (b), (c) no texto |
| Figura 3: cor e fluxo pelas setas | Fase 4 | setas numeradas, para o fluxo aparecer sem texto na imagem |
| Figuras 5 e 6 divididas | Fase 4 | cada figura nova citada no parágrafo certo |
| Figura 7 em 2 figuras | Fase 4 | sem pendência além da exportação pelo draw.io |
| Mais métricas e k-fold otimizando todos | Fases 1, 2 e 5 | ARMAX reestimado e otimizado em cada fold; tabela dos espaços de busca e dos valores escolhidos |

## Fase 1: protocolo e código (`mimo_sys_v2`)

**Folds** (janela crescente, prática de origem móvel: Tashman 2000; Bergmeir e
Benítez 2012; Cerqueira et al. 2020; conferir cada referência no DOI). São 2.208
linhas: bloco inicial de 888 h e blocos de 220 h.

| Fold | Treino | Validação | Teste |
|---|---|---|---|
| 1 | 0–887 | 888–1107 | 1108–1327 |
| 2 | 0–1107 | 1108–1327 | 1328–1547 |
| 3 | 0–1327 | 1328–1547 | 1548–1767 |
| 4 | 0–1547 | 1548–1767 | 1768–1987 |
| 5 | 0–1767 | 1768–1987 | 1988–2207 |

- Gravar cada fold como `data/processed/mimo_cv5_f{k}_{train,val,test}.parquet`.
  `TrainConfig.data_prefix` e `load_splits` já aceitam isso, e `train()`,
  `_windows()` e `run_improvements.evaluate()` funcionam sem mudança.
- **Falhas de sensor:** caem nas avaliações (linhas 1051–1057 na validação do
  fold 1; 1341–1349, 1373–1379 e a falha não marcada 1445–1488 no teste do
  fold 2). Estender a máscara com 1445–1488 e, para todos os modelos, excluir
  das métricas as origens cujo horizonte toca linha marcada. Nenhum fold é
  descartado.
- **Contrato igual ao da Tabela 5:** entradas futuras conhecidas, cenários
  `full` e `pumps`, cascata e combinação do nível ajustadas na validação do fold.

**Modelos novos**
- `src/mimo_sys/architectures/baselines.py`: MLP, LSTM e GRU, registrados em
  `ARCHITECTURES` (`train.py`).
- XGBoost e CatBoost: novo extra `boost` no `pyproject.toml`, atributos de
  defasagem mais entradas futuras, saída de 36 valores (3 saídas × 12 passos) e
  um adaptador para reaproveitar `forecast_origins`.

**Otimização** (`notebooks/training/scripts/run_cv_benchmark.py`)
- Optuna TPE com poda, 30 tentativas por modelo, cenário e fold, retomável em
  SQLite. Objetivo: R² médio em 12 passos na validação, com `_crossfit_val`.
- Espaços em `tuning.py`, reaproveitando `MODEL_SPACES` e acrescentando
  FourierGNN e os modelos novos.
- GOTAS: `n_regimes = 1` fixo, para continuar interpretável. Busca em bandas,
  resíduo, janela, taxa de aprendizado, *weight decay*, lote, peso da cascata e
  peso do lasso.
- Teste de cada fold usado uma vez, com o melhor conjunto e 3 sementes.
- **ARMAX também entra na otimização:** reestimado no treino de cada fold por
  mínimos quadrados estendidos, com as ordens $n_a, n_b, n_c$ de 1 a 5 escolhidas
  na validação. Os parâmetros publicados foram estimados nas linhas 550–1150, que
  se sobrepõem ao teste do fold 1 (1108–1327), então não servem para a validação
  cruzada. O ARMAX MOGWO publicado continua na Tabela 4: no protocolo dos autores e
  no teste do fold 5, fora da janela de estimação.
- Persistência e sazonal ingênuo não têm hiperparâmetros e são avaliados nos mesmos
  blocos.

**Métricas** (por saída, em 1 passo, 6 e 12 h, média ± desvio entre folds, e R² por
passo do horizonte): R², RMSE, MAE, sMAPE, MASE (escala: sazonal de 24 h no treino
do fold) e KGE.

**Testes:** folds sem sobreposição, máscara, formato de saída dos modelos novos,
perturbação do futuro (sem vazamento) e fórmulas das métricas.

## Fase 2: execução

1. Teste curto: 2 tentativas, fold 1, os 11 modelos, medindo tempo e RAM.
2. Rodada completa em segundo plano, um processo por vez: GOTAS, MLP, LSTM, GRU,
   XGBoost, CatBoost e depois as 5 GNNs. Estimativa: ~35 h, mais ~5 h de
   retreinos finais e ablações.
   **A máquina tem 7,7 GB de RAM, com 1,8 GB livres e swap em uso:** convém fechar
   Chrome e Discord durante a rodada.
3. **Ponto de parada:** apresento a tabela ao autor antes de reescrever o texto.
   Se algum modelo superar o GOTAS, objetivo, resumo e conclusão precisam refletir
   isso, e essa redação é decidida com o autor.

## Fase 3: modelo final e extração

- GOTAS final = configuração vencedora do **fold 5** (mais dados, teste mais
  recente, sem tocar no teste para escolher), 5 sementes, como nova variante em
  `run_improvements.VARIANTS`.
- Parametrizar `report_lean_model.py`, `report_extracted_model.py`,
  `figure_extracted_graph.py` e `figure_spectral_performance.py`, que hoje fixam
  início do teste (1876), 321 origens e `I0_ref`. Não rodar
  `extract_spectral_tf_net.py`.
- Refazer equações extraídas, filtros, grafo, respostas ao impulso, Laplace,
  balanço de massa, oráculos e decomposição por banda.
- Refazer as ablações nos 5 folds e o protocolo de 1 passo dos autores com a
  configuração final.

## Fase 4: figuras (largura final de 16 cm, texto ≥ 10 pt no tamanho impresso)

Gerar cada figura já em 6,3 pol de largura, sem reduzir no LaTeX.

**Regra pedida pelo autor: o texto explicativo sai da imagem e fica só no
Overleaf.**
- Sai da imagem: título, subtítulo, notas, anotações, caixas de legenda e frases
  explicativas. Esse conteúdo vai para o `\caption` e para o parágrafo que cita a
  figura.
- Fica na imagem só o indispensável para ler: nomes e unidades dos eixos, valores
  dos eixos, letras (a), (b), (c) dos painéis e rótulos curtos dos blocos.
- Séries e cores são identificadas na legenda do LaTeX (por exemplo, "preto:
  medido; vermelho: previsto"), com cor e tipo de linha distintos.
- Nomes em português, nada de `freq_b1` ou `nivel_res`.

| Figura atual | Mudança |
|---|---|
| 1, 2 e resultados | regenerar sem texto embutido e com fontes maiores |
| 4 | manter a figura original, por decisão do autor |
| 3 (fluxograma, TikZ) | cor por fase, caminho único sem cruzamentos, setas grossas e **numeradas** na ordem do fluxo, caixas com nomes curtos, incluir validação cruzada e comparação; a legenda explica as cores e o que passa em cada seta numerada |
| 5 | 2 figuras: coerência (entradas × saídas) e polos ARX de ordem 2 |
| 6 | 2 figuras: decomposição STL e variância de cada saída por banda |
| 7 | manter a arte atual; separar em fluxograma (esquerda) e painéis (direita), sem redesenhar |

- Novo `figure_eda_article.py`, portando os cálculos do notebook 03.
- Figura 7: editar o XML de cópias de `docs/figures/stf_arquitetura.drawio`; não
  rodar `figure_architecture_drawio.py`, que sobrescreve os ajustes manuais.
  **A exportação depende do MCP do drawio, hoje desconectado.** Sem ele, entrego
  os `.drawio` prontos para o autor exportar.
- As legendas no `main.tex` ficam mais completas, sem repetir o que o parágrafo já
  diz. Conferir o tamanho final do artigo depois das divisões.
- Atualizar as citações a painéis que deixam de existir (`fig:eda`,
  `fig:estrutura`, inclusive "Figura `estrutura`b" nos Resultados, `fig:espectro` e
  `fig:linear`) e citar cada figura nova no parágrafo que a discute.
- A figura de R² por passo passa a incluir o melhor modelo de cada família.

## Fase 5: texto (`TCC_MIMO_SYS_overleaf/main.tex`)

Antes: `git pull`, ler a skill `academic-paper-writing` e seguir `docs/14`.

- **Equações:** definir todo símbolo logo após cada equação. Pendências encontradas:

| Equação | Falta definir |
|---|---|
| `armax` | $k$, $l$, índices $i,j,m$, coeficientes $a_{il}, a_{ijl}, b_{iml}, c_{il}$ |
| `convolucao` | $t$, $\tau$, $f$, $X(f)$, $Y(f)$ |
| `mpnn` | $m_{ji}$ como mensagem, $h_i'$ |
| `kendall` | $\mathcal{L}_{\mathrm{K}}$ |
| `sgl` | $P(\theta)$, $\alpha$, $\theta_g$, normas $\lVert\cdot\rVert_1$ e $\lVert\cdot\rVert_2$ |
| `laplace` | $\mathcal{D}$, $\mathcal{N}$, $\Sigma$, $I$, índice $k$ |
| `sequencia` | $s_j$, $x_j$ |
| `bandas` | $H_{ij}(f)$, $m$ |
| `stf` | $\hat Y_i$, $S_j$, $b_i$, $\hat y_i$, $h_{ij}$ |
| `cascata`, `blend` | $N(t+k)$, $\hat q$, $k$, $h$, $\hat y_2^{\mathrm{direto}}$ |
| `perda` | $\lambda_c$, $\rho$, $N^{\mathrm{cascata}}$, MSE, $c$ |
| `metricas` | $y_i[k]$, $\hat y_i[k]$, $k$, mais os símbolos de MAE, sMAPE, MASE e KGE |
| `ext_*` | $\mu_i$, $\tilde u_m$, $r_i(k)$ |
| `continuidade`, `trapezio`, `balanco` | $h$, $t$, $Q_e$, $Q_s$, $q$, $t_k$, $\Delta y_2$ |
| No meio do texto | $j$ em $1/(jf)$, $\omega$ em $\cot(\omega/2)$, operador $*$ em $h_{ij} * \tilde s_j$ |

- **Colisões de notação a eliminar:** $h$ (estado do nó, resposta ao impulso,
  nível, passo), $N$ (número de nós e nível da cascata), $a,b$ (ARMAX e balanço de
  massa), $j$ (índice e unidade imaginária) e $\sigma$ (ruído por saída e variância
  do erro).
- **Em volta das equações, em todas, inclusive no Apêndice A:** remover a linha em
  branco antes e depois, para o texto continuar o parágrafo sem recuo, e pontuar as
  equações como parte da frase.
- **Introdução:** atualizar objetivos e o parágrafo de estrutura do trabalho.
- **Referencial:** nova subseção 2.5 com os modelos de comparação e a validação
  temporal, em prosa, sem equações não usadas e com as siglas definidas em inglês;
  atualizar o parágrafo que abre a seção.
- **Metodologia:** folds, tratamento das falhas, busca de hiperparâmetros, métricas
  novas na equação de métricas, tabela dos espaços de busca e dos valores escolhidos,
  Tabela de configuração do GOTAS final, bibliotecas e tempo de treino em Materiais.
  Remover "sem busca de hiperparâmetros" e "o teste foi usado uma vez" na forma atual.
- **Resultados:** tabela única com os 11 modelos, o ARMAX e as referências, tabela
  das métricas adicionais, ablações sem a linha "busca descartada", limitações
  atualizadas (hoje citam 27 janelas), e atualização de resumo (até 250 palavras),
  tabelas e conclusão.
- Conferir o "2.216 amostras" do texto, pois os splits somam 2.208.

## Arquivos principais

- Código: `src/mimo_sys/architectures/baselines.py` (novo),
  `src/mimo_sys/training/train.py`, `tuning.py`, `features.py`, `pyproject.toml`,
  `notebooks/training/scripts/run_cv_benchmark.py` (novo),
  `figure_eda_article.py` (novo) e os scripts de relatório e figuras citados.
- Artigo: `main.tex`, `referencias.bib`, `imagens/`.
- Documentação: `docs/changes/2026-11-18-validacao-cruzada-temporal.md` (nova),
  `CHANGELOG.md`, `docs/13`, `docs/14`, `AGENT.md` (comando com o extra `boost`).
- Memória: atualizar a preferência por painéis duplos, agora contrariada pela
  revisão.

Commits locais em Conventional Commits, sem atribuição a IA, nos dois
repositórios. O push fica com o autor. Não mexer nos arquivos não rastreados já
existentes (`docs/15`, `docs/16`, `Diagrama sem nome.drawio`, `gotas_canais.pdf`).

## Verificação

- `uv run ruff check src/ tests/` e `uv run --extra ml --extra tune pytest tests/`.
- Conferir que nenhum fold usa teste na escolha de hiperparâmetros ou nos pesos do
  nível, e que todos os modelos têm as mesmas origens avaliadas.
- Reproduzir o GOTAS `I0_ref` antes da rodada, para confirmar que o código novo não
  alterou os resultados atuais (0,862 e 0,833).
- Compilar `pdflatex → bibtex → pdflatex → pdflatex` no scratchpad: sem erros,
  citações ou referências indefinidas, nem `Overfull` acima de ~10 pt.
- Renderizar as páginas e conferir legibilidade e posição de cada figura.
- `grep` de cada número antigo no `main.tex`, para não sobrar valor desatualizado.
- Percorrer as equações uma a uma contra a tabela da Fase 5 e conferir no PDF que
  nenhum texto depois de equação ficou com recuo.
- Fechar a tabela "Cobertura da revisão" item por item antes de entregar.
