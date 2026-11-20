# Plano: GOTAS-W (componentes wavelet por aresta) e GOTAS mais robusto

> **Situação: aguardando liberação do autor.** Nada deste plano foi implementado ou
> treinado. O plano em execução é o dos ajustes do professor
> (`docs/17_plano-ajustes-da-revisao.md`).

## Quando decidir

A extração do modelo final, as figuras de resultado e o texto (Fases 3 a 5 do plano do
professor) dependem de qual GOTAS será o final. Quando a validação cruzada terminar, o
autor decide entre seguir com o GOTAS atual ou fazer antes o GOTAS-W. Se o GOTAS-W vier
depois dessas fases, elas precisarão ser refeitas.

## Contexto

Na validação cruzada parcial, o GOTAS ficou atrás de XGBoost e CatBoost no teste.
O autor quer melhorar o GOTAS preservando grafo, filtros e equações. A proposta
central dele: em vez de levar cada nó para a frequência, decompor o sinal em
componentes wavelet (por exemplo, 5), ter um filtro da aresta para cada componente e
somar tudo no final. Ele também pediu busca mais ampla e elastic net, desde que o
teste possa ser reconstruído.

**Fora do plano:** reduzir o tuning do XGBoost ou de qualquer concorrente, retirar
fold ou trocar a métrica depois de ver o resultado. Todos os modelos têm as mesmas
30 tentativas por fold e cenário (XGBoost busca 7 hiperparâmetros, GOTAS busca 8).

## O que os dados mostram

Diagnóstico de 2026-11-18, com a rodada ainda em andamento. Os modelos da tabela já
estavam completos.

R² em 12 h, todas as origens válidas, média de 5 folds e 3 sementes:

| Modelo | `full` validação | `full` teste | `pumps` validação | `pumps` teste |
|---|---|---|---|---|
| GOTAS | 0,867 | **0,716** | 0,729 | **0,608** |
| XGBoost | 0,856 | 0,830 | 0,716 | 0,703 |
| CatBoost | 0,852 | 0,818 | 0,732 | 0,710 |

- **Não parece faltar capacidade:** na validação o GOTAS empata ou lidera.
- **A diferença está no fold 2.** Validação → teste no `full`: GOTAS 0,888 → 0,272,
  XGBoost 0,854 → 0,806, MLP 0,837 → 0,405, MPNN 0,853 → −7,23. Sem esse fold, o
  teste fica em cerca de 0,83 / 0,84 / 0,82 no `full` e 0,73 / 0,73 / 0,74 no
  `pumps` (GOTAS / XGBoost / CatBoost).
- **O teste do fold 2 é o primeiro contato desse treino com o bloco de falhas de
  sensor**, e a validação dele não tem falhas: 114 das 116 origens avaliadas têm
  falha na janela de 96 h. Nível e pressão: GOTAS −0,06 e 0,07; XGBoost 0,78 e 0,75.

**Hipóteses a testar (nenhuma comprovada):**
1. O GOTAS centra a janela na média, que a falha contamina. XGBoost, CatBoost, MLP,
   LSTM e GRU partem do último valor observado. No fold 2, o erro do nível já no
   passo 1 é cerca de 0,85 desvio, contra 0,22 do XGBoost.
2. O balanço de massa acumulado não tem limite (`mass_balance_window=None`) e chega
   a −1,87 na escala do treino, em que a faixa normal é 0 a 1.
3. A busca do fold 2 escolheu uma configuração frágil (4 bandas, `lr` ≈ 0,0094) porque
   a validação não tinha falhas para penalizá-la.

**Já testado no split antigo (docs 9 a 11):** decompor **só as saídas** com à trous,
EWT corrigido ou sazonal (≤ +0,003 no `full`, ≤ +0,012 no `pumps`), Hammerstein e
limpar falhas só no treino. O GOTAS não usa STL: ela só aparece na figura
exploratória. Não há período de dados novo: `mimo_data.xlsx` tem os mesmos registros.

## Etapa 0: fechar a rodada atual

Esperar a rodada terminar, sem mexer em `src/` nem nos scripts, porque cada etapa abre
um processo novo que importa o código atual. Depois rodar `report` e guardar a tabela
como **GOTAS v1 e concorrentes**.

## Etapa 1: GOTAS-W, a arquitetura proposta pelo autor

Hoje cada nó passa pela RFFT e cada aresta multiplica o espectro por `H_ij(f)`. No
GOTAS-W, a RFFT sai:

1. **Componentes:** cada nó `j` é decomposto pela à trous causal de Haar em 5
   componentes: detalhes de cerca de 2–4, 4–8, 8–16 e 16–32 h, mais a aproximação
   acima de 32 h.
2. **Filtro por aresta e componente:** cada aresta `j → i` tem um filtro curto para
   cada componente `m`, com defasagens espaçadas conforme a escala (`d_m`).
3. **Soma no final:**
   `ŷ_i[t] = b_i + Σ_j Σ_m Σ_l a^m_ij(l) · c_j,m[t − l·d_m]`.
4. **Horizonte:** a simulação recursiva continua. A previsão volta para a sequência, e
   as componentes da linha nova são calculadas só com linhas passadas. Entradas futuras
   conhecidas entram do mesmo jeito.

Cascata física, combinação do nível e resíduo continuam como estão.

- **Reconstrução no teste:** as componentes somam exatamente o sinal e cada linha só
  usa linhas anteriores (`features.atrous_components`). A previsão já é a soma das
  contribuições, sem etapa de inversão.
- **Explicabilidade:** `a^m_ij` dá um grafo por componente, e o lasso por grupos passa
  a atuar em (aresta, componente), desligando componentes inteiras. A resposta
  `H_ij(f) = Σ_m A^m_ij(f) Ψ_m(f)` continua calculável, o que preserva impulso, Laplace
  e equações.
- **O que muda de fato:** sem não linearidade, isso ainda equivale a um filtro por
  aresta. A diferença está em como o filtro é parametrizado e regularizado, o que pode
  mudar a generalização. Os testes antigos decompunham só as saídas. Aqui entram todos
  os nós, com um filtro por componente.
- **Busca:** níveis de 3 a 5 (padrão 4, ou seja, 5 componentes), defasagens por
  componente, peso do lasso e os mesmos hiperparâmetros de treino do GOTAS.

## Etapa 2: ajustes de robustez e regularização

Entram sobre a melhor base (GOTAS-W, se aprovado, ou v1). Tudo fica desligado por
padrão, para o v1 continuar reproduzível.

| Id | Candidato | Por quê |
|---|---|---|
| R1 | Centro da janela: mediana, ou último valor nas saídas | hipótese 1 |
| R2 | Limite ±k·MAD nos detalhes wavelet do passado | cortar picos de falha por escala |
| R3 | Balanço de massa em janela móvel (`mass_balance_window`, já existe) | hipótese 2 |
| R4 | Treino com falhas simuladas nas entradas (zero, valor preso, pico) | ensinar o filtro a tolerar falhas |
| C1 | Elastic net: `alpha` do lasso por grupos e L2 nos filtros | pedido do autor |
| C2 | Resíduo maior: 1 a 3 camadas, largura até 64 | pedido do autor |
| C3 | Só no v1: centros das bandas nos picos do espectro do treino (ideia do EWT) | bandas com período físico |

Empilhar filtros lineares resulta em um filtro só, então a repetição de camadas entra
apenas no resíduo (C2). Registrar `nonlinearity_share` contra o v1.

## Etapa 3: triagem só na validação

- **GOTAS-W primeiro, com as mesmas 30 tentativas por fold e cenário do v1**, comparando
  melhor configuração contra melhor configuração. O teste não é carregado (prefixo
  `_tune`).
- Os ajustes da Etapa 2 partem da melhor configuração de cada fold: 5 folds × 2 cenários
  × 3 sementes.
- Duas métricas: validação normal e **validação sob estresse**, com falhas simuladas
  só nas entradas e a pontuação feita sobre os valores reais.
- **Regra fixada antes de rodar** (mesmos limites do doc 9): ganho acima de 0,012 na
  métrica-alvo em algum cenário, sem perder mais de 0,005 em nenhum. A métrica-alvo é
  o estresse para R1 a R4 e a validação normal para C1 a C3. O GOTAS-W passa se atingir
  o ganho em qualquer uma das duas, sem perder mais de 0,005 na validação normal.
- Os aprovados são combinados um a um, e a regra é reaplicada a cada acréscimo.
- Se R3 for aprovado, ele muda uma entrada comum, e **todos os modelos são refeitos**
  com ela (cerca de 8 h a mais). Os demais candidatos são internos ao GOTAS.
- Tempo estimado: 3 a 5 h para o GOTAS-W e cerca de 3 h para os ajustes.

## Etapa 4: GOTAS v2 no mesmo protocolo

- Novo identificador `gotas2`, com os mesmos folds, objetivo, 30 tentativas e 3
  sementes, usando a arquitetura e os ajustes aprovados.
- **O teste é aberto uma única vez.** Depois disso a arquitetura não muda, seja qual
  for o resultado.

## Etapa 5: relatório e artigo

- Tabela com v1, v2 e concorrentes, em origens a cada 12 h e em todas as origens, com
  bootstrap pareado e Diebold–Mariano (`evaluation.py`). O resultado do GOTAS-W aparece
  mesmo se ele não for aprovado.
- **Expectativa:** mesmo recuperando o fold 2, a projeção é empate com os modelos de
  árvore (cerca de 0,82 no `full` e 0,70 no `pumps`). O artigo só diz "melhor" onde a
  tabela sustentar. Caso contrário, a tese é a melhor GNN avaliada, interpretável e
  próxima dos modelos de árvore, redigida com o autor.
- **Declarar na Metodologia** que o v2 foi desenvolvido depois da primeira rodada. A
  validação de cada fold é o teste do fold anterior, então só o teste do fold 5 ficou
  fora da triagem, e mesmo ele já foi visto em resultado agregado. Um período novo da
  operação, se o coorientador conseguir, seria o teste mais limpo.

## Depois do v2

Retomar as Fases 3 a 5 de `docs/17_plano-ajustes-da-revisao.md` com o modelo escolhido.
Se o GOTAS-W for adotado, atualizar também as Figuras 9 e 10 e as equações da
arquitetura no artigo.

## Arquivos principais

- `src/mimo_sys/architectures/gotas_wavelet.py` (novo): GOTAS-W, registrado em
  `ARCHITECTURES` (`training/train.py`) e nas penalidades de `forecaster.py`.
- `src/mimo_sys/architectures/gotas.py`: R1, R2, C1, C2 e C3, desligados por padrão.
- `src/mimo_sys/training/train.py` e `augmentations.py`: configuração e falhas
  simuladas (R4).
- `notebooks/training/scripts/run_gotas_v2.py` (novo): triagem, reaproveitando
  `fit_predict` e `evaluate`.
- `run_cv_benchmark.py` e `run_cv_all.py`: identificadores e espaços de busca.
- `uncertainty.py` e scripts de extração, se o GOTAS-W for adotado.
- `tests/test_architectures.py`, `tests/test_causality.py`, `docs/changes/` e
  `CHANGELOG.md`.

Commits locais em Conventional Commits, sem atribuição a IA. O push fica com o autor.

## Verificação

- Com tudo desligado, o GOTAS reproduz as previsões do v1 na mesma semente.
- Componentes do GOTAS-W somam o sinal (erro ≤ 1e-10), e a atualização passo a passo
  coincide com a decomposição da sequência inteira.
- Perturbar o futuro não muda a previsão
  (`test_previsao_em_producao_e_causal_e_exata` com as novas variantes).
- `frequency_response()` do GOTAS-W coincide com a resposta medida por impulso.
- A triagem falha se tentar ler o teste real de um fold.
- `uv run ruff check src/ tests/` e `uv run --extra ml --extra tune --extra boost pytest tests/`.
