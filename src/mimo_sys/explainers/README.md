# `mimo_sys.explainers`

Métodos para extrair conhecimento interpretável de uma GNN treinada
(`mimo_sys.architectures`). Este arquivo explica **o código e a
metodologia**: o que cada técnica faz, passo a passo, de onde ela vem na
literatura, e onde estão as armadilhas. As referências numeradas `[n]`
estão no fim do arquivo.

Documentos relacionados:

- *Por que* existem duas rotas e qual pergunta cada uma responde:
  [`docs/4_rotas-de-extracao.md`](../../../docs/4_rotas-de-extracao.md).
- *Por que* cada arquitetura serve (ou não) a cada rota:
  [`docs/3_investigacao-arquiteturas.md`](../../../docs/3_investigacao-arquiteturas.md).
- Fichas de leitura dos papers: [`docs/0_referencias-tcc.md`](../../../docs/0_referencias-tcc.md).
  PDFs em [`docs/pappers/`](../../../docs/pappers/).

---

## 1. Visão geral

Há duas formas de tirar conhecimento interpretável de uma GNN treinada.
Elas respondem perguntas diferentes e **não são substitutas**:

| | Rota A — Equação | Rota B — Subgrafo |
|---|---|---|
| Pergunta | **Como** `y2` depende de `u4` e `y1`? | **Quem** depende de quem? |
| Entregável | expressão fechada, ex. `A·dy2/dt = u4 − y1` | matriz/máscara de acoplamento `7×7` |
| Técnica | destilação simbólica [2, 3, 4] | peso de aresta (B1) ou explainer [5, 6, 7] (B2) |
| Onde age | num **bloco interno** (`ψ`, `φ`) | nas **arestas** do grafo |
| Verificação contra a física | quantitativa (coeficientes) | qualitativa ("as arestas certas acenderam") |
| Serve para projeto de controle? | sim | não |
| Módulo | `mimo_sys.explainers.equation` | `mimo_sys.explainers.graph` |

A lacuna declarada do TCC é *"caixa-preta pura, sem equações
interpretáveis"*. **Só a Rota A fecha essa lacuna.** A Rota B é barata e
vale como corroboração independente (§5).

### Qual arquitetura serve a qual rota

De `docs/4_rotas-de-extracao.md` §Mapa modelo × rota. Este módulo só
precisa de código para as células marcadas **sim**:

| Modelo (`src/mimo_sys/architectures/`) | Rota A | Rota B | Por quê |
|---|---|---|---|
| `MPNNForecaster` | **sim** | **sim** | `message_fn` recebe `(janela de i, janela de j)`: tudo nomeado. Grafo completo, sem base derivada. |
| `SymbolicGraphNetwork` | **sim** | **sim** | O mesmo MPNN com warm-start S-G e injeção de ruído; protocolo de 2 estágios do paper [10]. |
| `StemGNN` | não | não | Pós-transformada, a computação vive numa base espectral da Laplaciana aprendida. Falha nos dois critérios. |
| `Seq2SeqLatentGNN` | não | fraco | O GCN vê um latente pós-filtro-de-Fourier (sem nome). A adjacência é *aprendida*: explicar é explicar um grafo que o modelo inventou. |
| FourierGNN (**não implementado**) | não | **sim** | Nó = `(variável, lag)` e base **DFT, fixa** ⇒ máscara bem definida. Mas o FGO é multiplicador compartilhado: sem MLP por aresta para destilar. |

`StemGNN` e `Seq2SeqLatentGNN` entram no TCC só como baseline de
acurácia (já implementados em `architectures/`). **Nunca** passe esses
modelos para `equation/` ou `graph/` (ver "Pares a não fazer" em
`4_rotas-de-extracao.md`). FourierGNN, quando implementado, usa os
mesmos wrappers de `graph/` sem código novo.

---

## 2. Fundamento comum: a MPNN e por que ela é "explicável"

As duas rotas operam sobre o `MPNNForecaster`, uma *Message Passing
Neural Network* [11] no formalismo de *Graph Networks* [12]. Cada uma das
7 séries do reservatório é um **nó**; a feature de cada nó é a sua
própria **janela de lags**:

| Nó | Coluna | Papel |
|---|---|---|
| `u1`, `u2`, `u3` | `freq_b1`, `freq_b2`, `freq_b3` | frequência das bombas (Hz) |
| `u4` | `vazao_entrada` | vazão de entrada (l/s) |
| `y1` | `vazao_distribuicao` | vazão de distribuição (l/s) |
| `y2` | `nivel_res` | nível do reservatório (m) |
| `y3` | `pressao` | pressão (mca) |

Uma rodada de *message passing* tem três passos:

$$
\begin{aligned}
m_{ij} &= \psi(h_i,\, h_j) && \text{mensagem do emissor } j \text{ para o receptor } i\\
\bar m_i &= \textstyle\sum_{j \neq i} m_{ij} && \text{agregação (soma)}\\
\hat y_i &= \phi(h_i,\, \bar m_i) && \text{atualização / previsão}
\end{aligned}
$$

`ψ` é `message_fn` e `φ` é `update_fn`, ambos MLPs pequenos. O grafo é
**completo** (42 arestas dirigidas, sem self-loops) e **fixo**: a
topologia não é aprendida.

O que torna esse modelo explicável é um **viés indutivo** [2]: a
computação foi forçada a passar por funções pequenas e separáveis cujos
**argumentos têm nome**. `ψ` recebe `(janela de y2, janela de u4)`, não
um vetor latente. Isso vale para as duas rotas:

- Rota A roda regressão simbólica em `ψ` e `φ` **separadamente**.
- Rota B mede quanto passa por cada aresta `(i, j)`, e mascarar uma
  aresta é uma perturbação **local** porque não há base derivada dos
  dados (ao contrário do `StemGNN`).

---

## 3. Rota A — Extração de equação (destilação simbólica)

[`equation/distill.py`](equation/distill.py)

### 3.1 A ideia: destilar a rede por partes

A **regressão simbólica** (SR) busca, no espaço de expressões
matemáticas, a fórmula que melhor ajusta um conjunto de dados.
Diferente da regressão clássica, a *forma* da expressão não é fixada a
priori. Ela tem origem na programação genética [13], e ganhou destaque em
física com a redescoberta de leis de conservação a partir de dados [14].

O problema é que a SR **escala muito mal com o número de variáveis**: o
espaço de expressões cresce combinatorialmente. Cranmer et al. [2]
resolvem isso em quatro passos:

1. Treinar uma rede com estrutura interna separável (uma Graph Network).
2. Forçar as funções internas a ficarem esparsas/de baixa dimensão.
3. Rodar SR em **cada função interna separadamente**, com poucas
   entradas cada.
4. Recompor as expressões e reajustar as constantes.

A fatoração é o ganho: no exemplo deles, o espaço de busca cai de
~10¹⁸ (modelo inteiro) para ~2×10⁹ (subproblemas). Com N=7 nós esse
argumento é mais fraco, e o que justifica a GNN aqui é **descobrir a
topologia do acoplamento** em vez de assumi-la (ver `3_investigacao-arquiteturas.md` §5).

### 3.2 O motor: PySR

A busca simbólica usa o PySR [3], cujo backend é `SymbolicRegression.jl`
(Julia). Em resumo:

- **Algoritmo**: programação genética com múltiplas populações
  evoluindo em paralelo. Cada expressão é uma árvore (operadores nos
  nós internos, variáveis e constantes nas folhas), e as mutações
  trocam operadores, inserem ou podam subárvores, e simplificam
  algebricamente.
- **Constantes**: otimizadas numericamente (BFGS) dentro de cada
  expressão candidata, então a busca evolutiva cuida só da *forma*.
- **Saída**: uma **frente de Pareto** de complexidade × erro, não uma
  única expressão. Para cada complexidade (número de nós da árvore),
  a melhor expressão encontrada.
- **Seleção**: o `model_selection="best"` padrão escolhe a expressão
  com maior *score*, a maior queda de log-erro por unidade de
  complexidade adicionada, entre as que têm erro próximo ao mínimo.
  Na prática: a expressão onde "mais um termo deixa de compensar".

O SymTorch [4] é uma camada PyTorch sobre o PySR, do mesmo autor, que
automatiza a coleta de I/O e o passo 4 (ver §3.4). **Este módulo usa o
PySR direto**, porque `MPNNForecaster` já expõe a coleta de I/O
(`collect_message_io`/`collect_update_io`).

### 3.3 Por que a mensagem precisa de L1 (e o que esperar da saída)

Sem restrição, `ψ` pode codificar a interação de forma arbitrária e
distribuída em todas as dimensões de `m_ij`. Nenhuma expressão simples
vai ajustar isso.

Cranmer et al. [2] mostram que basta **penalizar a norma L1 das
mensagens** durante o treino (peso 1e-2) para que `ψ` se torne uma
**transformação linear da interação verdadeira**. É o mesmo princípio do
LASSO [15]: L1 empurra componentes irrelevantes para exatamente zero.
Na Tabela 1 do paper, o R² entre mensagem e força verdadeira vai de
≈ 0,000 (sem regularização) para ≈ 1,000 (com L1).

Consequência direta: **espere uma rotação, não a equação limpa.** Em
[2], a lei da mola saiu como `1,36Δy + 0,60Δx − (0,60Δx + 1,37Δy)/r`,
que é uma combinação linear das componentes de `F = −(r−1)r̂`. Com
`message_dim=2` a rotação é inspecionável à mão.

> O treino com L1 é responsabilidade do notebook de treino
> (`loss = mse + 1e-2 * model.message_l1()`), não deste módulo. Sem ele,
> os passos abaixo rodam, mas não recuperam nada.

### 3.4 Passo a passo

**Passo 0 — Treinar o veículo.** `MPNNForecaster` (ou
`SymbolicGraphNetwork`, se o ruído atrapalhar) com `input_window=3` e
L1 na mensagem. A janela de 3 casa com a ordem do ARMAX de referência
[1] e deixa `ψ` com **6 entradas** em vez de 48. A SR degrada rápido
acima de um punhado de variáveis, então este é o parâmetro que decide
se a rota funciona.

**Passo 1 — Coletar os pares entrada/saída de `ψ`.**

```python
inputs, messages = model.collect_message_io(x)
# inputs:   (n_amostras * 42 arestas, 2 * input_window)
# messages: (n_amostras * 42 arestas, message_dim)
```

Cada linha é uma aresta de uma amostra. Use dados **fora do treino**
(validação) para a destilação não herdar overfit.

**Passo 2 — Escolher quais componentes da mensagem destilar.** Com L1,
só algumas das `message_dim` componentes carregam informação; as
outras ficam perto de zero. Em [2], destilam-se as componentes de
**maior desvio-padrão**:

```python
std = messages.std(dim=0)
keep = std.argsort(descending=True)[:k]
messages = messages[:, keep]
```

`distill_message_fn` destila todas as colunas que receber, então esse
filtro fica a cargo de quem chama.

**Passo 3 — Estágio I: destilar `ψ`.**

```python
from mimo_sys.explainers.equation import distill_message_fn

psi = distill_message_fn(
    inputs, messages,
    input_window=3,
    niterations=100,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["square"],
)
psi.equations_   # frente de Pareto (complexidade, loss, expressão)
psi.sympy()      # expressão escolhida, como objeto sympy
psi.latex()      # pronta para o TCC
```

Qualquer argumento extra vai direto para `pysr.PySRRegressor`.

**Passo 4 — Estágio II: destilar `φ`.**

```python
from mimo_sys.explainers.equation import distill_update_fn

inputs, outputs = model.collect_update_io(x)
# inputs:  (n_amostras * 7 nós, input_window + message_dim)
# outputs: (n_amostras * 7 nós, output_window)
phi = distill_update_fn(
    inputs, outputs, input_window=3, message_dim=model.message_dim
)
```

Esta é a busca barata: a interação já foi comprimida para
`message_dim` escalares no estágio I. É o protocolo de dois estágios do
paper de Symbolic Graph Networks [10].

**Passo 5 — Recompor e reajustar.** A equação final é `φ ∘ Σ ∘ ψ`:
substitua `msg0..msg{d-1}` em `φ` pela soma de `ψ` sobre os emissores.
As constantes que saem da SR foram ajustadas para cada bloco isolado,
então o passo 4 de [2] **reajusta as constantes end-to-end** sobre a
expressão composta. O SymTorch [4] faz isso com `switch_to_symbolic`
(substitui o bloco por uma expressão diferenciável e treina de novo).
**Isso não está implementado aqui**: hoje, o reajuste precisa ser feito
no notebook.

**Passo 6 — Interpretar contra a física.** A hipótese física é o
balanço de massa do reservatório (`docs/1_analise-dados-reservatorio.md`),
em tempo discreto com `Ts = 1 h`:

$$
y_2(t+1) \approx y_2(t) + \frac{T_s}{A}\,\big(u_4(t) - y_1(t)\big)
$$

Se a rota funcionar, no receptor `y2` deve aparecer um termo
proporcional ao lag mais recente de `u4` (sinal positivo) e de `y1`
(sinal negativo), e `φ` deve ter um termo ≈ 1 · `own_lag2` (o
integrador). A EDA reforça essa expectativa: `vazao_entrada → nivel_res`
é a causalidade de Granger mais forte do dataset (lag 1, p ≈ 5e-124) e
tem coerência de 0,98 (`docs/5_eda-series-temporais.md` §5 e §7).

### 3.5 Nomeação das colunas

Os nomes passados ao PySR (`variable_names`) seguem a ordem que
`MPNNForecaster` produz:

| Função | Colunas |
|---|---|
| `default_message_feature_names(w)` | `recv_lag0..recv_lag{w-1}`, depois `send_lag0..send_lag{w-1}` |
| `default_update_feature_names(w, d)` | `own_lag0..own_lag{w-1}`, depois `msg0..msg{d-1}` |

- **`lag{k}` é a posição na janela, não a defasagem.** A janela está em
  ordem cronológica, então `lag0` é a amostra **mais antiga**
  (`t − w + 1`) e `lag{w-1}` é a **mais recente** (`t`). Com
  `input_window=3`, `send_lag2` é `u(t)`. Atenção ao traduzir a
  expressão para o TCC.
- **Receptor antes do emissor.** É a ordem de `φ^e` em [2]. Inverter
  troca o sentido de toda a expressão.
- Os nomes falam do **papel** (receptor/emissor), não do nó físico,
  porque `ψ` é a mesma função para as 42 arestas. Para nomear por nó
  físico, filtre as linhas de uma aresta específica e passe
  `feature_names`.

### 3.6 Como validar o resultado

- **Contra o ARMAX** de referência [1]: a equação destilada, com ~10
  termos, deve ficar perto da acurácia da GNN e bater o ARMAX (72
  parâmetros estimados).
- **Fora da distribuição**: em [2], a expressão simbólica generalizou
  **melhor que a própria GNN** (0,0892 contra 0,142). Vale testar no
  conjunto de teste temporal.
- **Contra o SINDy** [8] direto nos sinais. É baseline obrigatório: uma
  banca vai perguntar se a GNN era necessária.
- **Robustez a ruído**: repetir com `SymbolicGraphNetwork` [10] e ver se
  a equação sobrevive à injeção de ruído.

### 3.7 Armadilhas

- **`ψ` não sabe qual nó é qual.** A mesma `message_fn` atende as 42
  arestas e recebe só os *valores* das janelas, sem identificador de
  nó. Para escrever `+u4 − y1` em `y2`, ela precisa distinguir `u4` de
  `y1` pelos valores (escala, faixa). Em [2] isso não é problema porque
  as partículas carregam atributos físicos (massa, carga) na feature.
  Se o estágio I não recuperar sinais opostos, este é o primeiro
  suspeito; a correção seria acrescentar um embedding/one-hot de tipo
  de nó à feature, ao custo de mais entradas para a SR.
- **`pysr` não é dependência padrão.** Extra opcional `symbolic`
  (`uv add --extra symbolic pysr`, requer Julia). Import lazy:
  `import mimo_sys` funciona sem ele, e só a chamada levanta
  `ImportError` explicando o que instalar.
- **Unidades.** Se o modelo foi treinado com dados normalizados, a
  expressão sai em unidades normalizadas. Desnormalize antes de
  comparar coeficientes com `Ts/A`.
- **Não destile `StemGNN` nem `Seq2SeqLatentGNN`.** Roda, mas produz
  expressão sobre latente sem nome.

---

## 4. Rota B — Extração de subgrafo (atribuição)

[`graph/topology.py`](graph/topology.py)

A Rota B responde **quais arestas importam** para a previsão. O
entregável é uma matriz ou máscara de pesos sobre as 42 arestas: é
**atribuição, não equação**. Há duas versões, e a barata vem primeiro.

### 4.1 B1 — Peso de aresta (`edge_strength_frame`)

**Ideia.** Se a mensagem foi treinada com L1 (§3.3), arestas que não
carregam física já têm mensagem perto de zero. A topologia sai **de
graça**, sem treinar nada além do modelo:

$$
S_{ij} = \frac{1}{B}\sum_{b=1}^{B} \big\lVert m_{ij}^{(b)} \big\rVert_2
$$

ou seja, a norma média da mensagem de `j` para `i` sobre `B` amostras.

**Passo a passo.**

1. Treinar o `MPNNForecaster` com L1 na mensagem (igual à Rota A).
2. Escolher um conjunto de janelas `x`, de preferência validação ou
   teste, no **mesmo pré-processamento** usado no treino.
3. Rodar:

   ```python
   from mimo_sys.explainers.graph import edge_strength_frame

   FEATURE_COLS = [
       "freq_b1", "freq_b2", "freq_b3", "vazao_entrada",
       "vazao_distribuicao", "nivel_res", "pressao",
   ]
   frame = edge_strength_frame(model, x, feature_names=FEATURE_COLS)
   frame.loc["nivel_res", "vazao_entrada"]   # quanto u4 empurra em y2
   ```

4. Ler por **linha** (o que chega em cada nó) e comparar com a
   validação clássica da §4.3. A expectativa física é que a linha de
   `nivel_res` seja dominada por `vazao_entrada` e
   `vazao_distribuicao`.

**O que B1 mede, e o que não mede.** `S_ij` é o **tamanho** do que `j`
envia a `i`. Não é a sensibilidade da saída a `j`:

- `φ` pode atenuar ou amplificar a mensagem agregada.
- A agregação é uma **soma**: duas mensagens grandes de sinais opostos
  se cancelam, e cada uma ainda aparece com `S_ij` alto.

Para sensibilidade de verdade, use B2 ou perturbação direta.

### 4.2 B2 — Explainers de GNN (`PGExplainerTopology`)

Só usar **se B1 discordar da validação clássica** (§4.3). Os
explainers otimizam uma máscara sobre as arestas para descobrir o menor
subgrafo que preserva a previsão. Há três referências principais:

**GNNExplainer [5].** Para **uma** instância, busca o subgrafo `G_S`
que maximiza a informação mútua com a predição:

$$
\max_{G_S}\; MI\big(Y, (G_S, X_S)\big) = H(Y) - H\big(Y \mid G = G_S,\, X = X_S\big)
$$

Como `H(Y)` é constante, isso equivale a minimizar a incerteza da
predição dado o subgrafo. A máscara discreta é relaxada para
`M ∈ [0,1]^{|E|}` (sigmoide) e otimizada por gradiente, com
regularização de **tamanho** (poucas arestas) e de **entropia** (máscara
perto de 0 ou 1). Defeitos para este TCC: é *instance-level* (uma
otimização por amostra; com ~1.100 amostras exige agregar máscaras) e
não-determinístico.

**PGExplainer [6] — o escolhido.** Em vez de otimizar uma máscara por
instância, **treina uma rede** que prevê a máscara a partir dos
embeddings dos nós:

1. Para cada aresta `(i, j)`, um MLP recebe `[z_i ; z_j]` (embeddings
   dos dois extremos) e produz um logit `ω_ij`.
2. A máscara é amostrada com a relaxação *Concrete* (Gumbel-Softmax
   binária) [16], para ser diferenciável:
   `ê_ij = σ((log ε − log(1−ε) + ω_ij) / τ)`, com `ε ~ U(0,1)`.
3. A temperatura `τ` **decai ao longo das épocas**, deixando a máscara
   cada vez mais próxima de binária.
4. A loss combina fidelidade (a predição com a máscara deve ficar
   próxima da original) com as mesmas regularizações de tamanho e
   entropia do GNNExplainer.

Vantagens: **amortizado** (uma rede explica todas as instâncias),
**indutivo** (explica amostras novas sem reotimizar) e mais estável. Por
isso é preferido.

**SubgraphX [7].** Explora subgrafos conectados com *Monte Carlo Tree
Search* e pontua cada um pelo **valor de Shapley** [17] (a contribuição
marginal média do subgrafo sobre todas as coalizões de nós), aproximado
por amostragem. É o mais fiel dos três, mas o mais caro. Não
implementado.

**Como o wrapper funciona.**

```python
from mimo_sys.explainers.graph import PGExplainerTopology

topology = PGExplainerTopology(wrapped_model, n_nodes=7, epochs=30)
topology.fit(x, target)          # treino do explainer
mask = topology.explain(x, target)   # (42,), uma entrada por aresta
```

1. No construtor, monta o `edge_index` do grafo completo sem
   self-loops (42 arestas) e o `Explainer` do `torch_geometric` com
   `PGExplainer`, modo regressão, nível de nó.
2. `fit` chama o passo de treino do `PGExplainer`.
3. `explain` devolve a máscara de arestas aprendida, na ordem do
   `edge_index`.

### 4.3 Validação clássica (obrigatória antes de confiar em B1 ou B2)

Para uma banca de controle, as ferramentas clássicas de identificação
MIMO valem mais que um método de XAI, e custam pouco:

- **Causalidade de Granger** [18]: `j` causa `i` se o passado de `j`
  melhora a previsão de `i` além do passado do próprio `i`. **Já
  calculada** na EDA (`docs/5_eda-series-temporais.md` §5): é a
  comparação mais direta com a matriz de B1.
- **RGA (Relative Gain Array)** [19, 20]: com o ganho estático
  `G(0)`, `Λ = G(0) ∘ (G(0)⁻¹)ᵀ` (produto elemento a elemento). Mede o
  quanto cada par entrada-saída é acoplado aos demais. O sistema aqui é
  **não-quadrado** (4 entradas, 3 saídas), então usa-se a
  pseudo-inversa `G(0)⁺`.
- **Resposta ao degrau / impulso** [21]: a resposta ao impulso empírica
  da EDA (§10) já mostra `vazao_entrada → nivel_res` com pico em 1-2 h.

Há também literatura mostrando que explainers de GNN nem sempre superam
baselines triviais (grau, peso de aresta) quando avaliados com
cuidado [9]. Uma banca pode cobrar isso: mais um motivo para começar
por B1 e validar contra Granger e RGA.

### 4.4 Armadilhas

- **`fit` faz um único passo, com `epoch=0`.** O `PGExplainer` espera
  ser treinado por várias épocas, porque a temperatura `τ` depende da
  época. Do jeito atual, o explainer fica subtreinado: para uso real,
  chame `topology.explainer.algorithm.train(epoch, ...)` num laço sobre
  épocas e batches.
- **O `model` do B2 precisa da convenção do `torch_geometric`**
  (`forward(x, edge_index) -> (n_nodes, out_dim)`), diferente do
  `forward(x) -> (batch, window, nodes)` deste projeto. Escreva um
  wrapper antes de instanciar.
- **B2 não foi testado contra o `torch_geometric` real** (não é
  dependência do projeto). Confira na versão instalada se o
  `PGExplainer` aceita `mode="regression"` em nível de nó.
- **`torch_geometric` é extra opcional** (`explain`), com import lazy,
  mesma política do PySR.
- **Nunca em `StemGNN`.** A base é a Laplaciana aprendida: mascarar uma
  aresta muda a base inteira e a atribuição perde sentido.

---

## 5. Corroboração entre as rotas

As rotas são independentes, então o acordo entre elas é evidência:

| Se... | então... |
|---|---|
| a Rota A recupera `y2 ← +u4 − y1` **e** B1 mostra `S[y2, u4]`, `S[y2, y1]` altos **e** Granger concorda | três métodos independentes apontam para o balanço de massa |
| A recupera algo que B1 não acende | suspeitar de cancelamento na soma (§4.1) ou de a SR ter achado uma rotação ruim |
| B1 acende arestas que A não usa | mensagens grandes que `φ` atenua; checar com B2 |

Isso responde à objeção *"você só achou o que já sabia que estava lá"*:
concordância entre métodos independentes é evidência, não suposição.

---

## 6. Notas de implementação

- Nenhuma rota lê ou escreve dados: operam sobre tensores de um modelo
  já treinado (`collect_message_io`, `collect_update_io`,
  `edge_strength`).
- Dependências pesadas (`pysr`, `torch-geometric`) ficam como extras
  (`symbolic`, `explain`), importadas dentro das funções que as usam,
  para `import mimo_sys` continuar funcionando sem elas.
- Testes de contrato em
  [`tests/test_explainers.py`](../../../tests/test_explainers.py):
  nomeação de features, rotulagem e validação de `edge_strength_frame`,
  e a mensagem de `ImportError` quando `pysr` falta. Não cobrem
  `PGExplainerTopology` nem a qualidade das equações, que dependem de
  um modelo treinado e das dependências opcionais.

---

## Referências

Formato ABNT. Confira os metadados nos PDFs de `docs/pappers/` antes de
copiar para o TCC.

[1] FERRARI, L.; LEANDRO, G. V.; COELHO, L. S. Multi-objective
metaheuristics applied for the multivariable system identification.
**International Journal of Parallel, Emergent and Distributed Systems**,
2025. DOI: 10.1080/17445760.2025.2508174.

[2] CRANMER, M. et al. Discovering symbolic models from deep learning
with inductive biases. In: **Advances in Neural Information Processing
Systems (NeurIPS)**, v. 33, 2020. arXiv:2006.11287.

[3] CRANMER, M. Interpretable machine learning for science with PySR
and SymbolicRegression.jl. **arXiv preprint** arXiv:2305.01582, 2023.

[4] TAN, E.; SOUBKI, A.; CRANMER, M. SymTorch: symbolic distillation of
neural networks. **arXiv preprint** arXiv:2602.21307, 2026.

[5] YING, R. et al. GNNExplainer: generating explanations for graph
neural networks. In: **Advances in Neural Information Processing Systems
(NeurIPS)**, v. 32, 2019. arXiv:1903.03894.

[6] LUO, D. et al. Parameterized explainer for graph neural network.
In: **Advances in Neural Information Processing Systems (NeurIPS)**,
v. 33, 2020. arXiv:2011.04573.

[7] YUAN, H. et al. On explainability of graph neural networks via
subgraph explorations. In: **International Conference on Machine
Learning (ICML)**, 2021. arXiv:2102.05152.

[8] BRUNTON, S. L.; PROCTOR, J. L.; KUTZ, J. N. Discovering governing
equations from data by sparse identification of nonlinear dynamical
systems. **Proceedings of the National Academy of Sciences**, v. 113,
n. 15, p. 3932–3937, 2016. DOI: 10.1073/pnas.1517384113.

[9] FABER, L.; MOGHADDAM, A. K.; WATTENHOFER, R. When comparing to
ground truth is wrong: on evaluating GNN explanation methods. In:
**ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)**,
2021.

[10] CHEN, X.; AN, J.; GUO, J.; ZHOU, Y. Symbolic graph networks for
robust PDE discovery from noisy sparse data. **arXiv preprint**
arXiv:2603.22380, 2026.

[11] GILMER, J. et al. Neural message passing for quantum chemistry.
In: **International Conference on Machine Learning (ICML)**, 2017.
arXiv:1704.01212.

[12] BATTAGLIA, P. W. et al. Relational inductive biases, deep
learning, and graph networks. **arXiv preprint** arXiv:1806.01261,
2018.

[13] KOZA, J. R. **Genetic programming**: on the programming of
computers by means of natural selection. Cambridge: MIT Press, 1992.

[14] SCHMIDT, M.; LIPSON, H. Distilling free-form natural laws from
experimental data. **Science**, v. 324, n. 5923, p. 81–85, 2009.

[15] TIBSHIRANI, R. Regression shrinkage and selection via the lasso.
**Journal of the Royal Statistical Society: Series B**, v. 58, n. 1,
p. 267–288, 1996.

[16] MADDISON, C. J.; MNIH, A.; TEH, Y. W. The concrete distribution:
a continuous relaxation of discrete random variables. In:
**International Conference on Learning Representations (ICLR)**, 2017.

[17] SHAPLEY, L. S. A value for n-person games. In: KUHN, H. W.;
TUCKER, A. W. (ed.). **Contributions to the Theory of Games II**.
Princeton: Princeton University Press, 1953. p. 307–317.

[18] GRANGER, C. W. J. Investigating causal relations by econometric
models and cross-spectral methods. **Econometrica**, v. 37, n. 3,
p. 424–438, 1969.

[19] BRISTOL, E. On a new measure of interaction for multivariable
process control. **IEEE Transactions on Automatic Control**, v. 11,
n. 1, p. 133–134, 1966.

[20] SKOGESTAD, S.; POSTLETHWAITE, I. **Multivariable feedback
control**: analysis and design. 2. ed. Chichester: Wiley, 2005.

[21] LJUNG, L. **System identification**: theory for the user. 2. ed.
Upper Saddle River: Prentice Hall, 1999.
