# `mimo_sys.explainers`

Duas rotas para extrair conhecimento interpretável de uma GNN treinada
(`mimo_sys.architectures`). Este arquivo documenta **o código**: o que
cada função/classe faz, o que cada parâmetro controla e onde estão as
armadilhas.

Para *por que* existem duas rotas e qual pergunta cada uma responde, ver
[`docs/4_rotas-de-extracao.md`](../../../docs/4_rotas-de-extracao.md).
Para *por que* as arquiteturas foram escolhidas para servir (ou não) a
cada rota, ver
[`docs/3_investigacao-arquiteturas.md`](../../../docs/3_investigacao-arquiteturas.md).

As duas rotas **não são substitutas**: respondem perguntas diferentes
(como `y2` depende de `u4`, vs. quem depende de quem) e nenhuma delas
exige a outra ter rodado.

### Qual arquitetura serve a qual rota

De `docs/4_rotas-de-extracao.md` §Mapa modelo × rota — este módulo só
precisa de código para as células marcadas **sim**:

| Modelo (`src/mimo_sys/architectures/`) | Rota A | Rota B | Por quê |
|---|---|---|---|
| `MPNNForecaster` | **sim** | **sim** | `message_fn` recebe `(janela de i, janela de j)` — tudo nomeado. Grafo completo, sem base derivada. |
| `SymbolicGraphNetwork` | **sim** | **sim** | O mesmo MPNN com warm-start S-G e injeção de ruído; protocolo de 2 estágios do paper. |
| `StemGNN` | não | não | Pós-transformada, a computação vive numa base espectral da Laplaciana aprendida. Falha nos dois critérios. |
| `Seq2SeqLatentGNN` | não | fraco | O GCN vê um latente pós-filtro-de-Fourier (sem nome). A adjacência é *aprendida*: explicar é explicar um grafo que o modelo inventou. |
| FourierGNN (**não implementado**) | não | **sim** | Nó = `(variável, lag)` e base **DFT, fixa** ⇒ máscara bem definida. Mas o FGO é multiplicador compartilhado: sem MLP por aresta para destilar. |

`StemGNN` e `Seq2SeqLatentGNN` entram no TCC só como baseline de
acurácia (já implementados em `architectures/`) — **nunca** passe o
resultado deles para `equation/` ou `graph/` (ver "Pares a não fazer"
em `4_rotas-de-extracao.md`). FourierGNN, quando implementado, usa os
mesmos wrappers de `graph/` sem precisar de código novo (base DFT fixa,
igual ao MPNN).

---

## `mimo_sys.explainers.equation` — Rota A

[`equation/distill.py`](equation/distill.py). Destilação simbólica de
`message_fn` (`psi`) e `update_fn` (`phi`) do `MPNNForecaster`/
`SymbolicGraphNetwork`, via [PySR](https://github.com/MilesCranmer/PySR).
É a rota que fecha a lacuna do TCC ("caixa-preta pura, sem equações
interpretáveis") — a única das duas que entrega expressão fechada,
verificável quantitativamente contra a física e que projeta controle.

```python
from mimo_sys.explainers.equation import distill_message_fn, distill_update_fn

inputs, messages = model.collect_message_io(x)        # estágio I
psi = distill_message_fn(inputs, messages, input_window=3)

inputs, outputs = model.collect_update_io(x)           # estágio II
phi = distill_update_fn(
    inputs, outputs, input_window=3, message_dim=model.message_dim
)
```

A expressão final é `phi . Agg . psi` (soma sobre vizinhos entre as duas
fases). Rode o estágio I primeiro: o estágio II regride sobre a mensagem
já reduzida, então é a busca barata.

| Função | Entrada | O que destila |
|---|---|---|
| `distill_message_fn` | `model.collect_message_io(x)` | `psi` |
| `distill_update_fn` | `model.collect_update_io(x)` | `phi` |

### Nomeação de colunas

`default_message_feature_names(input_window)` e
`default_update_feature_names(input_window, message_dim)` geram os nomes
passados ao PySR (`variable_names`), na ordem que `MPNNForecaster` já
documenta:

- Mensagem: `recv_lag0..recv_lag{w-1}` (janela do **receptor**), depois
  `send_lag0..send_lag{w-1}` (janela do **emissor**). Inverter essa
  ordem faz a expressão saída sair trocada — `message_fn` é compartilhada
  entre todas as arestas, então os nomes são agnósticos ao nó físico, não
  ao papel receptor/emissor.
- Atualização: `own_lag0..own_lag{w-1}` (janela do próprio nó), depois
  `msg0..msg{d-1}` (mensagem agregada).

Passe `feature_names` para sobrescrever (ex. se quiser nomear pelos nós
físicos de uma aresta específica em vez do papel genérico receptor/emissor).

### Armadilhas

- **`pysr` não é dependência padrão.** É o extra opcional `symbolic`
  (`uv add --extra symbolic pysr` — requer Julia local). O import é
  lazy: `import mimo_sys` nunca falha por falta dele, só a chamada a
  `distill_message_fn`/`distill_update_fn`, com `ImportError` explicando
  o que instalar.
- **Espere uma rotação, não a equação limpa.** Em Cranmer et al., a lei
  da mola saiu como `1.36Δy + 0.60Δx - (0.60Δx+1.37Δy)/r`, rotação de
  `F = -(r-1)r̂`. Com `message_dim=2` a rotação é inspecionável à mão.
- **`model.message_l1()` precisa ter sido treinado antes.** Sem
  penalidade L1 na loss de treino, `message_fn` vira uma codificação
  arbitrária de alta dimensão e a destilação não recupera nada — isso é
  responsabilidade do notebook de treino, não deste módulo (ver
  `architectures/README.md`).
- **`**pysr_kwargs` vai direto para `PySRRegressor`** (ex.
  `niterations`, `binary_operators`, `maxsize`) — não há default
  opinativo aqui além do que o PySR já traz.

---

## `mimo_sys.explainers.graph` — Rota B

[`graph/topology.py`](graph/topology.py). Extrai **atribuição**, não
equação: "as arestas `u4→y2` e `y1→y2` importam, peso 0.8/0.7". Serve
como corroboração independente da Rota A (se a máscara concordar com a
equação, é evidência mais forte que qualquer uma isolada) ou como
diagnóstico barato de topologia — não como entregável principal do TCC.

### `edge_strength_frame` — B1, começar por aqui

```python
from mimo_sys.explainers.graph import edge_strength_frame

frame = edge_strength_frame(model, x, feature_names=FEATURE_COLS)
# DataFrame [7, 7]; frame.loc["y2", "u4"] = quanto u4 empurra em y2
```

Só rotula `model.edge_strength(x)` com nomes de nó — **grátis**, porque
é o mesmo mecanismo de L1 que já torna a mensagem interpretável (ver
`architectures/README.md`). Não há matriz de adjacência separada para
treinar. `feature_names` precisa ter `n_nodes` entradas, ou levanta
`ValueError`.

### `PGExplainerTopology` — B2, só se B1 não bastar

Wrapper fino e opcional sobre `torch_geometric.explain.PGExplainer`:
monta o grafo completo sem self-loops implícito em
`MPNNForecaster`/`SymbolicGraphNetwork` e delega treino/explicação ao
`torch_geometric` — não reimplementa o algoritmo.

```python
from mimo_sys.explainers.graph import PGExplainerTopology

topology = PGExplainerTopology(wrapped_model, n_nodes=7)
topology.fit(x, target)
mask = topology.explain(x, target)   # [n_edges]
```

| Método | O que faz |
|---|---|
| `fit(x, target)` | uma passada de treino do explainer amortizado |
| `explain(x, target)` | máscara de aresta aprendida, `(n_edges,)` |

### Armadilhas

- **`torch_geometric` não é dependência padrão.** É o extra opcional
  `explain` (`uv add --extra explain torch-geometric`). Import lazy,
  mesma política do PySR na Rota A.
- **`model` precisa seguir a convenção node-level do `torch_geometric`**
  (`forward(x, edge_index) -> (n_nodes, out_dim)`), diferente do
  `forward(x) -> (batch, window, nodes)` nativo das arquiteturas deste
  projeto. `PGExplainerTopology` não faz essa adaptação — escreva um
  wrapper de `forward` antes de instanciar.
- **Nunca em `StemGNN`.** A base é a Laplaciana aprendida (base móvel):
  mascarar uma aresta muda a base inteira, e a atribuição perde sentido.
  Funciona em `MPNNForecaster`/`SymbolicGraphNetwork` porque o grafo é
  completo e fixo, não aprendido.
- **Validação obrigatória antes de confiar no resultado:** compare
  contra RGA, resposta ao degrau/impulso ou causalidade de Granger — uma
  banca de controle confia mais nessas ferramentas clássicas que em XAI
  de 2020, e elas custam pouco (`notebooks/eda/03_eda_series_temporais.ipynb`
  já traz Granger).

---

## Notas de implementação

- Nenhuma das duas rotas lê ou escreve dados — operam sobre tensores já
  produzidos por um modelo treinado (`collect_message_io`,
  `collect_update_io`, `edge_strength`).
- Dependências pesadas e específicas de cada rota (`pysr`,
  `torch-geometric`) ficam como extras (`symbolic`, `explain`),
  importadas lazy dentro das funções/classes que as usam — nunca no
  topo do módulo — para `import mimo_sys` continuar funcionando sem
  elas.
- Testes de contrato em
  [`tests/test_explainers.py`](../../../tests/test_explainers.py):
  cobrem nomeação de features, rotulagem/validação de
  `edge_strength_frame` e a mensagem de `ImportError` quando `pysr`
  falta. Não cobrem `PGExplainerTopology` nem a qualidade das equações
  destiladas — isso depende de um modelo treinado e de `torch_geometric`
  instalado, fora do escopo de um teste de contrato.
