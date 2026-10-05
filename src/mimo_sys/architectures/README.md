# `mimo_sys.architectures`

Quatro arquiteturas para identificação do sistema MIMO do reservatório São José,
sob um contrato único. Este arquivo documenta **o código**: o que cada classe faz,
o que cada parâmetro controla e onde estão as armadilhas.

Para *por que* cada arquitetura foi escolhida ou descartada, ver
[`docs/3_investigacao-arquiteturas.md`](../../../docs/3_investigacao-arquiteturas.md).
Para *qual serve a qual rota de extração*, ver
[`docs/4_rotas-de-extracao.md`](../../../docs/4_rotas-de-extracao.md).

---

## Contrato comum

```python
forward(x) : (batch, input_window, n_nodes) -> (batch, output_window, n_targets)
```

- **Nó = uma série.** `n_nodes=7` no sistema São José, na ordem `FEATURE_COLS` dos
  notebooks de processing: `freq_b1`, `freq_b2`, `freq_b3`, `vazao_entrada`
  (`u1..u4`), depois `vazao_distribuicao`, `nivel_reservatorio`, `pressao` (`y1..y3`).
- **`target_idx`** seleciona os nós de saída, na ordem em que devem voltar. Para o
  sistema do artigo: `[4, 5, 6]`.
- **Feature de nó = a janela de lags daquela série.** É o que mantém os argumentos
  dos blocos internos nomeáveis — ver "Por que `input_window=3`" abaixo.
- **Batch-first**, ao contrário do `notebooks/training/01_seq2seq_attention.ipynb`,
  que permuta para seq-first internamente.
- Todos os hiperparâmetros depois de `target_idx` são **keyword-only**.
- Sem `torch_geometric`: com N=7 a adjacência densa com `einsum` é mais simples e
  evita uma dependência pesada.

```python
from mimo_sys.architectures import MPNNForecaster

model = MPNNForecaster(
    n_nodes=7,
    input_window=3,
    output_window=1,
    target_idx=[4, 5, 6],
    message_dim=2,
)
```

`torch` está no extra `ml`, não nas dependências padrão:
`uv run --extra ml pytest tests/`.

---

## `MPNNForecaster` — [`mpnn.py`](mpnn.py)

MPNN totalmente conectada. **É o veículo de extração** (Modelo B da triagem): a razão
de existir é que `message_fn` e `update_fn` são submódulos isolados cujos argumentos
são variáveis físicas nomeadas, então um regressor simbólico roda em cada um
separadamente.

```
h_i' = update_fn( h_i , Σ_{j≠i} message_fn(h_i, h_j) )
```

| Parâmetro | O que controla |
|---|---|
| `hidden_size` | largura dos MLPs de mensagem e update |
| `message_dim` | **o bottleneck**. Cranmer recupera leis de força com 2-3 |
| `n_layers` | rodadas de message passing. `1` mantém `update_fn` destilável de uma vez; `>1` exige uma regressão simbólica por rodada |

**Grafo completo, adjacência não aprendida.** A topologia de acoplamento se lê em
`edge_strength()`: com penalidade L1 nas mensagens, arestas que não carregam física
colapsam para perto de zero. É o mesmo mecanismo que torna a mensagem interpretável,
então a topologia sai de graça — não há matriz de adjacência separada para treinar.

### API de extração

```python
model.message_l1()                  # somar à loss; Cranmer et al. usam peso 1e-2
model.edge_strength(x)              # [7, 7]; [i, j] = quanto j empurra em i
model.collect_message_io(x)         # Estágio I: (inputs, messages)
model.collect_update_io(x)          # Estágio II: (inputs, outputs)
```

- **Ordem das colunas em `collect_message_io`:** janela do **receptor** `i`, depois do
  **emissor** `j` — mesma ordem de argumentos do `phi^e` do Cranmer. Nomeie nessa ordem
  ao passar para o PySR/SymTorch, ou a expressão sai invertida.
- **`collect_update_io`:** janela do próprio nó, depois a mensagem agregada.
- `message_l1()` levanta `RuntimeError` se chamado antes de um `forward`.

### Armadilhas

- **`message_l1()` não é opcional.** Sem L1 na loss, `message_fn` vira codificação
  arbitrária de alta dimensão e a destilação não recupera nada. Tabela 1 do Cranmer:
  R² mensagem↔força de 1.000 com L1 contra 0.000 sem.
- **Por que `input_window=3`:** `message_fn` vê `2 * input_window` escalares. Com 3
  (a ordem do ARMAX do artigo) são 6 entradas nomeadas, tratável para regressão
  simbólica; com 24 são 48 e a busca degrada. Este é o parâmetro que decide se a rota
  da equação funciona.
- Diagonal das mensagens é zerada: um nó já vê a própria janela via `h_i`.

---

## `SymbolicGraphNetwork` — [`symbolic_graph_network.py`](symbolic_graph_network.py)

`MPNNForecaster` mais os dois estabilizadores do arXiv:2603.22380, voltados a registros
curtos e ruidosos. Mesma API de extração, delegada ao MPNN interno (`self.mpnn`).

1. **Warm-start geométrico** — filtro polinomial Savitzky-Golay na janela crua, para a
   rede não ter de aprender gradientes a partir de jitter (o "cold-start" do paper).
   Implementado como `conv1d` depthwise com coeficientes congelados de
   `scipy.signal.savgol_coeffs`, para rodar on-device dentro do `forward` em vez de
   ser um passo numpy de pré-processamento.
2. **Injeção dinâmica de ruído** — perturbação gaussiana das features **só em treino**,
   escalada pelo ruído *estimado*: `sigma = (x - savgol(x)).std()`. É o resíduo de alta
   frequência que o filtro removeu, o que torna a injeção adaptativa em vez de um
   hiperparâmetro fixo. Só formas de operador que sobrevivem à perturbação são retidas.

| Parâmetro | O que controla |
|---|---|
| `savgol_window` / `savgol_order` | comprimento (ímpar) e ordem do ajuste polinomial local |
| `noise_scale` | multiplicador sobre o ruído estimado; `0` desliga a injeção |

**Armadilhas:**

- O paper restringe `message_dim` a 1 ou 2 pelo argumento da hipótese da variedade;
  o default aqui é 2, por paridade com o `MPNNForecaster`.
- **O warm-start não age na configuração de extração, e a correção não é óbvia.**
  `SavitzkyGolayFilter` devolve a entrada intacta quando
  `input_window < savgol_window`, então com `input_window=3` e o default
  `savgol_window=5` não há filtragem. Mas baixar para `savgol_window=3` mantendo
  `savgol_order=2` **também** é identidade: um polinômio de grau 2 passa exatamente por
  3 pontos, e os coeficientes são `[0, 1, 0]`. Para filtro ativo com janela 3 use
  `savgol_order=1` (coeficientes `[1/3, 1/3, 1/3]`, ou seja, média móvel). Verifique com
  `scipy.signal.savgol_coeffs(window, order)` antes de confiar na configuração.
- O framing de PDE do paper não se aplica aqui: não há grade espacial, o grafo são as
  7 variáveis do sistema. O que importa deste paper é a robustez a ruído.

---

## `StemGNN` — [`stemgnn.py`](stemgnn.py)

Reprodução de Cao et al. (NeurIPS 2020). **Baseline de acurácia, não veículo de
extração** — depois da transformada a computação vive numa base espectral da Laplaciana
aprendida, então uma expressão destilada estaria escrita sobre modos espectrais, não
sobre `u1..u4` / `y1..y3`.

Componentes:

- **`LatentCorrelationLayer`** — GRU sobre o eixo de **nós** (como no código dos
  autores: sequência de comprimento `n_nodes`, features `input_window`), depois
  self-attention de uma cabeça dá `W = softmax(QKᵀ/√d)`. Dispensa topologia prévia.
- **`chebyshev_basis`** — `T_0 = I`, `T_1 = L`, `T_k = 2L·T_{k-1} - T_{k-2}` sobre a
  Laplaciana normalizada simétrica.
- **`SpeSeqCell`** — DFT → Conv1D → GLU → DFT inversa no eixo do tempo. Parte real e
  imaginária são empilhadas como canais para uma convolução real poder misturá-las.
- **Blocos com forecast + backcast**, encadeados pelo resíduo do backcast.

| Parâmetro | O que controla |
|---|---|
| `hidden_size` | largura do GRU/atenção da camada de correlação |
| `cheb_order` | polinômios de Chebyshev usados, que é também o número de canais espectrais no bloco |
| `n_blocks` | blocos empilhados |
| `kernel_size` | largura da convolução da Spe-Seq (ímpar) |

### Duas armadilhas documentadas, ambas achadas por teste

**1. Não use eigendecomposição exata.** O paper descreve GFT por autovetores; o
`microsoft/StemGNN` usa Chebyshev, e essa escolha é estrutural aqui, não otimização de
velocidade. Com N=7 a atenção aprendida fica quase uniforme (0,1429 ≈ 1/7), então a
Laplaciana tem autovalor 0 uma vez e autovalor ~1 com **multiplicidade 6** (gaps da
ordem de 1e-4). Autovetores nesse subespaço quase degenerado são numericamente
arbitrários e, como a Spe-Seq cell fica *entre* a transformada direta e a inversa, os
sinais arbitrários **não se cancelam**: duas chamadas dos mesmos pesos diferiam em até
0,21. Polinômios de Chebyshev são funções da própria Laplaciana e nunca referenciam uma
base de autovetores. Coberto por `test_stemgnn_is_reproducible`.

**2. `backcast_loss` é obrigatória no treino.**

```python
loss = mse(model(x), y) + model.backcast_loss(x)
```

O StemGNN treina forecasting e backcasting conjuntamente. Sem o termo de reconstrução,
o resíduo do último bloco é descartado ao fim do laço e **aqueles parâmetros não recebem
gradiente nenhum**. Coberto por `test_backward_reaches_every_parameter`.

Expõe também `last_adjacency` (`[7, 7]`) e `last_backcast`, preenchidos pelo último
`forward`.

---

## `Seq2SeqLatentGNN` — [`seq2seq_latent_gnn.py`](seq2seq_latent_gnn.py)

Reprodução de Seman, Stefenon, Yow, **Coelho** & Mariani (2026) — o trabalho do mesmo
grupo do artigo-alvo, no mesmo domínio (reservatórios). Baseline de "o que o grupo já
fez". Como o StemGNN, é caixa-preta para extração: o GCN vê um latente já filtrado
espectralmente, cujos argumentos não têm nome.

- **`FourierFilter`** — filtro espectral low-rank: ganhos complexos aprendidos nos
  modos `[k_low, k_high)`, todo o resto zerado. É a banda `κ1, κ2` do paper.
- **`LatentCorrelationGCN`** — `A_dyn = softmax(QKᵀ/√d)` com **máscara k-NN** (só os
  `top_k` vizinhos mais fortes por nó) mais uma `A_static` aprendida.
- **Seq2Seq LSTM** encoder-decoder com dot-attention e teacher forcing.
- **`GaussianSmoothing`** — suavização de kernel fixo ao longo do horizonte.

| Parâmetro | O que controla |
|---|---|
| `hidden_size` / `num_layers` | LSTM do encoder e do decoder |
| `top_k` | vizinhos retidos pela máscara do GCN |
| `k_low` / `k_high` | banda de Fourier retida; `k_high` é clipado ao número de bins do `rfft` |
| `smoothing_kernel` | largura da suavização gaussiana (ímpar) |

### Teacher forcing

```python
model(x, target=y, teacher_forcing_ratio=0.5)
```

Levanta `ValueError` se `teacher_forcing_ratio > 0` sem `target`. Em avaliação, chame
sem `target` (razão 0, autoregressivo puro).

### Armadilhas

- **`last_adjacency` é a média sobre o batch.** Cada amostra mascara um conjunto
  diferente de vizinhos, então a média tem mais não-zeros por linha que `top_k`. Para
  verificar a esparsidade da máscara, rode uma amostra por vez.
- **`GaussianSmoothing` é identidade quando `output_window < kernel_size`** — logo não
  age em previsão de um passo. Proposital, para a configuração single-step continuar
  usável.
- O **detrending híbrido adaptativo** do original (Savitzky-Golay + regressão
  polinomial com peso adaptativo nas bordas) é pré-processamento, não arquitetura, e
  vive em `mimo_sys.preprocessors`. Deliberadamente não duplicado aqui.

---

## Comparação

Parâmetros medidos com `output_window=1`, `n_nodes=7`, `hidden_size=64`,
`message_dim=2`:

| Classe | `input_window=24` | `input_window=3` | Destilável? | Subgrafo? |
|---|---|---|---|---|
| `MPNNForecaster` | 13.379 | 9.347 | **sim** | **sim** |
| `SymbolicGraphNetwork` | 13.379 | 9.347 | **sim** | **sim** |
| `StemGNN` | 27.658 | 22.408 | não | não |
| `Seq2SeqLatentGNN` | 46.764 | 44.916 | não | fraco |

O ARMAX do artigo tem 72 parâmetros estimados (eq. 15 / Tabela 3) — razão a mais para o
entregável ser a **equação destilada**, não o modelo.

## Notas de implementação

- **Nunca use operações in-place em tensores do grafo de autograd.** `adjacency += ...`
  sobre a saída de um softmax quebra o backward (`one of the variables needed for
  gradient computation has been modified by an inplace operation`). O `ruff
  --unsafe-fixes` sugere exatamente essa conversão para `PLR6104`; o único caso que
  precisa de reatribuição (`residual = residual - backcast` no `StemGNN`) tem
  `noqa: PLR6104` com a razão no comentário.
- Construtores de modelo passam do limite default de argumentos do Pylint; o
  `pyproject.toml` sobe `max-args` para 12, e os hiperparâmetros são keyword-only para
  os call sites continuarem explícitos.
- Testes de contrato em [`tests/test_architectures.py`](../../../tests/test_architectures.py):
  shapes, determinismo por seed, backward alcançando todo parâmetro, e as propriedades
  específicas de cada arquitetura. Não testam qualidade de previsão — isso é dos
  notebooks de treino.
