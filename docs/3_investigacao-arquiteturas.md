# Investigação de arquiteturas para extração de equações

Data: 2026-10-05. Complementa [0_referencias-tcc.md](0_referencias-tcc.md), que fica como
ficha de leitura por referência. Aqui ficam o **critério de triagem** e as **decisões**.

Pergunta: qual arquitetura usar, dado que o objetivo não é só prever as 3 saídas do
reservatório São José, mas **extrair uma equação interpretável** (e, secundariamente, a
topologia do acoplamento entre as séries).

---

## 1. O critério de triagem

A premissa inicial da investigação estava errada. Procurava-se "um StemGNN mais atual",
assumindo que os defeitos relevantes eram os defeitos de *performance* do StemGNN —
custo O(N³) da decomposição da Laplaciana e o GRU da latent correlation layer.

**Esses defeitos são irrelevantes neste problema.** Com N=7 nós, O(N³) não é custo nenhum;
e com 600 amostras de estimação o GRU não é "pesado", é **superdimensionado**. O gargalo
aqui é dado escasso, não compute.

O defeito que importa é estrutural:

> **A interpretabilidade é limitada por os inputs do bloco destilado terem nome.**

Esse é o critério único que organiza toda a triagem abaixo. O método do Cranmer (ref. 2)
exige uma função interna separável (`φ^e`, `φ^v`, `φ^u`) cujos **argumentos sejam variáveis
físicas nomeadas**. Em `φ^e` do Cranmer os inputs são `Δx`, `Δy`, `r`, `m1`, `m2` — e só a
*saída* é latente. É por isso que a ambiguidade colapsa para uma rotação, e é por isso que
eles conseguiram reconhecer `1.36Δy + 0.60Δx - (0.60Δx+1.37Δy)/r` como rotação de
`F = -(r-1)r̂`.

Corolário prático: **"arquitetura complexa" não é o obstáculo.** SymTorch envolve qualquer
`nn.Module` — o problema nunca é o wrapper. O obstáculo é o bloco receber latentes sem nome
na entrada. Uma arquitetura profunda cujos blocos recebam `u1..u4, y1..y3` é destilável;
uma arquitetura rasa cujo bloco receba coeficientes espectrais não é.

---

## 2. Triagem das arquiteturas

| Arquitetura | Inputs do bloco interno têm nome? | Tem `φ^e` destilável? | Base de perturbação fixa? | Papel no TCC |
|---|---|---|---|---|
| **MPNN simples** (planejada) | **sim** (`u`, `y` com lags) | **sim** | sim | **Modelo B — extração** |
| **FourierGNN** (2023) | **sim** (nó = `(variável, lag)`) | não (FGO é multiplicador) | **sim** (DFT) | **Modelo A — acurácia** |
| StemGNN (2020) | não (autovetores da Laplaciana) | parcial, pós-GFT | **não** (GFT é data-dependent) | contexto / estado da arte |
| Seq2SeqLatentGNN (2026) | não (latente pós-Fourier+detrend) | sim, mas sobre grafo latente | não | trabalho do grupo |
| Autoformer (2021) | n/a — sem arestas | não | n/a | citado e descartado |
| Autoencoder GNN profundo | **não** (latente nos dois lados) | n/a | n/a | ver §4 |

### StemGNN — por que sai

A GFT coloca o acoplamento inter-séries nos **autovetores da Laplaciana** e o acoplamento
temporal nos coeficientes de Fourier. Rodar SR ali extrai uma equação sobre *modos
espectrais*, não sobre `y1, y2, y3`. Não dá para ler "isto é o balanço de massa".

O mesmo defeito derruba a rota de subgrafo: mascarar a aresta `(i,j)` perturba a Laplaciana
e portanto **muda a base de autovetores inteira**. A máscara não é perturbação local, e a
explicação fica instável. Um obstáculo, duas rotas bloqueadas, mesma causa.

### FourierGNN — a melhor das arquiteturas complexas, por dois motivos

Avaliada como candidata a substituir o StemGNN no papel de "modelo de acurácia". Ganha em
dois pontos concretos, e ambos importam aqui:

1. **O grafo hipervariado tem nós nomeados.** Cada valor `x^(n)_t` é um nó — ou seja, um nó
   é o par `(variável, timestamp)`. Uma aresta é `(u4, t-k) → (y2, t)`: isso é *exatamente*
   um coeficiente de lag no sentido ARMAX/NARMAX. Alinhamento muito melhor com identificação
   de sistemas do que o StemGNN.
2. **A base é fixa.** O FGO usa **DFT sobre os nós, não GFT** — o paper evita explicitamente
   a eigendecomposição. Base fixa e independente dos dados ⇒ mascarar aresta é perturbação
   bem definida ⇒ **análise de subgrafo volta a fazer sentido**. O paper já traz os heatmaps
   de adjacência aprendida (em METR-LA correlacionam com proximidade física das vias; e
   padrões temporais distintos por série), então o entregável "quem depende de quem" sai
   quase de graça.

Escala favorável: N=7 e janela T≈24 dá NT≈168 nós, e o FGO é **n-invariante** (parâmetros
`S ∈ C^{d×d}` compartilhados, não por aresta). Menor contagem de parâmetros entre os
baselines GNN do paper — o que, com 600 amostras, é a propriedade que mais importa.

**Mas não serve para destilação simbólica, e a razão é interessante.** O FGO é uma
multiplicação diagonal em espaço de Fourier, compartilhada entre nós: **não existe MLP por
aresta** para destilar, não existe vetor de mensagem. Só há multiplicadores complexos.

A ressalva é que isso não é *perda total* de interpretabilidade: um multiplicador diagonal no
domínio da frequência **é uma função de transferência**. Ler `S` bin a bin é, na prática,
identificação por resposta em frequência (ETFE) — linguagem nativa de uma banca de controle.
O problema é que isso entrega um modelo **linear**, e linear-e-interpretável é precisamente o
que o ARMAX do Ferrari já é. Não bate o baseline *em espécie*, só em ajuste.

**Papel decidido:** Modelo A (referência de acurácia + topologia do acoplamento), substituindo
o StemGNN nessa função. Não é veículo de extração.

### Autoformer — descartado

Documentado na seção 7 de [0_referencias-tcc.md](0_referencias-tcc.md). Resumo: sem
acoplamento inter-séries explícito (Auto-Correlation age no eixo temporal; variáveis são
canais misturados por projeções lineares), **não é modelo de identificação** (não tem onde
encaixar `u1..u4`, que são exógenas *conhecidas no futuro*), e DLinear mostra Transformer
perdendo de camada linear em LTSF com datasets ordens de grandeza maiores. Aproveita-se só a
ideia do *series decomposition block* como pré-processamento.

---

## 3. Rota alternativa avaliada: subgrafo relevante (GNNExplainer)

Hipótese testada: se o entregável for subgrafo em vez de equação, dá para usar arquitetura
mais complexa?

**Parcialmente sim.** GNNExplainer/PGExplainer são post-hoc e agnósticos dentro da família
message-passing — otimizam máscara sobre arestas/features maximizando informação mútua com a
predição. Funciona sobre stack de GAT, TGN, MPNN profunda, adjacência dinâmica por atenção.
E, pelo §2, funciona em FourierGNN (base fixa) mas **não** em StemGNN (base móvel).

**Mas o entregável degrada, e é esse o motivo de não adotar como rota principal.** A saída é
uma máscara — *"as arestas `u4→y2` e `y1→y2` importam, peso 0.8 / 0.7"*. Isso é **atribuição,
não equação**:

- Não dá para projetar controle em cima.
- Só verifica `A·dy2/dt = u4 - y1` qualitativamente ("as arestas certas acenderam"), nunca
  quantitativamente.
- Perde o resultado mais forte do Cranmer: a expressão simbólica **generalizou melhor que a
  própria GNN** fora da distribuição (0.0892 vs 0.142). Máscara não generaliza nada.

A lacuna declarada do TCC é *"caixa-preta pura, sem equações interpretáveis"*. GNNExplainer
não fecha essa lacuna — acende uma lanterna dentro da caixa.

Ressalvas técnicas, se for usado: GNNExplainer é **instance-level** (uma máscara por amostra —
com 1.100 amostras exige esquema de agregação) e não-determinístico. Preferir **PGExplainer**
(amortizado, indutivo, estável) ou **SubgraphX** (Shapley, mais fiel, caro). Existe literatura
mostrando GNNExplainer não superando baselines triviais de grau/peso de aresta; uma banca
pode cobrar isso.

**Aviso de domínio:** para identificação MIMO, "qual entrada afeta qual saída" já tem
ferramental clássico que uma banca de controle confia mais que XAI de 2019 — resposta ao
degrau/impulso, matriz de ganhos estáticos, **RGA (Relative Gain Array)**, causalidade de
Granger. Se o objetivo é só topologia, isso é mais barato e mais defensável.

**Papel decidido:** ferramenta de *diagnóstico* sobre o Modelo B (checar se a máscara concorda
com a equação extraída), não entregável.

---

## 4. Rota alternativa avaliada: extrair do bottleneck (autoencoder)

Hipótese testada: destilar do gargalo de um autoencoder GNN profundo.

**O obstáculo é identificabilidade.** O latente é definido a menos de transformação
arbitrária: `E` e `D` compostos com `T` e `T⁻¹` dão a mesma reconstrução. A equação extraída
é a equação verdadeira **em coordenadas desconhecidas** — e aqui os *dois* lados do bloco são
latentes, contra só a saída no bottleneck de mensagem (§1).

Não é hipotético: é o resultado do experimento Lorenz do SymTorch. Encoder linear → MLP em
latente baixo → decoder linear, destilando o MLP. As equações recuperadas **não eram as
equações de Lorenz** — reproduziam a geometria do atrator de dois lóbulos num sistema de
coordenadas arbitrário. Interessante como dinâmica, inútil como lei nomeada.

### Mas há argumento físico a favor, neste sistema

Registrado porque é bom e porque reforça a crítica ao baseline: **a ordem verdadeira do
sistema é baixa**. `y2` (nível) é o único estado genuíno — integra o desbalanço de vazão.
`y1` (vazão) e `y3` (pressão) são quase-algébricos em estado + entradas (curva das bombas;
`y3 ≈ ρg·y2 + perdas(y1)`). Ou seja: **3 saídas observadas, ~1–2 estados reais** — exatamente
o cenário que motiva autoencoder (muitas observações, poucos estados). E é mais um argumento
contra o ARMAX de ordem 3 do Ferrari.

Então a pergunta não é "autoencoder sim ou não", é **como fixar a ambiguidade**.

### As três travas que tornam a rota defensável

1. **Encoder/decoder lineares.** Aí `z = W·y` e a equação sai em combinação linear de
   variáveis nomeadas — inspeciona-se `W` e lê-se. Foi por isso que o SymTorch usou enc/dec
   linear no Lorenz. **"Deep" é o botão errado aqui**: profundidade compra expressividade que
   se paga em identificabilidade *e* em overfit com n=600.
2. **Dimensão do latente fixada pela física, não aprendida.** Sabe-se que é 1–2. Não deixar a
   regularização descobrir.
3. **Ancorar um eixo**: termo supervisionado forçando `z1 = y2` (nível). O balanço de massa
   fica legível direto no latente; a ambiguidade restante fica só nos eixos auxiliares.

Com as três travas isso deixa de ser "autoencoder GNN" e passa a ser **SINDy-Autoencoder**
(Champion, Lusch, Kutz & Brunton, 2019, PNAS) — a referência canônica de extrair do
bottleneck, agora em `pappers/SINDyAutoencoder_2019.pdf`.

**Aviso:** o treino conjunto autoencoder + SINDy é sensível a inicialização e ao peso dos
termos da loss; os próprios autores usam múltiplos restarts. Com 600 amostras, esperar briga.

Nota de passagem: o StemGNN **já tem** autoencoder embutido (ramo de *backcasting*), mas ali
é regularizador, não bottleneck interpretável — e está pós-GFT, logo recai no §2.

---

## 5. Decisão: dois modelos, papéis distintos

Não escolher entre as rotas — elas atacam objetos diferentes, e é essa a oportunidade.

- **Modelo A — FourierGNN.** Caixa-preta assumida. Estabelece competitividade com o estado da
  arte e com o trabalho do próprio grupo (Seq2SeqLatentGNN). Entrega métrica + heatmap de
  adjacência (topologia do acoplamento).
- **Modelo B — MPNN simples + bottleneck de mensagem + SymTorch.** Entrega a equação.
  Grafo completo sobre os 7 nós (`u1..u4`, `y1..y3`), features de nó = janela de lags
  (ordem 3, para comparar direto com o ARMAX). Validado contra RGA / resposta ao degrau.

**A tese:** B, com ~10 termos legíveis, chega perto de A e bate o ARMAX de 120 parâmetros do
Ferrari — **logo a complexidade de A não estava comprando física, estava comprando ajuste.**

### Dois bottlenecks, duas recuperações independentes

| Bottleneck | O que extrai | Ambiguidade |
|---|---|---|
| **Mensagem** (Cranmer/SymTorch) | `y1 = f(u1,u2,u3,y3)` — acoplamento entre variáveis nomeadas | rotação, inspecionável |
| **Latente** (SINDy-AE linear, dim 1–2) | `A·dz/dt = u4 - y1` — dinâmica do estado | fixada pelas 3 travas (§4) |

Se as duas convergirem para o balanço de massa, isso é evidência bem mais forte do que
qualquer uma isolada — e é a defesa direta contra a objeção *"você só achou o que já sabia
que estava lá"*, porque acordo entre dois métodos independentes é evidência, não suposição.

### Ferramenta

**SymTorch** (Tan, Soubki & Cranmer, Cambridge, fev/2026) — sucessora direta do Cranmer 2020,
pelo mesmo autor. Dois motivos para usar em vez de PySR cru:

- **Regularização por pruning** descobre a dimensionalidade intrínseca da mensagem sozinha,
  com recuperação de lei de força comparável ao bottleneck explícito. Elimina o
  hiperparâmetro mais chato do paper de 2020 (escolher a dimensão a priori).
- Substitui o bloco por **expressão diferenciável**, então o passo 4 do Cranmer (refitar as
  constantes end-to-end) sai sem escrever plumbing.

### Baselines obrigatórios

- **SINDy** direto nos sinais. Com 7 nós e física conhecida, uma banca vai perguntar se a GNN
  era necessária. O argumento de fatorização do Cranmer (10¹⁸ → 2×10⁹) é **fraco com N=7** —
  o que justifica a GNN aqui é *descobrir a topologia do acoplamento* em vez de assumi-la.
  Rodar SINDy como baseline, não como alternativa.
- **ARMAX/MOGWO** já reproduzido (`notebooks/modeling/01_mogwo_armax_reproducao.ipynb`).
- **RGA / resposta ao degrau** para validar a topologia independentemente da GNN.

---

## 6. Papers baixados nesta investigação

Em `docs/pappers/`:

| Arquivo | Referência | Para quê |
|---|---|---|
| `SymTorch_2026.pdf` | Tan, Soubki & Cranmer — arXiv:2602.21307 | ferramenta central da extração |
| `FourierGNN_2023.pdf` | Yi et al. — arXiv:2311.06190 | Modelo A |
| `SINDyAutoencoder_2019.pdf` | Champion et al. — arXiv:1904.02107 | rota do bottleneck latente |
| `PySR_2023.pdf` | Cranmer — arXiv:2305.01582 | engine de SR sob o SymTorch |
| `SymbolicGraphNetworks_2026.pdf` | arXiv:2603.22380 | SR em grafo com dado ruidoso e esparso |
| `GNNExplainer_2019.pdf` | Ying et al. — arXiv:1903.03894 | rota de subgrafo (diagnóstico) |
| `PGExplainer_2020.pdf` | Luo et al. — arXiv:2011.04573 | idem, versão amortizada (preferir) |
| `SubgraphX_2021.pdf` | Yuan et al. — arXiv:2102.05152 | idem, Shapley |
| `Autoformer_2021.pdf` | Wu et al. — arXiv:2106.13008 | citado e descartado |
| `DLinear_2023.pdf` | Zeng et al. — arXiv:2205.13504 | sustenta o descarte do Autoformer |
| `TFT_2021.pdf` | Lim et al. — arXiv:1912.09363 | único Transformer com exógenas conhecidas |

**Falta baixar (não está no arXiv):** SINDy — Brunton, Proctor & Kutz (2016), PNAS 113(15),
*Discovering governing equations from data by sparse identification of nonlinear dynamical
systems*. DOI 10.1073/pnas.1517384113.

---

## 7. Pendências

- [ ] Baixar SINDy (PNAS 2016) manualmente — é baseline obrigatório.
- [ ] Decidir ordem de execução: Modelo B primeiro (é o entregável) ou A primeiro (é a métrica).
- [ ] Verificar se `Ts = 1 h` basta para estimar `dy2/dt` com qualidade — se o nível variar
      pouco entre amostras, a derivada fica dominada por ruído e o SINDy sofre. Checar na EDA
      antes de investir na rota do §4.
- [ ] Confirmar a hipótese do integrador em `y2` (polos perto de 1 no ARMAX ajustado) — é o
      argumento que explica o colapso em `y2` relatado no artigo.
