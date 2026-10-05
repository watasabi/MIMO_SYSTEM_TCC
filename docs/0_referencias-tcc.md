# Referências do TCC — síntese

Tema em construção: **identificação de sistemas + extração de física via regressão simbólica**,
aplicado ao sistema MIMO do reservatório de água (paper Ferrari/Leandro/Coelho).

Este arquivo é a **ficha de leitura por referência**. O critério de triagem das arquiteturas e
as decisões tomadas estão em [3_investigacao-arquiteturas.md](3_investigacao-arquiteturas.md).

---

## 1. Ferrari, Leandro & Coelho (2025) — *Multi-objective metaheuristics applied for the multivariable system identification*
Int. J. Parallel, Emergent and Distributed Systems, DOI 10.1080/17445760.2025.2508174. UFPR.

**Este é o paper-alvo.**

- **Planta:** reservatório de água, dados da Sanepar (Curitiba/PR).
- **Entradas (4):** `u1,u2,u3` = frequência dos inversores das bombas P1,P2,P3 [Hz]; `u4` = vazão de entrada [l/s].
- **Saídas (3):** `y1` = vazão [l/s]; `y2` = nível do reservatório [m]; `y3` = pressão [mca].
- **Dados:** 600 amostras (estimação) + 500 (validação), Ts = 1 h, **sem pré-processamento**.
- **Modelo:** ARMAX MIMO, ordem 3, 3 equações acopladas, **~120 parâmetros** (Eq. 15, Tabela 3).
- **Método:** estimação por metaheurísticas — mono-objetivo (GA, GWO, CS) vs multi-objetivo (NSGA-II, MOGWO, MOCS). 100 runs, 1000 iterações, espaço de busca [-1,1].
- **Populações:** GA 200, GWO 800, CS 100 (mesma para as versões multi).
- **Métricas:** MSE, R² por saída; MSE₃ e R²₃ (versões "múltiplas", Eqs. 18-19) para agregar.

**Resultados-chave (Tabela 2, melhor modelo na validação):**

| | MOCS (melhor) | MOGWO (2º) | NSGA-II | GA | GWO | CS |
|---|---|---|---|---|---|---|
| R² y1 | 0.9394 | 0.9478 | 0.9421 | 0.9324 | 0.9489 | 0.9456 |
| R² y2 | **0.9596** | 0.9526 | 0.9495 | **-15.860** | 0.4108 | **-3.2297** |
| R² y3 | 0.8633 | 0.8364 | 0.8374 | 0.6310 | 0.8286 | 0.8223 |
| Tempo (s) | 32.2 | 440.2 | 2150.6 | 47.3 | 29.0 | 16.2 |

- **`y2` (nível) é a saída problemática.** Mono-objetivo dá R² **negativo** (GA -15.9 / -62.8; CS -3.2 / -3.99).
- Explicação do paper: a agregação mono-objetivo é dominada por `y1` (maior ordem de magnitude), enviesando MSE₃/R²₃.
- Nenhuma saída atinge o critério "R² entre 0.9 e 1" recomendado para controle em `y3`.
- Código MATLAB (2021b) + dados no repositório citado como ref. [34].

**Lacuna explorável:** caixa-preta pura, sem equações interpretáveis, sem projeto de controle.
A física é conhecida e simples — balanço de massa: `A·dy2/dt = u4 - y1`; `y3 ≈ ρg·y2 + perdas(y1)`;
`y1 = f(u1,u2,u3,y3)` (curva das bombas). Um ARMAX de ordem 3 aproxima mal um **integrador**
(polos perto de 1) — provável causa real do colapso em `y2`.

---

## 2. Cranmer et al. (2020) — *Discovering Symbolic Models from Deep Learning with Inductive Biases*
NeurIPS 2020, arXiv:2006.11287. Código: github.com/MilesCranmer/symbolic_deep_learning

**Referência metodológica central.** Framework em 4 passos:
1. Modelo com estrutura interna separável (Graph Network: `φ^e` edge/mensagem, `φ^v` node, `φ^u` global).
2. Treino end-to-end.
3. Regressão simbólica **em cada função interna separadamente** (eureqa; PySR é a alternativa do próprio autor).
4. Substituir as MLPs pelas expressões e **refitar as constantes**.

**O truque central — bottleneck na mensagem:** regularizar a dimensão do vetor de mensagem
(L1 com peso 1e-2, ou KL contra prior gaussiano, ou bottleneck explícito) força `φ^e` a virar
uma rotação linear da **força verdadeira**. Tabela 1: L1 dá R² ≈ 1.000 vs Standard ≈ 0.000
na correlação mensagem↔força. **L1 foi a melhor estratégia** (melhor até que bottleneck explícito).

**Por que isso importa aqui:** fatoriza o problema — em vez de buscar no espaço
combinatório do modelo inteiro (10^18 no exemplo deles), busca 2×10^9 em sub-problemas.
Esse é o argumento que justifica usar GNN em vez de SINDy direto.

**Resultados:** recupera lei de mola (`φ1 ≈ 1.36Δy + 0.60Δx - (0.60Δx+1.37Δy)/r - 0.0025`, que é
rotação de `F = -(r-1)r̂`), 1/r², 1/r³, forças descontínuas (com IF), Hamiltonianos (FlatHGN),
e **descobre fórmula nova** para overdensity de matéria escura (loss 0.0882 vs 0.121 da fórmula humana).

**Generalização simbólica:** mascarando 20% dos dados (δ>1), a expressão simbólica generaliza
**melhor que a própria GNN** de onde foi extraída (0.0892 vs 0.142 out-of-distribution).
Esse é um resultado forte de vender num TCC.

---

## 3. Cao et al. (2020) — *StemGNN: Spectral Temporal Graph Neural Network*
NeurIPS 2020, arXiv:2103.07719. Código: github.com/microsoft/StemGNN

- **GFT** (correlações inter-séries) + **DFT** (dependências temporais) **conjuntamente no domínio espectral**.
- **Latent Correlation Layer:** GRU → self-attention → matriz de adjacência `W = Softmax(QKᵀ/√d)`. Dispensa topologia prévia.
- **Spe-Seq Cell:** DFT → Conv1D → GLU → IDFT, aplicada sobre a saída da GFT.
- **Backcasting** (auto-encoder) + forecasting, com skip/residual entre 2 blocos StemGNN.
- SOTA em 9 datasets; +8.1% MAE, +13.3% RMSE sobre o melhor baseline.
- Ablação (Tabela 3): todos os componentes importam; `w/o Spe-Seq Cell` é o pior (2.612 vs 2.144 MAE).
- **Complexidade O(N³)** (decomposição da Laplaciana) — limitação declarada.

**Escala dos datasets:** 137–358 nós, 5.000–52.116 timestamps.
→ **Comparar com os 7 nós e 1.100 amostras do reservatório do Leandro.** 2 a 3 ordens de grandeza menos.

---

## 4. Seman, Stefenon, Yow, **Coelho**, Mariani (2026) — *Fourier-enhanced Seq2Seq latent GNN (Seq2SeqLatentGNN)*
Eng. Appl. of AI 167:113939. Código: github.com/lseman/Seq2SeqLatentGNN

Essencialmente StemGNN industrializado, e **com o Leandro Coelho como coautor** — é a
linha de pesquisa do grupo aplicada a reservatórios. Ler como "o que o grupo já fez".

- **Dados:** 19 reservatórios hidrelétricos do sul do Brasil (ONS), horário, multi-ano.
  Variáveis: volume útil (%), inflow/outflow (m³/s), ENA (MWmed), precipitação (mm).
- **Componentes:** (i) camada de Fourier (filtro espectral low-rank, só modos `κ1,κ2`);
  (ii) latent correlation GCN com atenção + máscara k-NN (adjacência dinâmica `A_dyn = QKᵀ/√d + A_static`);
  (iii) Seq2Seq LSTM encoder-decoder com atenção e teacher forcing; (iv) suavização gaussiana na saída.
- **Detrending híbrido adaptativo** (Savitzky-Golay + regressão polinomial com peso α(t) adaptativo nas bordas) — β=0.25, p=2, λ=2.0.
- **Tuning bayesiano** via Optuna, loss de Huber.
- Baselines: LGBMRegressor, NGBoost, Random Forest, XGBoost.

**Nota crítica:** também é caixa-preta. Nenhum desses trabalhos entrega equação.

---

## 5. Longa et al. (2023) — *GNNs for temporal graphs: state of the art, open challenges, opportunities*
arXiv:2302.01018v4. **Survey** — usar para fundamentação teórica e taxonomia.

- Formalização: Static Graph, **Temporal Graph (TG)**, **DTTG** (discrete-time), **STG** (snapshot-based), **ETG** (event-based).
- Settings de aprendizado (Fig. 1): transdutivo/indutivo × passado/futuro.
- Tarefas: node/edge/graph classification, **regressão**, link prediction, event time prediction, clustering, LDE.
- **Taxonomia TGNN (Fig. 2):** Snapshot-based {Model Evolution: EvolveGCN | Embedding Evolution: VGRNN, DySAT, DynGESN, Roland, SSGNN} × Event-based {Temporal Embedding: TGAT, NAT, TGL | Temporal Neighborhood: APAN, DGNN, TGN}.
- **Lacuna declarada e útil para o TCC:** "limited research has been conducted on the application of TGNNs to **regression tasks**" — só 2 exceções (tráfego, catapora). Clustering e LDE: nenhum método.

---

## 6. Pacheco, Seman, Rigo, Camponogara, Bezerra, **Coelho** (2025) — *GNNs for the Offline Nanosatellite Task Scheduling Problem (SatGNN)*
arXiv:2303.13773v4. UFSC + UFPR.

Também do grupo do Coelho, mas **domínio diferente** (otimização combinatória, não identificação).
GNN sobre representação bipartida de MILP, como heurística primal para o solver SCIP:
+45% no valor objetivo esperado, -35% no tempo até solução factível. Generaliza para instâncias maiores.

**Relevância para o TCC: baixa/indireta.** Serve como evidência de que o grupo trabalha com GNN,
e como referência sobre representação bipartida de problemas de otimização — mas não sobre
identificação de sistemas nem sobre interpretabilidade.

---

## 7. Autoformer — Wu et al. (2021) — *Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting*
NeurIPS 2021, arXiv:2106.13008. **Avaliado e descartado como arquitetura candidata.**

- **Auto-Correlation** substitui self-attention: agrega sub-séries por período descoberto via FFT, O(L log L).
- **Series decomposition block:** separa trend-cyclical / seasonal *progressivamente dentro* da arquitetura,
  em vez de pré-processar uma vez.

**Por que não serve para a extração de equações:**

1. **Sem acoplamento inter-séries explícito.** A Auto-Correlation age no eixo temporal; as variáveis
   entram como canais, misturados apenas pelas projeções lineares. Não há aresta nem vetor de mensagem —
   não existe função interna separável para destilar. Perde-se exatamente o que motiva usar GNN (2).
2. **Não é modelo de identificação.** É forecasting puro (futuro da série a partir do passado dela).
   Aqui `u1..u4` são exógenas e *conhecidas no futuro* (setpoints das bombas); Autoformer não tem
   onde encaixá-las. ARMAX tem.
3. **Escala de dados.** DLinear (Zeng et al., 2023, *Are Transformers Effective for Time Series
   Forecasting?*, AAAI) mostra uma única camada linear batendo Autoformer/Informer/FEDformer em LTSF,
   com datasets ordens de grandeza maiores. Com 600 amostras de treino é overfit garantido.

**O que vale aproveitar:** só o *series decomposition block* como viés indutivo. O detrending híbrido
(Savitzky-Golay + polinomial) do Seq2SeqLatentGNN (4) é a versão externa do que o Autoformer internaliza —
**e nenhum dos dois produz equação.** Relevante para a hipótese do integrador em `y2`: o nível é dominado
por tendência (integral do desbalanço de vazão), e separar trend de seasonal antes do message passing pode
limpar a mensagem na aresta o suficiente para a SR achar `A·dy2/dt = u4 - y1`. É um pré-processamento de
uma linha, não motivo para importar a arquitetura.

**Se precisar de baseline Transformer com exógenas conhecidas:** TFT (Lim et al., 2021) ou TiDE —
mas é trocar uma caixa-preta por outra.

---

## 8. Tan, Soubki & Cranmer (2026) — *SymTorch: Symbolic Distillation of Neural Networks*
arXiv:2602.21307, DAMTP Cambridge. `pappers/SymTorch_2026.pdf`

**Sucessora direta de (2), pelo mesmo autor. Ferramenta central da etapa de extração.**

- Biblioteca PyTorch-native sobre PySR. `SymbolicModel` envolve qualquer `nn.Module`, registra
  hooks para coletar I/O do bloco, roda SR, e `switch_to_symbolic` **substitui o bloco por
  expressão diferenciável** — gradiente continua fluindo, então o passo 4 do Cranmer (refitar
  constantes end-to-end) sai sem plumbing. `switch_to_block` reverte.
- Seleção na frente de Pareto por maior queda fracional de log-MAE por unidade de complexidade.
- **Contribuição que mais importa aqui: regularização por *pruning* dinâmico.** Descobre a
  dimensionalidade intrínseca da mensagem **sem conhecimento prévio**, com recuperação de lei
  de força comparável à arquitetura de bottleneck explícito. Elimina o hiperparâmetro mais
  chato de (2).
- Recupera leis de força exatas de edge models de GNN (Coulomb, mola, repulsiva power-law).
  Confirma que a GNN é o que torna a SR tratável em sistemas multipartícula.

**Experimento Lorenz — a ressalva sobre bottleneck latente.** Encoder linear → MLP em latente
baixo → decoder linear; destila o MLP. As equações recuperadas **não eram as de Lorenz** (o
sistema de coordenadas aprendido é arbitrário), mas reproduziam a geometria do atrator de dois
lóbulos. Fora de domínio, o modelo simbólico e o híbrido refitado tiveram erro
**substancialmente menor que a rede original** — indício de que a MLP tinha decorado heurísticas
por trajetória. Ver §4 de [3_investigacao-arquiteturas.md](3_investigacao-arquiteturas.md).

- PINNs: destila a camada de saída e recupera solução + constantes físicas (difusividade,
  velocidade de onda) com dado esparso. Contraste útil "impor física" vs "descobrir física".
- Também: SLIME (LIME com surrogate simbólico), análise de aritmética de LLM, surrogates
  simbólicos para MLP de transformer. **Irrelevante para o TCC** — não citar.

---

## 9. Yi et al. (2023) — *FourierGNN: Rethinking MTS Forecasting from a Pure Graph Perspective*
NeurIPS 2023, arXiv:2311.06190. BIT / Tongji / Oxford / Macquarie. `pappers/FourierGNN_2023.pdf`

**Avaliada como substituta do StemGNN no papel de modelo de acurácia. Decisão: adotada como
Modelo A.** Racional completo em §2 de [3_investigacao-arquiteturas.md](3_investigacao-arquiteturas.md).

- **Crítica ao paradigma de (3) e (4):** combinar rede de grafo (espacial) + rede temporal
  (LSTM/GRU) assume uma separação artificial e tem compatibilidade incerta entre os dois blocos.
- **Grafo hipervariado:** cada valor `x^(n)_t` é um **nó** — nó = par `(variável, timestamp)`,
  `NT` nós, adjacência inicial totalmente conectada. Intra-série e inter-série viram
  dependências nó-nó num único grafo.
- **FGO (Fourier Graph Operator):** multiplicação em espaço de Fourier. Pelo teorema da
  convolução, `F(X)·S_{A,W} = F(AXW)` — equivale a convolução de grafo no domínio do tempo.
  Versão **n-invariante** (`S ∈ C^{d×d}` compartilhado) para o grafo completo.
  Empilhar K camadas ⇄ convolução de grafo de ordem K (Proposição 1).
- **Complexidade O(NT·log(NT) + K·NT·d²)** — usa **DFT, não GFT**: evita explicitamente a
  eigendecomposição de (3). Menor contagem de parâmetros entre os baselines GNN.
- Resultados: +9.4% MAE / +10.9% RMSE médio sobre o melhor baseline em 7 datasets (incl.
  StemGNN, MTGNN, GraphWaveNet, AGCRN, Autoformer, FEDformer). Em COVID-19 multi-step, +30%.
- Ablação: node embedding e FGO dinâmico são os componentes mais significativos.
- **Visualizações de adjacência aprendida** (METR-LA correlaciona com proximidade de vias;
  padrões temporais distintos por série; dependências variando no tempo).

**Por que serve e por que não serve, para o TCC:**
- **Serve** — nós nomeados (aresta `(u4,t-k) → (y2,t)` é um coeficiente de lag no sentido
  NARMAX) e **base de perturbação fixa** (DFT não depende dos dados), então análise de
  subgrafo/máscara é bem definida, ao contrário de (3). Escala favorável: N=7, T≈24 → NT≈168.
- **Não serve para destilação** — o FGO é multiplicador diagonal compartilhado: **não existe
  MLP por aresta** nem vetor de mensagem para destilar. O que se lê de `S` é uma **função de
  transferência** (identificação por resposta em frequência), ou seja, um modelo **linear** —
  e linear-e-interpretável é o que o ARMAX de (1) já é.

---

## Mapa: quem serve para quê

| Papel no TCC | Referência |
|---|---|
| Problema, dados, baseline a bater | Ferrari/Leandro (1) |
| Método central (GNN + bottleneck + regressão simbólica) | Cranmer (2) |
| Estado da arte em GNN espectral p/ séries temporais | StemGNN (3) |
| Trabalho do grupo no mesmo domínio (reservatórios) | Seq2SeqLatentGNN (4) |
| Fundamentação teórica de TGNN + lacuna em regressão | Longa survey (5) |
| Contexto do grupo (aplicação de GNN) | SatGNN (6) |
| Baseline citado e descartado (sem exógenas, sem aresta) | Autoformer (7) |
| Ferramenta da extração simbólica (Modelo B) | SymTorch (8) |
| Modelo de acurácia + topologia do acoplamento (Modelo A) | FourierGNN (9) |

## Baixados, ainda não fichados

Em `pappers/`, lidos em diagonal durante a investigação de 2026-10-05 — precisam de ficha própria:

- **SINDy-Autoencoder** — Champion, Lusch, Kutz & Brunton (2019), PNAS, arXiv:1904.02107.
  Rota do bottleneck latente. `SINDyAutoencoder_2019.pdf`
- **PySR** — Cranmer (2023), arXiv:2305.01582. Engine de SR sob o SymTorch. `PySR_2023.pdf`
- **Symbolic Graph Networks for PDE Discovery** — arXiv:2603.22380. Cenário ruidoso e esparso.
- **GNNExplainer** — Ying et al. (2019), arXiv:1903.03894. Rota de subgrafo (só diagnóstico).
- **PGExplainer** — Luo et al. (2020), arXiv:2011.04573. Versão amortizada/indutiva; preferir a esta.
- **SubgraphX** — Yuan et al. (2021), arXiv:2102.05152. Shapley, mais fiel, caro.
- **DLinear** — Zeng et al. (2023), AAAI, arXiv:2205.13504. Sustenta o descarte de (7).
- **TFT** — Lim et al. (2021), arXiv:1912.09363. Único Transformer com exógenas conhecidas.

## Lacunas ainda a cobrir (referências que faltam)

- **SINDy** — Brunton, Proctor & Kutz (2016), *Discovering governing equations from data by sparse
  identification of nonlinear dynamical systems*, PNAS 113(15), DOI 10.1073/pnas.1517384113.
  **Baseline obrigatório. Não está no arXiv — baixar manualmente.**
- **Neural ODE** / **Universal Differential Equations** (Rackauckas et al.) — alternativa direta ao message passing.
- **Identificação de sistemas clássica** — Ljung, *System Identification: Theory for the User*. Fundamentação de ARX/ARMAX.
- **Hammerstein-Wiener / NARMAX (Billings)** — famílias não-lineares interpretáveis, meio-termo entre ARMAX e caixa-preta.
- **RGA (Relative Gain Array)** — Bristol (1966) / Skogestad & Postlethwaite. Validação clássica da topologia de acoplamento, independente da GNN.
- **Physics-Informed Neural Networks (Raissi et al., 2019)** — a abordagem "impor a física" vs "descobrir a física". Precisa contrastar.
