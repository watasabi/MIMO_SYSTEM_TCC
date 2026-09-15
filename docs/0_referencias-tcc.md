# Referências do TCC — síntese

Tema em construção: **identificação de sistemas + extração de física via regressão simbólica**,
aplicado ao sistema MIMO do reservatório de água (paper Ferrari/Leandro/Coelho).

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

## Mapa: quem serve para quê

| Papel no TCC | Referência |
|---|---|
| Problema, dados, baseline a bater | Ferrari/Leandro (1) |
| Método central (GNN + bottleneck + regressão simbólica) | Cranmer (2) |
| Estado da arte em GNN espectral p/ séries temporais | StemGNN (3) |
| Trabalho do grupo no mesmo domínio (reservatórios) | Seq2SeqLatentGNN (4) |
| Fundamentação teórica de TGNN + lacuna em regressão | Longa survey (5) |
| Contexto do grupo (aplicação de GNN) | SatGNN (6) |

## Lacunas ainda a cobrir (referências que faltam)

- **SINDy** — Brunton, Proctor & Kutz (2016), *Discovering governing equations from data*, PNAS. Baseline obrigatório.
- **PySR** — Cranmer (2023), arXiv:2305.01582. Ferramenta da etapa 2.
- **Neural ODE** / **Universal Differential Equations** (Rackauckas et al.) — alternativa direta ao message passing.
- **Identificação de sistemas clássica** — Ljung, *System Identification: Theory for the User*. Fundamentação de ARX/ARMAX.
- **Hammerstein-Wiener / NARMAX (Billings)** — famílias não-lineares interpretáveis, meio-termo entre ARMAX e caixa-preta.
- **Physics-Informed Neural Networks (Raissi et al., 2019)** — a abordagem "impor a física" vs "descobrir a física". Precisa contrastar.
