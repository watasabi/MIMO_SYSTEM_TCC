# Duas rotas de extração, e qual modelo serve a cada uma

Data: 2026-10-05. Operacionaliza a triagem de
[3_investigacao-arquiteturas.md](3_investigacao-arquiteturas.md): lá está *por que*
cada arquitetura entrou ou saiu; aqui está *o que fazer com cada uma*, com a API
já implementada em `src/mimo_sys/architectures/`.

Existem duas formas de tirar conhecimento interpretável de uma GNN treinada. Elas
respondem perguntas diferentes e **não são substitutas**.

| | Rota A — Equação | Rota B — Subgrafo |
|---|---|---|
| Pergunta que responde | **Como** `y2` depende de `u4` e `y1` | **Quem** depende de quem |
| Entregável | expressão fechada, ex. `A·dy2/dt = u4 - y1` | matriz/máscara de acoplamento |
| Técnica | destilação simbólica (PySR / SymTorch) | peso de aresta aprendido, ou GNNExplainer / PGExplainer / SubgraphX |
| Onde age | num **bloco interno** (`ψ` da aresta, `φ` do nó) | nas **arestas** do grafo de entrada |
| Verificação contra a física | **quantitativa** (compara coeficientes) | qualitativa ("as arestas certas acenderam") |
| Projeta controle? | sim | não |
| Generaliza fora da distribuição | sim — em Cranmer, melhor que a própria GNN (0.0892 vs 0.142) | não |
| Custo | treino + busca simbólica | só inferência (peso de aresta) ou otimização de máscara (explainers) |

**Consequência para o TCC:** a lacuna declarada é *"caixa-preta pura, sem equações
interpretáveis"*. Só a Rota A fecha essa lacuna. A Rota B é barata e vale como
**corroboração independente** — se a máscara concordar com a equação, você tem dois
métodos apontando para a mesma topologia. Mas Rota B sozinha não é entregável de TCC.

---

## O critério que liga modelo e rota

Dois requisitos, um por rota. Eles são independentes, e é por isso que um modelo pode
servir a uma rota e não à outra.

**Rota A exige um bloco interno separável cujos inputs tenham nome.** Não basta
existir um MLP: os argumentos dele precisam ser variáveis físicas. Se o bloco recebe
um latente pós-GFT ou pós-filtro-de-Fourier, a expressão que sai está escrita em
termos de nada que se possa nomear.

**Rota B exige que a computação flua por arestas discretas numa base fixa.** Mascarar
a aresta `(i,j)` tem de ser uma perturbação *local*. Se a base de representação é
derivada dos dados (autovetores de uma Laplaciana aprendida), mascarar uma aresta
muda a base inteira e a atribuição perde sentido.

---

## Mapa modelo × rota

| Modelo (`src/mimo_sys/architectures/`) | Rota A | Rota B | Por quê |
|---|---|---|---|
| `MPNNForecaster` | **sim** | **sim** | `message_fn` recebe `(janela de i, janela de j)` — tudo nomeado. Grafo completo, sem base derivada. |
| `SymbolicGraphNetwork` | **sim** | **sim** | O mesmo MPNN com warm-start S-G e injeção de ruído; protocolo de 2 estágios do paper. |
| `StemGNN` | não | não | Pós-transformada, a computação vive numa base espectral da Laplaciana aprendida. Falha nos dois critérios. |
| `Seq2SeqLatentGNN` | não | fraco | O GCN vê um latente pós-filtro-de-Fourier (sem nome). A adjacência é *aprendida*: explicar é explicar um grafo que o modelo inventou. |
| FourierGNN (**não implementado**) | não | **sim** | Nó = `(variável, lag)` e base **DFT, fixa** ⇒ máscara bem definida. Mas o FGO é multiplicador compartilhado: sem MLP por aresta para destilar. |

Duas leituras importantes dessa tabela:

- **StemGNN e Seq2SeqLatentGNN estão no repositório como baselines de acurácia**, não
  como veículos de extração. Implementá-los serve para a comparação de métrica e para
  mostrar o que a complexidade compra (e o que não compra).
- **FourierGNN é o único caso de "serve para B mas não para A"**, e é justamente por
  isso que ele é o candidato a Modelo A da §5 da triagem. Ainda não implementado.

---

## Rota A — receita

Veículo: `MPNNForecaster` (ou `SymbolicGraphNetwork`, se o ruído atrapalhar).

1. **Use `input_window=3`.** Casa com a ordem do ARMAX do artigo e, mais importante,
   `message_fn` passa a ver 6 escalares nomeados em vez de 48. Regressão simbólica
   degrada rápido acima de um punhado de entradas — este é o parâmetro que decide se
   a rota funciona.
2. **Treine com L1 na mensagem.** No training step, some `model.message_l1()` à loss
   (Cranmer et al. usam peso 1e-2). Sem isso, `message_fn` vira uma codificação
   arbitrária de alta dimensão e a destilação não recupera nada — na Tabela 1 do paper,
   R² mensagem↔força de 1.000 com L1 contra 0.000 sem.
3. **Estágio I — destile `ψ`:**
   ```python
   inputs, messages = model.collect_message_io(x)
   ```
   Colunas de `inputs`, em ordem: janela do **receptor** `i`, depois do **emissor** `j`.
   Nomeie-as nessa ordem ao passar para o PySR/SymTorch, ou a expressão sai invertida.
4. **Estágio II — destile `φ`:**
   ```python
   inputs, outputs = model.collect_update_io(x)
   ```
   Colunas: janela do próprio nó, depois a mensagem agregada. Barato, porque a
   interação já foi comprimida no estágio I.
5. **Recomponha e refite.** A equação final é `φ ∘ Agg ∘ ψ`. Com SymTorch, o
   `switch_to_symbolic` substitui o bloco por expressão **diferenciável** — refite as
   constantes end-to-end (passo 4 do Cranmer) em vez de aceitar as da busca.

Esperar uma **rotação**, não a equação limpa: em Cranmer, a lei da mola saiu como
`1.36Δy + 0.60Δx - (0.60Δx+1.37Δy)/r`, que é rotação de `F = -(r-1)r̂`. Com
`message_dim=2` a rotação é inspecionável à mão.

## Rota B — receita

Duas versões, e a barata vem primeiro.

**B1 — peso de aresta (grátis, começar por aqui):**
```python
topology = model.edge_strength(x)   # [7, 7]; [i, j] = quanto j empurra em i
```
Com L1 treinado, arestas sem física colapsam para perto de zero. É o mesmo mecanismo
que torna a mensagem interpretável — você não paga nada a mais pela topologia.

**B2 — explainer (só se B1 não bastar):** preferir **PGExplainer** (amortizado,
indutivo, estável) a GNNExplainer, que é *instance-level* (uma máscara por amostra —
com 1.100 amostras exige esquema de agregação) e não-determinístico. Há literatura
mostrando GNNExplainer não superando baselines triviais de grau/peso de aresta, o que
uma banca pode cobrar. PDFs dos três em `pappers/`.

**Validação independente, obrigatória:** antes de confiar em B1 ou B2, confira contra
o ferramental clássico de identificação MIMO — **resposta ao degrau/impulso, matriz de
ganhos estáticos, RGA (Relative Gain Array), causalidade de Granger**. Uma banca de
controle confia mais nesses que num método de XAI, e eles custam pouco.

---

## Pares a não fazer

- **Destilar `Seq2SeqLatentGNN` ou `StemGNN`.** Tecnicamente roda — SymTorch envolve
  qualquer `nn.Module`. Produz expressão sobre latente sem nome. Não é interpretabilidade.
- **GNNExplainer sobre `StemGNN`.** Base móvel: a máscara não é perturbação local.
- **Rota B como entregável principal.** Atribuição não é equação (ver a primeira tabela).
- **`input_window` grande no veículo de extração.** Mata a busca simbólica.

---

## Onde isso entra no TCC

| Capítulo | Modelo | Rota | Entrega |
|---|---|---|---|
| Baseline reproduzido | ARMAX/MOGWO | — | 72 parâmetros, R² por saída (já feito) |
| Referência de acurácia | `StemGNN`, `Seq2SeqLatentGNN`, FourierGNN | — | métrica; "o que a complexidade compra" |
| Topologia do acoplamento | `MPNNForecaster` | B1 | matriz 7×7 vs. RGA |
| **Resultado central** | `MPNNForecaster` + SymTorch | **A** | equação com ~10 termos |
| Robustez a ruído | `SymbolicGraphNetwork` | A | a equação sobrevive à injeção de ruído? |
| Corroboração | `MPNNForecaster` | A + B | máscara concorda com a equação? |

A tese segue a §5 da triagem: a equação da Rota A, com ~10 termos legíveis, chega
perto dos baselines de acurácia e bate o ARMAX de 120 parâmetros — logo a complexidade
não estava comprando física, estava comprando ajuste.
