# Guia de edição do artigo do TCC (GOTAS)

Contexto para editar o artigo (ATCC) e o relatório sem refazer decisões
já tomadas. Vale para qualquer agente de IA e para o autor. Leia antes de
mexer no `main.tex` ou no doc 13. As regras gerais de escrita e do
modelo ATCC estão também na skill `academic-paper-writing`
(`~/.claude/skills/academic-paper-writing/`).

Estado em 2026-10-26: artigo com 21 páginas, compilando sem erros.

---

## 1. Onde está cada coisa

| O quê | Onde |
|---|---|
| Artigo (fonte da verdade) | repositório `watasabi/TCC_MIMO_SYS_overleaf`, clonado em `~/code/TCC_MIMO_SYS_overleaf`; integrado ao Overleaf pelo GitHub |
| Arquivos do artigo | `main.tex`, `referencias.bib`, `imagens/` |
| Cópia local de trabalho | `overleaf/` deste repositório (ignorada pelo git; serve de espelho) |
| Relatório completo (tudo o que saiu do artigo) | `docs/13_relatorio-artigo-vs-modelo-final.md` |
| Modelo Word da Unicuritiba | `~/Downloads/TCC_3 - Modelo ATCC.docx` (com os comentários da coordenação) |
| Artigo de referência | `docs/pappers/Multi-objective metaheuristics applied for the multivariable system identification.pdf` |
| Explicação peça por peça do modelo | `docs/.explanation_code/arquitetura_spectral_tf_net.ipynb` (local, fora do git) |

## 2. Fluxo de edição

1. **Antes de editar:** `git pull` no repositório do Overleaf. O autor
   também edita direto no Overleaf, e as mudanças chegam como commits
   "Updates from Overleaf".
2. **Compilar localmente** numa pasta do scratchpad: `pdflatex → bibtex →
   pdflatex → pdflatex`. O TeX local não tem `brazil.ldf`: baixe o
   `portuges.zip` do CTAN, rode `tex portuges.ins` e copie os `.ldf` para
   a pasta de compilação (o Overleaf já tem). Use `timeout` no `pdflatex`:
   uma tabela com especificação de colunas quebrada trava a compilação.
3. **Conferir:** `grep -E "^!" main.log` (erros), referências e citações
   indefinidas, `Overfull \hbox` acima de ~10 pt. Renderize as páginas
   alteradas (`pdftoppm -r 40` e `montage`) e confira a posição das
   figuras.
4. **Toda imagem usada no `main.tex` precisa estar no commit**; uma
   imagem faltando quebra a compilação no Overleaf.
5. **Commit** em Conventional Commits, em português, sem atribuição a IA
   (nem `Co-Authored-By`). Commitar nos dois repositórios quando a
   mudança também toca o doc 13 ou os scripts. O autor faz o `git push`
   e, no Overleaf, "Pull GitHub changes into Overleaf", salvo quando pedir
   o contrário.
6. Copiar o `main.tex` e o PDF para `overleaf/` deste repositório, para o
   espelho não ficar desatualizado.

## 3. Estrutura: igual ao modelo da Unicuritiba, sem acréscimos

O autor pediu explicitamente: **a estrutura tem que ser a mesma do
template**. Não crie títulos que o template não tem, nem para ajudar a
navegar.

```
Cabeçalho (título, autor¹, e-mail, "Professor orientador: Dr. ..." x2, curso)
Resumo + Palavras-chave
1. INTRODUÇÃO            sem subtítulos
2. REFERENCIAL TEÓRICO   2.1 ... 2.5 (subtítulos permitidos)
3. METODOLOGIA           fluxograma; 3.1 Materiais; 3.2 Procedimentos (sem níveis 3 e 4)
4. RESULTADOS E DISCUSSÕES  sem subtítulos; abre com parágrafo-roteiro
5. CONCLUSÕES            texto corrido (o autor não quer tópicos)
AGRADECIMENTOS
REFERÊNCIAS BIBLIOGRÁFICAS
```

- **Questão norteadora, objetivo geral e objetivos específicos** fecham a
  Introdução em texto corrido, nos dois últimos parágrafos: sem rótulo em
  negrito, sem citação em bloco, sem lista e sem subseção (decisão do
  autor; o modelo Word só exige que os objetivos constem). Não há
  parágrafo de roteiro das seções depois deles.
- **A conclusão responde à questão norteadora e a cada objetivo**, em
  parágrafos, sem lista.
- **Toda equação usada no trabalho é explicada no referencial teórico**
  (comentário da coordenação). A metodologia aplica e cita as equações
  do referencial; equação que não é usada não entra. Hoje o referencial
  tem: ARMAX (1), R² e RMSE (2), continuidade, trapézio e balanço de
  massa, convolução/filtro na frequência, passagem de mensagens, perda
  de Kendall, sparse-group lasso e aproximação de Laplace.

## 4. Formatação (copiada do .docx, não do template LaTeX)

O template LaTeX antigo divergia do Word em vários pontos; vale o Word.

| Item | Como é |
|---|---|
| Corpo | Times 12, espaço simples, justificado, recuo de 1,25 cm |
| Títulos | nível 1 `1. INTRODUÇÃO` (com ponto, caixa alta, negrito); nível 2 `2.1 Subtítulo` (negrito, caixa baixa); 12 pt antes e depois |
| Cabeçalho | título 14 pt negrito; autor 10 pt negrito; e-mail 9 pt; orientadores 10 pt; curso 9 pt; nota de rodapé ¹ "Graduação em Engenharia de Controle e Automação – Centro Universitário Curitiba (Unicuritiba)." |
| Resumo | título "Resumo" à esquerda, negrito; 100 a 250 palavras; **sem nomes de autores nem de obras**; "Palavras-chave:" sem negrito, 3 a 5, separadas e terminadas com ponto |
| Equações | à esquerda (`fleqn`, recuo 1,25 cm), número à direita |
| Figuras | legenda acima, 12, negrito, à esquerda; **imagem à esquerda**; "Fonte: Elaborado pelo autor." 10, à esquerda; figura de terceiros: "Fonte: \cite{chave}." → (SOBRENOME, ano) |
| Tabelas | legenda acima, 12, negrito, centralizada; corpo fonte 10, **grade completa**, células centralizadas, **nada em negrito**; "Fonte: Autor." |
| Posição | `[H]`, logo depois do parágrafo que cita a figura ou tabela (com `[htb]` o LaTeX corta parágrafos e afasta as figuras) |
| Referências | título "REFERÊNCIAS BIBLIOGRÁFICAS" no estilo de nível 1; ABNT autor-data (`alf`); à esquerda; linha em branco entre entradas |
| Cabeçalho e rodapé | logo centralizado, **sem linha** sob o logo e **sem número de página** (como no Word) |
| Orientadores | `\orientador{Dr. Aron Luiz Oliveira dos Santos}` e `\coorientador{Dr. Allan Christian Krainski Ferrari}`, uma linha "Professor orientador: ..." para cada |

Tamanho: o modelo é um artigo de ~12 páginas (referencial ~3,
metodologia ~3, resultados ~4). Hoje são 21: referencial ~4 (p. 2–6),
metodologia ~7 (p. 6–13), resultados ~5 (p. 13–18). O que sai do artigo
continua no doc 13.

## 5. Decisões de conteúdo (não refazer)

**O modelo.** Chama-se **GOTAS** (*Graph Operator for Transfer-function
ARX in the Spectral domain*); no código a classe continua
`SpectralTFNet`. O modelo final é o SpectralTFNet do doc 8 com a
**combinação do nível por passo**: um único modelo, **sem ensemble e sem
busca de hiperparâmetros** (os dois rendiam +0,003 ou menos). A perda
com pesos por saída (Kendall) **fica**: trocar por MSE simples custa
−0,020 (`full`) e −0,037 (`pumps`). LightGBM, intervalos conformal e
PyMC estão **fora** do artigo. A aproximação de Laplace entra só para a
significância das arestas do modelo final.

**A comparação com o artigo de referência.** O foco é **simulação livre
de 6 e 12 h**, o uso operacional; a comparação em 1 passo é só uma
verificação breve, em texto, porque nela até a persistência explica mais
de 75% da variância. O artigo de referência é de **Ferrari, Leandro e
Coelho (2025)**: são três autores, A. C. K. Ferrari, G. V. Leandro e
L. dos S. Coelho (conferido no PDF); "Leandro" é G. V. Leandro, não o
prenome de Leandro dos Santos Coelho. O primeiro autor, **A. C. K. Ferrari, é o
coorientador** do TCC: nunca em tom de crítica. A diferença entre os
trabalhos é de pergunta e de protocolo; o foco do texto é o GOTAS.

**Números principais** (teste; média de 5 treinos;
`outputs/lean_model_I0_ref.json`, `outputs/extracted_model.json`,
`outputs/extracted_model_laplace.json`):

| Medida | Valor |
|---|---|
| R² médio, simulação livre 6 h | GOTAS 0,862 (`full`) / 0,775 (`pumps`); ARMAX 0,606 |
| R² médio, simulação livre 12 h | GOTAS 0,833 / 0,719; ARMAX 0,333 |
| 12 h por saída (`full`) | vazão 0,925, nível 0,811, pressão 0,762 |
| R² por passo (`full`) | 0,89 no passo 1 → 0,79 no passo 12; ARMAX < 0 a partir de 8 h |
| 1 passo (protocolo dos autores) | GOTAS 0,902; ARMAX MOGWO 0,909 (MOCS 0,921 publicado); persistência 0,829 |
| Componentes (12 h) | entradas futuras +0,147; resíduo +0,036; Kendall +0,020/+0,037; combinação do nível +0,006/+0,014 |
| Balanço de massa | a = 0,00277 m por l/s → A ≈ 1.300 m²; aresta entrada → nível soma 0,0029 m por l/s nos atrasos 1–3 |
| Laplace (seed 42, ganho em 3 h) | 12 de 24 arestas com \|z\| > 3, todas com o sinal da física; entrada → nível +0,0035 (z = 29,5) |
| Tamanho | 387 parâmetros no grafo + 2.908 no resíduo + 3 + 2 + 12 ≈ 3,3 mil; ~1,3 MFLOP por previsão |

Ao mudar um número, mude no resumo, no texto, nas tabelas e na
conclusão, e confira na fonte (`grep` pelo número no `main.tex`).

## 6. Figuras: o que o autor quer e de onde vêm

O autor gosta de **painéis duplos (a)/(b) que relacionam duas figuras** e
economizam espaço. Todas as figuras de resultado são do **modelo final**,
nunca do modelo linear antigo do doc 8.

| Figura no artigo | Arquivo | Gerada por |
|---|---|---|
| GNN: conceito (a)(b)(c) | `gnn_conceito.pdf` | `notebooks/training/scripts/figure_gnn_concept.py` |
| DCRNN (terceiros) | `dcrnn_li2018.png` | print da Figura 2 de Li et al. (2018), com "Fonte: (LI et al., 2018)". Rascunho para substituí-la, ainda fora do artigo: `stgnn_generica` de `figure_gnn_concept.py` (`stgnn()`), GNN espaço-temporal genérica (séries sintéticas, sem o reservatório, para não confundir com o GOTAS) |
| Fluxograma da metodologia | TikZ no próprio `main.tex` | — |
| Coerência (a) + polos ARX (b) | `eda_15_coherence.png`, `eda_18_arx_poles.png` | `notebooks/eda/03_eda_series_temporais.ipynb` |
| STL (a) + variância por banda (b) | `eda_06_stl_decomposition.png`, `decomp_variancia_por_banda.png` | notebook de EDA 03 e `run_decomposition.py` |
| Arquitetura | `stf_arquitetura_drawio.png` | `figure_architecture_drawio.py` (gera o `.drawio`) + exportação pelo MCP do draw.io |
| Série prevista 12 h | `stf_modelo_serie_full.png` | `report_lean_model.py` |
| Espectro (a) + R² por banda (b) | `stf_modelo_espectro_full.png`, `stf_modelo_r2_banda_full.png` | `figure_spectral_performance.py` (só lê as previsões salvas e copia para o Overleaf) |
| R² por passo | `stf_modelo_por_horizonte.pdf` | `report_lean_model.py` |
| Grafo extraído | `stf_grafo_extraido_full.pdf` | `report_extracted_model.py` |
| Respostas ao impulso | `stf_kernels_extraidos.png` | `report_extracted_model.py` |
| Força por banda (a) + Laplace (b) | `stf_modelo_band_strength.png`, `stf_modelo_laplace_z.png` | `figure_extracted_graph.py` |

A Tabela de outras GNNs (`tab:gnns`) vem de
`run_architectures_tcc.py report` (configuração escolhida na validação;
`report all` lista as quatro configurações de cada arquitetura).

Saíram do artigo, mas o autor pode pedir de volta (as imagens ainda estão
em `imagens/` do repositório do Overleaf): correlação e Granger,
ACF/PACF, Welch, espectrograma, cointegração/DTW, resposta ao impulso
empírica, grafo do reservatório, protocolos em barras, filtros H(f) e
erro por banda.

## 7. Armadilhas já vistas

- **`extract_spectral_tf_net.py` treina o modelo do doc 8 do zero e salva
  por cima de `models/spectral_tf_net_h12_w168_s42.pth`.** Não rode para
  gerar figuras; use `figure_extracted_graph.py` e
  `report_extracted_model.py`, que só leem o modelo final.
- O draw.io só funciona com o MCP conectado (`list-documents`); sem ele,
  dá para editar o `.drawio` pelo gerador, mas não exportar o PNG.
- Ao tirar títulos do texto, as referências cruzadas para eles quebram:
  confira `\ref{sec:...}` depois.
- O BibTeX abrevia "Di Nino" como prenome; use `{DI NINO}, Leonardo`.
- Referências: confira autores, iniciais e edição no próprio PDF ou DOI.
  Já houve título inventado (iCSNN), autor faltando (Ferrari, Leandro e
  Coelho) e edição não confirmada (Chapra e Canale, hoje na 7. ed. em
  inglês, de 2015).
- A pasta do scratchpad é limpa entre sessões; os `.ldf` precisam ser
  baixados de novo.

## 8. Pendências

- **Um "x" sobrando em `\begin{SingleSpace}x`**, vindo de uma edição no
  Overleaf: imprime um "x" no início do resumo. Confirmar com o autor e
  remover.
- **Agradecimentos:** incluir a Sanepar (dados) e laboratórios ou
  grupos? Aguardando o autor.
- **Nota de rodapé do autor:** o Word termina com "CAMPUS"; a linha do
  orientador termina com ", XXXX XX XXXX". Saber o que vai ali.
- **E-mail** no cabeçalho: confirmar se é o institucional.
- **Tamanho:** 21 páginas contra ~12. Próximos cortes possíveis: reduzir
  o diagrama da arquitetura a meia página, tirar a figura da DCRNN
  (só citar), condensar a tabela de configuração.
- **Notebook explicativo** ainda usa o nome SpectralTFNet.
