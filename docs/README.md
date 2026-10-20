# Documentação

Esta pasta reúne a documentação viva do projeto, em Markdown (`.md`).

O `README.md` na raiz do projeto é a porta de entrada (visão geral, setup, convenções). Aqui em `docs/` fica documentação mais profunda ou de referência, que não precisa estar no primeiro contato com o projeto. Exemplos de conteúdo comum:

- `architecture.md` — decisões de arquitetura e desenho do pipeline
- `data_dictionary.md` — dicionário de dados (colunas, tipos, origem)
- `decisions.md` — registro de decisões técnicas (ADRs)
- `0_referencias-tcc.md` — ficha de leitura de cada referência do TCC (PDFs em `pappers/`)
- `2_split_artigo_vs_dataset_completo.md` — recorte do artigo vs. dataset completo e splits gerados
- `3_investigacao-arquiteturas.md` — triagem de arquiteturas para extração de equações e decisões tomadas
- `4_rotas-de-extracao.md` — as duas rotas de extração (equação e subgrafo) e qual arquitetura serve a cada uma
- `5_eda-series-temporais.md` — EDA de série temporal (tendência, frequência, identificação de sistemas)
- `6_tecnicas-de-treino.md` — toolkit de treino (augmentations, losses, RevIN, contrastivo, ensemble, tuning)
- `7_tecnica-cascata-fisica.md` — cascata física entre a previsão de `vazao_distribuicao` e o modelo de `nivel_res`, na busca por R² ≥0,85
- `8_arquitetura-spectral-tf-net.md` — SpectralTFNet (arestas = funções de transferência no domínio da frequência): desenho, resultados compilados e extração

Documentação de código fica junto do código: `src/mimo_sys/architectures/README.md` descreve cada arquitetura, seus parâmetros e armadilhas.
- `uv_subprojects.md` — como usar UV sub-projects para isolar dependências

Não há uma estrutura obrigatória: adicione arquivos `.md` conforme o projeto precisar, mantendo nomes curtos e em `snake_case`.
