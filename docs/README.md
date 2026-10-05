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

Documentação de código fica junto do código: `src/mimo_sys/architectures/README.md` descreve cada arquitetura, seus parâmetros e armadilhas.
- `uv_subprojects.md` — como usar UV sub-projects para isolar dependências

Não há uma estrutura obrigatória: adicione arquivos `.md` conforme o projeto precisar, mantendo nomes curtos e em `snake_case`.
