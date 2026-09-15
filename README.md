<a name="readme-top"></a>

<div align="center">
  <h1 align="center">mimo_sys</h1>
  <p align="center">
    https://github.com/watasabi/mimo_sys.git
    <br />
    <br />
    <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/Status-Development-yellow?style=for-the-badge" alt="Status">
  </p>
</div>


<details>
  <summary>Tabela de Conteúdos</summary>
  <ol>
    <li><a href="#sobre-o-projeto">Sobre o Projeto</a>
      <ul>
        <li><a href="#documentacao">Documentação</a></li>
        <li><a href="#principais-stakeholders">Principais Stakeholders</a></li>
      </ul>
    </li>
    <li><a href="#organizacao-e-estrutura">Organização e Estrutura</a></li>
    <li><a href="#filosofia-de-codigo">Filosofia de Código</a></li>
    <li><a href="#configuracao-de-ambiente">Configuração de Ambiente</a></li>
    <li><a href="#convencao-de-commits">Convenção de Commits</a></li>
    <li><a href="#autor">Autor</a></li>
  </ol>
</details>

---

## Sobre o Projeto

Reprodução e exploração do artigo [Ferrari, Leandro & Coelho (2025) — *Multi-objective metaheuristics applied for the multivariable system identification*](docs/pappers/Multi-objective%20metaheuristics%20applied%20for%20the%20multivariable%20system%20identification.pdf), que identifica um sistema MIMO (Multiple-Input Multiple-Output) do tipo ARMAX referente a um reservatório de distribuição de água (São José), comparando metaheurísticas mono-objetivo (GA, GWO, CS) e suas versões multiobjetivo (NSGA-II, MOGWO, MOCS).

Os dados brutos (`data/raw/`) trazem dois sistemas reais de reservatório:

- **São José** (`mimo_data.xlsx`) — sistema MIMO usado no artigo: 3 entradas (frequência das bombas de distribuição) e 4 saídas (vazão de entrada, vazão de distribuição, nível e pressão do reservatório).
- **Tarumã** (`miso_data.xls`) — sistema MISO usado no trabalho anterior citado pelo artigo, mantido apenas como referência.

### Documentação

| Recurso | Link |
|---------|------|
| Artigo de referência | [docs/pappers/Multi-objective metaheuristics applied for the multivariable system identification.pdf](docs/pappers/Multi-objective%20metaheuristics%20applied%20for%20the%20multivariable%20system%20identification.pdf) |
| Notebook de exploração | [notebooks/eda/01_exploracao_reservatorio_mimo.ipynb](notebooks/eda/01_exploracao_reservatorio_mimo.ipynb) |
| GitHub Repo | https://github.com/watasabi/mimo_sys |

### Principais Stakeholders
* **Rodrigo Watanabe Pisaia** — autor

<p align="right">(<a href="#readme-top">voltar ao topo</a>)</p>

## Organização e Estrutura

Este projeto segue uma estrutura padronizada para garantir reprodutibilidade.

> **Nota sobre Convenção de Nomes:**
> Arquivos numerados (ex: `01_load_data.py`) indicam **ordem de execução** em pipelines ou análises.
> Código reutilizável (funções/classes) deve residir em `src/` ou `utils/` e ser importado.

```text
.
├── config/                 # Configurações e variáveis de ambiente
│   ├── .env                # Variáveis de ambiente (NÃO commitar!)
│   └── .env.example        # Template com as variáveis necessárias
│
├── data/                   # Dados do projeto (Geralmente ignorados pelo Git)
│   ├── external/           # Dados de fontes terceiras
│   ├── interim/            # Dados transformados intermediários
│   ├── processed/          # Dados finais prontos para modelagem
│   └── raw/                # Dados originais imutáveis
│
├── docs/                   # Documentação do projeto (Markdown)
│
├── notebooks/              # Jupyter Notebooks
│   ├── eda/                # Análise exploratória de dados
│   ├── get_data/           # Extração de dados (usa queries/)
│   ├── processing/         # Transformação e feature engineering
│   ├── training/           # Treinamento de modelos
│   ├── modeling/           # Experimentos e avaliação de modelos
│   └── qa/                 # Validação e quality assurance
│
├── queries/                # Queries SQL (.txt/.sql) para Databricks
│   └── get_data/           # Queries usadas por notebooks/get_data/
│
├── models/                 # Artefatos de modelos (ignorados pelo Git)
│
├── reports/                # Relatórios gerados, html, pdf
│   └── figures/            # Gráficos e imagens geradas pelos códigos
│
├── scripts/                # Scripts utilitários (bash): setup, lint, etc.
│
├── src/                    # Código Fonte Reutilizável (Library do projeto)
│   └── __init__.py         # Funções de engenharia de features
│
├── .gitignore              # Arquivos a serem ignorados pelo git
├── LICENSE                 # Licença do projeto
├── pyproject.toml          # Dependências e config (UV workspace)
└── README.md               # Documentação principal
```

<p align="right">(<a href="#readme-top">voltar ao topo</a>)</p>

## Filosofia de Código

Este é um projeto de ciência de dados: a maior parte do código é exploratória ou analítica, e é lida com muito mais frequência do que reutilizada. Por isso, priorize **código simples e fácil de ler**, mesmo que isso signifique sacrificar parte da modularidade — um script linear e claro é melhor do que uma abstração prematura. Extraia funções, classes ou padrões (registry, base classes) apenas quando o reuso já é real, não como preparação para um reuso hipotético.

<p align="right">(<a href="#readme-top">voltar ao topo</a>)</p>

## Configuração de Ambiente

As variáveis de ambiente do projeto ficam em `config/.env`. Para configurar:

```bash
cp config/.env.example config/.env
```

Edite o arquivo `config/.env` com as credenciais necessárias:

| Variável | Descrição |
|----------|-----------|
| `DATABRICKS_TOKEN` | Token de acesso ao Databricks (DAPI) |
| `DATABRICKS_HOSTNAME` | Host do workspace Databricks |
| `DATABRICKS_HTTP_PATH` | HTTP Databricks Warehouse |

> **IMPORTANTE:** O arquivo `config/.env` está no `.gitignore` e **nunca** deve ser commitado. Use `config/.env.example` como referência.

As dependências são gerenciadas via **UV**. As bibliotecas de visualização (`matplotlib`, `seaborn`, `plotly`) ficam no grupo opcional `plot` para manter o ambiente padrão enxuto:

```bash
uv sync                # dependências principais
uv sync --group plot   # inclui também as libs de visualização
uv run --group plot jupyter lab
```

<p align="right">(<a href="#readme-top">voltar ao topo</a>)</p>

## Convenção de Commits

Este projeto segue o padrão **Conventional Commits**. Todas as mensagens de commit devem seguir o formato:

```
<tipo>(<escopo opcional>): <descrição>
```

### Tipos permitidos

| Tipo | Descrição |
|------|-----------|
| `feat` | Nova funcionalidade |
| `fix` | Correção de bug |
| `docs` | Alterações na documentação |
| `style` | Formatação (sem alteração de lógica) |
| `refactor` | Refatoração de código |
| `perf` | Melhoria de performance |
| `test` | Adição ou correção de testes |
| `chore` | Tarefas de manutenção |
| `infra` | Mudanças de infraestrutura |
| `imp` | Melhorias gerais |
| `breaking` | Mudança com quebra de compatibilidade |


### Exemplos

```bash
git commit -m "feat: adiciona modelo de classificação"
git commit -m "fix(pipeline): corrige leitura de dados raw"
git commit -m "docs: atualiza README com instruções de deploy"
git commit -m "refactor(src): simplifica feature engineering"
```

<p align="right">(<a href="#readme-top">voltar ao topo</a>)</p>

## Autor

| Nome | Email |
|------|-------|
| **Rodrigo Watanabe Pisaia** | rodrigo.watanabe0107@gmail.com |

<p align="right">(<a href="#readme-top">voltar ao topo</a>)</p>

## UV Sub-projects

Para usar UV sub-projects (dependências isoladas por modelo, ex: um modelo legado com versões conflitantes de libs), veja [docs/uv_subprojects.md](docs/uv_subprojects.md).

<p align="right">(<a href="#readme-top">voltar ao topo</a>)</p>
