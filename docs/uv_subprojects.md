# UV Sub-projects

Este projeto usa **UV workspaces**, o que permite criar sub-projetos com dependências isoladas dentro do mesmo repositório. Isso é útil quando você precisa, por exemplo, carregar um modelo legado que depende de versões específicas de bibliotecas que conflitam com o projeto principal.

## Quando usar sub-projects?

- Modelo antigo que requer versões específicas (ex: `scikit-learn==0.24`, `xgboost==1.5`)
- Serviço auxiliar com stack diferente
- Experimentação isolada sem afetar o ambiente principal

## Como criar um sub-project

```bash
# Dentro da raiz do projeto, crie o sub-project
uv init models/modelo_legado_v1

# Entre no sub-project e adicione as dependências específicas
cd models/modelo_legado_v1
uv add scikit-learn==0.24.2 xgboost==1.5.0
```

A estrutura resultante fica assim:

```text
.
├── pyproject.toml                  # Projeto principal (workspace root)
├── models/
│   └── modelo_legado_v1/           # Sub-project com deps isoladas
│       ├── pyproject.toml          # Dependências do modelo legado
│       └── src/
│           └── ...
├── src/                            # Código do projeto principal
└── ...
```

## Executando código dentro de um sub-project

```bash
# Rodar um script com as dependências do sub-project
uv run --package modelo_legado_v1 python predict.py

# Ou entre no diretório do sub-project
cd models/modelo_legado_v1
uv run python predict.py
```

## Referência do workspace no pyproject.toml

O UV detecta automaticamente sub-projetos. Para configuração explícita, adicione no `pyproject.toml` raiz:

```toml
[tool.uv.workspace]
members = ["models/*"]
```

> **Dica:** Cada sub-project tem seu próprio `pyproject.toml` e `.venv`, garantindo isolamento total de dependências.
