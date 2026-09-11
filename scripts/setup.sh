#!/usr/bin/env bash
set -euo pipefail

echo "Configurando ambiente do projeto..."

if [ ! -f "config/.env" ]; then
    cp config/.env.example config/.env
    echo "Criado config/.env a partir de config/.env.example"
fi

uv sync

echo "Ambiente pronto. Ative com: source .venv/bin/activate"
