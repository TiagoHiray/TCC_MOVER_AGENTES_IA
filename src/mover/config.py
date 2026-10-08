"""Configuração compartilhada: raiz do repositório e leitura de arquivos YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# src/mover/config.py -> parents[0] = mover, [1] = src, [2] = raiz do repositório
RAIZ = Path(__file__).resolve().parents[2]


def caminho(relativo: str | Path) -> Path:
    """Resolve um caminho relativo à raiz do repositório (absolutos ficam como estão)."""
    p = Path(relativo)
    return p if p.is_absolute() else RAIZ / p


def carregar_yaml(relativo: str | Path) -> dict[str, Any]:
    """Lê um arquivo YAML em UTF-8 e devolve um dicionário (vazio se o arquivo estiver vazio)."""
    with caminho(relativo).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
