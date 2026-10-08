"""Coloca a pasta src/ no sys.path para os testes rodarem a partir da raiz do repositório."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
