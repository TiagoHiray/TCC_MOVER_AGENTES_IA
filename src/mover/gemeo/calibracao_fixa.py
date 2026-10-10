"""Calibração fixa do celular no veículo, lida de uma volta de calibração já tratada.

Fluxo: grava-se uma volta com o celular na posição definitiva, roda-se a Etapa 1
(`python -m mover.tratamento.tratar_dados`) e o estimador ao vivo reaproveita o
`relatorio_tratamento.json` (montagem, sinal, ganhos, defasagem do GPS e referência geográfica)
e, se existir, o `alinhamento.json` (plano local -> plano do mapa).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mover.tratamento.geo import ReferenciaGeo


@dataclass(frozen=True)
class TransformacaoMapa:
    """p_mapa = escala * R(rotacao) * (leste, norte) + (tx, ty); mesma convenção do alinhamento.py."""

    escala: float
    rotacao_rad: float
    tx: float
    ty: float

    def aplicar(self, leste: float, norte: float) -> tuple[float, float]:
        c, s = math.cos(self.rotacao_rad), math.sin(self.rotacao_rad)
        return (self.escala * (c * leste - s * norte) + self.tx, self.escala * (s * leste + c * norte) + self.ty)


@dataclass(frozen=True)
class CalibracaoFixa:
    sinal: float
    R: np.ndarray  # linhas: frente, esquerda, cima (no referencial do aparelho)
    ganho_long: float
    ganho_lat: float
    defasagem_s: float  # t_real = t_gps - defasagem_s
    ref: ReferenciaGeo
    mapa: TransformacaoMapa | None = None
    origem: str = ""

    @classmethod
    def carregar(cls, relatorio: Path, alinhamento: Path | None = None) -> "CalibracaoFixa":
        with open(relatorio, encoding="utf-8") as f:
            r = json.load(f)
        mont = r["montagem"]
        g_long, g_lat = mont.get("ganhos_aplicados") or [1.0, 1.0]
        ref = r["referencia_geo"]
        mapa = None
        if alinhamento is not None and Path(alinhamento).exists():
            with open(alinhamento, encoding="utf-8") as f:
                t = json.load(f)["transformacao"]
            mapa = TransformacaoMapa(float(t["escala"]), math.radians(float(t["rotacao_graus"])),
                                     float(t["tx_m"]), float(t["ty_m"]))
        return cls(
            sinal=float(r["sinais"]["sinal_aceleracao"]),
            R=np.asarray(mont["R_aparelho_para_veiculo"], float),
            ganho_long=float(g_long), ganho_lat=float(g_lat),
            defasagem_s=float(r["defasagem_gps"]["segundos"]),
            ref=ReferenciaGeo(float(ref["lat0"]), float(ref["lon0"]), str(ref.get("origem", ""))),
            mapa=mapa,
            origem=f"{r['entrada'].get('dispositivo', '?')} ({r['entrada'].get('plataforma', '?')}), "
                   f"tratado em {r.get('gerado_em', '?')}",
        )
