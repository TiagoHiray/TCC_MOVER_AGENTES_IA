"""Rota da volta no campus: linha central da faixa, fechada e parametrizada pela distância.

Coordenadas no referencial do OpenDRIVE (x para leste, y para o norte do mapa). O CARLA inverte
o eixo y: (x, y)_CARLA = (x, -y)_OpenDRIVE e yaw_CARLA = -rumo (em graus).

Duas medidas de distância:
- s: posição dentro da volta, em [0, comprimento);
- d: distância percorrida desde a largada, que cresce sem voltar a zero (d = volta * L + s).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import uniform_filter1d


@dataclass(frozen=True)
class PontoRota:
    s: float
    x: float
    y: float
    rumo: float        # rad, anti-horário a partir de +x (OpenDRIVE)
    curvatura: float   # 1/m, positiva em curva à esquerda


class Rota:
    """Laço fechado reamostrado a cada `passo_m` metros, com rumo e curvatura suavizados."""

    def __init__(self, x: Any, y: Any, passo_m: float = 1.0, suavizar_m: float = 6.0):
        pontos = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
        if len(pontos) < 4:
            raise ValueError("a rota precisa de pelo menos 4 pontos")
        if np.hypot(*(pontos[0] - pontos[-1])) < 1e-6:  # o fechamento é implícito
            pontos = pontos[:-1]
        segmentos = np.hypot(*np.diff(np.vstack([pontos, pontos[:1]]), axis=0).T)
        manter = np.concatenate([[True], segmentos[:-1] > 1e-6])
        pontos = pontos[manter]
        fechado = np.vstack([pontos, pontos[:1]])
        s_bruto = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(fechado, axis=0).T))])
        self.comprimento = float(s_bruto[-1])
        n = max(8, int(round(self.comprimento / passo_m)))
        self.passo = self.comprimento / n
        self.s = np.arange(n) * self.passo
        self.x = np.interp(self.s, s_bruto, fechado[:, 0])
        self.y = np.interp(self.s, s_bruto, fechado[:, 1])
        janela = max(1, int(round(suavizar_m / self.passo)))
        if janela > 1:  # média móvel circular: tira o serrilhado dos pontos de origem
            self.x = uniform_filter1d(self.x, janela, mode="wrap")
            self.y = uniform_filter1d(self.y, janela, mode="wrap")
        dx = (np.roll(self.x, -1) - np.roll(self.x, 1)) / 2
        dy = (np.roll(self.y, -1) - np.roll(self.y, 1)) / 2
        self.rumo = np.arctan2(dy, dx)
        drumo = (np.roll(self.rumo, -1) - np.roll(self.rumo, 1) + math.pi) % (2 * math.pi) - math.pi
        curvatura = drumo / (2 * self.passo)
        self.curvatura = uniform_filter1d(curvatura, max(1, janela), mode="wrap")
        self._xy = np.column_stack([self.x, self.y])

    # ------------------------------------------------------------------ consultas
    @property
    def n(self) -> int:
        return len(self.s)

    def s_da_distancia(self, d: float) -> float:
        return float(d % self.comprimento)

    def volta_da_distancia(self, d: float) -> int:
        return int(math.floor(d / self.comprimento + 1e-9))

    def ponto(self, s: float) -> PontoRota:
        """Ponto interpolado em s (aceita qualquer s: vale o resto da divisão pela volta)."""
        s = float(s) % self.comprimento
        i = int(s // self.passo) % self.n
        j = (i + 1) % self.n
        f = (s - self.s[i]) / self.passo
        drumo = (self.rumo[j] - self.rumo[i] + math.pi) % (2 * math.pi) - math.pi
        return PontoRota(s=s, x=float(self.x[i] + f * (self.x[j] - self.x[i])),
                         y=float(self.y[i] + f * (self.y[j] - self.y[i])),
                         rumo=float(self.rumo[i] + f * drumo),
                         curvatura=float(self.curvatura[i] + f * (self.curvatura[j] - self.curvatura[i])))

    def projetar(self, x: float, y: float, s_dica: float | None = None, janela_m: float = 40.0) -> tuple[float, float]:
        """(s, desvio lateral) do ponto mais próximo da rota; com `s_dica`, procura só perto dele.

        O desvio é positivo à esquerda do sentido da rota.
        """
        if s_dica is None:
            candidatos = np.arange(self.n)
        else:
            meia = int(math.ceil(janela_m / self.passo))
            centro = int(round((s_dica % self.comprimento) / self.passo))
            candidatos = np.arange(centro - meia, centro + meia + 1) % self.n
        d2 = (self.x[candidatos] - x) ** 2 + (self.y[candidatos] - y) ** 2
        i = int(candidatos[int(np.argmin(d2))])
        melhor_s, melhor_dist, melhor_lado = float(self.s[i]), math.inf, 0.0
        for a in ((i - 1) % self.n, i):  # projeta nos dois segmentos vizinhos
            b = (a + 1) % self.n
            ax, ay, bx, by = self.x[a], self.y[a], self.x[b], self.y[b]
            vx, vy = bx - ax, by - ay
            comp2 = vx * vx + vy * vy
            u = 0.0 if comp2 < 1e-12 else min(1.0, max(0.0, ((x - ax) * vx + (y - ay) * vy) / comp2))
            px, py = ax + u * vx, ay + u * vy
            dist = math.hypot(x - px, y - py)
            if dist < melhor_dist:
                lado = vx * (y - ay) - vy * (x - ax)
                melhor_s = float((self.s[a] + u * self.passo) % self.comprimento)
                melhor_dist, melhor_lado = dist, math.copysign(dist, lado)
        return melhor_s, melhor_lado

    def avancar_distancia(self, d_anterior: float, s_novo: float) -> float:
        """Nova distância percorrida a partir do s projetado (trata a passagem pela largada)."""
        s_ant = d_anterior % self.comprimento
        delta = (s_novo - s_ant + self.comprimento / 2) % self.comprimento - self.comprimento / 2
        return d_anterior + delta

    def amostrar(self, s_ini: float, distancia: float, passo: float) -> list[PontoRota]:
        n = max(1, int(distancia // passo) + 1)
        return [self.ponto(s_ini + k * passo) for k in range(n)]

    def limites(self) -> tuple[float, float, float, float]:
        return float(self.x.min()), float(self.y.min()), float(self.x.max()), float(self.y.max())

    # ------------------------------------------------------------------ serialização
    def para_dict(self, passo_m: float = 5.0) -> dict[str, Any]:
        pontos = self.amostrar(0.0, self.comprimento - passo_m * 0.5, passo_m)
        return {"passo_m": passo_m, "comprimento_m": round(self.comprimento, 2),
                "x": [round(p.x, 2) for p in pontos], "y": [round(p.y, 2) for p in pontos]}

    @classmethod
    def de_dict(cls, dados: dict[str, Any], passo_m: float = 1.0) -> "Rota":
        return cls(dados["x"], dados["y"], passo_m=passo_m, suavizar_m=float(dados.get("suavizar_m", 4.0)))


def para_carla(x: float, y: float, rumo: float) -> tuple[float, float, float]:
    """OpenDRIVE (y para o norte, rumo anti-horário em rad) -> CARLA (x, y, yaw em graus)."""
    return x, -y, -math.degrees(rumo)


def de_carla(x: float, y: float, yaw_graus: float) -> tuple[float, float, float]:
    """CARLA -> OpenDRIVE (inversa de para_carla)."""
    return x, -y, -math.radians(yaw_graus)
