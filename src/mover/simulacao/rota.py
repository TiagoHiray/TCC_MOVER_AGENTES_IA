"""Rota fixa da Fase X: o mesmo trajeto da volta gravada no campus, sempre a partir do mesmo ponto.

O Traffic Manager só anda no sentido das faixas do mapa, e as vias do mapa_final.xodr foram
desenhadas no sentido contrário ao da volta gravada. Por isso, nesta rota, quem dirige é um
controlador próprio: pure pursuit no volante e PI na velocidade, seguindo as poses da volta real
alinhadas ao mapa (alinhamento.py) com a velocidade da volta real vezes um fator sorteado por volta.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from mover.config import caminho


@dataclass
class Rota:
    """Pontos do trajeto no referencial do CARLA, comprimento de arco s (m) e velocidade de referência (m/s)."""

    x: np.ndarray
    y: np.ndarray
    s: np.ndarray
    v: np.ndarray
    duracao_s: float

    @property
    def comprimento_m(self) -> float:
        return float(self.s[-1])

    def ponto_em(self, s: float) -> tuple[float, float]:
        return float(np.interp(s, self.s, self.x)), float(np.interp(s, self.s, self.y))

    def velocidade_em(self, s: float) -> float:
        return float(np.interp(s, self.s, self.v))

    def pose_inicial(self, adiante_m: float = 3.0) -> tuple[float, float, float]:
        """(x, y, yaw) da partida, com o yaw apontando para o ponto `adiante_m` à frente."""
        x0, y0 = float(self.x[0]), float(self.y[0])
        x1, y1 = self.ponto_em(min(adiante_m, self.comprimento_m))
        return x0, y0, math.degrees(math.atan2(y1 - y0, x1 - x0))


def montar_rota(x: np.ndarray, y: np.ndarray, v: np.ndarray, duracao_s: float, vel_min_mps: float = 2.0,
                passo_min_m: float = 0.2) -> Rota:
    """Rota a partir de poses a 20 Hz: descarta pontos parados e põe um piso na velocidade (a rota não para)."""
    x, y, v = (np.asarray(a, float) for a in (x, y, v))
    manter, ultimo = [0], (x[0], y[0])
    for i in range(1, len(x)):
        if math.hypot(x[i] - ultimo[0], y[i] - ultimo[1]) >= passo_min_m:
            manter.append(i)
            ultimo = (x[i], y[i])
    idx = np.asarray(manter)
    x, y = x[idx], y[idx]
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    return Rota(x, y, s, np.maximum(v[idx], vel_min_mps), float(duracao_s))


def rota_da_volta_real(cfg_simulacao: dict[str, Any], vel_min_mps: float = 2.0) -> Rota:
    """Poses da volta gravada (Etapa 1) alinhadas ao mapa do config, com a velocidade medida."""
    from mover.simulacao.verificar_mapa import alinhar_volta

    telemetria = pd.read_csv(caminho(cfg_simulacao["telemetria"]["arquivo"]))
    _, alinhamento, _ = alinhar_volta(cfg_simulacao, telemetria=telemetria)
    poses = alinhamento.poses
    t = telemetria["sim_time"].to_numpy(float)
    return montar_rota(poses["x_carla"], poses["y_carla"], telemetria["speed_mps"], t[-1] - t[0], vel_min_mps)


class SeguidorDeRota:
    """Pure pursuit no volante e PI na velocidade; o CARLA dá a física, o seguidor só decide os comandos."""

    def __init__(self, rota: Rota, fator_velocidade: float = 1.0, entre_eixos_m: float = 4.0,
                 angulo_max_roda_graus: float = 35.0, max_desvio_m: float = 8.0, kp: float = 0.4, ki: float = 0.05):
        self.rota = rota
        self.fator = float(fator_velocidade)
        self.entre_eixos_m = float(entre_eixos_m)
        self.angulo_max_roda_graus = float(angulo_max_roda_graus)
        self.max_desvio_m = float(max_desvio_m)
        self.kp, self.ki = float(kp), float(ki)
        self.i = 0
        self.integral = 0.0
        self.desvio_m = 0.0
        self.v_alvo = 0.0
        self.fim: str | None = None

    def comando(self, x: float, y: float, yaw_graus: float, v: float, dt: float) -> tuple[float, float, float]:
        """(throttle, steer, brake) para o próximo passo; `fim` fica preenchido ao acabar a rota ou sair dela."""
        r = self.rota
        # busca só adiante: a volta termina perto da partida e o ponto mais próximo não pode pular para o início
        fim_busca = min(len(r.x), self.i + 200)
        distancias = np.hypot(r.x[self.i:fim_busca] - x, r.y[self.i:fim_busca] - y)
        self.i += int(np.argmin(distancias))
        self.desvio_m = float(distancias.min())
        s_aqui = float(r.s[self.i])
        if self.desvio_m > self.max_desvio_m:
            self.fim = "fora da rota"
        elif s_aqui >= r.comprimento_m - 1.0:
            self.fim = "fim da rota"
        if self.fim:
            return 0.0, 0.0, 1.0

        visada = float(np.clip(4.0 + 0.8 * v, 5.0, 15.0))
        alvo_x, alvo_y = r.ponto_em(s_aqui + visada)
        alfa = math.atan2(alvo_y - y, alvo_x - x) - math.radians(yaw_graus)
        alfa = math.atan2(math.sin(alfa), math.cos(alfa))
        angulo_roda = math.degrees(math.atan2(2.0 * self.entre_eixos_m * math.sin(alfa), visada))
        steer = float(np.clip(angulo_roda / self.angulo_max_roda_graus, -1.0, 1.0))

        self.v_alvo = self.fator * r.velocidade_em(s_aqui + v * 1.0)
        erro = self.v_alvo - v
        self.integral = float(np.clip(self.integral + erro * dt, -5.0, 5.0))
        u = self.kp * erro + self.ki * self.integral
        if u >= 0.0:
            return min(u, 1.0), steer, 0.0
        return 0.0, steer, min(-u, 1.0)
