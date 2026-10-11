"""Simulador substituto (modo --sem-carla): o caminhão anda sobre a rota sem o CARLA.

A posição é sempre a da linha central da faixa (não há dinâmica lateral) e a velocidade vem do
controle do piloto (estilo Traffic Manager) e da dinâmica longitudinal do caminhão. Serve para
rodar e testar o laço inteiro (plano, agentes, intervenções, painel e métricas) numa máquina sem
o simulador. Os números que valem para o TCC são os do CARLA.

A telemetria de cada tick usa as mesmas chaves que o lado CARLA (carla_tm.py) devolve.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any

from mover.autonomo.rota import Rota
from mover.autonomo.veiculo import DinamicaCaminhao, ParametrosCaminhao, ParametrosPiloto, PilotoTM

DISTANCIA_EIXOS_M = 4.5   # para estimar o volante a partir da curvatura
VOLANTE_MAX_RAD = 0.6


class SimuladorSubstituto:
    nome = "substituto"

    def __init__(self, rota: Rota, caminhao: ParametrosCaminhao, piloto: ParametrosPiloto, dt: float = 0.05,
                 hora_inicio: datetime | None = None):
        self.rota = rota
        self.caminhao = caminhao
        self.p_piloto = piloto
        self.dt = dt
        self.hora_inicio = hora_inicio or datetime.now().astimezone()
        self.reiniciar()

    @property
    def massa_kg(self) -> float:
        return self.caminhao.massa_kg

    def iniciar(self) -> None:
        self.reiniciar()

    def reiniciar(self, d: float = 0.0) -> None:
        self.dinamica = DinamicaCaminhao(self.caminhao)
        self.piloto = PilotoTM(self.p_piloto, self.caminhao)
        self.frame = 0
        self.t = 0.0
        self.d = float(d)
        self.odom = 0.0

    def passo(self, v_desejada: float) -> dict[str, Any]:
        """Um tick: aplica a velocidade desejada (m/s) e devolve a telemetria."""
        v0 = self.dinamica.v
        acelerador, freio = self.piloto.comando(v_desejada, v0, self.dt)
        saida = self.dinamica.passo(acelerador, freio, self.dt)
        avanco = 0.5 * (v0 + saida["v"]) * self.dt
        self.d += avanco
        self.odom += avanco
        self.t += self.dt
        self.frame += 1
        ponto = self.rota.ponto(self.d)
        v = saida["v"]
        return {
            "frame": self.frame,
            "sim_time": round(self.t, 4),
            "hora_local": (self.hora_inicio + timedelta(seconds=self.t)).isoformat(timespec="milliseconds"),
            "x": ponto.x,
            "y": ponto.y,
            "z": 0.0,
            "rumo_graus": math.degrees(ponto.rumo),
            "speed_mps": v,
            "acc_long": saida["a"],
            "acc_lat": v * v * ponto.curvatura,
            "yaw_rate_dps": -math.degrees(v * ponto.curvatura),  # convenção do CARLA: positivo à direita
            "throttle": acelerador,
            "brake": freio,
            "steer": max(-1.0, min(1.0, -math.atan(DISTANCIA_EIXOS_M * ponto.curvatura) / VOLANTE_MAX_RAD)),
            "gear": 0,
            "rpm": 0.0,
            "d": self.d,
            "s": ponto.s,
            "volta": self.rota.volta_da_distancia(self.d),
            "odom_m": self.odom,
            "rampa": 0.0,
            "v_desejada": v_desejada,
            "fonte": self.nome,
        }

    def encerrar(self) -> None:
        pass
