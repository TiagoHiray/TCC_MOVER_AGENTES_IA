"""Dinâmica longitudinal do caminhão e o controle de velocidade no estilo do Traffic Manager.

Usados em dois lugares:
- no simulador substituto (modo --sem-carla), como a "física" do caminhão;
- na previsão da camada agêntica, como o modelo interno do gêmeo digital para antecipar o que o
  piloto vai fazer nos próximos segundos (também quando o caminhão de verdade roda no CARLA).

O Traffic Manager do CARLA controla a velocidade com um PID sobre o erro de velocidade e satura o
acelerador em 0,85 e o freio em 0,7. Com erro grande (uma mudança em degrau da velocidade
desejada), o comando satura: é daí que vêm as acelerações e frenagens bruscas do plano bruto.
Com uma velocidade desejada que muda devagar, o mesmo controle acompanha sem trancos.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any


@dataclass
class ParametrosCaminhao:
    massa_kg: float = 9000.0
    area_frontal_m2: float = 7.5
    cd: float = 0.65
    cr: float = 0.008                    # resistência ao rolamento
    potencia_max_kw: float = 180.0
    forca_tracao_max_n: float = 18000.0  # limita a aceleração em baixa velocidade
    desacel_freio_max_mps2: float = 8.0  # freio no fundo (o piloto usa no máximo 0,7)
    tau_motor_s: float = 0.30            # atraso do torque do motor
    tau_freio_s: float = 0.15            # atraso da pressão de freio
    densidade_ar: float = 1.2
    g: float = 9.81

    @classmethod
    def de_dict(cls, dados: dict[str, Any] | None) -> "ParametrosCaminhao":
        nomes = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in (dados or {}).items() if k in nomes})

    def resistencia_n(self, v: float, rampa: float = 0.0) -> float:
        """Arrasto + rolamento + rampa (rampa = tangente da inclinação, positiva em subida)."""
        aero = 0.5 * self.densidade_ar * self.cd * self.area_frontal_m2 * v * v
        rolamento = self.massa_kg * self.g * self.cr if v > 1e-3 else 0.0
        return aero + rolamento + self.massa_kg * self.g * rampa

    def tracao_max_n(self, v: float) -> float:
        return min(self.forca_tracao_max_n, self.potencia_max_kw * 1000.0 / max(v, 0.5))


@dataclass
class ParametrosPiloto:
    """Controle do piloto automático (emulação do PID longitudinal do Traffic Manager)."""

    kp: float = 2.5             # aceleração pedida (m/s²) por m/s de erro
    ki: float = 0.15
    acelerador_max: float = 0.85
    freio_max: float = 0.70

    @classmethod
    def de_dict(cls, dados: dict[str, Any] | None) -> "ParametrosPiloto":
        nomes = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in (dados or {}).items() if k in nomes})


class PilotoTM:
    """Transforma a velocidade desejada em acelerador e freio, como o Traffic Manager."""

    def __init__(self, p: ParametrosPiloto, caminhao: ParametrosCaminhao):
        self.p = p
        self.c = caminhao
        self.integral = 0.0

    def reiniciar(self) -> None:
        self.integral = 0.0

    def comando(self, v_desejada: float, v: float, dt: float, rampa: float = 0.0) -> tuple[float, float]:
        erro = v_desejada - v
        a_pedida = self.p.kp * erro + self.p.ki * self.integral
        forca = self.c.massa_kg * a_pedida + self.c.resistencia_n(v, rampa)
        if forca >= 0.0:
            acelerador = forca / self.c.tracao_max_n(v)
            saturado = acelerador >= self.p.acelerador_max
            acelerador, freio = min(acelerador, self.p.acelerador_max), 0.0
        else:
            freio = -forca / (self.c.massa_kg * self.c.desacel_freio_max_mps2)
            saturado = freio >= self.p.freio_max
            acelerador, freio = 0.0, min(freio, self.p.freio_max)
        if v_desejada <= 0.05 and v <= 0.05:  # parado: segura no freio e zera a memória
            self.integral = 0.0
            return 0.0, self.p.freio_max
        if not saturado:  # anti-windup: só integra fora da saturação
            self.integral = max(-5.0, min(5.0, self.integral + erro * dt))
        return acelerador, freio


class DinamicaCaminhao:
    """Integra a velocidade com o atraso do motor e do freio (Euler, passo do simulador)."""

    def __init__(self, p: ParametrosCaminhao, v0: float = 0.0, a0: float = 0.0, rampa: float = 0.0):
        self.p = p
        self.v = max(0.0, float(v0))
        liquida = p.massa_kg * a0 + p.resistencia_n(self.v, rampa)  # forças coerentes com a0
        self.f_motor = max(0.0, liquida)
        self.f_freio = max(0.0, -liquida)
        self.a = float(a0)

    def passo(self, acelerador: float, freio: float, dt: float, rampa: float = 0.0) -> dict[str, float]:
        p = self.p
        alvo_motor = max(0.0, min(1.0, acelerador)) * p.tracao_max_n(self.v)
        alvo_freio = max(0.0, min(1.0, freio)) * p.massa_kg * p.desacel_freio_max_mps2
        self.f_motor += (alvo_motor - self.f_motor) * min(1.0, dt / p.tau_motor_s)
        self.f_freio += (alvo_freio - self.f_freio) * min(1.0, dt / p.tau_freio_s)
        resistencia = p.resistencia_n(self.v, rampa)
        if self.v <= 1e-3:
            # parado: o freio e o atrito seguram até a tração vencer o rolamento e a rampa
            rolamento = p.massa_kg * p.g * p.cr
            empurra = self.f_motor - p.massa_kg * p.g * rampa
            if empurra <= rolamento or self.f_freio >= self.f_motor:
                self.v, self.a = 0.0, 0.0
                return self._saida(acelerador, freio)
        a = (self.f_motor - self.f_freio - resistencia) / p.massa_kg
        v_nova = self.v + a * dt
        if v_nova < 0.0:  # o freio para o caminhão; não anda de ré
            a, v_nova = -self.v / dt, 0.0
        self.v, self.a = v_nova, a
        return self._saida(acelerador, freio)

    def _saida(self, acelerador: float, freio: float) -> dict[str, float]:
        return {"v": self.v, "a": self.a, "acelerador": acelerador, "freio": freio,
                "f_motor_n": self.f_motor, "f_freio_n": self.f_freio,
                "p_motor_kw": self.f_motor * self.v / 1000.0, "p_freio_kw": self.f_freio * self.v / 1000.0}


def tempo_para_frear(v0: float, v1: float, desacel: float) -> float:
    return max(0.0, (v0 - v1) / max(desacel, 1e-6))


def distancia_para_frear(v0: float, v1: float, desacel: float) -> float:
    return max(0.0, (v0 * v0 - v1 * v1) / (2.0 * max(desacel, 1e-6)))


def kmh(v_mps: float) -> float:
    return v_mps * 3.6


def mps(v_kmh: float) -> float:
    return v_kmh / 3.6


def rampa_de_pitch(pitch_graus: float) -> float:
    return math.tan(math.radians(pitch_graus))
