"""Eventos sintéticos de teste (--injetar-eventos): frenagens e arrancadas bruscas.

A volta gravada no campus não tem nenhum evento de condução brusca (todos os picos de jerk
coincidem com lombadas). Para testar a detecção, este módulo soma pulsos de aceleração
longitudinal à telemetria tratada, numa cópia: os dados reais nunca são alterados.

Cada pulso sobe até a intensidade com rampa de cosseno, se mantém e volta a zero. Ele passa
pelo mesmo passa-baixa de 1,5 Hz usado no tratamento antes de virar jerk. São alterados
acc_long, jerk_long, imu_acc_x, acc_x/acc_y (referencial do mundo do CARLA), os comandos
estimados e a manobra; as linhas afetadas recebem fonte = sintetico. A velocidade e a pose
não mudam, porque o replay segue a trajetória gravada.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from mover.tratamento.sinais import passa_baixa

log = logging.getLogger("mover.agentes")

SINAL_DO_TIPO = {"frenagem_brusca": -1.0, "arrancada_brusca": +1.0}


def pulso(t: np.ndarray, inicio: float, duracao: float, rampa: float) -> np.ndarray:
    """Pulso unitário com rampas de cosseno de `rampa` s na subida e na descida."""
    rampa = min(rampa, duracao / 2)
    x = np.zeros_like(t, dtype=float)
    subida = (t >= inicio) & (t < inicio + rampa)
    x[subida] = 0.5 * (1 - np.cos(np.pi * (t[subida] - inicio) / rampa))
    x[(t >= inicio + rampa) & (t < inicio + duracao - rampa)] = 1.0
    descida = (t >= inicio + duracao - rampa) & (t < inicio + duracao)
    x[descida] = 0.5 * (1 + np.cos(np.pi * (t[descida] - (inicio + duracao - rampa)) / rampa))
    return x


def injetar(telemetria: pd.DataFrame, eventos: list[dict[str, Any]], veiculo: dict[str, Any],
            taxa_hz: float = 20.0, corte_jerk_hz: float = 1.5, acel_limiar_manobra: float = 0.5) -> pd.DataFrame:
    """Devolve uma cópia da telemetria com os eventos sintéticos somados."""
    df = telemetria.copy()
    t = df["sim_time"].to_numpy(dtype=float)
    yaw = np.radians(df["yaw"].to_numpy(dtype=float))
    acel_tracao = float(veiculo.get("acel_max_tracao_mps2", 2.0))
    desacel_freio = float(veiculo.get("desacel_max_freio_mps2", 6.0))

    for ev in eventos:
        if ev["tipo"] not in SINAL_DO_TIPO:
            raise ValueError(f"tipo de evento sintético desconhecido: {ev['tipo']!r}")
        amplitude = SINAL_DO_TIPO[ev["tipo"]] * float(ev["intensidade_mps2"])
        bruto = amplitude * pulso(t, float(ev["tempo_s"]), float(ev["duracao_s"]), float(ev.get("rampa_s", 0.4)))
        a = passa_baixa(bruto, taxa_hz, corte_jerk_hz)
        jerk = np.gradient(a, 1.0 / taxa_hz)
        afetadas = np.abs(a) > 0.05 * abs(amplitude)

        df["acc_long"] += a
        df["jerk_long"] += jerk
        df["imu_acc_x"] += a
        df["acc_x"] += a * np.cos(yaw)  # frente do veículo no mundo do CARLA = (cos yaw, sin yaw)
        df["acc_y"] += a * np.sin(yaw)

        forte = afetadas & (np.abs(a) >= acel_limiar_manobra)
        if amplitude < 0:
            df.loc[forte, "brake_est"] = np.clip(df.loc[forte, "brake_est"] - a[forte] / desacel_freio, 0, 1)
            df.loc[forte, "throttle_est"] = 0.0
            df.loc[forte, "manobra"] = "frear"
        else:
            df.loc[forte, "throttle_est"] = np.clip(df.loc[forte, "throttle_est"] + a[forte] / acel_tracao, 0, 1)
            df.loc[forte, "brake_est"] = 0.0
            df.loc[forte, "manobra"] = "acelerar"
        df.loc[afetadas, "fonte"] = "sintetico"
        log.info("Evento sintético: %s de %.1f m/s² em %.1f s (%d linhas, jerk de %.1f a %.1f m/s³)",
                 ev["tipo"], abs(amplitude), ev["tempo_s"], int(afetadas.sum()), jerk.min(), jerk.max())

    # Colunas de compatibilidade com o dashboard e a POC (cópias das estimativas)
    df["throttle"], df["brake"] = df["throttle_est"], df["brake_est"]
    df["cmd_throttle"], df["cmd_brake"], df["cmd_manobra"] = df["throttle_est"], df["brake_est"], df["manobra"]
    return df
