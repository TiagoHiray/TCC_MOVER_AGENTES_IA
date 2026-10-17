"""Telemetria ao vivo do caminhão autônomo para a camada agêntica.

Cada tick do simulador (CARLA ou substituto) vira uma linha com as mesmas chaves. Aqui ficam:
- MedidorConsumo: consumo e custo acumulados tick a tick (definido em mover.autonomo.consumo e
  reexportado aqui: é o mesmo medidor do painel, da corrida e do benchmark);
- rotular_manobras: rótulo de manobra no vocabulário do coleta_carla (acelerar, frear, curva...);
- para_agentes: o trecho de telemetria no formato que os fatos e os especialistas esperam, com a
  aceleração suavizada em 1,5 Hz e o jerk (os mesmos limiares da camada agêntica).
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd

from mover.autonomo.consumo import MedidorConsumo
from mover.tratamento.sinais import moda_movel, passa_baixa

# colunas mandadas ao servidor a cada tick (o resto da linha fica só no CSV local)
COLUNAS_TELEMETRIA = ("frame", "sim_time", "hora_local", "x", "y", "z", "rumo_graus", "speed_mps", "acc_long", "acc_lat",
                      "yaw_rate_dps", "throttle", "brake", "steer", "gear", "rpm", "d", "s", "volta", "odom_m", "rampa",
                      "v_desejada", "v_alvo_kmh", "modo_tick", "limitado_por", "combustivel_l", "custo_r", "fonte")

__all__ = ["COLUNAS_TELEMETRIA", "MedidorConsumo", "para_agentes", "resumo_numerico", "rotular_manobras"]


def rotular_manobras(v: np.ndarray, a: np.ndarray, yaw_rate_dps: np.ndarray, taxa_hz: float) -> np.ndarray:
    rotulos = np.full(len(v), "cruzeiro", dtype=object)
    rotulos[a > 0.35] = "acelerar"
    rotulos[a < -0.35] = "frear"
    rotulos[yaw_rate_dps > 8.0] = "curva_dir"   # convenção do CARLA: positivo à direita
    rotulos[yaw_rate_dps < -8.0] = "curva_esq"
    rotulos[v < 0.5] = "parar"
    return moda_movel(rotulos.astype(str), max(1, int(round(taxa_hz))))


def para_agentes(linhas: Iterable[dict[str, Any]] | pd.DataFrame, taxa_hz: float = 20.0, fc_hz: float = 1.5) -> pd.DataFrame:
    """Linhas de telemetria ao vivo -> DataFrame com as colunas dos fatos e dos especialistas."""
    bruto = linhas if isinstance(linhas, pd.DataFrame) else pd.DataFrame(list(linhas))
    if bruto.empty:
        return pd.DataFrame()
    t = bruto["sim_time"].to_numpy(float)
    v = bruto["speed_mps"].to_numpy(float)
    a_suave = passa_baixa(bruto["acc_long"].to_numpy(float), taxa_hz, fc_hz, ordem=4)
    jerk = np.gradient(a_suave, 1.0 / taxa_hz) if len(a_suave) > 1 else np.zeros(len(a_suave))
    yaw = bruto["yaw_rate_dps"].to_numpy(float) if "yaw_rate_dps" in bruto else np.zeros(len(v))
    rampa = bruto["rampa"].to_numpy(float) if "rampa" in bruto else np.zeros(len(v))
    acc_vert = bruto["acc_vert"].to_numpy(float) if "acc_vert" in bruto else np.zeros(len(v))
    dados = pd.DataFrame({
        "sim_time": t,
        "hora_local": bruto["hora_local"].astype(str) if "hora_local" in bruto else "",
        "speed_mps": v,
        "speed_kmh": v * 3.6,
        "acc_long": a_suave,
        "acc_lat": bruto["acc_lat"].to_numpy(float) if "acc_lat" in bruto else np.zeros(len(v)),
        "acc_vert": acc_vert,
        "jerk_long": jerk,
        "yaw_rate_dps": yaw,
        "alt_m": bruto["z"].to_numpy(float) if "z" in bruto else np.zeros(len(v)),
        "odom_m": bruto["odom_m"].to_numpy(float) if "odom_m" in bruto else np.cumsum(v) / taxa_hz,
        "grade_pct": 100.0 * rampa,
        "manobra": rotular_manobras(v, a_suave, yaw, taxa_hz),
        "fonte": bruto["fonte"].astype(str) if "fonte" in bruto else "simulacao",
        "throttle": bruto["throttle"].to_numpy(float) if "throttle" in bruto else np.zeros(len(v)),
        "brake": bruto["brake"].to_numpy(float) if "brake" in bruto else np.zeros(len(v)),
    })
    return dados


def resumo_numerico(valor: float) -> float:
    return 0.0 if not math.isfinite(valor) else round(float(valor), 4)
