"""Crenças da camada agêntica: fatos medidos em cada trecho de telemetria.

Os fatos são números já arredondados, com nome e unidade explícitos, porque vão direto
para o prompt do Supervisor e para a guarda de números.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Nome legível de cada rótulo da coluna "manobra" (vocabulário do coleta_carla v7)
NOMES_MANOBRA = {
    "cruzeiro": "velocidade constante",
    "acelerar": "aceleração",
    "frear": "frenagem",
    "curva_esq": "curva à esquerda",
    "curva_dir": "curva à direita",
    "parar": "parado",
}

NOMES_NIVEL = {"info": "informativo", "atencao": "atenção", "critico": "crítico"}
ORDEM_NIVEL = {"info": 0, "atencao": 1, "critico": 2}


def hora_curta(hora_iso: Any) -> str:
    """'2026-10-03T14:36:35.427-03:00' -> '14:36:35' (texto vazio se não houver hora)."""
    texto = str(hora_iso)
    return texto[11:19] if len(texto) >= 19 else ""


def faixa_velocidade(v_kmh: float, limites: list[float]) -> str:
    """Rótulo da faixa de velocidade em km/h: '<10', '10-20', '20-30' ou '>=30'."""
    if v_kmh < limites[0]:
        return f"<{limites[0]:g}"
    for baixo, alto in zip(limites, limites[1:]):
        if v_kmh < alto:
            return f"{baixo:g}-{alto:g}"
    return f">={limites[-1]:g}"


def fonte_do_trecho(fontes: pd.Series) -> str:
    """'real', 'sintetico' ou 'misto', conforme a coluna fonte das linhas do trecho."""
    valores = set(fontes.astype(str))
    if valores <= {"real"}:
        return "real"
    if valores <= {"sintetico"}:
        return "sintetico"
    return "misto"


def _r(valor: float, casas: int) -> float:
    """round() sem o "-0.0" (somar 0.0 transforma -0.0 em 0.0)."""
    return round(float(valor), casas) + 0.0


def calcular_fatos(trecho: pd.DataFrame, taxa_hz: float, reto_ate_graus: float = 15.0) -> dict[str, Any]:
    """Resume um trecho (uma janela ou janelas mescladas) em fatos arredondados.

    A mudança de rumo é a integral de yaw_rate_dps (convenção do CARLA: positivo à direita).
    """
    dt = 1.0 / taxa_hz
    t = trecho["sim_time"].to_numpy(dtype=float)
    v = trecho["speed_kmh"].to_numpy(dtype=float)
    a_long = trecho["acc_long"].to_numpy(dtype=float)
    jerk = trecho["jerk_long"].to_numpy(dtype=float)
    giro = float(np.sum(trecho["yaw_rate_dps"].to_numpy(dtype=float)) * dt)
    alt = trecho["alt_m"].to_numpy(dtype=float)
    odom = trecho["odom_m"].to_numpy(dtype=float)
    contagem = trecho["manobra"].astype(str).value_counts()

    if abs(giro) < reto_ate_graus:
        lado = "reto"
    else:
        lado = "direita" if giro > 0 else "esquerda"

    return {
        "t_ini": _r(t[0], 2),
        "t_fim": _r(t[-1] + dt, 2),
        "duracao_s": _r(t[-1] + dt - t[0], 1),
        "hora_ini": hora_curta(trecho["hora_local"].iloc[0]),
        "hora_fim": hora_curta(trecho["hora_local"].iloc[-1]),
        "vel_media_kmh": _r(v.mean(), 1),
        "vel_min_kmh": _r(v.min(), 1),
        "vel_max_kmh": _r(v.max(), 1),
        "variacao_vel_kmh": _r(v[-1] - v[0], 1),
        "acel_long_min_mps2": _r(a_long.min(), 2),
        "acel_long_max_mps2": _r(a_long.max(), 2),
        "acel_lat_max_abs_mps2": _r(np.abs(trecho["acc_lat"].to_numpy(dtype=float)).max(), 2),
        "acel_vert_max_abs_mps2": _r(np.abs(trecho["acc_vert"].to_numpy(dtype=float)).max(), 2),
        "jerk_long_min_mps3": _r(jerk.min(), 2),
        "jerk_long_max_mps3": _r(jerk.max(), 2),
        "mudanca_rumo_graus": int(round(giro)),
        "lado": lado,
        "manobra_predominante": str(contagem.index[0]),
        "manobra_predominante_pct": int(round(100.0 * contagem.iloc[0] / len(trecho))),
        "rampa_media_pct": _r(trecho["grade_pct"].mean(), 1),
        "variacao_alt_m": _r(alt[-1] - alt[0], 1),
        "distancia_m": int(round(float(odom[-1] - odom[0] + v[-1] / 3.6 * dt))),
        "fonte": fonte_do_trecho(trecho["fonte"]),
    }


def estado_do_trecho(fatos: dict[str, Any], faixas_kmh: list[float]) -> tuple[str, str, str, bool]:
    """Estado usado na mesclagem: (manobra predominante, faixa de velocidade, nível, atípica pelo ML)."""
    return (
        fatos["manobra_predominante"],
        faixa_velocidade(fatos["vel_media_kmh"], faixas_kmh),
        fatos.get("nivel", "info"),
        bool(fatos.get("ml_atipica", False)),
    )
