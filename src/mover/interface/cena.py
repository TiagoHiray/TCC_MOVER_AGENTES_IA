"""Cena 2D para a interface: vias do mapa e trajeto alinhado, no referencial do mapa.

A página desenha com ela o mapa da volta (dashboard) e, sem o CARLA, a visão de perseguição do
painel 1. Referencial do mapa OpenDRIVE: x = leste, y = norte, em metros. O CARLA usa y invertido
(y_carla = -y_mapa e yaw_carla = -rumo), por isso a página converte as poses do WebSocket.

Quem monta a cena:
- o replay, depois de alinhar a volta (PUT /cena), com os mesmos ajustes da linha de comando;
- o servidor, se a página pedir a cena antes do replay (GET /cena), a partir do config/simulacao.yaml.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mover.simulacao.alinhamento import ResultadoAlinhamento
from mover.simulacao.opendrive import MapaOpenDrive

SALTO_MAX_M = 2.0  # pontos seguidos de uma via mais distantes que isso são de faixas diferentes


def montar_cena(mapa: MapaOpenDrive, alinhamento: ResultadoAlinhamento, origem: str = "replay",
                passo_trajeto: int = 2, casas: int = 2) -> dict[str, Any]:
    """Vias como polilinhas (com a largura da faixa) e o trajeto com tempo e rumo, a cada `passo_trajeto` quadros."""
    vias: list[dict[str, Any]] = []
    for via in dict.fromkeys(mapa.vias.tolist()):  # ordem de aparição no .xodr
        indices = np.flatnonzero(mapa.vias == via)
        pontos = mapa.pontos[indices]
        cortes = np.flatnonzero(np.hypot(*np.diff(pontos, axis=0).T) > SALTO_MAX_M) + 1
        for trecho, idx in zip(np.split(pontos, cortes), np.split(indices, cortes)):
            if len(trecho) >= 2:
                vias.append({"via": str(via), "largura_m": round(float(np.median(mapa.larguras[idx])), 2),
                             "pontos": np.round(trecho, casas).tolist()})

    poses = alinhamento.poses.iloc[::max(1, int(passo_trajeto))]
    x, y = poses["x_mapa"].to_numpy(float), poses["y_mapa"].to_numpy(float)
    trajeto = {
        "t": np.round(poses["sim_time"].to_numpy(float), 3).tolist(),
        "x": np.round(x, casas).tolist(),
        "y": np.round(y, casas).tolist(),
        "rumo_graus": np.round(-poses["yaw_carla"].to_numpy(float), 1).tolist(),  # anti-horário a partir do leste
    }
    todos = np.vstack([mapa.pontos, np.column_stack([x, y])])
    limites = [round(float(v), casas) for v in (*todos.min(axis=0), *todos.max(axis=0))]
    return {"referencial": "mapa OpenDRIVE (x = leste, y = norte, metros)", "origem": origem,
            "vias": vias, "trajeto": trajeto, "limites": limites}


def cena_do_yaml(cfg_simulacao: dict[str, Any]) -> dict[str, Any]:
    """Alinha a volta com o mapa e a telemetria do YAML (sem ajustes manuais) e monta a cena."""
    from mover.simulacao.verificar_mapa import alinhar_volta  # importa o matplotlib só se precisar

    mapa, alinhamento, _ = alinhar_volta(cfg_simulacao)
    return montar_cena(mapa, alinhamento, origem="config")
