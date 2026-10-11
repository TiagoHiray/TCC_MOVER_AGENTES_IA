"""Cena 2D para a interface: vias do mapa, rota da volta, zonas e paradas do plano de missão.

A página desenha com ela o mapa da volta (dashboard) e, sem o CARLA, a visão de perseguição do
painel 1. O rastro do caminhão não vem daqui: ele cresce na página com a telemetria ao vivo.
Referencial do mapa OpenDRIVE: x = leste, y = norte, em metros (o CARLA usa y invertido).

Quem monta a cena:
- o piloto (mover.autonomo.dirigir), antes da largada (PUT /cena);
- o servidor, se a página pedir antes (GET /cena), a partir do config/autonomo.yaml e do mapa.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mover.autonomo.missao import PlanoMissao
from mover.simulacao.opendrive import MapaOpenDrive

SALTO_MAX_M = 2.0  # pontos seguidos de uma via mais distantes que isso são de faixas diferentes


def vias_do_mapa(mapa: MapaOpenDrive, casas: int = 2) -> list[dict[str, Any]]:
    """Vias como polilinhas, com a largura da faixa."""
    vias: list[dict[str, Any]] = []
    for via in dict.fromkeys(mapa.vias.tolist()):  # ordem de aparição no .xodr
        indices = np.flatnonzero(mapa.vias == via)
        pontos = mapa.pontos[indices]
        cortes = np.flatnonzero(np.hypot(*np.diff(pontos, axis=0).T) > SALTO_MAX_M) + 1
        for trecho, idx in zip(np.split(pontos, cortes), np.split(indices, cortes)):
            if len(trecho) >= 2:
                vias.append({"via": str(via), "largura_m": round(float(np.median(mapa.larguras[idx])), 2),
                             "pontos": np.round(trecho, casas).tolist()})
    return vias


def montar_cena(plano: PlanoMissao, mapa: MapaOpenDrive | None = None, origem: str = "piloto",
                passo_rota_m: float = 2.0, casas: int = 2) -> dict[str, Any]:
    rota = plano.rota
    L = rota.comprimento
    pontos = rota.amostrar(0.0, L, passo_rota_m)
    zonas = []
    for z in plano.zonas:
        fim = z.s_fim if z.s_fim >= z.s_ini else z.s_fim + L
        amostras = rota.amostrar(z.s_ini, max(fim - z.s_ini, 0.5), 1.0)
        zonas.append({"id": z.id, "tipo": z.tipo, "nome": z.nome or z.tipo, "s_ini": round(z.s_ini, 1),
                      "s_fim": round(z.s_fim, 1), "v_max_kmh": z.v_max_kmh,
                      "pontos": [[round(p.x, casas), round(p.y, casas)] for p in amostras]})
    paradas = []
    for p in plano.paradas:
        ponto = rota.ponto(p.s)
        paradas.append({"id": p.id, "nome": p.nome or "parada", "s": round(p.s, 1), "duracao_s": p.duracao_s,
                        "x": round(ponto.x, casas), "y": round(ponto.y, casas)})
    largada = rota.ponto(0.0)
    vias = vias_do_mapa(mapa, casas) if mapa is not None else []
    todos = [np.array([[p.x, p.y] for p in pontos])]
    if mapa is not None and len(mapa.pontos):
        todos.append(mapa.pontos)
    todos = np.vstack(todos)
    limites = [round(float(v), casas) for v in (*todos.min(axis=0), *todos.max(axis=0))]
    return {
        "referencial": "mapa OpenDRIVE (x = leste, y = norte, metros)",
        "origem": origem,
        "vias": vias,
        "rota": {"comprimento_m": round(L, 2), "s": [round(p.s, 1) for p in pontos],
                 "x": [round(p.x, casas) for p in pontos], "y": [round(p.y, casas) for p in pontos]},
        "largada": {"x": round(largada.x, casas), "y": round(largada.y, casas), "rumo_graus": round(np.degrees(largada.rumo), 1)},
        "sentido": plano.sentido,
        "zonas": zonas,
        "paradas": paradas,
        "limites": limites,
    }


def cena_do_config(cfg_simulacao: dict[str, Any], cfg_autonomo: dict[str, Any]) -> dict[str, Any]:
    """Plano de missão do config/autonomo.yaml e vias do mapa do config/simulacao.yaml (se existir)."""
    from mover.config import caminho
    from mover.simulacao.opendrive import ler_xodr

    arquivo = caminho((cfg_autonomo.get("missao") or {}).get("arquivo", "config/missao.yaml"))
    if not arquivo.exists():
        raise FileNotFoundError(f"plano de missão não encontrado ({arquivo})")
    plano = PlanoMissao.carregar(arquivo)
    mapa = None
    cm = cfg_simulacao.get("mapa") or {}
    arquivo_mapa = caminho(plano.mapa or cm.get("arquivo", "data/mapas/mapa_final.xodr"))
    if arquivo_mapa.exists():
        mapa = ler_xodr(arquivo_mapa, float(cm.get("passo_m", 0.5)),
                        tuple(cm.get("tipos_faixa", ("driving", "restricted", "bidirectional", "parking"))))
    return montar_cena(plano, mapa, origem="config")


__all__ = ["cena_do_config", "montar_cena", "vias_do_mapa"]
