"""Rota da volta tirada só do mapa do CARLA (OpenDRIVE), sem nenhuma gravação.

A volta é dada por um itinerário: as vias do .xodr em que o caminhão passa, na ordem (os
conectores das junções podem ficar de fora, porque são achados aqui). Daqui sai:
1. as faixas do mapa, cada uma orientada no sentido de tráfego;
2. o grafo de ligação entre faixas (o fim de uma encosta no começo da outra);
3. o caminho mais curto no grafo entre cada via do itinerário e a seguinte, fechando a volta;
4. a rota (linha central das faixas, reamostrada), com s = 0 no ponto de largada.

Uso, para ver as vias do mapa e escolher um itinerário:
    python -m mover.autonomo.rota_mapa                    # lista as vias (id, comprimento, extremos)
    python -m mover.autonomo.rota_mapa --figura vias.png  # e desenha os ids no mapa
"""

from __future__ import annotations

import argparse
import heapq
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # execução direta: python src/mover/autonomo/rota_mapa.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from mover.autonomo.rota import Rota
from mover.simulacao.opendrive import MapaOpenDrive

log = logging.getLogger("mover.autonomo")


@dataclass
class Faixa:
    """Uma faixa de uma via, com os pontos no sentido de tráfego."""

    via: int
    indice: int
    pontos: np.ndarray  # (N, 2)

    @property
    def chave(self) -> tuple[int, int]:
        return self.via, self.indice

    @property
    def comprimento(self) -> float:
        return float(np.sum(np.hypot(*np.diff(self.pontos, axis=0).T))) if len(self.pontos) > 1 else 0.0

    @property
    def inicio(self) -> np.ndarray:
        return self.pontos[0]

    @property
    def fim(self) -> np.ndarray:
        return self.pontos[-1]

    def rumo_inicio(self) -> float:
        p = self.pontos[: min(4, len(self.pontos))]
        return math.atan2(p[-1, 1] - p[0, 1], p[-1, 0] - p[0, 0])

    def rumo_fim(self) -> float:
        p = self.pontos[-min(4, len(self.pontos)):]
        return math.atan2(p[-1, 1] - p[0, 1], p[-1, 0] - p[0, 0])


def faixas_do_mapa(mapa: MapaOpenDrive) -> dict[tuple[int, int], Faixa]:
    """Separa os centros de faixa por via e por faixa, orientados no sentido de tráfego.

    O leitor do .xodr devolve, para cada via, as faixas uma depois da outra, cada uma no sentido
    crescente de s. Um salto entre pontos seguidos marca o começo da faixa seguinte.
    """
    faixas: dict[tuple[int, int], Faixa] = {}
    for via in np.unique(mapa.vias):
        sel = np.flatnonzero(mapa.vias == via)
        pts, rumos = mapa.pontos[sel], mapa.rumos[sel]
        if len(pts) < 2:
            continue
        passos = np.hypot(*np.diff(pts, axis=0).T)
        tipico = float(np.median(passos)) if len(passos) else 1.0
        cortes = np.flatnonzero(passos > max(3.0 * tipico, 1.5)) + 1
        for k, (a, b) in enumerate(zip(np.r_[0, cortes], np.r_[cortes, len(pts)])):
            trecho, rumo = pts[a:b], rumos[a:b]
            if len(trecho) < 2:
                continue
            direcao = np.arctan2(*np.gradient(trecho, axis=0)[:, ::-1].T)
            if float(np.mean(np.cos(direcao - rumo))) < 0:  # faixa com tráfego contra o sentido de s
                trecho = trecho[::-1]
            faixas[(int(via), k)] = Faixa(int(via), k, trecho.copy())
    return faixas


def grafo_de_faixas(faixas: dict[tuple[int, int], Faixa], tolerancia_m: float = 1.0,
                    max_giro_graus: float = 75.0) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Liga a faixa A à faixa B quando o fim de A encosta no começo de B, sem virar demais."""
    chaves = list(faixas)
    inicios = np.array([faixas[c].inicio for c in chaves])
    grafo: dict[tuple[int, int], list[tuple[int, int]]] = {c: [] for c in chaves}
    for a in chaves:
        fa = faixas[a]
        perto = np.flatnonzero(np.hypot(*(inicios - fa.fim).T) <= tolerancia_m)
        for j in perto:
            b = chaves[j]
            if b == a:
                continue
            giro = abs((faixas[b].rumo_inicio() - fa.rumo_fim() + math.pi) % (2 * math.pi) - math.pi)
            if math.degrees(giro) <= max_giro_graus:
                grafo[a].append(b)
    return grafo


def _caminho_mais_curto(faixas: dict[tuple[int, int], Faixa], grafo: dict[tuple[int, int], list[tuple[int, int]]],
                        origem: tuple[int, int], destinos: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
    """Dijkstra pelo comprimento das faixas: devolve as faixas entre a origem e um destino (sem os dois)."""
    fila = [(0.0, origem)]
    anterior: dict[tuple[int, int], tuple[int, int] | None] = {origem: None}
    custo = {origem: 0.0}
    while fila:
        c, atual = heapq.heappop(fila)
        if c > custo.get(atual, math.inf):
            continue
        for prox in grafo.get(atual, []):
            if prox in destinos:
                caminho = []
                no = atual
                while no is not None and no != origem:
                    caminho.append(no)
                    no = anterior[no]
                return [*reversed(caminho), prox]
            novo = c + faixas[prox].comprimento
            if novo < custo.get(prox, math.inf):
                custo[prox], anterior[prox] = novo, atual
                heapq.heappush(fila, (novo, prox))
    return None


def itinerario_de_faixas(faixas: dict[tuple[int, int], Faixa], grafo: dict[tuple[int, int], list[tuple[int, int]]],
                         vias: list[int]) -> list[tuple[int, int]]:
    """As faixas da volta: as do itinerário e os conectores achados entre elas, fechando o laço."""
    if len(vias) < 2:
        raise ValueError("o itinerário precisa de pelo menos 2 vias")
    por_via: dict[int, list[tuple[int, int]]] = {}
    for chave in faixas:
        por_via.setdefault(chave[0], []).append(chave)
    faltando = [v for v in vias if v not in por_via]
    if faltando:
        raise ValueError(f"vias do itinerário que não existem no mapa (ou sem faixa transitável): {faltando}")
    # a faixa de cada via: a que liga à via seguinte pelo caminho mais curto
    escolha: list[tuple[int, int]] = []
    for i, via in enumerate(vias):
        candidatas = por_via[via]
        if len(candidatas) == 1 or not escolha and i + 1 >= len(vias):
            escolha.append(candidatas[0])
            continue
        seguintes = set(por_via[vias[(i + 1) % len(vias)]])
        melhor, melhor_c = candidatas[0], math.inf
        for c in candidatas:
            caminho = _caminho_mais_curto(faixas, grafo, c, seguintes)
            if caminho is not None:
                comp = sum(faixas[k].comprimento for k in caminho)
                if comp < melhor_c:
                    melhor, melhor_c = c, comp
        escolha.append(melhor)
    sequencia: list[tuple[int, int]] = []
    for i, atual in enumerate(escolha):
        prox = escolha[(i + 1) % len(escolha)]
        caminho = _caminho_mais_curto(faixas, grafo, atual, {prox})
        if caminho is None:
            raise ValueError(f"não há ligação no mapa da via {atual[0]} para a via {prox[0]} (confira a ordem do itinerário)")
        sequencia.append(atual)
        sequencia.extend(caminho[:-1])  # os conectores; a próxima via entra na volta seguinte do laço
    return sequencia


def rota_do_itinerario(faixas: dict[tuple[int, int], Faixa], sequencia: list[tuple[int, int]],
                       largada: tuple[float, float] | None = None, passo_m: float = 1.0, suavizar_m: float = 4.0) -> Rota:
    """Junta as faixas da sequência numa rota fechada; s = 0 no ponto da rota mais perto da largada."""
    partes = []
    for chave in sequencia:
        pts = faixas[chave].pontos
        if partes and np.hypot(*(pts[0] - partes[-1][-1])) < 0.3:
            pts = pts[1:]  # o fim de uma faixa é o começo da outra
        partes.append(pts)
    pontos = np.vstack(partes)
    if np.hypot(*(pontos[-1] - pontos[0])) < 0.3:
        pontos = pontos[:-1]
    rota = Rota(pontos[:, 0], pontos[:, 1], passo_m=passo_m, suavizar_m=suavizar_m)
    if largada is not None:
        s0, _ = rota.projetar(float(largada[0]), float(largada[1]))
        k = int(round(s0 / rota.passo)) % rota.n
        rota = Rota(np.roll(rota.x, -k), np.roll(rota.y, -k), passo_m=passo_m, suavizar_m=0.0)
    return rota


def rota_do_mapa(mapa: MapaOpenDrive, vias: list[int], largada: tuple[float, float] | None = None,
                 passo_m: float = 1.0, suavizar_m: float = 4.0, tolerancia_m: float = 1.0) -> tuple[Rota, list[tuple[int, int]]]:
    """Rota fechada da volta a partir do itinerário de vias do mapa. Devolve a rota e as faixas usadas."""
    faixas = faixas_do_mapa(mapa)
    grafo = grafo_de_faixas(faixas, tolerancia_m)
    sequencia = itinerario_de_faixas(faixas, grafo, [int(v) for v in vias])
    return rota_do_itinerario(faixas, sequencia, largada, passo_m, suavizar_m), sequencia


def resumo_das_vias(mapa: MapaOpenDrive) -> list[dict[str, Any]]:
    """Id, comprimento e extremos de cada faixa do mapa (para montar um itinerário à mão)."""
    faixas = faixas_do_mapa(mapa)
    grafo = grafo_de_faixas(faixas)
    saida = []
    for chave, f in sorted(faixas.items()):
        saida.append({"via": f.via, "faixa": f.indice, "comprimento_m": round(f.comprimento, 1),
                      "inicio": [round(float(v), 1) for v in f.inicio], "fim": [round(float(v), 1) for v in f.fim],
                      "segue_para": sorted({b[0] for b in grafo[chave]})})
    return saida


def desenhar_vias(mapa: MapaOpenDrive, destino: Path, rota: Rota | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    faixas = faixas_do_mapa(mapa)
    fig, ax = plt.subplots(figsize=(9, 8), dpi=130)
    for f in faixas.values():
        ax.plot(f.pontos[:, 0], f.pontos[:, 1], c="#9aa0a6", lw=1)
        meio = f.pontos[len(f.pontos) // 2]
        if f.comprimento >= 12:
            ax.annotate(str(f.via), meio, fontsize=6, color="#1a73e8", ha="center")
            fim, antes = f.pontos[-1], f.pontos[max(0, len(f.pontos) - 4)]
            ax.annotate("", fim, antes, arrowprops={"arrowstyle": "->", "color": "#5f6368", "lw": 0.8})
    if rota is not None:
        ax.plot(np.append(rota.x, rota.x[0]), np.append(rota.y, rota.y[0]), c="#d93025", lw=1.6, alpha=0.7,
                label=f"volta ({rota.comprimento:.0f} m)")
        ax.legend(loc="best", fontsize=7)
    ax.set_aspect("equal")
    ax.set_title("Vias do mapa (id) no sentido de tráfego")
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destino)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    from mover.config import caminho, carregar_yaml
    from mover.simulacao.opendrive import ler_xodr

    parser = argparse.ArgumentParser(description="Lista as vias do mapa para montar o itinerário da volta.")
    parser.add_argument("--config", default="config/simulacao.yaml")
    parser.add_argument("--config-autonomo", default="config/autonomo.yaml")
    parser.add_argument("--mapa", help="arquivo .xodr (padrão: o do config/simulacao.yaml)")
    parser.add_argument("--figura", help="PNG com os ids das vias e a volta do itinerário atual")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    cfg_sim = carregar_yaml(args.config)
    mapa = ler_xodr(caminho(args.mapa or cfg_sim["mapa"]["arquivo"]), 1.0,
                    tuple(cfg_sim["mapa"].get("tipos_faixa", ("driving",))))
    for v in resumo_das_vias(mapa):
        print(f"via {v['via']:>5} faixa {v['faixa']}  {v['comprimento_m']:7.1f} m  início {v['inicio']}  fim {v['fim']}  "
              f"-> {v['segue_para']}")
    rota = None
    cr = ((carregar_yaml(args.config_autonomo).get("gerar_missao") or {}).get("rota") or {})
    if cr.get("vias"):
        rota, seq = rota_do_mapa(mapa, cr["vias"], tuple(cr["largada"]) if cr.get("largada") else None)
        print(f"Itinerário do config ({len(cr['vias'])} vias + conectores): {[v for v, _ in seq]} -> {rota.comprimento:.1f} m")
    if args.figura:
        desenhar_vias(mapa, caminho(args.figura) if not Path(args.figura).is_absolute() else Path(args.figura), rota)
        print(f"Figura: {args.figura}")


if __name__ == "__main__":
    main()
