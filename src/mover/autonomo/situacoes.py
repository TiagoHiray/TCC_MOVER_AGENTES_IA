"""Situações da apresentação (curva, pedestre, veículo à frente, lombada, parada) e as métricas de cada uma.

As situações saem do plano de missão: a curva mais fechada, uma faixa de pedestre longe das outras zonas
(com um pedestre atravessando), a reta mais longa (com um veículo lento à frente), uma lombada e uma parada,
quando o mapa as tem. As métricas de uma situação são as da telemetria dentro da janela dela, em cada volta.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from mover.autonomo.consumo import Limiares, suavizar
from mover.autonomo.diagnostico import eventos_da_corrida
from mover.autonomo.missao import PlanoMissao, Zona

MARGEM_LARGADA_M = 40.0  # nada de situação colada na largada ou na chegada


@dataclass
class Situacao:
    id: str
    nome: str
    pergunta: str
    s_ini: float          # janela das métricas na volta (m); pode cruzar a largada
    s_fim: float
    figurantes: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def para_dict(self) -> dict[str, Any]:
        return asdict(self)


def _comprimento(a: float, b: float, L: float) -> float:
    return (b - a) % L


def _livres(plano: PlanoMissao, ocupados: list[tuple[float, float]], tipos: tuple[str, ...] | None = None) -> list[tuple[float, float]]:
    """Trechos da volta sem zona (dos tipos dados; todos se None), sem parada e sem situação já escolhida."""
    L = plano.comprimento
    n = int(L)
    livre = np.ones(n, bool)
    livre[: int(MARGEM_LARGADA_M)] = livre[n - int(MARGEM_LARGADA_M):] = False
    zonas = [z for z in plano.zonas if tipos is None or z.tipo in tipos]
    intervalos = [(z.s_ini, z.s_fim) for z in zonas] + [(p.s - 20.0, p.s + 20.0) for p in plano.paradas] + ocupados
    for a, b in intervalos:
        idx = (np.arange(int(np.floor(a)), int(np.floor(a)) + int(np.ceil(_comprimento(a, b, L))) + 1)) % n
        livre[idx] = False
    trechos, inicio = [], None
    for i, ok in enumerate(livre):
        if ok and inicio is None:
            inicio = i
        elif not ok and inicio is not None:
            trechos.append((float(inicio), float(i)))
            inicio = None
    if inicio is not None:
        trechos.append((float(inicio), float(n)))
    return sorted(trechos, key=lambda t: t[1] - t[0], reverse=True)


def _longe(s: float, ocupados: list[tuple[float, float]], L: float) -> float:
    """Menor distância (m, na volta) do ponto s às janelas já ocupadas."""
    if not ocupados:
        return L
    dist = []
    for a, b in ocupados:
        dentro = _comprimento(a, s, L) <= _comprimento(a, b, L)
        dist.append(0.0 if dentro else min(_comprimento(s, a, L), _comprimento(b, s, L)))
    return min(dist)


def escolher_situacoes(plano: PlanoMissao, faixa_kmh: float = 15.0) -> list[Situacao]:
    """As situações da volta. Se o mapa não tiver faixa de pedestre, uma é criada na reta mais longa."""
    L = plano.comprimento
    situacoes: list[Situacao] = []
    ocupados: list[tuple[float, float]] = []

    def janela(a: float, b: float) -> tuple[float, float]:
        return a % L, b % L

    curvas = [z for z in plano.zonas if z.tipo == "curva"]
    # de preferência uma curva longe da largada: perto dela, a volta do vídeo está saindo ou chegando
    curvas = [z for z in curvas if MARGEM_LARGADA_M + 40.0 <= z.s_ini <= z.s_fim <= L - MARGEM_LARGADA_M - 20.0] or curvas
    if curvas:
        c = min(curvas, key=lambda z: (z.v_max_kmh, -_comprimento(z.s_ini, z.s_fim, L)))
        a, b = janela(c.s_ini - 40.0, c.s_fim + 20.0)
        situacoes.append(Situacao("curva", f"Curva fechada ({c.nome}, limite de {c.v_max_kmh:.0f} km/h)",
                                  "Como o caminhão se comporta numa curva fechada?", a, b))
        ocupados.append((a, b))
    parada = None
    if plano.paradas:
        p = plano.paradas[0]
        a, b = janela(p.s - 60.0, p.s + 30.0)
        parada = Situacao("parada", f"Parada obrigatória ({p.nome}, {p.duracao_s:.0f} s)",
                          "Como o caminhão para e retoma (portão)?", a, b)
        ocupados.append((a, b))

    faixas = [z for z in plano.zonas if z.tipo == "faixa"
              and MARGEM_LARGADA_M + 25.0 < (z.s_ini + z.s_fim) / 2 < L - MARGEM_LARGADA_M]
    if faixas:
        faixa = max(faixas, key=lambda z: _longe((z.s_ini + _comprimento(z.s_ini, z.s_fim, L) / 2) % L, ocupados, L))
        centro = (faixa.s_ini + _comprimento(faixa.s_ini, faixa.s_fim, L) / 2) % L
    else:
        livres = _livres(plano, ocupados)
        if not livres:
            centro = None
        else:
            a, b = livres[0]
            centro = (a + b) / 2
            plano.zonas.append(Zona("faixa_cenario", "faixa", round(centro - 6.0, 1), round(centro + 6.0, 1), faixa_kmh,
                                    "faixa de pedestre", "faixa de pedestre do cenário (o mapa não tem faixa nesta volta)"))
            plano.zonas.sort(key=lambda z: z.s_ini)
    if centro is not None:
        a, b = janela(centro - 60.0, centro + 25.0)
        situacoes.append(Situacao("pedestre", "Pedestre atravessando na faixa",
                                  "Como o caminhão se comporta com um pedestre atravessando?", a, b,
                                  {"pedestres": [{"s_m": round(centro, 1), "gatilho_m": 30.0, "lado_m": 6.0,
                                                  "atravessar_m": 14.0, "velocidade_mps": 1.0}]}))
        ocupados.append((a, b))

    livres = [t for t in _livres(plano, ocupados) if t[1] - t[0] >= 120.0] or \
        [t for t in _livres(plano, ocupados, ("curva",)) if t[1] - t[0] >= 120.0]  # reta com faixa/lombada
    if livres:
        a, b = livres[0]
        percurso = min(b - a - 50.0, 220.0)
        situacoes.append(Situacao("veiculo", "Veículo lento à frente (12 km/h)",
                                  "Como o caminhão se comporta com outro veículo à frente?", a, (a + 35.0 + percurso) % L,
                                  {"veiculos": [{"s_m": round(a + 5.0, 1), "a_frente_m": 35.0, "percurso_m": round(percurso, 1),
                                                 "velocidade_kmh": 12.0}]}))
        ocupados.append((a, (a + 35.0 + percurso) % L))

    lombadas = [z for z in plano.zonas if z.tipo == "lombada"]
    if lombadas:
        lomb = max(lombadas, key=lambda z: _longe(z.s_ini, ocupados, L))
        a, b = janela(lomb.s_ini - 40.0, lomb.s_fim + 20.0)
        situacoes.append(Situacao("lombada", f"Lombada ({lomb.nome})", "Como o caminhão passa por uma lombada?", a, b))
        ocupados.append((a, b))
    if parada is not None:
        situacoes.append(parada)
    return situacoes


def figurantes_do_cenario(situacoes: list[Situacao]) -> dict[str, list[dict[str, Any]]]:
    """A seção cenario do config/simulacao.yaml com os figurantes de todas as situações."""
    cenario: dict[str, list[dict[str, Any]]] = {"pedestres": [], "veiculos": []}
    for s in situacoes:
        for tipo, lista in s.figurantes.items():
            cenario[tipo].extend(lista)
    return cenario


def metricas_situacao(tel: pd.DataFrame, sit: Situacao, L: float, limiares: Limiares,
                      eventos: pd.DataFrame | None = None) -> dict[str, Any]:
    """Médias por volta dentro da janela da situação: tempo, diesel, eventos, velocidade mínima, pior
    desaceleração e, com figurante, a menor distância a ele."""
    if tel.empty or "d" not in tel:
        return {}
    t = tel["sim_time"].to_numpy(float)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.05
    a_suave = suavizar(tel["acc_long"].to_numpy(float), dt, limiares.fc_hz)
    if eventos is None:
        eventos = eventos_da_corrida(tel, limiares, L)
    d = tel["d"].to_numpy(float)
    largura = _comprimento(sit.s_ini, sit.s_fim, L)
    voltas = []
    for k in range(int(d.max() // L) + 1):
        ini = k * L + sit.s_ini
        sel = (d >= ini) & (d <= ini + largura)
        if sel.sum() < 5 or d.max() < ini + largura:
            continue
        idx = np.flatnonzero(sel)
        t0, t1 = t[idx[0]], t[idx[-1]]
        ev = eventos[(eventos["t_s"] >= t0) & (eventos["t_s"] <= t1)] if len(eventos) else eventos
        linha = {
            "tempo_s": t1 - t0,
            "diesel_ml": float(tel["combustivel_ml"].iloc[idx[-1]] - tel["combustivel_ml"].iloc[idx[0]])
            if "combustivel_ml" in tel else np.nan,
            "eventos": int(len(ev)), "criticos": int((ev["nivel"] == "critico").sum()) if len(ev) else 0,
            "v_min_kmh": float(tel["speed_mps"].iloc[idx].min() * 3.6),
            "desacel_max_mps2": float(-min(0.0, a_suave[idx].min())),
        }
        if "dist_figurante_m" in tel and tel["dist_figurante_m"].iloc[idx].notna().any():
            linha["dist_min_figurante_m"] = float(tel["dist_figurante_m"].iloc[idx].min())
        voltas.append(linha)
    if not voltas:
        return {}
    media = pd.DataFrame(voltas).mean(numeric_only=True).round(2).to_dict()
    media["voltas"] = len(voltas)
    return media


__all__ = ["Situacao", "escolher_situacoes", "figurantes_do_cenario", "metricas_situacao"]
