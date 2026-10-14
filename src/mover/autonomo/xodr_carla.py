"""Ajusta o .xodr antes de o CARLA gerar o mundo: junções de passagem com conector minúsculo viram ligação direta.

O mapa_final.xodr (convertido do SUMO) tem junções de passagem: uma via chega, um conector de 0,1 a
0,6 m liga à via seguinte, e nada mais. Na primeira corrida no CARLA, o Traffic Manager removeu o
caminhão logo antes da primeira junção com conector de 0,11 m (via 257 -> conector 287 -> via 258).
Aqui cada uma delas vira uma ligação direta via -> via (links da via e das faixas); o conector e a
junção saem. Se a última geometria da via que chega for uma reta no rumo do conector, ela cresce o
comprimento do conector, sem deixar fresta. O arquivo em disco não muda.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from typing import Any


def _faixa(secao: ET.Element | None, faixa_id: str) -> ET.Element | None:
    if secao is None:
        return None
    for lado in ("left", "center", "right"):
        grupo = secao.find(lado)
        for faixa in grupo.findall("lane") if grupo is not None else []:
            if faixa.get("id") == faixa_id:
                return faixa
    return None


def _ligar_faixa(faixa: ET.Element, tipo: str, alvo: str) -> None:
    link = faixa.find("link")
    if link is None:
        link = ET.Element("link")
        faixa.insert(0, link)  # no OpenDRIVE, o link vem antes da largura
    for antigo in link.findall(tipo):
        link.remove(antigo)
    novo = ET.Element(tipo, {"id": alvo})
    if tipo == "predecessor":
        link.insert(0, novo)
    else:
        link.append(novo)


def _ref(estrada: ET.Element, tipo: str) -> ET.Element | None:
    link = estrada.find("link")
    return link.find(tipo) if link is not None else None


def colapsar_juncoes_de_passagem(texto: str, limiar_m: float = 1.0) -> tuple[str, list[dict[str, Any]]]:
    """Devolve o texto ajustado e a lista das junções colapsadas (o texto original, se nada mudou)."""
    if limiar_m <= 0.0 or "<junction" not in texto:
        return texto, []
    raiz = ET.fromstring(texto)
    estradas = {e.get("id"): e for e in raiz.findall("road")}
    referencias: dict[str, int] = {}
    for e in estradas.values():
        for tipo in ("predecessor", "successor"):
            r = _ref(e, tipo)
            if r is not None and r.get("elementType") == "junction":
                referencias[r.get("elementId")] = referencias.get(r.get("elementId"), 0) + 1
    feitas: list[dict[str, Any]] = []
    for juncao in list(raiz.findall("junction")):
        jid = juncao.get("id")
        conexoes = juncao.findall("connection")
        if len(conexoes) != 1 or referencias.get(jid, 0) != 2:
            continue
        conexao = conexoes[0]
        conector = estradas.get(conexao.get("connectingRoad"))
        if conector is None or float(conector.get("length", "inf")) >= limiar_m:
            continue
        pre, suc = _ref(conector, "predecessor"), _ref(conector, "successor")
        if pre is None or suc is None or pre.get("elementType") != "road" or suc.get("elementType") != "road":
            continue
        if pre.get("contactPoint") != "end" or suc.get("contactPoint") != "start" or conexao.get("contactPoint") != "start":
            continue
        entrada, saida = estradas.get(pre.get("elementId")), estradas.get(suc.get("elementId"))
        if entrada is None or saida is None or entrada is saida:
            continue
        ref_entrada, ref_saida = _ref(entrada, "successor"), _ref(saida, "predecessor")
        if (ref_entrada is None or ref_saida is None or ref_entrada.get("elementId") != jid
                or ref_saida.get("elementId") != jid):
            continue
        # faixas: entrada -(laneLink)-> conector -(successor da faixa do conector)-> saída
        secao_entrada = (entrada.find("lanes").findall("laneSection") or [None])[-1]
        secao_conector = (conector.find("lanes").findall("laneSection") or [None])[-1]
        secao_saida = (saida.find("lanes").findall("laneSection") or [None])[0]
        pares = []
        for ll in conexao.findall("laneLink"):
            faixa_conector = _faixa(secao_conector, ll.get("to"))
            suc_faixa = faixa_conector.find("link/successor") if faixa_conector is not None else None
            if suc_faixa is None:
                pares = []
                break
            pares.append((ll.get("from"), suc_faixa.get("id")))
        if not pares or any(_faixa(secao_entrada, a) is None or _faixa(secao_saida, b) is None for a, b in pares):
            continue
        for a, b in pares:
            _ligar_faixa(_faixa(secao_entrada, a), "successor", b)
            _ligar_faixa(_faixa(secao_saida, b), "predecessor", a)
        ref_entrada.attrib = {"elementType": "road", "elementId": saida.get("id"), "contactPoint": "start"}
        ref_saida.attrib = {"elementType": "road", "elementId": entrada.get("id"), "contactPoint": "end"}
        comprimento = float(conector.get("length"))
        emendada = False
        ultima = entrada.find("planView").findall("geometry")[-1]
        geo_conector = conector.find("planView/geometry")
        if ultima.find("line") is not None and geo_conector is not None:
            giro = (float(geo_conector.get("hdg")) - float(ultima.get("hdg")) + math.pi) % (2 * math.pi) - math.pi
            if abs(giro) < 0.05:
                ultima.set("length", f"{float(ultima.get('length')) + comprimento:.8f}")
                entrada.set("length", f"{float(entrada.get('length')) + comprimento:.8f}")
                emendada = True
        raiz.remove(juncao)
        raiz.remove(conector)
        feitas.append({"juncao": jid, "conector": conector.get("id"), "conector_m": round(comprimento, 3),
                       "de": entrada.get("id"), "para": saida.get("id"), "sem_fresta": emendada})
    if not feitas:
        return texto, []
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(raiz, encoding="unicode"), feitas


__all__ = ["colapsar_juncoes_de_passagem"]
