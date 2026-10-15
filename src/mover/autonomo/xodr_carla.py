"""Ajusta o .xodr antes de o CARLA gerar o mundo (o arquivo em disco não muda).

1. Acostamentos: todas as vias do mapa_final.xodr têm uma faixa só, de 3,2 m, sem nada ao lado; fora
   dela o CARLA não gera chão. Um caminhão de ~2,9 m que corta a curva põe as rodas de dentro fora da
   malha e fica preso (na VM: parado a 82,5 m, z = -0,11 m, 1 m à direita da rota, e o Traffic Manager
   acabou removendo-o). Cada via ganha um acostamento (faixa "shoulder", que o TM não usa) de cada
   lado; a faixa de direção continua no mesmo lugar.
2. Junções de passagem (convertidas do SUMO): uma via chega, um conector de 0,1 a 0,6 m liga à via
   seguinte, e nada mais. Cada uma vira uma ligação direta via -> via (links da via e das faixas); o
   conector e a junção saem. Se a última geometria da via que chega for uma reta no rumo do conector,
   ela cresce o comprimento do conector, sem deixar fresta.
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


def _acostamento(faixa_id: int, largura_m: float) -> ET.Element:
    faixa = ET.Element("lane", {"id": str(faixa_id), "type": "shoulder", "level": "true"})
    ET.SubElement(faixa, "link")
    ET.SubElement(faixa, "width", {"sOffset": "0", "a": f"{largura_m:.2f}", "b": "0", "c": "0", "d": "0"})
    return faixa


def acrescentar_acostamentos(texto: str, largura_m: float = 2.0) -> tuple[str, int]:
    """Um acostamento de `largura_m` de cada lado de cada via. Devolve o texto e quantas seções mudaram."""
    if largura_m <= 0.0 or "<laneSection" not in texto:
        return texto, 0
    raiz = ET.fromstring(texto)
    secoes = 0
    for secao in raiz.iter("laneSection"):
        esquerda, centro, direita = secao.find("left"), secao.find("center"), secao.find("right")
        if centro is None:
            continue
        if esquerda is None:
            esquerda = ET.Element("left")
            secao.insert(list(secao).index(centro), esquerda)
        if direita is None:
            direita = ET.SubElement(secao, "right")
        ids_esq = [int(f.get("id")) for f in esquerda.findall("lane")]
        ids_dir = [int(f.get("id")) for f in direita.findall("lane")]
        esquerda.insert(0, _acostamento(max(ids_esq, default=0) + 1, largura_m))  # o OpenDRIVE lista da borda para o centro
        direita.append(_acostamento(min(ids_dir, default=0) - 1, largura_m))
        secoes += 1
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(raiz, encoding="unicode"), secoes


def sem_georreferencia_ilegivel(texto: str) -> tuple[str, bool]:
    """Tira o geoReference que o CARLA não lê (ele só entende +lat_0/+lon_0; o resto vira aviso e (0, 0)).

    O projeto não usa a geolocalização do CARLA: as coordenadas saem do próprio .xodr (simulacao/projecao).
    """
    if "<geoReference" not in texto:
        return texto, False
    raiz = ET.fromstring(texto)
    cabecalho = raiz.find("header")
    geo = cabecalho.find("geoReference") if cabecalho is not None else None
    if geo is None or "+lat_0" in (geo.text or ""):
        return texto, False
    cabecalho.remove(geo)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(raiz, encoding="unicode"), True


def preparar_para_carla(texto: str, cfg_opendrive: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    """O texto que vai para generate_opendrive_world e o resumo dos ajustes (config: carla.opendrive)."""
    cfg = cfg_opendrive or {}
    texto, juncoes = colapsar_juncoes_de_passagem(texto, float(cfg.get("colapsar_conectores_m", 1.0)))
    largura = float(cfg.get("acostamento_m", 2.0))
    texto, secoes = acrescentar_acostamentos(texto, largura)
    texto, sem_geo = sem_georreferencia_ilegivel(texto)
    return texto, {"juncoes_colapsadas": juncoes, "acostamento_m": largura if secoes else 0.0, "secoes_com_acostamento": secoes,
                   "georreferencia_removida": sem_geo}


__all__ = ["acrescentar_acostamentos", "colapsar_juncoes_de_passagem", "preparar_para_carla"]
