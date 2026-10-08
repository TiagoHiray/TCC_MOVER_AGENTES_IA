"""Leitura leve de mapas OpenDRIVE (.xodr), sem precisar do CARLA.

Serve para conferir, antes de abrir o simulador, se a trajetória gravada cabe nas vias do
mapa e para calcular o alinhamento entre o referencial local da Etapa 1 e o do mapa
(alinhamento.py).

Cobre o que o Osm2Odr do CARLA (netconvert do SUMO) e os editores comuns geram:
- planView: line, arc, spiral, poly3 e paramPoly3;
- lanes: laneOffset e laneSection com faixas à esquerda e à direita (larguras polinomiais);
- header: geoReference (proj4) e offset. O netconvert escreve o offset quando o mapa é
  centralizado (center_map, padrão do Osm2Odr): coordenada do mapa = projetada + offset.

As coordenadas devolvidas estão no referencial do OpenDRIVE (x = leste, y = norte, destro).
O CARLA inverte o y: (x, y)_CARLA = (x, -y)_OpenDRIVE.
"""

from __future__ import annotations

import logging
import math
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger("mover.simulacao")

PASSO_FINO_M = 0.1  # resolução usada para integrar as curvas antes de reamostrar

# Faixas onde o caminhão pode estar. O Osm2Odr transforma as vias "service" do OSM (as do
# campus) em faixas "restricted"; o CARLA 0.9.16 gera asfalto para todas as faixas
# (Map::GenerateChunkedMesh), então elas contam como pista.
TIPOS_PADRAO: tuple[str, ...] = ("driving", "restricted", "bidirectional", "parking")


@dataclass
class MapaOpenDrive:
    """Centros das faixas e metadados de um .xodr."""

    georeferencia: str | None
    offset: tuple[float, float, float, float] | None  # x, y, z, hdg do <header><offset>
    pontos: np.ndarray                                 # (N, 2) centros das faixas escolhidas
    rumos: np.ndarray                                  # (N,) sentido de tráfego (rad, anti-horário a partir de x)
    vias: np.ndarray                                   # (N,) id da via de cada ponto
    larguras: np.ndarray                               # (N,) largura da faixa em cada ponto (m)
    n_vias: int = 0
    tipos_faixa: Counter = field(default_factory=Counter)
    avisos: list[str] = field(default_factory=list)

    @property
    def limites(self) -> tuple[float, float, float, float]:
        """(x_min, y_min, x_max, y_max) dos centros de faixa."""
        return (*self.pontos.min(axis=0), *self.pontos.max(axis=0))


# ---------------------------------------------------------------------------------------------
# Geometria da linha de referência
# ---------------------------------------------------------------------------------------------
def _f(no: ET.Element, nome: str, padrao: float = 0.0) -> float:
    valor = no.get(nome)
    return padrao if valor in (None, "") else float(valor)


def _curva_local(filho: ET.Element, comprimento: float) -> tuple[np.ndarray, np.ndarray]:
    """Pontos (u, v) da geometria no referencial local (origem no início, u no rumo inicial)."""
    tipo = filho.tag
    n = max(2, int(math.ceil(comprimento / PASSO_FINO_M)) + 1)
    if tipo == "line":
        u = np.linspace(0.0, comprimento, n)
        return u, np.zeros_like(u)
    if tipo in ("arc", "spiral"):
        s = np.linspace(0.0, comprimento, n)
        if tipo == "arc":
            rumo = _f(filho, "curvature") * s
        else:
            k0, k1 = _f(filho, "curvStart"), _f(filho, "curvEnd")
            rumo = k0 * s + 0.5 * (k1 - k0) / max(comprimento, 1e-9) * s**2
        # integração trapezoidal de (cos, sin) do rumo ao longo do comprimento de arco
        du = np.concatenate([[0.0], np.cumsum(0.5 * (np.cos(rumo[1:]) + np.cos(rumo[:-1])) * np.diff(s))])
        dv = np.concatenate([[0.0], np.cumsum(0.5 * (np.sin(rumo[1:]) + np.sin(rumo[:-1])) * np.diff(s))])
        return du, dv
    if tipo == "poly3":
        a, b, c, d = (_f(filho, k) for k in "abcd")
        u = np.linspace(0.0, comprimento, n)  # u <= s, então u = comprimento cobre a curva inteira
        return u, a + b * u + c * u**2 + d * u**3
    if tipo == "paramPoly3":
        coef_u = [_f(filho, k) for k in ("aU", "bU", "cU", "dU")]
        coef_v = [_f(filho, k) for k in ("aV", "bV", "cV", "dV")]
        # Como no CARLA 0.9.16 (MapBuilder.cpp): só pRange="arcLength" usa p em metros; ausente = normalizado
        p_max = comprimento if filho.get("pRange") == "arcLength" else 1.0
        p = np.linspace(0.0, p_max, n)
        u = coef_u[0] + coef_u[1] * p + coef_u[2] * p**2 + coef_u[3] * p**3
        v = coef_v[0] + coef_v[1] * p + coef_v[2] * p**2 + coef_v[3] * p**3
        return u, v
    raise ValueError(f"geometria OpenDRIVE não suportada: <{tipo}>")


def _cortar_por_comprimento(u: np.ndarray, v: np.ndarray, comprimento: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Comprimento de arco acumulado; no poly3 corta a curva onde o arco atinge o comprimento."""
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(u), np.diff(v)))])
    if s[-1] > comprimento * 1.001:
        fim = int(np.searchsorted(s, comprimento)) + 1
        u, v, s = u[:fim], v[:fim], s[:fim]
    return u, v, s


def linha_de_referencia(plan_view: ET.Element) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """s, x, y e rumo (rad) ao longo da linha de referência de uma via, com passo fino."""
    partes_s, partes_x, partes_y = [], [], []
    for geo in plan_view.findall("geometry"):
        filho = next(iter(geo), None)
        if filho is None:
            continue
        s0, x0, y0, h0, comp = (_f(geo, k) for k in ("s", "x", "y", "hdg", "length"))
        if comp <= 0:
            continue
        u, v = _curva_local(filho, comp)
        u, v, s_local = _cortar_por_comprimento(u, v, comp)
        if s_local[-1] > 0:
            s_local = s_local * (comp / s_local[-1])  # fecha exatamente no comprimento declarado
        cos_h, sin_h = math.cos(h0), math.sin(h0)
        partes_s.append(s0 + s_local)
        partes_x.append(x0 + u * cos_h - v * sin_h)
        partes_y.append(y0 + u * sin_h + v * cos_h)
    if not partes_s:
        vazio = np.zeros(0)
        return vazio, vazio, vazio, vazio
    s, x, y = (np.concatenate(p) for p in (partes_s, partes_x, partes_y))
    ordem = np.argsort(s, kind="stable")
    s, x, y = s[ordem], x[ordem], y[ordem]
    rumo = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x))) if s.size > 1 else np.zeros_like(s)
    return s, x, y, rumo


# ---------------------------------------------------------------------------------------------
# Faixas
# ---------------------------------------------------------------------------------------------
def _polinomio_por_trecho(registros: list[tuple[float, float, float, float, float]], s: np.ndarray) -> np.ndarray:
    """Avalia registros (s_inicio, a, b, c, d) ordenados: vale o último com s_inicio <= s."""
    saida = np.zeros_like(s)
    if not registros:
        return saida
    inicios = np.array([r[0] for r in registros])
    idx = np.clip(np.searchsorted(inicios, s, side="right") - 1, 0, len(registros) - 1)
    for i, (s_ini, a, b, c, d) in enumerate(registros):
        sel = idx == i
        ds = s[sel] - s_ini
        saida[sel] = a + b * ds + c * ds**2 + d * ds**3
    return saida


def _larguras(lane: ET.Element, s_secao: float) -> list[tuple[float, float, float, float, float]]:
    registros = [(s_secao + _f(w, "sOffset"), _f(w, "a"), _f(w, "b"), _f(w, "c"), _f(w, "d")) for w in lane.findall("width")]
    return sorted(registros)


def _centros_das_faixas(road: ET.Element, passo_m: float, tipos: set[str], contagem: Counter, avisos: list[str]):
    comprimento = _f(road, "length")
    plan_view = road.find("planView")
    lanes = road.find("lanes")
    if plan_view is None or lanes is None or comprimento <= 0:
        return []
    s_fino, x_fino, y_fino, rumo_fino = linha_de_referencia(plan_view)
    if s_fino.size < 2:
        return []
    deslocamentos = sorted((_f(o, "s"), _f(o, "a"), _f(o, "b"), _f(o, "c"), _f(o, "d")) for o in lanes.findall("laneOffset"))
    secoes = sorted(lanes.findall("laneSection"), key=lambda sec: _f(sec, "s"))
    resultado = []
    for i, secao in enumerate(secoes):
        s_ini = _f(secao, "s")
        s_fim = _f(secoes[i + 1], "s") if i + 1 < len(secoes) else comprimento
        if s_fim - s_ini <= 1e-6:
            continue
        n = max(2, int(math.ceil((s_fim - s_ini) / passo_m)) + 1)
        s = np.linspace(s_ini, s_fim, n)
        if i + 1 < len(secoes):
            s = s[:-1]  # o fim da seção é o início da próxima
        x = np.interp(s, s_fino, x_fino)
        y = np.interp(s, s_fino, y_fino)
        rumo = np.interp(s, s_fino, rumo_fino)
        normal_x, normal_y = -np.sin(rumo), np.cos(rumo)  # t > 0 à esquerda da linha de referência
        base = _polinomio_por_trecho(deslocamentos, s)
        for lado, sinal in (("left", 1.0), ("right", -1.0)):
            no_lado = secao.find(lado)
            if no_lado is None:
                continue
            faixas = sorted(no_lado.findall("lane"), key=lambda ln: abs(int(ln.get("id", "0"))))
            acumulado = np.zeros_like(s)
            for lane in faixas:
                tipo = lane.get("type", "none")
                contagem[tipo] += 1
                registros = _larguras(lane, s_ini)
                if not registros and lane.find("border") is not None:
                    avisos.append(f"via {road.get('id')}: faixa {lane.get('id')} usa <border>, não suportado (largura 0)")
                largura = _polinomio_por_trecho(registros, s)
                t = base + sinal * (acumulado + largura / 2)
                acumulado = acumulado + largura
                if tipo in tipos:
                    # Tráfego pela direita: faixas à direita (id < 0) seguem o sentido de s
                    sentido = rumo if sinal < 0 else rumo + math.pi
                    resultado.append((x + t * normal_x, y + t * normal_y, sentido, largura))
    return resultado


# ---------------------------------------------------------------------------------------------
# Leitura do arquivo
# ---------------------------------------------------------------------------------------------
def ler_xodr(fonte: str | Path, passo_m: float = 1.0, tipos_faixa: tuple[str, ...] = TIPOS_PADRAO) -> MapaOpenDrive:
    """Lê um .xodr (caminho ou texto XML) e amostra os centros das faixas a cada `passo_m`."""
    texto = Path(fonte).read_text(encoding="utf-8") if not str(fonte).lstrip().startswith("<") else str(fonte)
    raiz = ET.fromstring(texto)
    header = raiz.find("header")
    georeferencia, offset = None, None
    if header is not None:
        geo = header.find("geoReference")
        if geo is not None and (geo.text or "").strip():
            georeferencia = " ".join(geo.text.split())
        off = header.find("offset")
        if off is not None:
            offset = (_f(off, "x"), _f(off, "y"), _f(off, "z"), _f(off, "hdg"))

    contagem: Counter = Counter()
    avisos: list[str] = []
    xs, ys, rumos, vias, larguras = [], [], [], [], []
    roads = raiz.findall("road")
    for road in roads:
        id_via = int(road.get("id", "-1")) if road.get("id", "").lstrip("-").isdigit() else -1
        for x, y, sentido, largura in _centros_das_faixas(road, passo_m, set(tipos_faixa), contagem, avisos):
            xs.append(x)
            ys.append(y)
            rumos.append(sentido)
            larguras.append(largura)
            vias.append(np.full(x.size, id_via))
    if not xs:
        raise ValueError(f"nenhuma faixa dos tipos {tipos_faixa} encontrada no .xodr "
                         f"(tipos presentes: {dict(contagem)})")
    pontos = np.column_stack([np.concatenate(xs), np.concatenate(ys)])
    mapa = MapaOpenDrive(georeferencia, offset, pontos, np.mod(np.concatenate(rumos), 2 * math.pi),
                         np.concatenate(vias), np.concatenate(larguras), len(roads), contagem, avisos)
    log.info("Mapa OpenDRIVE: %d vias, %d pontos de centro de faixa (%s); geoReference: %s; offset: %s",
             mapa.n_vias, len(pontos), ", ".join(tipos_faixa), georeferencia or "ausente", offset)
    return mapa


def para_carla(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OpenDRIVE (y para o norte) -> CARLA (y para o sul)."""
    return np.asarray(x, float), -np.asarray(y, float)
