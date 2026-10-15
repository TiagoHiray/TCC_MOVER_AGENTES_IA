"""Gera o plano de missão da volta (config/missao.yaml) só a partir do mapa do CARLA e do OSM.

Uso, a partir da pasta src/:
    python -m mover.autonomo.gerar_missao             # escreve config/missao.yaml (e uma figura)
    python -m mover.autonomo.rota_mapa --figura vias.png   # ids das vias, para mudar o itinerário

O que sai de cada fonte (nenhuma gravação de celular entra no plano):
- mapa do CARLA (mapa_final.xodr): a rota. O itinerário (gerar_missao.rota.vias no
  config/autonomo.yaml) diz por quais vias a volta passa; os conectores das junções são achados
  no grafo de faixas, e a rota segue o centro das faixas no sentido de tráfego (anti-horário no campus_maua.xodr);
- geometria da rota: zonas de curva, com o limite dado pela aceleração lateral máxima;
- objetos do mapa (campus_maua.xodr): lombadas (speedBump) e faixas de pedestre (crosswalk);
- OSM do campus: portões (barrier=gate) perto da rota e, se o mapa não tiver crosswalk, as faixas;
- config/autonomo.yaml: lombadas como pontos fixos, só para mapas sem speedBump (o mapa_final.xodr).

O arquivo gerado é editável à mão: zonas e paradas podem ser mudadas sem rodar isto de novo.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from mover.autonomo.missao import Parada, PlanoMissao, Zona
from mover.autonomo.rota import Rota
from mover.autonomo.rota_mapa import rota_do_mapa
from mover.config import caminho, carregar_yaml
from mover.simulacao.opendrive import ler_xodr, linha_de_referencia
from mover.simulacao.projecao import criar_projecao

log = logging.getLogger("mover.autonomo")

PADRAO = {
    "cruzeiro_kmh": 30.0,
    "rota": {"vias": [], "largada": None, "passo_m": 1.0, "suavizar_m": 4.0, "tolerancia_m": 1.0},
    "lombadas": [],
    "portoes": [],
    "lombada": {"v_max_kmh": 10.0, "meia_largura_m": 8.0, "distancia_max_m": 10.0},
    "faixa": {"v_max_kmh": 15.0, "meia_largura_m": 6.0, "distancia_max_m": 10.0},
    "portao": {"duracao_s": 4.0, "distancia_max_m": 10.0},
    "curva": {"acel_lateral_max_mps2": 1.6, "v_min_kmh": 10.0, "folga_kmh": 2.0, "antes_m": 5.0, "juntar_m": 10.0},
}


def _mesclar(padrao: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    saida = dict(padrao)
    for chave, valor in (cfg or {}).items():
        saida[chave] = {**padrao[chave], **valor} if isinstance(padrao.get(chave), dict) and isinstance(valor, dict) else valor
    return saida


def sentido_da_volta(rota: Rota) -> str:
    """Horário ou anti-horário, pela área com sinal do laço (x para leste, y para o norte)."""
    area = 0.5 * float(np.sum(rota.x * np.roll(rota.y, -1) - np.roll(rota.x, -1) * rota.y))
    return "anti-horário" if area > 0 else "horário"


# ---------------------------------------------------------------------------------------------
# Elementos do plano
# ---------------------------------------------------------------------------------------------
def objetos_do_mapa(arquivo: Path, tipo: str) -> list[dict[str, Any]]:
    """Objetos OpenDRIVE de um tipo (speedBump, crosswalk...) com a posição (x, y) no mapa."""
    raiz = ET.parse(arquivo).getroot()
    saida = []
    for via in raiz.findall("road"):
        objetos = [o for o in via.iter("object") if o.get("type") == tipo]
        if not objetos:
            continue
        s_ref, x_ref, y_ref, rumo = linha_de_referencia(via.find("planView"))
        for obj in objetos:
            s, t = float(obj.get("s", 0.0)), float(obj.get("t", 0.0))
            h = float(np.interp(s, s_ref, rumo))
            x = float(np.interp(s, s_ref, x_ref)) - t * math.sin(h)
            y = float(np.interp(s, s_ref, y_ref)) + t * math.cos(h)
            saida.append({"x": x, "y": y, "nome": obj.get("name") or obj.get("id"), "via": via.get("id")})
    return saida


def zonas_de_lombada(rota: Rota, pontos: list[Any], cfg: dict[str, Any], origem: str = "config/autonomo.yaml") -> list[Zona]:
    """Lombadas (x, y no mapa), numeradas na ordem em que a volta passa por elas."""
    projetadas = []
    for ponto in pontos:
        x, y = (float(ponto["x"]), float(ponto["y"])) if isinstance(ponto, dict) else (float(ponto[0]), float(ponto[1]))
        s, lateral = rota.projetar(x, y)
        nome = ponto.get("nome") if isinstance(ponto, dict) else None
        if abs(lateral) > cfg["distancia_max_m"]:
            # objetos do mapa fora da volta são normais (outras vias); ponto do config fora da rota é erro de cadastro
            (log.debug if nome else log.warning)("Lombada em (%.1f, %.1f) fica a %.1f m da rota: ignorada.", x, y, abs(lateral))
            continue
        projetadas.append((s, x, y, nome))
    zonas = []
    meia = cfg["meia_largura_m"]
    for k, (s, x, y, nome) in enumerate(sorted(projetadas), start=1):
        descricao = f"objeto {nome} do mapa" if nome else f"lombada do campus cadastrada no {origem}"
        zonas.append(Zona(f"lombada_{k}", "lombada", round((s - meia) % rota.comprimento, 1),
                          round((s + meia) % rota.comprimento, 1), cfg["v_max_kmh"], f"lombada {k}",
                          f"{descricao} em ({x:.1f}, {y:.1f})"))
    return zonas


def zonas_de_faixa_do_mapa(rota: Rota, objetos: list[dict[str, Any]], cfg: dict[str, Any]) -> list[Zona]:
    projetadas = []
    for obj in objetos:
        s, lateral = rota.projetar(obj["x"], obj["y"])
        if abs(lateral) <= cfg["distancia_max_m"]:
            projetadas.append((s, obj["nome"], abs(lateral)))
    zonas = []
    meia = cfg["meia_largura_m"]
    for k, (s, nome, lateral) in enumerate(sorted(projetadas), start=1):
        zonas.append(Zona(f"faixa_{k}", "faixa", round((s - meia) % rota.comprimento, 1),
                          round((s + meia) % rota.comprimento, 1), cfg["v_max_kmh"], f"faixa de pedestre {k}",
                          f"objeto {nome} do mapa, a {lateral:.1f} m da rota"))
    return zonas


def nos_osm(texto_osm: str, chave: str, valor: str) -> list[tuple[str, float, float, dict[str, str]]]:
    saida = []
    for m in re.finditer(r'<node id="(\d+)"[^>]*?lat="([-\d.]+)" lon="([-\d.]+)"[^>]*>(.*?)</node>', texto_osm, re.S):
        tags = dict(re.findall(r'<tag k="([^"]+)" v="([^"]+)"', m.group(4)))
        if tags.get(chave) == valor:
            saida.append((m.group(1), float(m.group(2)), float(m.group(3)), tags))
    return saida


def projetor_mapa(georeferencia: str, offset: tuple[float, ...] | None, mapa_pontos: np.ndarray,
                  amostra_latlon: list[tuple[float, float]]):
    """(lat, lon) -> (x, y) do mapa. O sinal do offset varia entre ferramentas: vale o que cai nas vias."""
    projetar, _ = criar_projecao(georeferencia)
    ox, oy = (offset[0], offset[1]) if offset else (0.0, 0.0)
    from scipy.spatial import cKDTree

    arvore = cKDTree(mapa_pontos)
    melhor, melhor_erro = 1.0, math.inf
    for sinal in (-1.0, 1.0):
        pts = [projetar(lat, lon) for lat, lon in amostra_latlon]
        xy = np.array([(float(e) + sinal * ox, float(n) + sinal * oy) for e, n in pts])
        erro = float(np.median(arvore.query(xy)[0])) if len(xy) else math.inf
        if erro < melhor_erro:
            melhor, melhor_erro = sinal, erro

    def para_mapa(lat: float, lon: float) -> tuple[float, float]:
        e, n = projetar(lat, lon)
        return float(e) + melhor * ox, float(n) + melhor * oy

    return para_mapa, melhor_erro


def zonas_de_faixa(rota: Rota, nos: list[tuple[str, float, float, dict[str, str]]], para_mapa, cfg: dict[str, Any]) -> list[Zona]:
    zonas = []
    for no_id, lat, lon, tags in nos:
        x, y = para_mapa(lat, lon)
        s, lateral = rota.projetar(x, y)
        if abs(lateral) > cfg["distancia_max_m"]:
            continue
        k = len(zonas) + 1
        meia = cfg["meia_largura_m"]
        tipo_faixa = tags.get("crossing", "marcada")
        zonas.append(Zona(f"faixa_{k}", "faixa", round((s - meia) % rota.comprimento, 1),
                          round((s + meia) % rota.comprimento, 1), cfg["v_max_kmh"], f"faixa de pedestre {k}",
                          f"OSM nó {no_id} (crossing={tipo_faixa}), a {abs(lateral):.1f} m da rota"))
    return zonas


def paradas_de_portao(rota: Rota, nos: list[tuple[str, float, float, dict[str, str]]], para_mapa, cfg: dict[str, Any]) -> list[Parada]:
    paradas = []
    for no_id, lat, lon, _tags in nos:
        x, y = para_mapa(lat, lon)
        s, lateral = rota.projetar(x, y)
        if abs(lateral) > cfg["distancia_max_m"]:
            continue
        k = len(paradas) + 1
        paradas.append(Parada(f"portao_{k}", round(s, 1), cfg["duracao_s"], f"portão {k}",
                              f"OSM nó {no_id} (barrier=gate), a {abs(lateral):.1f} m da rota"))
    return paradas


def zonas_de_curva(rota: Rota, cruzeiro_kmh: float, cfg: dict[str, Any]) -> list[Zona]:
    """Trechos onde a aceleração lateral passaria do limite no cruzeiro: limite = sqrt(a_lat / |k|)."""
    kappa = np.abs(rota.curvatura)
    v_curva = np.sqrt(cfg["acel_lateral_max_mps2"] / np.maximum(kappa, 1e-6)) * 3.6
    apertada = v_curva < cruzeiro_kmh - cfg["folga_kmh"]
    if not apertada.any():
        return []
    # trechos contíguos (o laço é fechado: começa num ponto folgado)
    inicio = int(np.flatnonzero(~apertada)[0]) if (~apertada).any() else 0
    ordem = np.roll(np.arange(rota.n), -inicio)
    trechos, atual = [], []
    for i in ordem:
        if apertada[i]:
            atual.append(i)
        elif atual:
            trechos.append(atual)
            atual = []
    if atual:
        trechos.append(atual)
    juntos: list[list[int]] = []
    for trecho in trechos:
        if juntos and ((trecho[0] - juntos[-1][-1]) % rota.n) * rota.passo <= cfg["juntar_m"]:
            juntos[-1] = juntos[-1] + trecho
        else:
            juntos.append(trecho)
    zonas = []
    for trecho in juntos:
        # degrau lateral numa junção (a rota troca de faixa sem virar): curvatura alta, giro total pequeno
        giro = abs(float(np.sum(rota.curvatura[trecho]) * rota.passo))
        if math.degrees(giro) < cfg.get("giro_minimo_graus", 25.0):
            continue
        k = len(zonas) + 1
        limite = max(cfg["v_min_kmh"], math.floor(float(v_curva[trecho].min())))
        s_ini = (rota.s[trecho[0]] - cfg["antes_m"]) % rota.comprimento
        s_fim = (rota.s[trecho[-1]] + rota.passo) % rota.comprimento
        raio = 1.0 / float(kappa[trecho].max())
        zonas.append(Zona(f"curva_{k}", "curva", round(s_ini, 1), round(s_fim, 1), float(limite), f"curva {k}",
                          f"raio mínimo {raio:.0f} m, giro de {math.degrees(giro):.0f}° "
                          f"(aceleração lateral até {cfg['acel_lateral_max_mps2']} m/s²)"))
    return zonas


# ---------------------------------------------------------------------------------------------
# Programa
# ---------------------------------------------------------------------------------------------
def gerar(cfg_sim: dict[str, Any], cfg_missao: dict[str, Any], texto_osm: str = "") -> PlanoMissao:
    cm = _mesclar(PADRAO, cfg_missao)
    cr = cm["rota"]
    if not cr.get("vias"):
        raise SystemExit("Falta o itinerário da volta (gerar_missao.rota.vias no config/autonomo.yaml). "
                         "Veja os ids das vias com: python -m mover.autonomo.rota_mapa --figura vias.png")
    arquivo_mapa = cfg_sim["mapa"]["arquivo"]
    mapa = ler_xodr(caminho(arquivo_mapa), 0.5, tuple(cfg_sim["mapa"].get("tipos_faixa", ("driving",))))
    rota, sequencia = rota_do_mapa(mapa, cr["vias"], tuple(cr["largada"]) if cr.get("largada") else None,
                                   float(cr["passo_m"]), float(cr["suavizar_m"]), float(cr["tolerancia_m"]))
    sentido = sentido_da_volta(rota)
    conectores = [v for v, _ in sequencia if v not in set(int(x) for x in cr["vias"])]
    observacoes = [
        f"Rota tirada só do mapa do CARLA ({Path(arquivo_mapa).name}): itinerário de {len(cr['vias'])} vias "
        f"e {len(conectores)} conectores de junção ({', '.join(map(str, conectores)) or 'nenhum'}), no centro das "
        f"faixas e no sentido de tráfego ({sentido}).",
    ]
    lombadas_mapa = objetos_do_mapa(caminho(arquivo_mapa), "speedBump")
    faixas_mapa = objetos_do_mapa(caminho(arquivo_mapa), "crosswalk")
    if lombadas_mapa:
        zonas = zonas_de_lombada(rota, lombadas_mapa, cm["lombada"])
        observacoes.append(f"Lombadas e faixas de pedestre são objetos do mapa (speedBump: {len(lombadas_mapa)}, "
                           f"crosswalk: {len(faixas_mapa)}; entram as que ficam na rota). Portões: OSM do campus ou config/autonomo.yaml.")
    else:
        zonas = zonas_de_lombada(rota, cm["lombadas"], cm["lombada"])
        observacoes.append("Nenhuma gravação de celular entra no plano: as lombadas são pontos fixos do config/autonomo.yaml, "
                           "e as faixas de pedestre e os portões vêm do OSM do campus.")
    if faixas_mapa:
        zonas += zonas_de_faixa_do_mapa(rota, faixas_mapa, cm["faixa"])
    paradas: list[Parada] = []
    if mapa.georeferencia and texto_osm:
        portoes = nos_osm(texto_osm, "barrier", "gate")
        vias = [(float(m.group(1)), float(m.group(2))) for m in re.finditer(r'lat="([-\d.]+)" lon="([-\d.]+)"', texto_osm)]
        para_mapa, erro = projetor_mapa(mapa.georeferencia, mapa.offset, mapa.pontos, vias[:400])
        observacoes.append(f"OSM projetado no mapa pela georreferência do .xodr (mediana {erro:.1f} m até a via mais próxima).")
        if not faixas_mapa:
            zonas += zonas_de_faixa(rota, nos_osm(texto_osm, "highway", "crossing"), para_mapa, cm["faixa"])
        paradas += paradas_de_portao(rota, portoes, para_mapa, cm["portao"])
    if not paradas:
        for x, y in cm["portoes"]:
            s, lateral = rota.projetar(float(x), float(y))
            if abs(lateral) <= cm["portao"]["distancia_max_m"]:
                k = len(paradas) + 1
                paradas.append(Parada(f"portao_{k}", round(s, 1), cm["portao"]["duracao_s"], f"portão {k}",
                                      f"portão cadastrado no config/autonomo.yaml em ({x:.1f}, {y:.1f})"))
    zonas += zonas_de_curva(rota, float(cm["cruzeiro_kmh"]), cm["curva"])
    zonas.sort(key=lambda z: z.s_ini)
    paradas.sort(key=lambda p: p.s)
    return PlanoMissao(rota=rota, cruzeiro_kmh=float(cm["cruzeiro_kmh"]), zonas=zonas, paradas=paradas,
                       mapa=str(arquivo_mapa), sentido=sentido, observacoes=observacoes)


def salvar_figura(plano: PlanoMissao, destino: Path, mapa_pontos: np.ndarray | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7), dpi=130)
    if mapa_pontos is not None:
        ax.scatter(mapa_pontos[:, 0], mapa_pontos[:, 1], s=0.3, c="#bbbbbb", label="faixas do mapa")
    r = plano.rota
    ax.plot(np.append(r.x, r.x[0]), np.append(r.y, r.y[0]), c="#1f77b4", lw=1.5, label=f"rota ({r.comprimento:.0f} m)")
    cores = {"lombada": "#d62728", "faixa": "#ff7f0e", "curva": "#9467bd"}
    for z in plano.zonas:
        s = np.arange(z.s_ini, z.s_ini + ((z.s_fim - z.s_ini) % r.comprimento), 1.0)
        pts = [r.ponto(v) for v in s]
        ax.plot([p.x for p in pts], [p.y for p in pts], c=cores.get(z.tipo, "k"), lw=4, alpha=0.8)
        p = r.ponto((z.s_ini + ((z.s_fim - z.s_ini) % r.comprimento) / 2))
        ax.annotate(f"{z.id} ({z.v_max_kmh:.0f})", (p.x, p.y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    for p in plano.paradas:
        q = r.ponto(p.s)
        ax.scatter([q.x], [q.y], marker="s", s=60, c="k")
        ax.annotate(p.id, (q.x, q.y), fontsize=7, xytext=(4, -10), textcoords="offset points")
    p0 = r.ponto(0.0)
    ax.annotate("largada", (p0.x, p0.y), fontsize=8, weight="bold", xytext=(6, 6), textcoords="offset points")
    seta = r.ponto(15.0)
    ax.annotate("", (seta.x, seta.y), (p0.x, p0.y), arrowprops={"arrowstyle": "->", "lw": 1.5})
    for tipo, cor in cores.items():
        ax.plot([], [], c=cor, lw=4, label=tipo)
    ax.set_aspect("equal")
    ax.set_title(f"Plano de missão: {len(plano.zonas)} zonas, {len(plano.paradas)} paradas, cruzeiro {plano.cruzeiro_kmh:.0f} km/h")
    ax.legend(loc="best", fontsize=7)
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(destino)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config/simulacao.yaml")
    parser.add_argument("--config-autonomo", default="config/autonomo.yaml")
    parser.add_argument("--osm", default="data/openstreetviewmap.osm")
    parser.add_argument("--saida", default=None, help="padrão: missao.arquivo do config/autonomo.yaml")
    parser.add_argument("--sem-figura", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    cfg_sim = carregar_yaml(args.config)
    cfg_auto = carregar_yaml(args.config_autonomo)
    texto_osm = caminho(args.osm).read_text(encoding="utf-8") if caminho(args.osm).exists() else ""
    if not texto_osm:
        log.warning("OSM não encontrado (%s): portões só os do config/autonomo.yaml (e faixas só as do mapa).", args.osm)
    plano = gerar(cfg_sim, cfg_auto.get("gerar_missao") or {}, texto_osm)
    saida = caminho(args.saida or cfg_auto["missao"]["arquivo"])
    cabecalho = ("# Plano de missão da volta autônoma (gerado por: python -m mover.autonomo.gerar_missao), só a partir\n"
                 "# do mapa do CARLA, do OSM e do config/autonomo.yaml. Editável à mão: zonas (s_ini, s_fim em m ao longo\n"
                 "# da rota; v_max_kmh) e paradas (s, duracao_s).")
    plano.salvar(saida, cabecalho)
    log.info("Plano de missão: %s", json.dumps(plano.resumo(), ensure_ascii=False))
    for z in plano.zonas:
        log.info("  %-10s %-8s s %6.1f-%6.1f m  %4.0f km/h  (%s)", z.id, z.tipo, z.s_ini, z.s_fim, z.v_max_kmh, z.origem)
    for p in plano.paradas:
        log.info("  %-10s parada   s %6.1f m  %4.0f s     (%s)", p.id, p.s, p.duracao_s, p.origem)
    log.info("Gravado %s", saida)
    if not args.sem_figura:
        figura = saida.with_suffix(".png")
        mapa = ler_xodr(caminho(cfg_sim["mapa"]["arquivo"]), 1.0)
        salvar_figura(plano, figura, mapa.pontos)
        log.info("Figura: %s", figura)


if __name__ == "__main__":
    main()
