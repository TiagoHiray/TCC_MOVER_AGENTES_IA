"""Alinhamento da volta gravada às vias do mapa OpenDRIVE (não precisa do CARLA).

1. Georreferência: projeta (gnss_lat, gnss_lon) da telemetria com o geoReference do mapa e
   aplica o <offset> do header. O sinal do offset muda de um conversor para outro (Osm2Odr do
   CARLA: local = projetado + offset; netconvert do SUMO 1.27, usado no mapa do Eduardo:
   local = projetado - offset), então os dois são testados e fica o que cai sobre as vias.
   Desses pontos sai uma similaridade (escala, rotação, translação) do plano local da Etapa 1
   para o plano do mapa. A rotação é a convergência meridiana da projeção do mapa (~0,6° no
   UTM 23 do campus) e a escala é ~1, a não ser que o mapa use "+proj=tmerc" sem lon_0.
2. ICP rígido ponto-a-reta: minimiza a distância lateral ao centro da faixa, com pesos de
   Cauchy para que os trechos onde a volta foge do desenho da via pesem pouco. Corrige
   deslocamentos sistemáticos de alguns metros entre o GPS e o desenho das vias. A rotação e a
   translação são limitadas no YAML.
3. Correção de borda: em cada quadro, o deslocamento lateral (perpendicular à faixa) tem de
   deixar o centro do caminhão a no máximo meia largura - margem do centro da faixa. Entre os
   deslocamentos que cumprem isso fica o menor e mais suave (um problema quadrático resolvido
   por ADMM): onde a volta já está dentro da faixa ele tende a zero, e a correção entra e sai
   em ~2 s. O CSV tratado não é alterado.

Convenções:
- plano local (Etapa 1): x = leste, y = -norte, yaw = rumo - 90 (graus, sentido horário);
- plano do mapa (OpenDRIVE): x = leste, y = norte, ângulos anti-horários;
- CARLA: x_c = x_mapa, y_c = -y_mapa, yaw_c = -(rumo anti-horário no mapa), em graus.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.ndimage import convolve1d
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

from mover.simulacao.opendrive import MapaOpenDrive
from mover.simulacao.projecao import criar_projecao

log = logging.getLogger("mover.simulacao")

PADRAO = {
    "icp": {"ativo": True, "max_rotacao_graus": 3.0, "max_translacao_m": 5.0, "escala_robusta_m": 1.5,
            "dist_max_par_m": 10.0, "max_iteracoes": 50, "subamostragem": 4},
    "peso_rumo_m": 3.0,
    "correcao_borda": {"ativa": True, "margem_m": 0.5, "suavizacao_s": 2.0, "peso_curvatura": 32.0, "dist_max_m": 15.0,
                       "max_desvio_rumo_graus": 25.0},
    "bloco_s": 10.0,
}


def _mesclar(base: dict[str, Any], extra: dict[str, Any] | None) -> dict[str, Any]:
    saida = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for chave, valor in (extra or {}).items():
        if isinstance(valor, dict) and isinstance(saida.get(chave), dict):
            saida[chave] = _mesclar(saida[chave], valor)
        else:
            saida[chave] = valor
    return saida


def _ang(rad: np.ndarray | float) -> np.ndarray | float:
    """Normaliza ângulos para (-pi, pi]."""
    return np.angle(np.exp(1j * np.asarray(rad)))


# ---------------------------------------------------------------------------------------------
# Transformação
# ---------------------------------------------------------------------------------------------
@dataclass
class Similaridade:
    """p_mapa = escala * R(rotacao) * p + (tx, ty)."""

    escala: float = 1.0
    rotacao_rad: float = 0.0
    tx: float = 0.0
    ty: float = 0.0

    def aplicar(self, pontos: np.ndarray) -> np.ndarray:
        c, s = math.cos(self.rotacao_rad), math.sin(self.rotacao_rad)
        rot = np.array([[c, -s], [s, c]])
        return self.escala * np.asarray(pontos, float) @ rot.T + np.array([self.tx, self.ty])

    def depois(self, outra: "Similaridade") -> "Similaridade":
        """Composição: aplica `self` e depois `outra`."""
        t = outra.aplicar(np.array([[self.tx, self.ty]]))[0]
        return Similaridade(self.escala * outra.escala, self.rotacao_rad + outra.rotacao_rad, float(t[0]), float(t[1]))

    def para_json(self) -> dict[str, float]:
        return {"escala": round(self.escala, 8), "rotacao_graus": round(math.degrees(self.rotacao_rad), 6),
                "tx_m": round(self.tx, 4), "ty_m": round(self.ty, 4)}


def ajustar_similaridade(origem: np.ndarray, destino: np.ndarray, com_escala: bool = True) -> Similaridade:
    """Mínimos quadrados de origem -> destino (Umeyama, 1991). Sem escala vira Kabsch."""
    mo, md = origem.mean(axis=0), destino.mean(axis=0)
    a, b = origem - mo, destino - md
    cov = b.T @ a / len(origem)
    u, d, vt = np.linalg.svd(cov)
    sinal = np.diag([1.0, np.sign(np.linalg.det(u @ vt)) or 1.0])
    rot = u @ sinal @ vt
    escala = float(np.trace(np.diag(d) @ sinal) / (a**2).sum(axis=1).mean()) if com_escala else 1.0
    t = md - escala * rot @ mo
    return Similaridade(escala, math.atan2(rot[1, 0], rot[0, 0]), float(t[0]), float(t[1]))


# ---------------------------------------------------------------------------------------------
# Busca nas faixas (posição + direção da via, sem sentido)
# ---------------------------------------------------------------------------------------------
class BuscaFaixas:
    """KD-tree em (x, y, c·cos 2h, c·sin 2h).

    A direção entra com o ângulo dobrado: uma faixa de mão única desenhada no sentido oposto
    ao da volta (caso do mapa do Eduardo) não é penalizada, mas uma via perpendicular custa 2c.
    """

    def __init__(self, mapa: MapaOpenDrive, peso_rumo_m: float):
        self.mapa = mapa
        self.peso = float(peso_rumo_m)
        self._arvore = cKDTree(self._caracteristicas(mapa.pontos, mapa.rumos))

    def _caracteristicas(self, pontos: np.ndarray, rumos: np.ndarray) -> np.ndarray:
        return np.column_stack([pontos, self.peso * np.cos(2 * rumos), self.peso * np.sin(2 * rumos)])

    def consultar(self, pontos: np.ndarray, rumos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Índice da faixa casada, deslocamento lateral com sinal (m, + à esquerda da faixa) e distância."""
        _, idx = self._arvore.query(self._caracteristicas(pontos, rumos))
        q = self.mapa.pontos[idx]
        h = self.mapa.rumos[idx]
        normal = np.column_stack([-np.sin(h), np.cos(h)])
        lateral = ((pontos - q) * normal).sum(axis=1)
        return idx, lateral, np.hypot(*(pontos - q).T)


def _estatisticas(lateral: np.ndarray, meia_largura: np.ndarray) -> dict[str, float]:
    d = np.abs(lateral)
    return {"mediana_m": round(float(np.median(d)), 3), "p95_m": round(float(np.percentile(d, 95)), 3),
            "max_m": round(float(d.max()), 3), "frac_dentro_da_faixa": round(float((d <= meia_largura).mean()), 4)}


# ---------------------------------------------------------------------------------------------
# Etapas
# ---------------------------------------------------------------------------------------------
def _por_georreferencia(mapa: MapaOpenDrive, telem: pd.DataFrame, enu: np.ndarray, busca: BuscaFaixas,
                        rumos_enu: np.ndarray, avisos: list[str]) -> tuple[Similaridade, dict[str, Any]]:
    info: dict[str, Any] = {"metodo": "identidade", "sinal_offset": None, "residuo_max_m": None, "projecao": None}
    tem_gnss = {"gnss_lat", "gnss_lon"} <= set(telem.columns)
    if not mapa.georeferencia or not tem_gnss:
        avisos.append("O mapa não tem geoReference (ou a telemetria não tem gnss_lat/gnss_lon): supondo que o "
                      "plano do mapa é o plano local da Etapa 1. Ajuste com --offset-x/--offset-y/--rotacao.")
        return Similaridade(), info
    try:
        projetar, metodo = criar_projecao(mapa.georeferencia)
    except ValueError as erro:
        avisos.append(f"geoReference não suportado ({erro}); usando o plano local da Etapa 1.")
        return Similaridade(), info
    validos = telem[["gnss_lat", "gnss_lon"]].notna().all(axis=1).to_numpy()
    leste, norte = projetar(telem["gnss_lat"].to_numpy()[validos], telem["gnss_lon"].to_numpy()[validos])
    projetados = np.column_stack([leste, norte])
    candidatos = [0] if mapa.offset is None else [-1, +1, 0]
    melhor = None
    for sinal in candidatos:
        destino = projetados + (sinal * np.array(mapa.offset[:2]) if sinal else 0.0)
        _, lateral, _ = busca.consultar(destino, rumos_enu[validos])
        custo = float(np.median(np.abs(lateral)))
        if melhor is None or custo < melhor[0]:
            melhor = (custo, sinal, destino)
    _, sinal, destino = melhor
    transf = ajustar_similaridade(enu[validos], destino, com_escala=True)
    residuo = np.hypot(*(transf.aplicar(enu[validos]) - destino).T)
    info.update(metodo="georreferencia", sinal_offset=sinal, projecao=metodo,
                residuo_max_m=round(float(residuo.max()), 4))
    if residuo.max() > 0.5:
        avisos.append(f"A similaridade não reproduz a projeção do mapa (resíduo máx. {residuo.max():.2f} m): "
                      "gnss_lat/gnss_lon podem não corresponder a x/y.")
    if abs(transf.escala - 1) > 0.01:
        avisos.append(f"O mapa está em escala {transf.escala:.3f} em relação ao terreno (projeção "
                      f"'{mapa.georeferencia}'). O caminhão vai parecer {abs(transf.escala - 1):.0%} "
                      f"{'mais rápido' if transf.escala > 1 else 'mais lento'} no CARLA.")
    return transf, info


def _icp(pontos: np.ndarray, rumos: np.ndarray, busca: BuscaFaixas, cfg: dict[str, Any]) -> tuple[Similaridade, dict[str, Any]]:
    """ICP rígido ponto-a-reta em torno do centroide: p' = R(θ)(p - c) + c + d.

    Minimiza só a distância lateral ao centro da faixa (a componente ao longo da via não diz
    nada), com pesos de Cauchy para que trechos onde a volta foge do desenho da via (curvas
    cortadas, erro do GPS) pesem pouco. θ e |d| ficam dentro dos limites do YAML.
    """
    centro = pontos.mean(axis=0)
    teta, desloc = 0.0, np.zeros(2)
    max_rot = math.radians(float(cfg["max_rotacao_graus"]))
    max_trans = float(cfg["max_translacao_m"])
    escala = float(cfg["escala_robusta_m"])
    limitado, iteracoes, convergiu = False, 0, False
    for iteracoes in range(1, int(cfg["max_iteracoes"]) + 1):
        c, s = math.cos(teta), math.sin(teta)
        atual = (pontos - centro) @ np.array([[c, -s], [s, c]]).T + centro + desloc
        idx, lateral, dist = busca.consultar(atual, rumos + teta)
        sel = dist <= float(cfg["dist_max_par_m"])
        if sel.sum() < 10:
            break
        h = busca.mapa.rumos[idx[sel]]
        normal = np.column_stack([-np.sin(h), np.cos(h)])
        rel = atual[sel] - centro
        jac = np.column_stack([(normal * np.column_stack([-rel[:, 1], rel[:, 0]])).sum(axis=1), normal])
        raiz_peso = 1.0 / np.sqrt(1.0 + (lateral[sel] / escala) ** 2)
        delta, *_ = np.linalg.lstsq(jac * raiz_peso[:, None], -lateral[sel] * raiz_peso, rcond=None)
        cd, sd = math.cos(delta[0]), math.sin(delta[0])
        novo_teta = teta + float(delta[0])
        novo_desloc = np.array([[cd, -sd], [sd, cd]]) @ desloc + delta[1:]
        if abs(novo_teta) > max_rot:
            novo_teta, limitado = math.copysign(max_rot, novo_teta), True
        if np.hypot(*novo_desloc) > max_trans:
            novo_desloc, limitado = novo_desloc * max_trans / np.hypot(*novo_desloc), True
        mudanca = abs(novo_teta - teta) * 100.0 + np.hypot(*(novo_desloc - desloc))  # 1 mrad ~ 10 cm a 100 m
        teta, desloc = novo_teta, novo_desloc
        if mudanca < 1e-3:
            convergiu = True
            break
    c, s = math.cos(teta), math.sin(teta)
    t = centro + desloc - np.array([[c, -s], [s, c]]) @ centro
    info = {"ativo": True, "rotacao_graus": round(math.degrees(teta), 4), "deslocamento_m": [round(float(v), 3) for v in desloc],
            "iteracoes": iteracoes, "convergiu": convergiu, "chegou_no_limite": limitado}
    return Similaridade(1.0, teta, float(t[0]), float(t[1])), info


def _suavizar(valores: np.ndarray, n_amostras: int) -> np.ndarray:
    if n_amostras < 3:
        return valores
    janela = np.hanning(n_amostras + 2)[1:-1]
    return convolve1d(valores, janela / janela.sum(), axis=0, mode="nearest")


def _matriz_suavidade(n: int, escala: float, peso_curvatura: float = 1.0) -> sparse.csc_matrix:
    """H = I + 2ℓ²·D1ᵀD1 + k·ℓ⁴·D2ᵀD2. Com k = 1 (amortecimento crítico) não há oscilação."""
    d1 = sparse.diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n))
    d2 = sparse.diags([np.ones(n - 2), -2.0 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n))
    return (sparse.identity(n) + 2.0 * escala**2 * (d1.T @ d1) + peso_curvatura * escala**4 * (d2.T @ d2)).tocsc()


def _corredor_suave(inferior: np.ndarray, superior: np.ndarray, escala: float, peso_curvatura: float = 1.0,
                    tolerancia: float = 1e-5, max_iteracoes: int = 5000) -> np.ndarray:
    """O menor e mais suave c com inferior <= c <= superior em todo quadro.

    Minimiza Σc² + 2ℓ²·Σ(Δc)² + k·ℓ⁴·Σ(Δ²c)² (ℓ = `escala`, em amostras; k = `peso_curvatura`)
    por ADMM. Com k = 1, longe das restrições a solução some como (1 + t/ℓ)·e^(-t/ℓ), sem oscilar;
    k maior penaliza mais a curvatura que a correção acrescenta ao caminho (aceleração lateral),
    com rampas mais longas. O resultado é a variável recortada do ADMM, então respeita o
    corredor mesmo se o laço parar antes de convergir.
    """
    n = len(inferior)
    zero = np.clip(np.zeros(n), inferior, superior)
    if n < 3 or not np.any(zero != 0.0):
        return zero
    rho = 2.0 * math.sqrt(peso_curvatura) * escala**2  # algumas centenas de iterações para ℓ de 4 a 12
    fatorada = splu((_matriz_suavidade(n, escala, peso_curvatura) + rho * sparse.identity(n)).tocsc())
    z, u = zero, np.zeros(n)
    for _ in range(max_iteracoes):
        x = fatorada.solve(rho * (z - u))
        z_anterior = z
        z = np.clip(x + u, inferior, superior)
        u += x - z
        if np.abs(x - z).max() < tolerancia and np.abs(z - z_anterior).max() < tolerancia:
            break
    else:
        log.warning("Correção de borda: o ADMM não convergiu em %d iterações; a pose fica dentro da faixa, "
                    "mas a correção pode ficar menos suave.", max_iteracoes)
    return z


def _acel_lateral(pontos: np.ndarray, taxa_hz: float) -> np.ndarray:
    """Aceleração lateral (m/s², + para a esquerda) do caminho amostrado a taxa_hz."""
    v = np.gradient(pontos, 1.0 / taxa_hz, axis=0)
    a = np.gradient(v, 1.0 / taxa_hz, axis=0)
    return (v[:, 0] * a[:, 1] - v[:, 1] * a[:, 0]) / np.maximum(np.hypot(v[:, 0], v[:, 1]), 0.1)


def _rumo_do_trajeto(pontos: np.ndarray) -> np.ndarray:
    v = np.gradient(pontos, axis=0)
    return np.arctan2(v[:, 1], v[:, 0])


def _correcao_borda(pontos: np.ndarray, rumos: np.ndarray, velocidade: np.ndarray, busca: BuscaFaixas,
                    cfg: dict[str, Any], taxa_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Correção (N, 2) que traz o centro do caminhão para dentro da faixa e o desvio de rumo que ela causa.

    Em cada quadro, o deslocamento lateral c (perpendicular à faixa) tem de deixar o centro a no
    máximo meia largura - margem do centro da faixa. Entre todos os c que cumprem isso, fica o
    menor e mais suave (_corredor_suave): onde a volta já está dentro da faixa c tende a zero, e a
    correção entra e sai em ~suavizacao_s (cai para ~4 % a essa distância do trecho fora da faixa).
    As passadas seguintes só limpam o efeito das curvas e das trocas de via.
    """
    n_suave = float(cfg["suavizacao_s"]) * taxa_hz
    escala = max(n_suave / 5.0, 1.0)  # (1 + 5)·e^-5 ≈ 4 %
    peso_curvatura = max(float(cfg.get("peso_curvatura", 1.0)), 1e-3)
    margem, dist_max = float(cfg["margem_m"]), float(cfg["dist_max_m"])
    correcao = np.zeros_like(pontos)
    for _ in range(3):
        idx, lateral, _ = busca.consultar(pontos + correcao, rumos)
        tolerancia = np.maximum(busca.mapa.larguras[idx] / 2 - margem, 0.05)
        na_via = np.abs(lateral) <= dist_max
        if not np.any(na_via & (np.abs(lateral) > tolerancia + 0.005)):
            break
        h = busca.mapa.rumos[idx]
        # lateral e normal em relação ao sentido do movimento: assim a normal não troca de sinal
        # quando o casamento pula para uma via desenhada no sentido oposto (conectores de junção)
        sentido = np.where(np.cos(h - rumos) >= 0.0, 1.0, -1.0)
        lateral = lateral * sentido
        normal = np.column_stack([-np.sin(h), np.cos(h)]) * sentido[:, None]  # à esquerda do movimento
        # normal suavizada em ~suavizacao_s: numa quina do desenho (ou troca de via) a direção da
        # correção gira aos poucos, sem dar uma guinada no caminhão
        normal = _suavizar(normal, max(int(round(n_suave)), 3))
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-6)
        c = _corredor_suave(np.where(na_via, lateral - tolerancia, -np.inf),
                            np.where(na_via, lateral + tolerancia, np.inf), escala, peso_curvatura)
        correcao = correcao - c[:, None] * normal
    desvio = _ang(_rumo_do_trajeto(pontos + correcao) - _rumo_do_trajeto(pontos))
    desvio = np.where(velocidade > 0.5, desvio, 0.0)
    limite = math.radians(float(cfg["max_desvio_rumo_graus"]))
    return correcao, np.clip(_suavizar(desvio, max(int(round(n_suave / 4)), 3)), -limite, limite)


# ---------------------------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------------------------
@dataclass
class ResultadoAlinhamento:
    transformacao: Similaridade
    georreferencia: dict[str, Any]
    icp: dict[str, Any]
    estatisticas: dict[str, dict[str, float]]
    por_bloco: list[dict[str, Any]]
    poses: pd.DataFrame
    avisos: list[str] = field(default_factory=list)
    ajuste_manual: dict[str, float] = field(default_factory=dict)
    suavidade: dict[str, Any] = field(default_factory=dict)

    def para_json(self) -> dict[str, Any]:
        return {"transformacao": self.transformacao.para_json(), "georreferencia": self.georreferencia,
                "ajuste_manual": self.ajuste_manual, "icp": self.icp, "estatisticas": self.estatisticas,
                "suavidade": self.suavidade, "por_bloco": self.por_bloco, "avisos": self.avisos}

    def resumo(self) -> str:
        t = self.transformacao.para_json()
        linhas = [
            f"Transformação local -> mapa: escala {t['escala']:.5f}, rotação {t['rotacao_graus']:.3f}°, "
            f"translação ({t['tx_m']:.2f}, {t['ty_m']:.2f}) m  [{self.georreferencia['metodo']}"
            + (f", sinal do offset {self.georreferencia['sinal_offset']:+d}" if self.georreferencia.get("sinal_offset") else "")
            + "]",
        ]
        if self.icp.get("ativo"):
            linhas.append(f"ICP: rotação {self.icp['rotacao_graus']:+.3f}°, deslocamento {self.icp['deslocamento_m']} m "
                          f"em {self.icp['iteracoes']} iterações" + (" (chegou no limite!)" if self.icp["chegou_no_limite"] else ""))
        for etapa, est in self.estatisticas.items():
            linhas.append(f"Distância ao centro da faixa [{etapa}]: mediana {est['mediana_m']:.2f} m, p95 {est['p95_m']:.2f} m, "
                          f"máx {est['max_m']:.2f} m; {est['frac_dentro_da_faixa']:.0%} dos quadros dentro da faixa")
        if self.suavidade:
            s = self.suavidade
            linhas.append(f"Aceleração lateral máx do caminho: gravado {s['acel_lateral_gravada_max_mps2']:.2f} m/s², "
                          f"imposto {s['acel_lateral_max_mps2']:.2f} m/s² (aos {s['t_acel_lateral_max_s']:.1f} s)")
        linhas.append("Bloco  dist. mediana  dist. máx  correção máx  quadros corrigidos  a_lat máx gravada -> imposta")
        for b in self.por_bloco:
            linhas.append(f"  {b['bloco']:02d}    {b['dist_mediana_m']:6.2f} m    {b['dist_max_m']:6.2f} m    "
                          f"{b['correcao_max_m']:6.2f} m      {b['frac_corrigida']:6.0%}            "
                          f"{b['acel_lateral_gravada_max_mps2']:4.1f} -> {b['acel_lateral_max_mps2']:4.1f} m/s²")
        linhas.extend(f"AVISO: {a}" for a in self.avisos)
        return "\n".join(linhas)


def alinhar(mapa: MapaOpenDrive, telemetria: pd.DataFrame, cfg: dict[str, Any] | None = None,
            ajuste_manual: dict[str, float] | None = None, taxa_hz: float = 20.0) -> ResultadoAlinhamento:
    """Alinha a telemetria tratada (x, y, yaw, gnss_lat, gnss_lon, sim_time) ao mapa."""
    cfg = _mesclar(PADRAO, cfg)
    ajuste_manual = {k: float(v) for k, v in (ajuste_manual or {}).items() if v}
    avisos = list(mapa.avisos)
    telem = telemetria.reset_index(drop=True)
    enu = np.column_stack([telem["x"].to_numpy(float), -telem["y"].to_numpy(float)])
    rumos_enu = np.radians(-telem["yaw"].to_numpy(float))
    busca = BuscaFaixas(mapa, float(cfg["peso_rumo_m"]))
    meia = lambda idx: mapa.larguras[idx] / 2  # noqa: E731

    # 1) georreferência (+ ajuste manual, que também serve de chute quando não há geoReference)
    transf, info_geo = _por_georreferencia(mapa, telem, enu, busca, rumos_enu, avisos)
    if ajuste_manual:
        centro = transf.aplicar(enu).mean(axis=0)
        rot = math.radians(ajuste_manual.get("rotacao_graus", 0.0))
        c, s = math.cos(rot), math.sin(rot)
        t = centro - np.array([[c, -s], [s, c]]) @ centro + np.array([ajuste_manual.get("offset_x", 0.0), ajuste_manual.get("offset_y", 0.0)])
        transf = transf.depois(Similaridade(1.0, rot, float(t[0]), float(t[1])))
    estat: dict[str, dict[str, float]] = {}
    idx, lateral, _ = busca.consultar(transf.aplicar(enu), rumos_enu + transf.rotacao_rad)
    estat["georreferencia"] = _estatisticas(lateral, meia(idx))

    # 2) ICP rígido
    info_icp: dict[str, Any] = {"ativo": False}
    if cfg["icp"]["ativo"]:
        passo = max(1, int(cfg["icp"]["subamostragem"]))
        correcao_icp, info_icp = _icp(transf.aplicar(enu)[::passo], (rumos_enu + transf.rotacao_rad)[::passo], busca, cfg["icp"])
        transf = transf.depois(correcao_icp)
        idx, lateral, _ = busca.consultar(transf.aplicar(enu), rumos_enu + transf.rotacao_rad)
        estat["icp"] = _estatisticas(lateral, meia(idx))
        if info_icp["chegou_no_limite"]:
            avisos.append("O ICP chegou no limite de rotação/translação: confira o mapa ou aumente os limites no YAML.")
    alinhados = transf.aplicar(enu)
    rumos_mapa = rumos_enu + transf.rotacao_rad
    lateral_rigido = lateral.copy()

    # 3) correção de borda
    correcao = np.zeros_like(alinhados)
    desvio = np.zeros(len(alinhados))
    cb = cfg["correcao_borda"]
    if cb["ativa"]:
        velocidade = telem["speed_mps"].to_numpy(float) if "speed_mps" in telem else np.full(len(telem), 5.0)
        correcao, desvio = _correcao_borda(alinhados, rumos_mapa, velocidade, busca, cb, taxa_hz)
    finais = alinhados + correcao
    idx, lateral, _ = busca.consultar(finais, rumos_mapa + desvio)
    if cb["ativa"]:
        estat["correcao_borda"] = _estatisticas(lateral, meia(idx))
    if estat[list(estat)[-1]]["mediana_m"] > 5:
        avisos.append("A trajetória não cai sobre as vias do mapa (mediana > 5 m): confira o mapa e o geoReference.")

    yaw_carla = -np.degrees(_ang(rumos_mapa + desvio))
    t = telem["sim_time"].to_numpy(float)
    bloco = np.floor((t - t[0]) / float(cfg["bloco_s"]) + 1e-9).astype(int)
    modulo_correcao = np.hypot(*correcao.T)
    acel_gravada = np.abs(_acel_lateral(alinhados, taxa_hz))
    acel_imposta = np.abs(_acel_lateral(finais, taxa_hz))
    poses = pd.DataFrame({
        "frame": telem["frame"] if "frame" in telem else np.arange(len(telem)),
        "sim_time": t, "bloco": bloco,
        "x_mapa": finais[:, 0], "y_mapa": finais[:, 1],
        "x_carla": finais[:, 0], "y_carla": -finais[:, 1], "yaw_carla": yaw_carla,
        "x_mapa_rigido": alinhados[:, 0], "y_mapa_rigido": alinhados[:, 1],
        "dist_faixa_m": np.abs(lateral), "dist_faixa_sem_correcao_m": np.abs(lateral_rigido),
        "correcao_m": modulo_correcao,
    })
    por_bloco = []
    for k, grupo in poses.groupby("bloco"):
        linhas = grupo.index.to_numpy()
        por_bloco.append({
            "bloco": int(k),
            "dist_mediana_m": round(float(grupo["dist_faixa_sem_correcao_m"].median()), 3),
            "dist_max_m": round(float(grupo["dist_faixa_sem_correcao_m"].max()), 3),
            "correcao_max_m": round(float(grupo["correcao_m"].max()), 3),
            "frac_corrigida": round(float((grupo["correcao_m"] > 0.05).mean()), 3),
            "dist_max_corrigida_m": round(float(grupo["dist_faixa_m"].max()), 3),
            "acel_lateral_gravada_max_mps2": round(float(acel_gravada[linhas].max()), 2),
            "acel_lateral_max_mps2": round(float(acel_imposta[linhas].max()), 2),
        })
    i_max = int(np.argmax(acel_imposta))
    suavidade = {"acel_lateral_gravada_max_mps2": round(float(acel_gravada.max()), 2),
                 "acel_lateral_max_mps2": round(float(acel_imposta.max()), 2),
                 "t_acel_lateral_max_s": round(float(t[i_max]), 2),
                 "peso_curvatura": float(cb.get("peso_curvatura", 1.0)), "suavizacao_s": float(cb["suavizacao_s"])}
    resultado = ResultadoAlinhamento(transf, info_geo, info_icp, estat, por_bloco, poses, avisos, ajuste_manual, suavidade)
    return resultado


def carregar_parametros(cfg_simulacao: dict[str, Any]) -> dict[str, Any]:
    """Parâmetros do alinhamento a partir do config/simulacao.yaml (com os padrões deste módulo)."""
    return _mesclar(PADRAO, cfg_simulacao.get("alinhamento", {}))


__all__ = ["Similaridade", "ajustar_similaridade", "BuscaFaixas", "ResultadoAlinhamento", "alinhar",
           "carregar_parametros", "PADRAO", "asdict"]
