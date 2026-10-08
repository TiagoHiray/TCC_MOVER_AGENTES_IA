"""Dados sintéticos compartilhados pelos testes (telemetria de 20 Hz e mapa OpenDRIVE de teste).

Não é um módulo de testes: o pytest só coleta arquivos test_*.py. Os testes importam daqui com
`from auxiliares import ...` (o pytest põe a pasta tests/ no sys.path).
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

TAXA_HZ = 20
INICIO = datetime(2026, 10, 3, 14, 36, 35, tzinfo=timezone(timedelta(hours=-3)))
FRENAGEM = {"tempo_s": 5.0, "tipo": "frenagem_brusca", "intensidade_mps2": 3.5, "duracao_s": 1.5, "rampa_s": 0.4}


# ---------------------------------------------------------------------------------------------
# Telemetria
# ---------------------------------------------------------------------------------------------
def telemetria(duracao_s: float, vel_kmh: float = 25.0) -> pd.DataFrame:
    """Cruzeiro em linha reta, sem ruído, só com as colunas que a camada agêntica usa."""
    n = int(round(duracao_s * TAXA_HZ))
    t = np.arange(n) / TAXA_HZ
    v = np.full(n, float(vel_kmh))
    zeros = {nome: np.zeros(n) for nome in ("yaw", "acc_long", "acc_lat", "acc_vert", "jerk_long", "acc_x", "acc_y",
                                            "imu_acc_x", "yaw_rate_dps", "grade_pct", "brake_est")}
    return pd.DataFrame({
        "sim_time": t,
        "hora_local": [(INICIO + timedelta(seconds=float(s))).isoformat(timespec="milliseconds") for s in t],
        "speed_kmh": v,
        "speed_mps": v / 3.6,
        **zeros,
        "alt_m": np.full(n, 760.0),
        "odom_m": v / 3.6 * t,
        "throttle_est": np.full(n, 0.2),
        "manobra": ["cruzeiro"] * n,
        "fonte": ["real"] * n,
    })


def bgra_azul(largura: int, altura: int) -> bytes:
    """Quadro azul no formato BGRA de 8 bits do carla.Image.raw_data."""
    pixels = np.zeros((altura, largura, 4), dtype=np.uint8)
    pixels[:, :, 0] = 255  # B
    pixels[:, :, 3] = 255  # A
    return pixels.tobytes()


def marcar(dados: pd.DataFrame, coluna: str, t_ini: float, t_fim: float, valor: float) -> None:
    """Escreve `valor` na coluna para as linhas com sim_time em [t_ini, t_fim)."""
    t = dados["sim_time"]
    dados.loc[(t >= t_ini - 1e-9) & (t < t_fim - 1e-9), coluna] = valor


# ---------------------------------------------------------------------------------------------
# Pista em forma de estádio (duas retas e duas curvas de 180°), para a simulação
# ---------------------------------------------------------------------------------------------
RETA_M = 60.0
RAIO_M = 20.0
LARGURA_M = 3.2


def _quarto_de_circulo_pp3(r: float, arc_length: bool) -> str:
    """Quarto de círculo à esquerda como paramPoly3: a Bézier cúbica clássica (k = 0,5523 r).

    Pontas e tangentes exatas; erro radial máximo de 0,03 % do raio (5 mm para r ~ 18 m).
    """
    k = 4.0 / 3.0 * math.tan(math.pi / 8) * r
    cu = np.array([0.0, 3 * k, 3 * r - 6 * k, 3 * k - 2 * r])
    cv = np.array([0.0, 0.0, 3 * (r - k), 3 * k - 2 * r])
    if arc_length:  # p em metros (0 a L) em vez de 0 a 1
        escala = np.array([1.0, 1.0, 1.0, 1.0]) / (math.pi * r / 2) ** np.arange(4)
        cu, cv = cu * escala, cv * escala
    faixa = ' pRange="arcLength"' if arc_length else ""  # sem pRange = normalizado (0 a 1)
    return ('<paramPoly3 aU="{:.12g}" bU="{:.12g}" cU="{:.12g}" dU="{:.12g}" '
            'aV="{:.12g}" bV="{:.12g}" cV="{:.12g}" dV="{:.12g}"{}/>').format(*cu, *cv, faixa)


def pista_estadio_xodr(georeferencia: str | None = None, offset: tuple[float, float] | None = None,
                       origem: tuple[float, float] = (0.0, 0.0)) -> str:
    """Uma via fechada, mão única, com uma faixa à direita (driving) de 3,2 m.

    O centro da faixa é o estádio com retas de RETA_M e curvas de raio RAIO_M, percorrido no
    sentido anti-horário a partir de `origem` (início da reta de baixo), como a volta gravada.
    A linha de referência fica 1,6 m à esquerda (raio RAIO_M - 1,6). A primeira curva é um
    <arc>; a segunda são dois quartos de círculo em paramPoly3 (um com pRange="arcLength" e
    outro normalizado), para exercitar esses caminhos do leitor.
    """
    r_ref = RAIO_M - LARGURA_M / 2  # a faixa da direita fica por fora da linha de referência
    x0, y0 = origem[0], origem[1] + LARGURA_M / 2
    comp_arco = math.pi * r_ref
    s1 = RETA_M + comp_arco
    s2 = s1 + RETA_M
    geometrias = [
        (0.0, x0, y0, 0.0, RETA_M, "<line/>"),
        (RETA_M, x0 + RETA_M, y0, 0.0, comp_arco, f'<arc curvature="{1 / r_ref:.12f}"/>'),
        (s1, x0 + RETA_M, y0 + 2 * r_ref, math.pi, RETA_M, "<line/>"),
        (s2, x0, y0 + 2 * r_ref, math.pi, comp_arco / 2, _quarto_de_circulo_pp3(r_ref, arc_length=True)),
        (s2 + comp_arco / 2, x0 - r_ref, y0 + r_ref, 1.5 * math.pi, comp_arco / 2,
         _quarto_de_circulo_pp3(r_ref, arc_length=False)),
    ]
    comprimento = 2 * RETA_M + 2 * comp_arco
    geo_xml = "\n".join(
        f'      <geometry s="{s:.6f}" x="{x:.6f}" y="{y:.6f}" hdg="{h:.12f}" length="{c:.6f}">{g}</geometry>'
        for s, x, y, h, c, g in geometrias)
    header_extra = ""
    if georeferencia:
        header_extra += f"\n    <geoReference><![CDATA[{georeferencia}]]></geoReference>"
    if offset is not None:
        header_extra += f'\n    <offset x="{offset[0]:.4f}" y="{offset[1]:.4f}" z="0" hdg="0"/>'
    return f"""<?xml version="1.0" standalone="yes"?>
<!-- pista de teste gerada por tests/auxiliares.py -->
<OpenDRIVE>
  <header revMajor="1" revMinor="4" name="estadio" version="1.00">{header_extra}
  </header>
  <road name="estadio" length="{comprimento:.6f}" id="1" junction="-1">
    <link/>
    <planView>
{geo_xml}
    </planView>
    <elevationProfile><elevation s="0" a="0" b="0" c="0" d="0"/></elevationProfile>
    <lanes>
      <laneOffset s="0" a="0" b="0" c="0" d="0"/>
      <laneSection s="0">
        <left>
          <lane id="1" type="sidewalk" level="false"><width sOffset="0" a="2.0" b="0" c="0" d="0"/></lane>
        </left>
        <center><lane id="0" type="none" level="false"/></center>
        <right>
          <lane id="-1" type="driving" level="false"><width sOffset="0" a="{LARGURA_M}" b="0" c="0" d="0"/></lane>
        </right>
      </laneSection>
    </lanes>
  </road>
</OpenDRIVE>
"""


def centro_da_faixa(s: np.ndarray, origem: tuple[float, float] = (0.0, 0.0)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ponto (x, y) e rumo (rad, anti-horário) do centro da faixa no comprimento de arco s (exato)."""
    perimetro = 2 * RETA_M + 2 * math.pi * RAIO_M
    s = np.mod(np.asarray(s, float), perimetro)
    x, y, h = np.zeros_like(s), np.zeros_like(s), np.zeros_like(s)
    cx1, cy = origem[0] + RETA_M, origem[1] + RAIO_M
    cx0 = origem[0]
    a1 = RETA_M + math.pi * RAIO_M
    a2 = a1 + RETA_M
    m = s < RETA_M
    x[m], y[m], h[m] = origem[0] + s[m], origem[1], 0.0
    m = (s >= RETA_M) & (s < a1)
    ang = (s[m] - RETA_M) / RAIO_M - math.pi / 2
    x[m], y[m], h[m] = cx1 + RAIO_M * np.cos(ang), cy + RAIO_M * np.sin(ang), ang + math.pi / 2
    m = (s >= a1) & (s < a2)
    x[m], y[m], h[m] = cx1 - (s[m] - a1), origem[1] + 2 * RAIO_M, math.pi
    m = s >= a2
    ang = (s[m] - a2) / RAIO_M + math.pi / 2
    x[m], y[m], h[m] = cx0 + RAIO_M * np.cos(ang), cy + RAIO_M * np.sin(ang), ang + math.pi / 2
    return x, y, h


def volta_no_estadio(duracao_s: float, vel_kmh: float = 25.0, lateral_m: np.ndarray | float = 0.0,
                     origem: tuple[float, float] = (0.0, 0.0)) -> pd.DataFrame:
    """Telemetria no padrão da Etapa 1 percorrendo o centro da faixa (+ desvio lateral à esquerda).

    Plano local da Etapa 1: x = leste, y = -norte, yaw = rumo da bússola - 90 (graus). Aqui o
    plano do mapa coincide com o leste/norte locais, como num mapa sem geoReference.
    """
    dados = telemetria(duracao_s, vel_kmh)
    s = dados["odom_m"].to_numpy(float)
    x, y, h = centro_da_faixa(s, origem)
    lateral = np.broadcast_to(np.asarray(lateral_m, float), s.shape)
    x, y = x - lateral * np.sin(h), y + lateral * np.cos(h)
    dados["x"], dados["y"] = x, -y
    dados["yaw"] = -np.degrees(np.angle(np.exp(1j * h)))
    dados["frame"] = np.arange(len(dados))
    return dados


# ---------------------------------------------------------------------------------------------
# Pista com uma quina (duas retas), como os conectores de junção do mapa do Eduardo
# ---------------------------------------------------------------------------------------------
TRECHO_M = 60.0


def pista_com_quina_xodr(angulo_graus: float) -> str:
    """Duas retas de TRECHO_M; a segunda vira `angulo_graus` à direita. Uma faixa à direita de 3,2 m."""
    h2 = -math.radians(angulo_graus)
    return f"""<?xml version="1.0" standalone="yes"?>
<OpenDRIVE>
  <header revMajor="1" revMinor="4" name="quina" version="1.00"/>
  <road name="quina" length="{2 * TRECHO_M:.6f}" id="1" junction="-1">
    <planView>
      <geometry s="0" x="0" y="0" hdg="0" length="{TRECHO_M:.6f}"><line/></geometry>
      <geometry s="{TRECHO_M:.6f}" x="{TRECHO_M:.6f}" y="0" hdg="{h2:.12f}" length="{TRECHO_M:.6f}"><line/></geometry>
    </planView>
    <lanes>
      <laneSection s="0">
        <center><lane id="0" type="none" level="false"/></center>
        <right><lane id="-1" type="driving" level="false"><width sOffset="0" a="{LARGURA_M}" b="0" c="0" d="0"/></lane></right>
      </laneSection>
    </lanes>
  </road>
</OpenDRIVE>
"""


def volta_suave_na_quina(angulo_graus: float, lateral_m: float, raio_m: float = 25.0,
                         vel_kmh: float = 25.0) -> pd.DataFrame:
    """Caminho suave (reta, arco de `raio_m`, reta) a `lateral_m` à esquerda do centro da faixa com quina.

    O arco arredonda a quina como faria o motorista; o desenho da faixa, não. Mesmo padrão de
    colunas de volta_no_estadio.
    """
    teta = math.radians(angulo_graus)
    deslocamento = lateral_m - LARGURA_M / 2  # em relação à linha de referência (+ à esquerda)
    tangente = raio_m * math.tan(teta / 2)
    comp_arco = raio_m * teta
    comprimento = 2 * TRECHO_M - 2 * tangente + comp_arco
    v = vel_kmh / 3.6
    dados = telemetria((comprimento - 1.0) / v, vel_kmh)
    s = dados["odom_m"].to_numpy(float)
    x, y, h = np.zeros_like(s), np.zeros_like(s), np.zeros_like(s)
    reta1 = s < TRECHO_M - tangente
    x[reta1], y[reta1] = s[reta1], 0.0
    arco = (s >= TRECHO_M - tangente) & (s < TRECHO_M - tangente + comp_arco)
    ang = (s[arco] - (TRECHO_M - tangente)) / raio_m  # quanto já virou à direita
    cx, cy = TRECHO_M - tangente, -raio_m  # centro do arco (curva à direita)
    x[arco], y[arco], h[arco] = cx + raio_m * np.sin(ang), cy + raio_m * np.cos(ang), -ang
    reta2 = ~(reta1 | arco)
    resto = s[reta2] - (TRECHO_M - tangente + comp_arco)
    x[reta2] = TRECHO_M + (tangente + resto) * math.cos(teta)
    y[reta2] = -(tangente + resto) * math.sin(teta)
    h[reta2] = -teta
    x, y = x - deslocamento * np.sin(h), y + deslocamento * np.cos(h)
    dados["x"], dados["y"] = x, -y
    dados["yaw"] = -np.degrees(h)
    dados["frame"] = np.arange(len(dados))
    return dados
