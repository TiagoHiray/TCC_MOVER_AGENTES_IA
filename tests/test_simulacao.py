"""Testes da simulação (sem CARLA): leitor OpenDRIVE, projeção, alinhamento, laço do replay e câmera do painel.

Rodar a partir da raiz do repositório:
    python -m pytest tests -q

O mapa de teste é um estádio (duas retas e duas curvas de 180°) gerado em tests/auxiliares.py.
O lado CARLA do replay é testado com um módulo `carla` falso, que registra as chamadas.
"""

from __future__ import annotations

import copy
import io
import logging
import math
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from auxiliares import (FRENAGEM, LARGURA_M, RAIO_M, RETA_M, bgra_azul, centro_da_faixa, pista_com_quina_xodr,
                        pista_estadio_xodr, telemetria, volta_no_estadio, volta_suave_na_quina)
from fastapi.testclient import TestClient

from mover.agentes import eventos_sinteticos
from mover.agentes.llm import FalsoLLM
from mover.agentes.supervisor import Supervisor
from mover.config import carregar_yaml
from mover.interface.cena import montar_cena
from mover.servidor.app import criar_app
from mover.simulacao.alinhamento import alinhar
from mover.simulacao.camera_painel import codificar_jpeg
from mover.simulacao.opendrive import ler_xodr
from mover.simulacao.projecao import criar_projecao
from mover.simulacao.replay_carla import (ClienteAgentes, ErroServidor, MundoCarla, MundoFalso, Relogio, Trajeto,
                                          rodar_replay, rotulo_problema)

PERIMETRO = 2 * RETA_M + 2 * math.pi * RAIO_M
LAT0, LON0 = -23.648254, -46.5738235  # centro do campus do IMT
TMERC_LOCAL = f"+proj=tmerc +lat_0={LAT0} +lon_0={LON0} +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
SEM_ICP = {"icp": {"ativo": False}}


def girar(pontos: np.ndarray, graus: float, centro: tuple[float, float]) -> np.ndarray:
    c, s = math.cos(math.radians(graus)), math.sin(math.radians(graus))
    return (pontos - centro) @ np.array([[c, -s], [s, c]]).T + centro


def lat_lon(leste: np.ndarray, norte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inversa local da projeção (raios de curvatura do WGS84); erro < 1 mm a centenas de metros."""
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    s = math.sin(math.radians(LAT0))
    meridiano = a * (1 - e2) / (1 - e2 * s * s) ** 1.5
    normal = a / math.sqrt(1 - e2 * s * s)
    return LAT0 + np.degrees(norte / meridiano), LON0 + np.degrees(leste / (normal * math.cos(math.radians(LAT0))))


# ---------------------------------------------------------------------------------------------
# Leitor OpenDRIVE e projeção
# ---------------------------------------------------------------------------------------------
def test_leitor_xodr_com_laneoffset_e_faixas_dos_dois_lados():
    xodr = """<OpenDRIVE><header/>
      <road length="100" id="7" junction="-1"><planView>
        <geometry s="0" x="0" y="0" hdg="0" length="100"><line/></geometry></planView>
        <lanes><laneOffset s="0" a="1.0" b="0" c="0" d="0"/>
          <laneSection s="0">
            <left><lane id="1" type="driving"><width sOffset="0" a="3.0" b="0" c="0" d="0"/></lane></left>
            <center><lane id="0" type="none"/></center>
            <right><lane id="-1" type="driving"><width sOffset="0" a="3.2" b="0" c="0" d="0"/></lane>
                   <lane id="-2" type="sidewalk"><width sOffset="0" a="2.0" b="0" c="0" d="0"/></lane></right>
          </laneSection></lanes></road></OpenDRIVE>"""
    mapa = ler_xodr(xodr, passo_m=1.0, tipos_faixa=("driving",))
    direita = mapa.larguras == 3.2
    assert np.allclose(mapa.pontos[direita, 1], 1.0 - 1.6) and np.allclose(mapa.rumos[direita], 0.0)
    assert np.allclose(mapa.pontos[~direita, 1], 1.0 + 1.5) and np.allclose(mapa.rumos[~direita], math.pi)
    assert dict(mapa.tipos_faixa) == {"driving": 2, "sidewalk": 1} and set(mapa.vias) == {7}


def test_leitor_xodr_reproduz_a_pista_com_arc_e_parampoly3():
    mapa = ler_xodr(pista_estadio_xodr(), passo_m=0.25)
    s = np.linspace(0.0, PERIMETRO, 40000, endpoint=False)
    x, y, h = centro_da_faixa(s)
    dist = np.hypot(mapa.pontos[:, None, 0] - x[None, ::20], mapa.pontos[:, None, 1] - y[None, ::20]).min(axis=1)
    assert dist.max() < 0.08  # passo da comparação: 0,12 m
    # o leitor amostra a cada passo_m da linha de referência, que fica 1,6 m por dentro do centro da faixa
    comprimento_referencia = 2 * RETA_M + 2 * math.pi * (RAIO_M - LARGURA_M / 2)
    assert abs(len(mapa.pontos) * 0.25 - comprimento_referencia) < 1.0
    assert np.allclose(mapa.larguras, LARGURA_M)


def test_projecao_utm_e_tmerc_batem_com_o_pyproj():
    pyproj = pytest.importorskip("pyproj")
    lat = LAT0 + np.array([0.0, 0.001, -0.002, 0.0015])
    lon = LON0 + np.array([0.0, -0.002, 0.001, 0.003])
    for proj4 in ("+proj=utm +zone=23 +ellps=WGS84 +datum=WGS84 +units=m +no_defs", TMERC_LOCAL):
        projetar, _ = criar_projecao(proj4)
        e, n = projetar(lat, lon)
        e_ref, n_ref = pyproj.Proj(proj4)(lon, lat)
        assert np.abs(e - e_ref).max() < 1e-6 and np.abs(n - n_ref).max() < 1e-6


# ---------------------------------------------------------------------------------------------
# Alinhamento
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("sinal", [-1, +1])
def test_georreferencia_acha_o_sinal_do_offset(sinal):
    """netconvert do SUMO: mapa = projetado - offset; Osm2Odr do CARLA: mapa = projetado + offset."""
    offset = np.array([-100.0, 50.0])
    origem = tuple(sinal * offset)  # onde o estádio (projetado em torno de 0, 0) cai no plano do mapa
    mapa = ler_xodr(pista_estadio_xodr(TMERC_LOCAL, tuple(offset), origem=origem), passo_m=0.5)
    dados = volta_no_estadio(36.0)
    leste, norte = dados["x"].to_numpy(), -dados["y"].to_numpy()
    dados["gnss_lat"], dados["gnss_lon"] = lat_lon(leste, norte)
    # o plano local da Etapa 1 não é o projetado: gira 0,6° e desloca, como o ENU do EKF
    enu = girar(np.column_stack([leste, norte]), 0.6, (0.0, 0.0)) + [3.0, -2.0]
    dados["x"], dados["y"], dados["yaw"] = enu[:, 0], -enu[:, 1], dados["yaw"] - 0.6

    res = alinhar(mapa, dados, SEM_ICP)
    assert res.georreferencia["sinal_offset"] == sinal
    assert res.georreferencia["residuo_max_m"] < 0.01
    assert abs(res.transformacao.rotacao_rad + math.radians(0.6)) < 1e-4
    assert res.estatisticas["georreferencia"]["max_m"] < 0.05
    assert not [a for a in res.avisos if "não cai sobre as vias" in a]


def test_icp_desfaz_rotacao_e_deslocamento_sem_georreferencia():
    mapa = ler_xodr(pista_estadio_xodr(), passo_m=0.5)
    dados = volta_no_estadio(35.0)
    centro = (RETA_M / 2, RAIO_M)
    p = girar(np.column_stack([dados["x"], -dados["y"]]), 1.5, centro) + [2.0, -1.5]
    dados["x"], dados["y"], dados["yaw"] = p[:, 0], -p[:, 1], dados["yaw"] - 1.5

    res = alinhar(mapa, dados, {"correcao_borda": {"ativa": False}})
    assert res.georreferencia["metodo"] == "identidade"  # sem geoReference no mapa
    assert res.icp["convergiu"] and not res.icp["chegou_no_limite"]
    assert abs(res.icp["rotacao_graus"] + 1.5) < 0.05
    assert res.estatisticas["georreferencia"]["max_m"] > 2.0
    assert res.estatisticas["icp"]["p95_m"] < 0.05


def _volta_com_desvio_rapido() -> tuple[Any, np.ndarray]:
    dados = telemetria(35.0)
    s = dados["odom_m"].to_numpy()
    rampa = lambda u: 0.5 - 0.5 * np.cos(np.pi * np.clip(u, 0.0, 1.0))  # noqa: E731
    lateral = 4.0 * rampa((s - 20.0) / 5.0) * rampa((40.0 - s) / 5.0)  # 4 m à esquerda entre 25 e 35 m
    return volta_no_estadio(35.0, lateral_m=lateral), s


def test_correcao_de_borda_so_mexe_no_trecho_fora_da_faixa():
    dados, s = _volta_com_desvio_rapido()
    mapa = ler_xodr(pista_estadio_xodr(), passo_m=0.5)

    # peso_curvatura = 1: a correção é a mínima necessária e some sem oscilar
    poses = alinhar(mapa, dados, {**SEM_ICP, "correcao_borda": {"peso_curvatura": 1.0}}).poses
    assert poses["dist_faixa_sem_correcao_m"].max() > 3.9
    assert poses["dist_faixa_m"].max() <= LARGURA_M / 2  # o caminhão fica dentro da faixa
    longe = (s < 20.0 - 25.0) | (s > 40.0 + 25.0)  # mais de ~3,5 s antes ou depois do desvio
    assert poses["correcao_m"].to_numpy()[longe].max() < 0.02
    patamar = (s > 27.0) & (s < 33.0)
    assert abs(poses["correcao_m"].to_numpy()[patamar].mean() - (4.0 - 1.1)) < 0.3


def test_peso_de_curvatura_suaviza_a_guinada_numa_quina_da_faixa():
    """Caso do conector de junção aos 54,6 s: a faixa tem uma quina e a volta gravada, não."""
    mapa = ler_xodr(pista_com_quina_xodr(25.0), passo_m=0.5)
    dados = volta_suave_na_quina(25.0, lateral_m=2.5)  # 1,4 m além da tolerância o tempo todo

    minimo = alinhar(mapa, dados, {**SEM_ICP, "correcao_borda": {"peso_curvatura": 1.0}})
    padrao = alinhar(mapa, dados, SEM_ICP)  # peso_curvatura do PADRAO (32)
    assert padrao.suavidade["peso_curvatura"] == 32.0
    for resultado in (minimo, padrao):
        assert resultado.poses["dist_faixa_m"].max() <= 1.1 + 0.01  # meia largura - margem
    gravada = padrao.suavidade["acel_lateral_gravada_max_mps2"]
    assert minimo.suavidade["acel_lateral_max_mps2"] > 1.5 * gravada  # seguir a quina dá uma guinada
    assert padrao.suavidade["acel_lateral_max_mps2"] < 0.6 * minimo.suavidade["acel_lateral_max_mps2"]
    assert padrao.suavidade["acel_lateral_max_mps2"] < 1.25 * gravada  # quase tão suave quanto a gravação


def test_yaw_do_carla_segue_a_direcao_do_movimento():
    poses = alinhar(ler_xodr(pista_estadio_xodr(), 0.5), volta_no_estadio(35.0)).poses
    dx, dy = np.diff(poses["x_carla"]), np.diff(poses["y_carla"])
    direcao = np.degrees(np.arctan2(dy, dx))  # CARLA: y para o sul, yaw horário
    diferenca = (poses["yaw_carla"].to_numpy()[:-1] - direcao + 180.0) % 360.0 - 180.0
    assert np.abs(diferenca).max() < 1.0


# ---------------------------------------------------------------------------------------------
# Replay sem CARLA
# ---------------------------------------------------------------------------------------------
def test_rotulo_do_problema_na_pista_e_ascii():
    texto = rotulo_problema({"nivel": "critico", "t_pico": 37.24, "evento": "soltura_freio",
                             "causa": "irregularidade_via", "fatos": {"jerk_max_abs_mps3": 6.61}})
    assert texto == "CRITICO 37.2s | soltura do freio (provavel lombada) | jerk 6.6 m/s3"
    assert rotulo_problema({"nivel": "atencao", "t_pico": 5, "evento": "arrancada_brusca",
                            "causa": "conducao_brusca"}).isascii()


def test_relogio_segue_o_fator_de_tempo_e_nao_tenta_recuperar_atraso():
    relogio = Relogio(fator_tempo=20.0)
    relogio.ancorar(0.0)
    inicio = time.monotonic()
    for t in np.arange(0.0, 2.0, 0.05):  # 2 s de simulação a 20x = 0,1 s
        relogio.esperar_ate(t)
    assert 0.08 < time.monotonic() - inicio < 0.5
    relogio.ancorar(0.0)
    time.sleep(0.4)
    antes = time.monotonic()
    relogio.esperar_ate(0.05)  # atrasado: reancora em vez de correr
    assert relogio.atrasos == 1 and time.monotonic() - antes < 0.05


def test_replay_sem_carla_anuncia_os_blocos_em_ordem_e_marca_os_problemas(tmp_path):
    cfg = carregar_yaml("config/agentes.yaml")
    dados = eventos_sinteticos.injetar(volta_no_estadio(25.0), [FRENAGEM], {})
    trajeto = Trajeto.de_poses(alinhar(ler_xodr(pista_estadio_xodr(), 0.5), dados, SEM_ICP).poses)
    app = criar_app(cfg, {"servidor": {"pasta_sessoes": str(tmp_path)}}, telemetria=dados,
                    fabrica_supervisor=lambda opcoes: Supervisor(FalsoLLM(), cfg), sem_ml=True)
    impressos: list[str] = []
    mundo = MundoFalso()
    with TestClient(app) as http:
        cliente = ClienteAgentes(http)
        sessao = cliente.abrir_sessao({"tempo_real": False})
        mundo.preparar(trajeto, None, dados)
        res = rodar_replay(trajeto, cliente, mundo, sessao, sem_espera=True, imprimir=impressos.append)
        fim = cliente.encerrar_sessao()
        cliente.fechar()

    assert res.quadros == mundo.quadros == len(dados) == 500
    assert [b["bloco"] for b in res.blocos] == [0, 1, 2] and not res.interrompido
    assert res.estados == cliente.estados_enviados == 125
    assert len(mundo.marcas) == 1
    quadro, problema = mundo.marcas[0]
    assert problema["evento"] == "frenagem_brusca" and abs(trajeto.t[quadro] - problema["t_pico"]) < 0.03
    assert fim["resumo"]["blocos_publicados"] == 3 and fim["sessao"]["encerrada"]
    assert len(Path(sessao["arquivo"]).read_text(encoding="utf-8").splitlines()) == len(res.entradas)
    assert impressos[0].startswith("=== Bloco 00 (0-10 s)") and any("PROBLEMA" in linha for linha in impressos)


# ---------------------------------------------------------------------------------------------
# Lado CARLA com um módulo `carla` falso
# ---------------------------------------------------------------------------------------------
def carla_falso() -> types.ModuleType:
    """O mínimo da API do CARLA 0.9.16 usada pelo replay, registrando as chamadas."""
    mod = types.ModuleType("carla")
    chamadas: list[tuple] = []

    class Location:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x, self.y, self.z = x, y, z

    class Rotation:
        def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
            self.pitch, self.yaw, self.roll = pitch, yaw, roll

    class Transform:
        def __init__(self, location=None, rotation=None):
            self.location, self.rotation = location or Location(), rotation or Rotation()

    class Color:
        def __init__(self, r=0, g=0, b=0, a=255):
            self.r, self.g, self.b, self.a = r, g, b, a

    class LaneType:
        Any, Driving = -2, 2

    class OpendriveGenerationParameters:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Configuracao:
        synchronous_mode, fixed_delta_seconds = False, None

    class Clima:
        sun_altitude_angle, cloudiness, precipitation = 10.0, 0.0, 0.0

    class Waypoint:
        transform = Transform(Location(0.0, 0.0, 1.5))

    class Mapa:
        name = "estadio"

        def get_waypoint(self, location, project_to_road=True, lane_type=LaneType.Driving):
            assert project_to_road and lane_type == LaneType.Any
            return Waypoint()

    class Caixa:
        location, extent = Location(0.0, 0.0, 1.9), Location(4.0, 1.3, 1.7)  # fundo 0,2 m acima da origem

    class Ator:
        type_id, id, bounding_box = "vehicle.carlamotors.firetruck", 42, Caixa()

        def __init__(self):
            self.transformacoes, self.fisica, self.destruido = [], True, False

        def set_simulate_physics(self, ligada):
            self.fisica = ligada

        def set_transform(self, t):
            self.transformacoes.append(t)

        def destroy(self):
            chamadas.append(("destruir_ator",))
            self.destruido = True
            return True

    class Blueprint:
        id = "vehicle.carlamotors.firetruck"

        def __init__(self):
            self.atributos = {}

        def has_attribute(self, nome):
            return nome == "role_name"

        def set_attribute(self, nome, valor):
            self.atributos[nome] = valor

    class BlueprintCamera(Blueprint):
        id = "sensor.camera.rgb"

        def has_attribute(self, nome):
            return nome in ("image_size_x", "image_size_y", "fov", "sensor_tick")

    class AttachmentType:
        Rigid, SpringArm, SpringArmGhost = 0, 1, 2

    class Sensor:
        """carla.Sensor: o callback recebe carla.Image (raw_data em BGRA) numa thread do cliente."""

        type_id, id = "sensor.camera.rgb", 43

        def __init__(self):
            self.callback, self.destruido = None, False

        def listen(self, callback):
            self.callback = callback

        def is_listening(self):
            return self.callback is not None

        def stop(self):
            chamadas.append(("parar_sensor",))
            self.callback = None

        def destroy(self):
            chamadas.append(("destruir_sensor",))
            self.destruido = True
            return True

    class Biblioteca:
        def find(self, nome):
            assert nome in ("vehicle.carlamotors.firetruck", "sensor.camera.rgb"), nome
            return mod.blueprint if nome == "vehicle.carlamotors.firetruck" else mod.blueprint_camera

    class Debug:
        def draw_point(self, location, size=0.1, color=None, life_time=-1.0):
            chamadas.append(("ponto", location, life_time))

        def draw_line(self, begin, end, thickness=0.1, color=None, life_time=-1.0):
            chamadas.append(("linha", begin, end))

        def draw_string(self, location, text, draw_shadow=False, color=None, life_time=-1.0):
            chamadas.append(("texto", text, color.r, life_time))

    class Mundo:
        def __init__(self):
            self.config, self.aplicadas, self.ticks, self.debug = Configuracao(), [], 0, Debug()
            self.clima, self.spectator, self.ator, self.sensores = Clima(), Ator(), None, []

        def get_settings(self):
            return copy.copy(self.config)

        def apply_settings(self, config):
            self.config = copy.copy(config)
            self.aplicadas.append((config.synchronous_mode, config.fixed_delta_seconds))
            chamadas.append(("apply_settings", config.synchronous_mode))

        def get_map(self):
            return Mapa()

        def get_blueprint_library(self):
            return Biblioteca()

        def try_spawn_actor(self, bp, transformacao):
            chamadas.append(("spawn", transformacao.location.z))
            self.ator = Ator()
            return self.ator

        def spawn_actor(self, bp, transformacao, attach_to=None, attachment_type=AttachmentType.Rigid):
            if mod.erro_spawn_sensor is not None:
                raise mod.erro_spawn_sensor
            chamadas.append(("spawn_sensor", bp.id, transformacao, attach_to, attachment_type))
            self.sensores.append(Sensor())
            return self.sensores[-1]

        def get_spectator(self):
            return self.spectator

        def tick(self):
            self.ticks += 1
            return self.ticks

        def get_weather(self):
            return self.clima

        def set_weather(self, clima):
            self.clima = clima

    class Client:
        def __init__(self, host, porta):
            self.mundo = Mundo()
            mod.cliente = self

        def set_timeout(self, segundos):
            pass

        def get_server_version(self):
            return "0.9.16"

        def get_client_version(self):
            return "0.9.16"

        def get_world(self):
            return self.mundo

        def generate_opendrive_world(self, texto, parametros):
            chamadas.append(("gerar", texto, parametros))
            self.mundo = Mundo()
            return self.mundo

    for nome, valor in list(locals().items()):
        if isinstance(valor, type):
            setattr(mod, nome, valor)
    mod.chamadas, mod.blueprint, mod.blueprint_camera = chamadas, Blueprint(), BlueprintCamera()
    mod.erro_spawn_sensor = None  # exceção a levantar ao criar a câmera (teste de falha)
    return mod


def test_mundo_carla_impoe_a_pose_marca_os_problemas_e_restaura_o_simulador(monkeypatch):
    carla = carla_falso()
    monkeypatch.setitem(sys.modules, "carla", carla)
    cfg = carregar_yaml("config/simulacao.yaml")
    dados = volta_no_estadio(12.0)
    dados["sun_altitude"] = 46.2
    trajeto = Trajeto.de_poses(alinhar(ler_xodr(pista_estadio_xodr(), 0.5), dados, SEM_ICP).poses)

    mundo = MundoCarla(cfg)
    mundo.conectar()
    mundo.preparar(trajeto, "<OpenDRIVE/>", dados)
    gerar = [c for c in carla.chamadas if c[0] == "gerar"]
    assert len(gerar) == 1 and gerar[0][1] == "<OpenDRIVE/>" and gerar[0][2].wall_height == 0.0
    world = carla.cliente.mundo
    assert world.aplicadas == [(True, 0.05)] and world.clima.sun_altitude_angle == pytest.approx(46.2)
    ator = world.ator
    assert not ator.fisica and carla.blueprint.atributos["role_name"] == "mover_gemeo"

    for i in (1, 2, 3):
        mundo.aplicar(i)
        mundo.avancar()
    ultima = ator.transformacoes[-1]
    assert (ultima.location.x, ultima.location.y) == (trajeto.x[3], trajeto.y[3])
    assert ultima.location.z == pytest.approx(1.5 - 0.2 + 0.05)  # asfalto - fundo da caixa + folga
    assert ultima.rotation.yaw == trajeto.yaw[3] and ultima.rotation.pitch == 0.0  # pista plana
    camera = world.spectator.transformacoes[-1]
    assert camera.location.z == pytest.approx(1.5 + 8) and camera.rotation.pitch == -20

    mundo.marcar_problemas([(100, {"nivel": "critico", "t_pico": 5.0, "evento": "frenagem_brusca",
                                   "causa": "conducao_brusca", "fatos": {"jerk_max_abs_mps3": 9.5}})])
    texto = [c for c in carla.chamadas if c[0] == "texto"][0]
    assert texto[1].startswith("CRITICO 5.0s | frenagem brusca") and texto[2] == 255 and texto[3] == 12.0

    mundo.encerrar()
    assert world.aplicadas[-1] == (False, None) and ator.destruido  # volta ao modo assíncrono
    mundo.encerrar()  # idempotente


def esperar(condicao, prazo_s: float = 5.0) -> None:
    """A câmera codifica e envia numa thread (FilaEnvio): espera a condição ficar verdadeira."""
    limite = time.monotonic() + prazo_s
    while not condicao():
        assert time.monotonic() < limite, "a condição não ficou verdadeira no prazo"
        time.sleep(0.01)


class ImagemFalsa:
    """carla.Image: raw_data em BGRA de 8 bits."""

    def __init__(self, largura: int, altura: int):
        self.width, self.height, self.raw_data = largura, altura, bgra_azul(largura, altura)


def _mundo_com_camera(monkeypatch, enviar, cfg_extra: dict[str, Any] | None = None):
    carla = carla_falso()
    monkeypatch.setitem(sys.modules, "carla", carla)
    cfg = carregar_yaml("config/simulacao.yaml")
    cfg["camera_painel"] = {**cfg.get("camera_painel", {}), **(cfg_extra or {})}
    trajeto = Trajeto.de_poses(alinhar(ler_xodr(pista_estadio_xodr(), 0.5), volta_no_estadio(12.0), SEM_ICP).poses)
    mundo = MundoCarla(cfg)
    mundo.conectar()
    mundo.ligar_camera_painel(enviar)
    mundo.preparar(trajeto, "<OpenDRIVE/>", volta_no_estadio(12.0))
    return carla, mundo


def test_camera_do_painel_presa_ao_caminhao_manda_jpeg_e_sai_antes_do_caminhao(monkeypatch, caplog):
    from PIL import Image

    quadros: list[bytes] = []
    carla, mundo = _mundo_com_camera(monkeypatch, quadros.append)
    world = carla.cliente.mundo
    camera = mundo.camera_painel
    assert camera is not None and len(world.sensores) == 1
    assert carla.blueprint_camera.atributos == {"image_size_x": "640", "image_size_y": "360", "fov": "90.0",
                                                "sensor_tick": "0.100"}  # 10 fps em tempo de simulação
    _, bp_id, posicao, preso_a, tipo = next(c for c in carla.chamadas if c[0] == "spawn_sensor")
    assert bp_id == "sensor.camera.rgb" and preso_a is world.ator and tipo == carla.AttachmentType.Rigid
    loc, rot = posicao.location, posicao.rotation
    assert (loc.x, loc.y, loc.z, rot.pitch) == (-12.0, 0.0, 5.0, -15.0)  # 12 m atrás e 5 m acima
    sensor = world.sensores[0]
    assert sensor.is_listening()

    sensor.callback(ImagemFalsa(64, 36))  # a thread do cliente do CARLA entrega um quadro
    esperar(lambda: camera.enviados == 1)
    imagem = Image.open(io.BytesIO(quadros[0]))
    assert quadros[0].startswith(b"\xff\xd8") and imagem.size == (64, 36) and imagem.getpixel((32, 18))[2] > 220

    def servidor_fora(jpeg: bytes) -> None:
        raise ConnectionError("servidor fora do ar")

    camera.enviar = servidor_fora  # falha ao mandar: só um aviso, o replay continua
    with caplog.at_level(logging.WARNING, logger="mover.simulacao"):
        sensor.callback(ImagemFalsa(64, 36))
        esperar(lambda: "Falha ao mandar o quadro da câmera" in caplog.text)
    assert (camera.recebidos, camera.enviados, len(quadros)) == (2, 1, 1)

    inicio = len(carla.chamadas)
    mundo.encerrar()
    ordem = [c[0] for c in carla.chamadas[inicio:]]
    # o CARLA volta ao modo assíncrono primeiro; a câmera sai antes do caminhão a que está presa
    assert ordem == ["apply_settings", "parar_sensor", "destruir_sensor", "destruir_ator"]
    assert mundo.camera_painel is None and sensor.destruido and not sensor.is_listening()
    mundo.encerrar()  # idempotente
    camera.encerrar()
    assert len(carla.chamadas) == inicio + 4


def test_replay_segue_sem_a_camera_do_painel_se_ela_falhar_ou_estiver_desligada(monkeypatch, caplog):
    carla = carla_falso()
    carla.erro_spawn_sensor = RuntimeError("blueprint sensor.camera.rgb indisponível")
    monkeypatch.setitem(sys.modules, "carla", carla)
    cfg = carregar_yaml("config/simulacao.yaml")
    trajeto = Trajeto.de_poses(alinhar(ler_xodr(pista_estadio_xodr(), 0.5), volta_no_estadio(12.0), SEM_ICP).poses)
    mundo = MundoCarla(cfg)
    mundo.conectar()
    mundo.ligar_camera_painel(lambda jpeg: None)
    with caplog.at_level(logging.WARNING, logger="mover.simulacao"):
        mundo.preparar(trajeto, "<OpenDRIVE/>", volta_no_estadio(12.0))
    assert mundo.camera_painel is None and "Câmera do painel desligada" in caplog.text
    world = carla.cliente.mundo
    assert world.ator is not None and world.ticks == 1  # o caminhão foi criado e o replay começou
    mundo.aplicar(1)
    mundo.encerrar()
    assert world.ator.destruido

    _, mundo = _mundo_com_camera(monkeypatch, lambda jpeg: None, {"ativa": False})
    assert mundo.camera_painel is None and not sys.modules["carla"].cliente.mundo.sensores
    mundo.encerrar()


def test_cliente_manda_a_cena_e_os_quadros_da_camera_ao_servidor(tmp_path):
    cfg = carregar_yaml("config/agentes.yaml")
    dados = volta_no_estadio(12.0)
    mapa = ler_xodr(pista_estadio_xodr(), 0.5)
    cena = montar_cena(mapa, alinhar(mapa, dados, SEM_ICP))
    cfg_sim = {"servidor": {"pasta_sessoes": str(tmp_path)}, "mapa": {"arquivo": str(tmp_path / "nao_existe.xodr")}}
    app = criar_app(cfg, cfg_sim, telemetria=dados, fabrica_supervisor=lambda opcoes: Supervisor(FalsoLLM(), cfg),
                    sem_ml=True, fabrica_llm_chat=lambda opcoes: FalsoLLM())
    with TestClient(app) as http:
        cliente = ClienteAgentes(http)
        assert http.get("/cena").status_code == 404  # sem mapa no config: só a cena mandada pelo replay
        assert cliente.enviar_cena(cena) is True
        assert http.get("/cena").json()["origem"] == "replay"
        ruim = copy.deepcopy(cena)
        ruim["trajeto"]["x"].pop()
        assert cliente.enviar_cena(ruim) is False  # 422: só um aviso, a cena boa continua lá
        assert len(http.get("/cena").json()["trajeto"]["x"]) == len(cena["trajeto"]["x"])

        cliente.postar_quadro(codificar_jpeg(bgra_azul(32, 16), 32, 16))
        assert http.get("/camera/info").json()["quadros"] == 1
        with pytest.raises(ErroServidor, match="HTTP 400"):
            cliente.postar_quadro(b"isto nao e um jpeg")
        cliente.fechar()
