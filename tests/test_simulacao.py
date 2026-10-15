"""Testes da simulação (sem CARLA): leitor OpenDRIVE, projeção, alinhamento e o lado CARLA da corrida autônoma.

Rodar a partir da raiz do repositório:
    python -m pytest tests -q

O mapa de teste é um estádio (duas retas e duas curvas de 180°) gerado em tests/auxiliares.py.
O lado CARLA (Traffic Manager, telemetria, câmera, gravador) é testado com um módulo `carla` falso,
que registra as chamadas e leva o caminhão pelo caminho passado ao Traffic Manager.
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
from auxiliares import (LARGURA_M, RAIO_M, RETA_M, bgra_azul, centro_da_faixa, pista_com_quina_xodr, pista_estadio_xodr,
                        plano_estadio, volta_no_estadio, volta_suave_na_quina, telemetria)
from fastapi.testclient import TestClient

from mover.agentes.llm import FalsoLLM
from mover.agentes.supervisor import Supervisor
from mover.config import carregar_yaml
from mover.servidor.app import criar_app
from mover.simulacao.alinhamento import alinhar
from mover.simulacao.camera_painel import codificar_jpeg
from mover.simulacao.opendrive import ler_xodr
from mover.simulacao.projecao import criar_projecao

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
# Lado CARLA da corrida autônoma, com um módulo `carla` falso
# ---------------------------------------------------------------------------------------------
def carla_falso() -> types.ModuleType:
    """O mínimo da API do CARLA 0.9.16 usada por carla_tm.py, registrando as chamadas.

    O Traffic Manager falso leva o caminhão pelo caminho recebido em set_path, com a velocidade
    seguindo a desejada (atraso de 0,3 s, até 6 m/s²): o bastante para o laço de condução rodar.
    """
    mod = types.ModuleType("carla")
    chamadas: list[tuple] = []

    class Vector3D:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x, self.y, self.z = x, y, z

    class Location(Vector3D):
        pass

    class Rotation:
        def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
            self.pitch, self.yaw, self.roll = pitch, yaw, roll

    class Transform:
        def __init__(self, location=None, rotation=None):
            self.location, self.rotation = location or Location(), rotation or Rotation()

        def get_forward_vector(self):
            r = math.radians(self.rotation.yaw)
            return Vector3D(math.cos(r), math.sin(r), 0.0)

        def get_right_vector(self):
            r = math.radians(self.rotation.yaw)
            return Vector3D(-math.sin(r), math.cos(r), 0.0)

    class VehicleControl:
        def __init__(self, throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False, manual_gear_shift=False, gear=0):
            self.throttle, self.steer, self.brake, self.hand_brake, self.gear = throttle, steer, brake, hand_brake, gear

    class LaneType:
        Any, Driving = -2, 2

    class OpendriveGenerationParameters:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Configuracao:
        synchronous_mode, fixed_delta_seconds, no_rendering_mode = False, None, False

    class WeatherParameters:
        ClearNoon = "ClearNoon"

    class Waypoint:
        transform = Transform(Location(0.0, 0.0, 0.4))

    class Mapa:
        name = "OpenDriveMap"

        def get_waypoint(self, location, project_to_road=True, lane_type=LaneType.Driving):
            return Waypoint()

        def to_opendrive(self):
            return "<OpenDRIVE/>"

    class Caixa:
        location, extent = Location(0.0, 0.0, 1.9), Location(4.0, 1.3, 1.7)

    class Ator:
        type_id, id, bounding_box = "vehicle.carlamotors.european_hgv", 42, Caixa()

        def __init__(self, transformacao):
            self.tf = transformacao
            self.v, self.a, self.s_caminho = 0.0, 0.0, 0.0
            self.autopiloto, self.controles, self.destruido = None, [], False

        def apply_control(self, controle):
            self.controles.append(controle)

        @property
        def is_alive(self):
            return not self.destruido

        def set_autopilot(self, ligado=True, porta=8000):
            chamadas.append(("autopiloto", ligado, porta))
            self.autopiloto = (ligado, porta)

        def get_transform(self):
            return self.tf

        def get_velocity(self):
            f = self.tf.get_forward_vector()
            return Vector3D(f.x * self.v, f.y * self.v, 0.0)

        def get_acceleration(self):
            f = self.tf.get_forward_vector()
            return Vector3D(f.x * self.a, f.y * self.a, 0.0)

        def get_angular_velocity(self):
            return Vector3D(0.0, 0.0, 0.0)

        def get_control(self):
            return VehicleControl(throttle=max(0.0, self.a) / 3.0, brake=max(0.0, -self.a) / 6.0)

        def destroy(self):
            chamadas.append(("destruir_ator",))
            self.destruido = True
            return True

    class TrafficManager:
        def __init__(self, porta):
            self.porta, self.sincrono, self.desejada, self.caminho, self.ajustes = porta, False, {}, [], {}

        def get_port(self):
            return self.porta

        def set_synchronous_mode(self, ligado):
            chamadas.append(("tm_sincrono", ligado))
            self.sincrono = ligado

        def set_random_device_seed(self, semente):
            self.ajustes["semente"] = semente

        def set_hybrid_physics_mode(self, ligado):
            self.ajustes["hibrida"] = ligado

        def auto_lane_change(self, ator, ligado):
            self.ajustes["troca_faixa"] = ligado

        def ignore_lights_percentage(self, ator, perc):
            self.ajustes["semaforos"] = perc

        def ignore_signs_percentage(self, ator, perc):
            self.ajustes["placas"] = perc

        def distance_to_leading_vehicle(self, ator, dist):
            self.ajustes["lider"] = dist

        def set_desired_speed(self, ator, kmh):
            self.desejada[ator.id] = kmh

        def set_path(self, ator, pontos, empty_buffer=True):
            chamadas.append(("caminho", len(pontos)))
            p0 = ator.tf.location
            self.caminho = [(p0.x, p0.y)] + [(p.x, p.y) for p in pontos]
            ator.s_caminho = 0.0

        def mover(self, ator, dt):
            """Avança o caminhão pelo caminho (a velocidade persegue a desejada)."""
            alvo = self.desejada.get(ator.id, 0.0) / 3.6 if ator.autopiloto and ator.autopiloto[0] else 0.0
            a = max(-6.0, min(3.0, (alvo - ator.v) / 0.3))
            ator.v = max(0.0, ator.v + a * dt)
            ator.a = a if ator.v > 0 or a > 0 else 0.0
            ator.s_caminho += ator.v * dt
            resto = ator.s_caminho
            for (x0, y0), (x1, y1) in zip(self.caminho, self.caminho[1:]):
                comp = math.hypot(x1 - x0, y1 - y0)
                if resto <= comp or (x1, y1) == self.caminho[-1]:
                    u = min(1.0, resto / comp) if comp > 1e-9 else 0.0
                    yaw = math.degrees(math.atan2(y1 - y0, x1 - x0))
                    ator.tf = Transform(Location(x0 + u * (x1 - x0), y0 + u * (y1 - y0), 0.4), Rotation(yaw=yaw))
                    return
                resto -= comp

    class Timestamp:
        def __init__(self, t):
            self.elapsed_seconds = t

    class Snapshot:
        def __init__(self, frame, t):
            self.frame, self.timestamp = frame, Timestamp(t)

    class Blueprint:
        id = "vehicle.carlamotors.european_hgv"

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
            assert nome in ("vehicle.carlamotors.european_hgv", "sensor.camera.rgb"), nome
            return mod.blueprint if nome == "vehicle.carlamotors.european_hgv" else mod.blueprint_camera

    class Espectador:
        def __init__(self):
            self.transformacoes = []

        def set_transform(self, t):
            self.transformacoes.append(t)

    class Mundo:
        def __init__(self):
            self.config, self.aplicadas, self.ticks, self.tempo = Configuracao(), [], 0, 1234.5
            self.clima, self.spectator, self.ator, self.sensores = None, Espectador(), None, []

        def get_settings(self):
            return copy.copy(self.config)

        def apply_settings(self, config):
            self.config = copy.copy(config)
            self.aplicadas.append((config.synchronous_mode, config.fixed_delta_seconds, config.no_rendering_mode))
            chamadas.append(("apply_settings", config.synchronous_mode))

        def get_map(self):
            return Mapa()

        def get_blueprint_library(self):
            return Biblioteca()

        def try_spawn_actor(self, bp, transformacao):
            chamadas.append(("spawn", transformacao.location.x, transformacao.location.y, transformacao.rotation.yaw))
            self.ator = Ator(transformacao)
            return self.ator

        def spawn_actor(self, bp, transformacao, attach_to=None, attachment_type=AttachmentType.Rigid):
            chamadas.append(("spawn_sensor", bp.id, attach_to))
            self.sensores.append(Sensor())
            return self.sensores[-1]

        def get_spectator(self):
            return self.spectator

        def set_weather(self, clima):
            self.clima = clima

        def tick(self):
            self.ticks += 1
            dt = self.config.fixed_delta_seconds or 0.05
            self.tempo += dt
            if self.ator is not None and not self.ator.destruido:
                mod.cliente.tm.mover(self.ator, dt)
            return self.ticks

        def get_snapshot(self):
            return Snapshot(self.ticks, self.tempo)

    class Client:
        def __init__(self, host, porta):
            self.mundo, self.tm, self.gravando = Mundo(), None, None
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

        def get_available_maps(self):
            return ["/Game/Carla/Maps/Town10HD_Opt", "/Game/campus_maua_pkg/Maps/campus_maua/campus_maua"]

        def load_world(self, nome):
            chamadas.append(("carregar", nome))
            self.mundo = Mundo()
            return self.mundo

        def get_trafficmanager(self, porta=8000):
            self.tm = TrafficManager(porta)
            return self.tm

        def start_recorder(self, nome, additional_data=False):
            chamadas.append(("gravar", nome, additional_data))
            self.gravando = nome
            return nome

        def stop_recorder(self):
            chamadas.append(("parar_gravacao",))
            self.gravando = None

    for nome, valor in list(locals().items()):
        if isinstance(valor, type):
            setattr(mod, nome, valor)
    mod.chamadas, mod.blueprint, mod.blueprint_camera = chamadas, Blueprint(), BlueprintCamera()
    return mod


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


def _simulador(monkeypatch, voltas: int = 1, enviar=None, mapa_carla: str = "", pasta_xodr=None):
    from mover.autonomo.carla_tm import SimuladorCarla

    carla = carla_falso()
    monkeypatch.setitem(sys.modules, "carla", carla)
    plano = plano_estadio()
    cfg = carregar_yaml("config/simulacao.yaml")
    cfg["carla"] = {**cfg["carla"], "mapa_carla": mapa_carla, "pasta_xodr_carla": str(pasta_xodr or "")}
    sim = SimuladorCarla(plano.rota, cfg, None, 0.05, "<OpenDRIVE/>", voltas=voltas)
    if enviar is not None:
        sim.ligar_camera_painel(enviar)
    sim.iniciar()
    return carla, plano, sim


def test_simulador_carla_carrega_o_mapa_instalado_como_ele_e_e_salva_o_xodr(monkeypatch, tmp_path):
    carla, _, sim = _simulador(monkeypatch, mapa_carla="campus_maua", pasta_xodr=tmp_path)
    assert ("carregar", "/Game/campus_maua_pkg/Maps/campus_maua/campus_maua") in carla.chamadas
    assert not [c for c in carla.chamadas if c[0] == "gerar"]  # nada de gerar do .xodr nem de preparar o mapa
    assert (tmp_path / "campus_maua_carla.xodr").read_text(encoding="utf-8") == "<OpenDRIVE/>"
    sim.encerrar()
    with pytest.raises(SystemExit, match="não está instalado.*Town10HD_Opt"):
        _simulador(monkeypatch, mapa_carla="outro_mapa", pasta_xodr=tmp_path)


def test_simulador_carla_liga_o_piloto_automatico_le_a_telemetria_e_restaura_o_simulador(monkeypatch, tmp_path):
    carla, plano, sim = _simulador(monkeypatch)
    gerar = [c for c in carla.chamadas if c[0] == "gerar"]
    assert len(gerar) == 1 and gerar[0][1] == "<OpenDRIVE/>" and gerar[0][2].wall_height == 0.0
    world, tm = carla.cliente.mundo, carla.cliente.tm
    assert world.aplicadas[0] == (True, 0.05, False) and world.clima == "ClearNoon"
    assert tm.porta == 8100 and tm.sincrono  # a 8000 é do servidor da página
    ator = world.ator
    assert ator.autopiloto == (True, 8100) and ator.controles[0].hand_brake and not ator.controles[-1].hand_brake
    assert (tm.ajustes["troca_faixa"], tm.ajustes["semaforos"], tm.ajustes["placas"]) == (False, 100.0, 100.0)
    _, x, y, yaw = next(c for c in carla.chamadas if c[0] == "spawn")
    p0 = plano.rota.ponto(0.0)
    assert (x, y, yaw) == pytest.approx((p0.x, -p0.y, -math.degrees(p0.rumo)), abs=1e-6)  # y e yaw do CARLA invertidos
    n_pontos = next(c[1] for c in carla.chamadas if c[0] == "caminho")
    assert n_pontos == math.ceil((plano.comprimento + 30.0 - 8.0) / 10.0)  # a corrida inteira de uma vez

    linhas = [sim.passo(5.0) for _ in range(60)]  # 3 s pedindo 18 km/h
    assert tm.desejada[ator.id] == pytest.approx(18.0)
    ultima = linhas[-1]
    assert ultima["sim_time"] == pytest.approx(3.0) and ultima["fonte"] == "carla"
    assert ultima["speed_mps"] == pytest.approx(5.0, abs=0.05) and ultima["acc_long"] == pytest.approx(0.0, abs=0.1)
    andado = sum(l["speed_mps"] for l in linhas) * 0.05
    assert ultima["d"] == pytest.approx(andado, abs=0.5) and abs(ultima["desvio_m"]) < 0.3  # o caminho do TM é a rota
    assert {"x", "y", "rumo_graus", "x_carla", "yaw_carla", "throttle", "brake", "steer", "volta", "rampa"} <= set(ultima)
    assert world.spectator.transformacoes  # a janela do CARLA segue o caminhão

    caminho = sim.gravar(tmp_path / "corrida" / "gravacao_carla.log")
    assert Path(caminho).is_absolute() and ("gravar", caminho, True) in carla.chamadas
    inicio = len(carla.chamadas)
    sim.encerrar()
    ordem = [c[0] for c in carla.chamadas[inicio:]]
    # gravador, CARLA de volta ao modo assíncrono, TM idem, e só então o caminhão sai
    assert ordem == ["parar_gravacao", "apply_settings", "tm_sincrono", "autopiloto", "destruir_ator"]
    assert world.aplicadas[-1][0] is False and ator.destruido
    sim.encerrar()  # idempotente
    assert len(carla.chamadas) == inicio + len(ordem) + 1  # só o tm_sincrono de novo


def test_corrida_completa_no_carla_falso_para_na_chegada(monkeypatch):
    from mover.autonomo.conducao import Conducao, OpcoesConducao
    from mover.autonomo.montagem import montar

    carla, plano, sim = _simulador(monkeypatch)
    m = montar(carregar_yaml("config/autonomo.yaml"), plano=plano)
    res = Conducao(plano, sim, m.governador("bruto", 1), None, OpcoesConducao(voltas=1, sem_espera=True)).rodar()
    sim.encerrar()
    assert res.completa and res.simulador == "carla"
    assert res.linhas[-1]["d"] == pytest.approx(plano.comprimento, abs=10.0) and res.linhas[-1]["speed_mps"] < 0.06
    assert min(l["speed_mps"] for l in res.linhas if abs(l["d"] - 200.0) < 6.0) < 0.05  # parou na portaria


def test_caminhao_removido_pelo_carla_encerra_a_corrida_com_o_diagnostico(monkeypatch, caplog):
    from mover.autonomo.conducao import Conducao, OpcoesConducao
    from mover.autonomo.montagem import montar

    carla, plano, sim = _simulador(monkeypatch)
    world = carla.cliente.mundo
    tick = world.tick

    def tick_que_remove():  # o Traffic Manager tira o caminhão do mundo no meio da volta
        frame = tick()
        if world.ticks > 400:
            world.ator.destruido = True
        return frame

    world.tick = tick_que_remove
    m = montar(carregar_yaml("config/autonomo.yaml"), plano=plano)
    with caplog.at_level(logging.ERROR, logger="mover.autonomo"):
        res = Conducao(plano, sim, m.governador("bruto", 1), None, OpcoesConducao(voltas=1, sem_espera=True)).rodar()
    sim.encerrar()
    assert res.motivo_fim == "removido" and not res.completa and len(res.linhas) == 400 - 20  # 20 ticks assentando
    assert "o CARLA removeu o caminhão" in caplog.text and f"d = {res.linhas[-1]['d']:.1f} m" in caplog.text


def test_verificacao_da_topologia_acha_o_beco_sem_saida_antes_da_largada(monkeypatch):
    from mover.autonomo.rota import para_carla

    carla, plano, sim = _simulador(monkeypatch)
    assert sim.topologia == {"verificada": False}  # o waypoint falso não tem next()

    class WaypointNaRota:
        road_id, lane_id, is_junction = 257, -1, False
        fim_m = 150.0

        def __init__(self, d):
            p = plano.rota.ponto(d)
            x, y, _ = para_carla(p.x, p.y, p.rumo)
            self.d, self.transform = d, carla.Transform(carla.Location(x, y, 0.0))

        def next(self, passo):
            return [] if self.d + passo > self.fim_m else [WaypointNaRota(self.d + passo)]

    sim.mapa = types.SimpleNamespace(get_waypoint=lambda loc, **kw: WaypointNaRota(0.0))
    problemas = sim.verificar_topologia()["problemas"]
    assert [(p["tipo"], p["via"], p["faixa"]) for p in problemas] == [("beco sem saída", 257, -1)]
    assert problemas[0]["s_m"] == pytest.approx(150.0, abs=2.5)
    WaypointNaRota.fim_m = math.inf
    resultado = sim.verificar_topologia()
    assert resultado["problemas"] == [] and resultado["percorrido_m"] >= plano.comprimento
    sim.encerrar()


def test_juncoes_de_passagem_com_conector_minusculo_viram_ligacao_direta_no_mapa_do_carla(tmp_path):
    import xml.etree.ElementTree as ET

    from mover.autonomo.rota_mapa import rota_do_mapa
    from mover.autonomo.xodr_carla import colapsar_juncoes_de_passagem, preparar_para_carla
    from mover.config import caminho

    original = caminho("data/mapas/mapa_final.xodr").read_text(encoding="utf-8")
    texto, feitas = colapsar_juncoes_de_passagem(original, 1.0)
    por_conector = {f["conector"]: f for f in feitas}
    assert por_conector["287"] == {"juncao": "11", "conector": "287", "conector_m": 0.112, "de": "257", "para": "258",
                                   "sem_fresta": True}
    assert all(f["conector_m"] < 1.0 for f in feitas) and "285" not in por_conector  # o conector de 11 m fica
    raiz = ET.fromstring(texto)
    estradas = {e.get("id"): e for e in raiz.findall("road")}
    assert "287" not in estradas and "285" in estradas and raiz.find("junction[@id='11']") is None
    assert estradas["257"].find("link/successor").attrib == {"elementType": "road", "elementId": "258", "contactPoint": "start"}
    assert estradas["258"].find("link/predecessor").attrib == {"elementType": "road", "elementId": "257", "contactPoint": "end"}
    assert estradas["257"].find(".//lane[@id='-1']/link/successor").get("id") == "-1"
    assert estradas["258"].find(".//lane[@id='-1']/link/predecessor").get("id") == "-1"
    assert float(estradas["257"].get("length")) == pytest.approx(106.9733078 + 0.11215534, abs=1e-6)
    # o mapa ajustado continua lido igual: a mesma volta, do mesmo tamanho
    texto, ajustes = preparar_para_carla(original, {"colapsar_conectores_m": 1.0, "acostamento_m": 2.0})
    assert ajustes["acostamento_m"] == 2.0 and len(ajustes["juncoes_colapsadas"]) == len(feitas)
    raiz = ET.fromstring(texto)
    for secao in raiz.iter("laneSection"):  # acostamento dos dois lados; a faixa de direção não muda
        faixas = {f.get("id"): f for f in secao.iter("lane")}
        assert faixas["1"].get("type") == faixas["-2"].get("type") == "shoulder"
        assert faixas["-1"].get("type") == "driving" and faixas["-2"].find("width").get("a") == "2.00"
    assert [c.tag for c in raiz.find("road[@id='257']/lanes/laneSection")] == ["left", "center", "right"]
    ajustado = tmp_path / "mapa_carla.xodr"
    ajustado.write_text(texto, encoding="utf-8")
    vias = [255, 250, 257, 258, 259, 260, 263, 252, 262, 272, 273, 274]  # itinerário da volta no mapa_final.xodr
    tipos = ("driving", "restricted", "bidirectional", "parking")
    antes, _ = rota_do_mapa(ler_xodr(caminho("data/mapas/mapa_final.xodr"), 0.5, tipos), vias)
    depois, _ = rota_do_mapa(ler_xodr(ajustado, 0.5, tipos), vias)
    assert depois.comprimento == pytest.approx(antes.comprimento, abs=1.0)
    assert colapsar_juncoes_de_passagem("<OpenDRIVE/>", 1.0) == ("<OpenDRIVE/>", [])
    assert colapsar_juncoes_de_passagem(original, 0.0) == (original, [])
    assert preparar_para_carla("<OpenDRIVE/>", {}) == ("<OpenDRIVE/>", {"juncoes_colapsadas": [], "acostamento_m": 0.0,
                                                                        "secoes_com_acostamento": 0,
                                                                        "georreferencia_removida": False})
    so_geo, info = preparar_para_carla(original, {"colapsar_conectores_m": 0, "acostamento_m": 0})
    assert info["georreferencia_removida"] and "<geoReference" not in so_geo and "<geoReference" in original
    assert ET.fromstring(so_geo).find("header").get("name") == ET.fromstring(original).find("header").get("name")


def test_caminhao_preso_no_carla_encerra_a_corrida_mesmo_com_o_governador_pedindo_pouco(monkeypatch, caplog):
    from mover.autonomo.conducao import Conducao, OpcoesConducao
    from mover.autonomo.montagem import montar

    carla, plano, sim = _simulador(monkeypatch)
    tm = carla.cliente.tm
    mover = tm.mover

    def mover_ate_encalhar(ator, dt):  # as rodas saem da malha: o TM acelera e o caminhão não sai do lugar
        if ator.s_caminho > 60.0:
            ator.v, ator.a = 0.0, 0.0
            return
        mover(ator, dt)

    tm.mover = mover_ate_encalhar
    m = montar(carregar_yaml("config/autonomo.yaml"), plano=plano)
    decisor = m.decisor(1)
    op = OpcoesConducao(voltas=1, usar_llm=False, assincrona=False, sem_espera=True, parado_max_s=5.0)
    with caplog.at_level(logging.ERROR, logger="mover.autonomo"):
        res = Conducao(plano, sim, m.governador("agentes", 1), decisor, op).rodar()
    decisor.encerrar()
    sim.encerrar()
    assert res.motivo_fim == "preso" and res.linhas[-1]["t"] < 60.0
    assert "parado com alvo de" in caplog.text and "d = 6" in caplog.text


def test_caminhao_que_sai_da_rota_encerra_a_corrida_com_a_posicao(monkeypatch, caplog):
    from mover.autonomo.conducao import Conducao, OpcoesConducao
    from mover.autonomo.montagem import montar

    carla, plano, sim = _simulador(monkeypatch)
    tm = carla.cliente.tm
    mover = tm.mover

    def mover_e_desviar(ator, dt):  # o TM pega outra faixa: o caminhão anda, mas longe da rota
        mover(ator, dt)
        if ator.s_caminho > 60.0:
            loc = ator.tf.location
            ator.tf = carla.Transform(carla.Location(loc.x, loc.y + 8.0, loc.z), ator.tf.rotation)

    tm.mover = mover_e_desviar
    m = montar(carregar_yaml("config/autonomo.yaml"), plano=plano)
    op = OpcoesConducao(voltas=1, sem_espera=True, max_desvio_m=5.0)
    with caplog.at_level(logging.WARNING, logger="mover.autonomo"):
        res = Conducao(plano, sim, m.governador("bruto", 1), None, op).rodar()
    sim.encerrar()
    assert res.motivo_fim == "fora_da_rota" and "m da rota em d = 6" in caplog.text
    assert "O caminhão saiu da rota" in caplog.text


def test_caminho_do_tm_nao_tem_pontos_dentro_das_juncoes(monkeypatch):
    from mover.autonomo.rota import para_carla

    carla, plano, sim = _simulador(monkeypatch)
    p = plano.rota.ponto(100.0)
    cx, cy, _ = para_carla(p.x, p.y, p.rumo)

    class WaypointFalso:
        def __init__(self, loc):  # uma junção de ~20 m em volta de d = 100 m
            self.is_junction = math.hypot(loc.x - cx, loc.y - cy) < 10.0

    sim.mapa = types.SimpleNamespace(get_waypoint=lambda loc, **kw: WaypointFalso(loc))
    sim._definir_caminho(0.0)
    caminho = carla.cliente.tm.caminho[1:]  # o 1º é a posição do caminhão
    assert caminho and all(math.hypot(x - cx, y - cy) >= 10.0 for x, y in caminho)
    assert len(caminho) == math.ceil((plano.comprimento + 30.0 - 8.0) / 10.0) - 2  # 98 e 108 m ficam de fora
    sim.encerrar()


def test_tick_sem_resposta_encerra_a_corrida_e_o_vigia_avisa_o_tick_travado(monkeypatch):
    from mover.autonomo.carla_tm import VigiaTick
    from mover.autonomo.conducao import Conducao, OpcoesConducao
    from mover.autonomo.montagem import montar

    carla, plano, sim = _simulador(monkeypatch)
    world = carla.cliente.mundo
    tick = world.tick

    def tick_que_estoura():  # o cliente do CARLA desiste depois do timeout
        if world.ticks > 100:
            raise RuntimeError("time-out of 60000ms while waiting for the simulator")
        return tick()

    world.tick = tick_que_estoura
    m = montar(carregar_yaml("config/autonomo.yaml"), plano=plano)
    res = Conducao(plano, sim, m.governador("bruto", 1), None, OpcoesConducao(voltas=1, sem_espera=True)).rodar()
    sim.encerrar()
    assert res.motivo_fim == "sem_resposta" and len(res.linhas) == 101 - 20  # ticks 21 a 101; o 102º estoura

    travados: list[float] = []
    vigia = VigiaTick(0.2, travados.append)
    vigia.comecar()
    esperar(lambda: travados, prazo_s=3.0)
    vigia.parar()
    assert travados[0] > 0.2
    calmo: list[float] = []
    vigia = VigiaTick(0.2, calmo.append)
    vigia.comecar()
    vigia.terminar()
    time.sleep(1.3)
    vigia.parar()
    assert calmo == []


def test_camera_do_painel_presa_ao_caminhao_manda_jpeg_e_sai_antes_do_caminhao(monkeypatch):
    from PIL import Image

    quadros: list[bytes] = []
    carla, _, sim = _simulador(monkeypatch, enviar=quadros.append)
    world = carla.cliente.mundo
    camera = sim.camera_painel
    assert camera is not None and len(world.sensores) == 1
    _, bp_id, preso_a = next(c for c in carla.chamadas if c[0] == "spawn_sensor")
    assert bp_id == "sensor.camera.rgb" and preso_a is world.ator
    sensor = world.sensores[0]
    sensor.callback(ImagemFalsa(64, 36))
    esperar(lambda: camera.enviados == 1)
    imagem = Image.open(io.BytesIO(quadros[0]))
    assert quadros[0].startswith(b"\xff\xd8") and imagem.size == (64, 36)
    inicio = len(carla.chamadas)
    sim.encerrar()
    ordem = [c[0] for c in carla.chamadas[inicio:]]
    assert ordem.index("destruir_sensor") < ordem.index("destruir_ator") and not sensor.is_listening()


def test_cliente_do_painel_manda_cena_telemetria_e_quadros(tmp_path):
    from mover.autonomo.cliente import ClientePainel, ErroServidor
    from mover.interface.cena import montar_cena

    cfg = carregar_yaml("config/agentes.yaml")
    app = criar_app(cfg, {"servidor": {"pasta_sessoes": str(tmp_path)}, "mapa": {"arquivo": str(tmp_path / "x.xodr")}},
                    fabrica_supervisor=lambda opcoes: Supervisor(FalsoLLM(), cfg),
                    fabrica_llm_chat=lambda opcoes: FalsoLLM(), cfg_autonomo={"missao": {"arquivo": str(tmp_path / "x.yaml")}})
    plano = plano_estadio()
    with TestClient(app) as http:
        cliente = ClientePainel(http, em_segundo_plano=False)
        assert http.get("/cena").status_code == 404
        assert cliente.enviar_cena(montar_cena(plano)) is True
        assert http.get("/cena").json()["origem"] == "piloto"
        assert cliente.enviar_cena({"rota": {}}) is False  # 422: só um aviso
        cliente.abrir_sessao({"modo": "bruto", "voltas": 1, "logs_medidos": False})
        cliente.enviar_telemetria([{"t": 0.05, "sim_time": 0.05, "x": 0.0, "y": 0.0, "d": 0.1, "speed_kmh": 1.0}])
        assert cliente.lotes_enviados == 1 and http.get("/saude").json()["sessao"]["modo"] == "bruto"
        # NaN na telemetria (visto no CARLA) não pode derrubar o lote
        cliente.enviar_telemetria([{"t": 0.1, "sim_time": 0.1, "x": 0.0, "y": 0.0, "d": 0.2, "speed_kmh": 1.0,
                                    "acc_long": float("nan"), "v_alvo_kmh": float("inf")}])
        assert cliente.lotes_enviados == 2
        cliente.postar_quadro(codificar_jpeg(bgra_azul(32, 16), 32, 16))
        assert http.get("/camera/info").json()["quadros"] == 1
        with pytest.raises(ErroServidor, match="HTTP 400"):
            cliente.postar_quadro(b"isto nao e um jpeg")
        fim = cliente.encerrar_sessao({"motivo_fim": "chegada"})
        assert fim["resumo"]["piloto"] == {"motivo_fim": "chegada"}
        cliente.fechar()
