"""Testes do novo escopo (sem CARLA e sem LLM): Fase X (voltas autônomas) e Fase Y (malha fechada).

Rodar a partir da raiz do repositório:
    python -m pytest tests -q

A coleta da Fase X usa um módulo `carla` falso com o Traffic Manager. A Fase Y roda de ponta a
ponta com o servidor (TestClient), o provedor falso e o mundo falso: os agentes recebem o plano
de uma volta com uma frenagem brusca, devolvem o ajuste e o caminhão Y passa sem o evento.
"""

from __future__ import annotations

import copy
import json
import math
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from auxiliares import FRENAGEM, marcar, pista_estadio_xodr, telemetria
from fastapi.testclient import TestClient

from mover.agentes import eventos_sinteticos
from mover.agentes.especialistas import EspecialistaJerk
from mover.agentes.grafo import CamadaAgentica
from mover.agentes.llm import FalsoLLM
from mover.agentes.supervisor import Supervisor
from mover.agentes.treinar_especialista_ml import treinar
from mover.config import carregar_yaml
from mover.interface.cena import montar_cena_poses
from mover.servidor.app import criar_app
from mover.simulacao import voltas_autonomas
from mover.simulacao.opendrive import ler_xodr
from mover.simulacao.plano_velocidade import PlanoVelocidade, perfil_ajustado, suavizar_abaixo
from mover.simulacao.replay_carla import ClienteAgentes, MundoFalso
from mover.simulacao.voltas_autonomas import ColetorVoltas, salvar_volta, telemetria_carla
from mover.simulacao.voltas_com_agentes import (comparar, poses_no_mapa, rodar_volta_y, telemetria_executada,
                                                trajeto_da_volta)
from mover.tratamento.tratar_dados import COLUNAS_CARLA

INICIO = datetime(2026, 10, 10, 14, 0, 0, tzinfo=timezone(timedelta(hours=-3)))
HZ = 20.0


@pytest.fixture(scope="module")
def cfg() -> dict[str, Any]:
    return carregar_yaml("config/agentes.yaml")


@pytest.fixture(scope="module")
def cfg_trat() -> dict[str, Any]:
    return carregar_yaml("config/tratamento.yaml")


def bruto(v: np.ndarray, yaw: np.ndarray | None = None) -> pd.DataFrame:
    """Amostras como as da coleta no CARLA (relógio do simulador começando em 100 s)."""
    n = len(v)
    yaw = np.zeros(n) if yaw is None else yaw
    r = np.radians(yaw)
    vx, vy = v * np.cos(r), v * np.sin(r)
    x = np.concatenate([[0.0], np.cumsum(vx[:-1]) / HZ])
    y = np.concatenate([[0.0], np.cumsum(vy[:-1]) / HZ])
    zeros = np.zeros(n)
    return pd.DataFrame({"t": 100.0 + np.arange(n) / HZ, "x": x, "y": y, "z": zeros, "yaw": yaw, "pitch": zeros,
                         "roll": zeros, "vx": vx, "vy": vy, "vz": zeros, "ax": zeros, "ay": zeros, "az": zeros,
                         "throttle": np.full(n, 0.3), "brake": zeros, "steer": zeros, "gear": np.ones(n),
                         "wp_x": x, "wp_y": y, "wp_road_id": np.full(n, 7), "wp_lane_id": np.full(n, -1),
                         "n_collisions": zeros, "n_lane_invasions": zeros})


def velocidade_com_frenagem(duracao_s: float = 40.0, t_frenagem: float = 22.0) -> np.ndarray:
    """30 km/h, frenagem brusca para 3 m/s em 1 s (-5,3 m/s²) e retomada lenta até 6 m/s."""
    t = np.arange(int(duracao_s * HZ)) / HZ
    v = np.full(t.size, 30 / 3.6)
    freia = (t >= t_frenagem) & (t < t_frenagem + 1.0)
    v[freia] = 30 / 3.6 - (30 / 3.6 - 3.0) * (t[freia] - t_frenagem)
    depois = t >= t_frenagem + 1.0
    v[depois] = np.minimum(3.0 + 0.3 * np.maximum(t[depois] - (t_frenagem + 4.0), 0.0), 6.0)
    return v


# ---------------------------------------------------------------------------------------------
# Plano de velocidade
# ---------------------------------------------------------------------------------------------
def test_suavizacao_nunca_passa_do_plano_e_so_muda_em_volta_da_zona():
    t = np.arange(1000) / HZ
    v = velocidade_com_frenagem(50.0)
    assert np.all(suavizar_abaixo(v, 40) <= v + 1e-12)
    zona = {"t_ini": 15.0, "t_fim": 30.0, "janela_s": 4.0, "rampa_s": 1.0}
    perfil = perfil_ajustado(t, v, [zona], HZ)
    assert np.all(perfil <= v + 1e-12)
    # a zona se alarga até o fim da retomada de X (36 s) para a saída não cortar a aceleração no meio
    fora = (t < 15.0) | (t > 45.0)
    assert np.array_equal(perfil[fora], v[fora])
    a_x, a_y = np.gradient(v, 1 / HZ), np.gradient(perfil, 1 / HZ)
    assert a_x.min() < -5.0 and a_y.min() > -2.5  # a frenagem começa antes e fica bem mais suave
    assert np.abs(np.gradient(a_y, 1 / HZ)).max() < 2.0
    assert perfil[np.searchsorted(t, 21.0)] < v[np.searchsorted(t, 21.0)] - 1.0

    lombada = {**zona, "vel_max_kmh": 12.0, "nucleo_ini": 9.0, "nucleo_fim": 11.0, "t_ini": 3.0, "t_fim": 17.0}
    perfil = perfil_ajustado(t, np.full(t.size, 30 / 3.6), [lombada], HZ)
    nucleo = (t >= 9.0) & (t <= 11.0)
    assert perfil[nucleo].max() <= 12 / 3.6 + 1e-9 and perfil[0] == pytest.approx(30 / 3.6)


def test_ajuste_atrasado_entra_a_partir_do_instante_atual_sem_salto():
    t = np.arange(800) / HZ
    v = velocidade_com_frenagem()
    plano = PlanoVelocidade(t, v, rampa_replanejamento_s=1.5)
    assert plano.taxa(5.0) == 1.0 and plano.velocidade(5.0) == pytest.approx(v[100])
    ajuste = {"tipo": "suavizar", "t_ini": 15.0, "t_fim": 30.0, "janela_s": 4.0, "rampa_s": 1.0}
    antes = plano.velocidade(19.0)
    novos = plano.atualizar([(7, ajuste), (8, None)], tau=19.0)
    assert [i for i, _ in novos] == [7] and plano.chegadas[0]["atrasado"]
    assert plano.velocidade(19.0) == pytest.approx(antes)  # sem salto no instante da chegada
    alvo = perfil_ajustado(t, v, [ajuste], HZ)
    depois = t >= 19.0 + 1.5
    assert np.allclose(plano.alvo[depois], alvo[depois])
    assert plano.atualizar([(7, ajuste)], tau=25.0) == []  # o mesmo ajuste não entra duas vezes
    assert all(0.0 < plano.taxa(x) <= 1.0 for x in np.arange(0.0, 40.0, 0.5))


# ---------------------------------------------------------------------------------------------
# Camada agêntica: ajuste executável em cada problema
# ---------------------------------------------------------------------------------------------
def test_problema_traz_o_ajuste_para_o_caminhao(cfg):
    dados = eventos_sinteticos.injetar(telemetria(25.0), [FRENAGEM], {})
    camada = CamadaAgentica(cfg, dados, Supervisor(FalsoLLM(), cfg), especialista_ml=None)
    entradas = [e for k in range(camada.n_blocos) for e in camada.processar_bloco(k)]
    problema = next(e for e in entradas if e["tipo"] == "problema")
    ajuste = problema["ajuste"]
    assert ajuste["tipo"] == "suavizar" and ajuste["vel_max_kmh"] is None
    assert ajuste["t_ini"] == pytest.approx(problema["t_ini"] - 6.0) and ajuste["t_fim"] == pytest.approx(problema["t_fim"] + 6.0)
    assert ajuste["descricao"].startswith("Suavizar a velocidade")
    assert all(e.get("ajuste") is None for e in entradas if e["tipo"] == "log")

    dados = telemetria(20.0)
    marcar(dados, "jerk_long", 8.0, 8.2, 4.0)
    marcar(dados, "acc_vert", 7.9, 8.3, 3.0)
    camada = CamadaAgentica(cfg, dados, Supervisor(FalsoLLM(), cfg), especialista_ml=None)
    lombada = next(e for k in range(camada.n_blocos) for e in camada.processar_bloco(k) if e["tipo"] == "problema")
    assert lombada["causa"] == "irregularidade_via"
    assert lombada["ajuste"]["tipo"] == "limitar_velocidade" and lombada["ajuste"]["vel_max_kmh"] == 12.0


# ---------------------------------------------------------------------------------------------
# Fase X: telemetria e coleta
# ---------------------------------------------------------------------------------------------
def test_telemetria_do_carla_sai_no_formato_da_etapa_1(cfg_trat):
    n = 600
    yaw = np.zeros(n)
    curva = (np.arange(n) >= 300) & (np.arange(n) < 390)  # 4,5 s a +20°/s: curva de 90° à direita
    yaw[curva] = 20.0 * (np.arange(n)[curva] - 300) / HZ
    yaw[np.arange(n) >= 390] = 90.0
    tel = telemetria_carla(bruto(np.full(n, 8.0), yaw), cfg_trat, INICIO, vel_desejada_kmh=28.8,
                           clima={"cloudiness": 5.0, "precipitation": 0.0, "sun_altitude": 40.0})
    assert list(tel.columns[:len(COLUNAS_CARLA)]) == COLUNAS_CARLA
    assert tel["sim_time"].iloc[0] == 0.0 and tel["speed_kmh"].iloc[100] == pytest.approx(28.8)
    no_meio = tel.iloc[340]
    assert no_meio["yaw_rate_dps"] == pytest.approx(20.0, abs=0.5)  # positivo à direita, como no CARLA
    assert no_meio["acc_lat"] == pytest.approx(8.0 * math.radians(20.0), abs=0.1)
    assert "curva_dir" in set(tel["manobra"]) and tel["fonte"].eq("real").all()
    assert tel["hora_local"].iloc[0] == "2026-10-10T14:00:00.000-03:00"
    assert tel["rumo_graus"].iloc[0] == pytest.approx(90.0) and tel["rumo_graus"].iloc[-1] == pytest.approx(180.0)
    assert tel["odom_m"].iloc[-1] == pytest.approx(8.0 * (n - 1) / HZ, rel=1e-3)
    assert not tel["jerk_long"].isna().any() and tel["cmd_target_speed_kmh"].iloc[0] == 28.8


def carla_da_coleta() -> types.ModuleType:
    """O mínimo da API do CARLA 0.9.16 usada pela coleta: mundo síncrono, Traffic Manager e sensores."""
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

    class OpendriveGenerationParameters:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Configuracao:
        synchronous_mode, fixed_delta_seconds, no_rendering_mode = False, None, False

    class Clima:
        cloudiness, precipitation, sun_altitude_angle = 10.0, 0.0, 45.0

    class Waypoint:
        road_id, lane_id, is_junction = 7, -1, False

        def __init__(self, loc):
            self.transform = Transform(Location(loc.x, 0.0, 0.0))

    class Mapa:
        def get_spawn_points(self):
            return [Transform(Location(0.0, 0.0, 0.5)), Transform(Location(500.0, 0.0, 0.5))]

        def get_waypoint(self, loc, project_to_road=True, lane_type=None):
            return Waypoint(loc)

    class Controle:
        throttle, brake, steer, gear = 0.4, 0.0, 0.0, 1

    class Veiculo:
        def __init__(self, ident, transform):
            self.id, self.attributes, self.is_alive = ident, {"role_name": "mover_autonomo"}, True
            self.x, self.v, self.autopilot = transform.location.x, 0.0, False

        def set_autopilot(self, ligado, porta):
            chamadas.append(("autopilot", ligado, porta))
            self.autopilot = ligado

        def get_control(self):
            return Controle()

        def destroy(self):
            chamadas.append(("destruir_veiculo",))
            self.is_alive = False
            return True

    class Sensor:
        def __init__(self, tipo):
            self.tipo, self.callback = tipo, None

        def listen(self, callback):
            self.callback = callback

        def stop(self):
            self.callback = None

        def destroy(self):
            chamadas.append(("destruir_sensor", self.tipo))
            return True

    class Blueprint:
        def __init__(self, ident):
            self.id, self.atributos = ident, {}

        def has_attribute(self, nome):
            return nome == "role_name"

        def set_attribute(self, nome, valor):
            self.atributos[nome] = valor

    class Biblioteca:
        def find(self, nome):
            return Blueprint(nome)

    class EstadoAtor:
        def __init__(self, veiculo):
            self.veiculo = veiculo

        def get_transform(self):
            return Transform(Location(self.veiculo.x, 0.0, 0.0))

        def get_velocity(self):
            return Vector3D(self.veiculo.v, 0.0, 0.0)

        def get_acceleration(self):
            return Vector3D()

    class Tempo:
        def __init__(self, segundos):
            self.elapsed_seconds = segundos

    class Snapshot:
        def __init__(self, mundo):
            self.timestamp, self.mundo = Tempo(mundo.tempo), mundo

        def find(self, ident):
            veiculo = self.mundo.veiculos.get(ident)
            return EstadoAtor(veiculo) if veiculo is not None and veiculo.is_alive else None

    class ListaAtores(list):
        def filter(self, padrao):
            return [a for a in self if a.is_alive]

    class Espectador:
        def set_transform(self, transformacao):
            pass

    class Mundo:
        def __init__(self):
            self.config, self.aplicadas, self.tempo = Configuracao(), [], 10.0
            self.veiculos: dict[int, Veiculo] = {}
            self.sensores: list[Sensor] = []

        def get_settings(self):
            return copy.copy(self.config)

        def apply_settings(self, config):
            self.config = copy.copy(config)
            self.aplicadas.append((config.synchronous_mode, config.fixed_delta_seconds))

        def get_map(self):
            return Mapa()

        def get_blueprint_library(self):
            return Biblioteca()

        def get_actors(self):
            return ListaAtores(self.veiculos.values())

        def try_spawn_actor(self, bp, transformacao):
            veiculo = Veiculo(100 + len(self.veiculos), transformacao)
            self.veiculos[veiculo.id] = veiculo
            return veiculo

        def spawn_actor(self, bp, transformacao, attach_to=None):
            self.sensores.append(Sensor(bp.id))
            return self.sensores[-1]

        def get_spectator(self):
            return Espectador()

        def get_weather(self):
            return Clima()

        def get_snapshot(self):
            return Snapshot(self)

        def tick(self):
            dt = self.config.fixed_delta_seconds or 0.05
            self.tempo += dt
            for veiculo in self.veiculos.values():
                if veiculo.is_alive and veiculo.autopilot:
                    veiculo.v = min(mod.tm.desejada[veiculo.id] / 3.6, veiculo.v + 2.0 * dt)
                    veiculo.x += veiculo.v * dt
            return int(round(self.tempo / dt))

    class TrafficManager:
        def __init__(self, porta):
            self.porta, self.desejada, self.sementes = porta, {}, []
            mod.tm = self

        def set_synchronous_mode(self, ligado):
            chamadas.append(("tm_sincrono", ligado))

        def set_hybrid_physics_mode(self, ligado):
            pass

        def set_osm_mode(self, ligado):
            chamadas.append(("osm", ligado))

        def set_random_device_seed(self, semente):
            self.sementes.append(semente)

        def ignore_lights_percentage(self, ator, perc):
            chamadas.append(("ignorar_semaforos", perc))

        def ignore_signs_percentage(self, ator, perc):
            pass

        def auto_lane_change(self, ator, ligado):
            pass

        def random_left_lanechange_percentage(self, ator, perc):
            pass

        def random_right_lanechange_percentage(self, ator, perc):
            pass

        def set_desired_speed(self, ator, kmh):
            self.desejada[ator.id] = kmh

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
            chamadas.append(("gerar", parametros.wall_height))
            self.mundo = Mundo()
            return self.mundo

        def get_trafficmanager(self, porta):
            return TrafficManager(porta)

    for nome, valor in list(locals().items()):
        if isinstance(valor, type):
            setattr(mod, nome, valor)
    mod.chamadas = chamadas
    return mod


def test_coleta_grava_voltas_autonomas_com_semente_e_devolve_o_carla_ao_modo_assincrono(monkeypatch, tmp_path):
    carla = carla_da_coleta()
    monkeypatch.setitem(sys.modules, "carla", carla)
    cfg_sim = carregar_yaml("config/simulacao.yaml")
    cfg_sim["voltas"].update(pasta=str(tmp_path), duracao_min_s=1.0, aquecimento_s=0.1)
    arquivo_cfg = tmp_path / "simulacao.yaml"
    arquivo_cfg.write_text(yaml.safe_dump(cfg_sim, allow_unicode=True), encoding="utf-8")

    assert voltas_autonomas.main(["--config", str(arquivo_cfg), "--voltas", "2", "--duracao", "4", "--semente", "5"]) == 0
    mundo = carla.cliente.mundo
    assert ("gerar", 0.0) in carla.chamadas and ("osm", True) in carla.chamadas
    assert carla.tm.porta == 8100 and carla.tm.sementes == [5, 6]
    assert ("autopilot", True, 8100) in carla.chamadas and ("ignorar_semaforos", 100.0) in carla.chamadas
    assert mundo.aplicadas[0] == (True, 0.05) and mundo.aplicadas[-1] == (False, None)  # volta ao modo assíncrono
    assert carla.chamadas[-1] == ("tm_sincrono", False)
    assert sum(c == ("destruir_veiculo",) for c in carla.chamadas) == 2
    assert sum(c[0] == "destruir_sensor" for c in carla.chamadas) == 4  # colisão e invasão de faixa, por volta

    for indice, semente in ((1, 5), (2, 6)):
        pasta = tmp_path / f"volta_{indice:03d}"
        meta = json.loads((pasta / "volta.json").read_text(encoding="utf-8"))
        tel = pd.read_csv(pasta / "telemetria.csv")
        assert meta["semente"] == semente and 20.0 <= meta["velocidade_desejada_kmh"] <= 30.0
        assert meta["encerrada_por"] == "duracao" and meta["volta"] == pasta.name
        assert len(tel) == 80 and tel["speed_kmh"].iloc[-1] > 20.0
        assert tel["gnss_lat"].between(-23.70, -23.60).all()  # inverso do alinhamento da Etapa 3


def test_coletor_descarta_volta_quando_o_traffic_manager_remove_o_caminhao(monkeypatch):
    carla = carla_da_coleta()
    monkeypatch.setitem(sys.modules, "carla", carla)
    coletor = ColetorVoltas(carregar_yaml("config/simulacao.yaml"), seguir_camera=False)
    coletor.conectar()
    coletor.preparar_mundo("<OpenDRIVE/>")
    tick_original = coletor.world.tick

    def tick_que_remove():
        frame = tick_original()
        if coletor.world.tempo > 12.0:  # via sem saída: o TM (modo OSM) remove o veículo
            for veiculo in coletor.world.veiculos.values():
                veiculo.is_alive = False
        return frame

    coletor.world.tick = tick_que_remove
    amostras, meta = coletor.gravar_volta(9, 10.0)
    assert meta["encerrada_por"] == "removido pelo Traffic Manager" and 0 < len(amostras) < 200
    coletor.encerrar()
    coletor.encerrar()  # idempotente


# ---------------------------------------------------------------------------------------------
# Fase Y: malha fechada de ponta a ponta (sem CARLA)
# ---------------------------------------------------------------------------------------------
def test_volta_com_agentes_recebe_o_ajuste_antes_e_passa_sem_a_frenagem_brusca(cfg, cfg_trat, tmp_path):
    tel_x = telemetria_carla(bruto(velocidade_com_frenagem()), cfg_trat, INICIO, vel_desejada_kmh=30.0)
    salvar_volta(tmp_path, "volta_001", tel_x, {"semente": 1})
    eventos_x = EspecialistaJerk(cfg["limiares"]).detectar(tel_x, -math.inf, math.inf)
    assert [(e.tipo, e.nivel) for e in eventos_x] == [("frenagem_brusca", "critico")]

    app = criar_app(cfg, {"servidor": {"pasta_sessoes": str(tmp_path / "sessoes")}, "voltas": {"pasta": str(tmp_path)}},
                    fabrica_supervisor=lambda opcoes: Supervisor(FalsoLLM(), cfg), sem_ml=True)
    impressos: list[str] = []
    trajeto = trajeto_da_volta(tel_x)
    mundo = MundoFalso()
    with TestClient(app) as http:
        assert http.post("/sessao", json={"volta": "../voltas"}).status_code == 400
        assert http.post("/sessao", json={"volta": "volta_999"}).status_code == 404
        assert http.get("/sessao/previsao").status_code == 404  # ainda sem sessão
        cliente = ClienteAgentes(http)
        sessao = cliente.abrir_sessao({"volta": "volta_001", "tempo_real": False})
        assert sessao["volta"] == "volta_001" and sessao["n_blocos"] == 4
        mundo.preparar(trajeto, None, tel_x)
        res = rodar_volta_y(trajeto, tel_x["speed_mps"].to_numpy(float), cliente, mundo, sessao, sem_espera=True,
                            imprimir=impressos.append)
        fim = cliente.encerrar_sessao()
        cliente.fechar()

    assert not res.interrompido and [b["bloco"] for b in res.blocos] == [0, 1, 2, 3]
    assert len(res.ajustes) == 1 and not res.ajustes[0]["atrasado"]  # chegou antes do início da zona
    assert res.ajustes[0]["tau_chegada_s"] <= 14.0 + 0.1  # no ponto de decisão do bloco 2 (20 s - 6 s)
    assert any("ajuste #" in linha for linha in impressos)
    tau = np.asarray(res.tau)
    assert np.all(np.diff(tau) > 0) and res.ticks == mundo.quadros > len(tel_x)  # Y anda mais devagar que X
    assert res.estados == cliente.estados_enviados > 0 and mundo.ultima_pose[1] == 0.0
    assert fim["sessao"]["encerrada"] and Path(fim["sessao"]["arquivo"]).parent == tmp_path / "volta_001"

    tel_y = telemetria_executada(tau, tel_x, cfg_trat, INICIO)
    v_x_no_ponto = np.interp(tau, tel_x["sim_time"], tel_x["speed_mps"])
    assert np.all(tel_y["speed_mps"].to_numpy() <= v_x_no_ponto + 0.35)  # Y nunca passa do plano de X
    comparacao = comparar(telemetria_executada(tel_x["sim_time"].to_numpy(float), tel_x, cfg_trat, INICIO), tel_y, cfg)
    assert comparacao["x"]["eventos"] == 1 and comparacao["y"]["eventos"] == 0
    assert comparacao["y"]["jerk_max_abs_mps3"] < 2.5 < comparacao["x"]["jerk_max_abs_mps3"]
    assert comparacao["acrescimo_tempo_s"] > 0


def test_cena_da_volta_e_treino_do_ml_com_as_voltas_da_fase_x(cfg, cfg_trat, tmp_path):
    tel = telemetria_carla(bruto(velocidade_com_frenagem(20.0, 10.0)), cfg_trat, INICIO)
    cena = montar_cena_poses(ler_xodr(pista_estadio_xodr(), 0.5), poses_no_mapa(tel), origem="volta_001")
    assert len(cena["trajeto"]["t"]) == len(tel) // 2 and cena["origem"] == "volta_001"

    for nome in ("volta_001", "volta_002"):
        salvar_volta(tmp_path, nome, tel, {})
    cfg_ml = copy.deepcopy(cfg)
    cfg_ml["especialista_ml"]["n_arvores"] = 20
    metadados = treinar(cfg_ml, [tmp_path], saida=tmp_path / "ml.joblib")
    assert metadados["amostras"] == 2 * len(tel) and metadados["origem"].startswith("2 arquivos")
    assert (tmp_path / "ml.joblib").exists()
