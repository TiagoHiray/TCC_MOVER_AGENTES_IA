"""Fase X: voltas autônomas do caminhão no CARLA, gravadas como o plano que os agentes vão ler.

O caminhão (firetruck) anda sozinho no mapa do campus com o autopilot do Traffic Manager (TM), em
modo síncrono a 20 Hz. Cada volta tem uma semente, que escolhe o ponto de partida e a velocidade
desejada e torna determinísticas as conversões do TM nas junções. Na Fase Y
(voltas_com_agentes.py) o caminhão refaz cada volta e a camada agêntica recebe os 10 s seguintes
deste plano antes de o caminhão executá-los.

A telemetria sai no formato da Etapa 1 (colunas do coleta_carla + extras), então a camada agêntica
a lê sem mudanças:

    data/voltas/volta_001/telemetria.csv   20 Hz, fonte = real
    data/voltas/volta_001/volta.json       semente, partida, velocidade desejada e resumo

Latitude/longitude saem do inverso do alinhamento da Etapa 3 (data/simulacao/alinhamento.json) e
da referência da Etapa 1, não da geolocalização do CARLA, que não entende o geoReference em UTM do
mapa do campus. Sem esses arquivos, gnss_lat/gnss_lon ficam vazios.

Uso, a partir da pasta src/, com o CarlaUE4 0.9.16 aberto (Python 3.12 + carla==0.9.16):
    python -m mover.simulacao.voltas_autonomas                           # 50 voltas (config/simulacao.yaml)
    python -m mover.simulacao.voltas_autonomas --voltas 3 --duracao 60   # teste rápido
    python -m mover.simulacao.voltas_autonomas --manter-mundo --sem-renderizacao
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # execução direta: python src/mover/simulacao/voltas_autonomas.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from mover.config import caminho, carregar_yaml
from mover.tratamento.sinais import G0, passa_baixa
from mover.tratamento.tratar_dados import CASAS_DECIMAIS, COLUNAS_CARLA, rotular_manobras

log = logging.getLogger("mover.simulacao")

ARQUIVO_TELEMETRIA = "telemetria.csv"
ARQUIVO_VOLTA = "volta.json"
ROLE_NAME = "mover_autonomo"

PADRAO_VOLTAS: dict[str, Any] = {
    "pasta": "data/voltas",
    "quantidade": 50,
    "semente_inicial": 1,
    "duracao_s": 150.0,
    "duracao_min_s": 30.0,
    "aquecimento_s": 1.0,
    "velocidade_kmh": [20.0, 30.0],
    "porta_tm": 8100,
    "max_parado_s": 30.0,
}

COLUNAS_BRUTAS = ("t", "x", "y", "z", "yaw", "pitch", "roll", "vx", "vy", "vz", "ax", "ay", "az",
                  "throttle", "brake", "steer", "gear", "wp_x", "wp_y", "wp_road_id", "wp_lane_id",
                  "n_collisions", "n_lane_invasions")


def cfg_voltas(cfg_simulacao: dict[str, Any]) -> dict[str, Any]:
    return {**PADRAO_VOLTAS, **(cfg_simulacao.get("voltas") or {})}


# ---------------------------------------------------------------------------------------------
# Sinais derivados (os mesmos filtros da Etapa 1, para os limiares de jerk valerem igual)
# ---------------------------------------------------------------------------------------------
def derivar_sinais(t: np.ndarray, v: np.ndarray, yaw_graus: np.ndarray, cfg_tratamento: dict[str, Any],
                   acc_vert: np.ndarray | None = None, pitch_graus: np.ndarray | None = None) -> dict[str, Any]:
    """Velocidade, acelerações, jerk, guinada, odometria e manobra a partir de v (m/s) e do yaw do CARLA.

    Convenção do CARLA: yaw cresce virando à direita, então a guinada e a aceleração lateral são
    positivas à direita, como no telemetria.csv da Etapa 1.
    """
    t = np.asarray(t, float)
    v = np.clip(np.asarray(v, float), 0.0, None)
    hz = 1.0 / float(np.median(np.diff(t))) if t.size > 1 else 20.0
    dt = 1.0 / hz
    f = cfg_tratamento.get("filtros", {})
    ordem = int(f.get("ordem", 4))
    corte, corte_jerk = float(f.get("imu_corte_hz", 4.0)), float(f.get("jerk_corte_hz", 1.5))
    corte_manobra = float(f.get("manobra_corte_hz", 1.0))

    gradiente = (lambda x: np.gradient(x, dt)) if t.size > 1 else np.zeros_like  # noqa: E731
    yaw_continuo = np.degrees(np.unwrap(np.radians(np.asarray(yaw_graus, float))))
    guinada_bruta = gradiente(yaw_continuo)  # graus/s, positiva à direita
    a_bruta = gradiente(v)
    lateral_bruta = v * np.radians(guinada_bruta)
    yaw_rate = passa_baixa(guinada_bruta, hz, corte, ordem)
    vert = np.zeros_like(v) if acc_vert is None else passa_baixa(np.asarray(acc_vert, float), hz, corte, ordem)
    grade = np.zeros_like(v) if pitch_graus is None else 100.0 * np.tan(np.radians(np.asarray(pitch_graus, float)))
    with np.errstate(divide="ignore", invalid="ignore"):
        curvatura = np.where(v >= 1.0, np.radians(yaw_rate) / v, np.nan)
    manobra = rotular_manobras(passa_baixa(a_bruta, hz, corte_manobra, ordem),
                               -passa_baixa(guinada_bruta, hz, corte_manobra, ordem), v,
                               cfg_tratamento["manobras"], hz)
    return {
        "speed_mps": v,
        "speed_kmh": v * 3.6,
        "acc_long": passa_baixa(a_bruta, hz, corte, ordem),
        "acc_lat": passa_baixa(lateral_bruta, hz, corte, ordem),
        "acc_vert": vert,
        "jerk_long": gradiente(passa_baixa(a_bruta, hz, corte_jerk, ordem)),
        "jerk_lat": gradiente(passa_baixa(lateral_bruta, hz, corte_jerk, ordem)),
        "yaw_rate_dps": yaw_rate,
        "curvatura": curvatura,
        "grade_pct": grade,
        "odom_m": np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * dt)]),
        "rumo_graus": np.mod(yaw_continuo + 90.0, 360.0),
        "manobra": manobra,
    }


def horas_locais(inicio: datetime, t: np.ndarray) -> list[str]:
    """'2026-10-10T14:36:35.427-03:00' para cada instante (inicio precisa ter fuso)."""
    return [(inicio + timedelta(seconds=float(s))).isoformat(timespec="milliseconds") for s in t]


class GeoDoMapa:
    """Mapa do CARLA -> latitude/longitude, invertendo o alinhamento local -> mapa da Etapa 3."""

    def __init__(self, ref: Any, transformacao: Any):
        self.ref, self.tr = ref, transformacao

    @classmethod
    def carregar(cls, cfg_simulacao: dict[str, Any]) -> "GeoDoMapa | None":
        from mover.gemeo.calibracao_fixa import CalibracaoFixa

        g = cfg_simulacao.get("gemeo", {})
        relatorio = caminho(g.get("calibracao", "data/tratado/relatorio_tratamento.json"))
        alinhamento = caminho(g.get("alinhamento", "data/simulacao/alinhamento.json"))
        if not (relatorio.exists() and alinhamento.exists()):
            log.warning("Sem %s ou %s: gnss_lat/gnss_lon ficam vazios.", relatorio.name, alinhamento.name)
            return None
        cal = CalibracaoFixa.carregar(relatorio, alinhamento)
        return cls(cal.ref, cal.mapa) if cal.mapa is not None else None

    def lat_lon(self, x_carla: np.ndarray, y_carla: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tr = self.tr
        c, s = math.cos(tr.rotacao_rad), math.sin(tr.rotacao_rad)
        dx, dy = np.asarray(x_carla, float) - tr.tx, -np.asarray(y_carla, float) - tr.ty
        return self.ref.de_enu((c * dx + s * dy) / tr.escala, (-s * dx + c * dy) / tr.escala)


def telemetria_carla(bruto: pd.DataFrame, cfg_tratamento: dict[str, Any], inicio: datetime,
                     geo: GeoDoMapa | None = None, vel_desejada_kmh: float | None = None,
                     clima: dict[str, float] | None = None) -> pd.DataFrame:
    """Amostras brutas do CARLA (COLUNAS_BRUTAS) -> telemetria no formato da Etapa 1."""
    t = bruto["t"].to_numpy(float) - float(bruto["t"].iloc[0])
    n = len(t)
    yaw = bruto["yaw"].to_numpy(float)
    pitch = bruto["pitch"].to_numpy(float)
    ry = np.radians(yaw)
    v_frente = bruto["vx"].to_numpy(float) * np.cos(ry) + bruto["vy"].to_numpy(float) * np.sin(ry)
    s = derivar_sinais(t, v_frente, yaw, cfg_tratamento, acc_vert=bruto["az"].to_numpy(float), pitch_graus=pitch)
    hz = 1.0 / float(np.median(np.diff(t))) if n > 1 else 20.0
    corte = float(cfg_tratamento.get("filtros", {}).get("imu_corte_hz", 4.0))
    vazio = np.full(n, np.nan)
    lat, lon = geo.lat_lon(bruto["x"], bruto["y"]) if geo is not None else (vazio, vazio)
    rp = np.radians(pitch)
    controles = {c: bruto[c].to_numpy(float) for c in ("throttle", "brake", "steer")}
    clima = clima or {}
    df = pd.DataFrame({
        "frame": np.arange(n), "sim_time": t,
        "x": bruto["x"], "y": bruto["y"], "z": bruto["z"], "yaw": yaw, "pitch": pitch, "roll": bruto["roll"],
        "speed_mps": s["speed_mps"], **controles, "gear": bruto["gear"],
        "acc_x": passa_baixa(bruto["ax"].to_numpy(float), hz, corte),
        "acc_y": passa_baixa(bruto["ay"].to_numpy(float), hz, corte),
        "acc_z": s["acc_vert"],
        "gnss_lat": lat, "gnss_lon": lon, "gnss_alt": bruto["z"],
        "imu_acc_x": s["acc_long"] + G0 * np.sin(rp), "imu_acc_y": s["acc_lat"],
        "imu_acc_z": G0 * np.cos(rp) + s["acc_vert"],
        "imu_gyro_x": vazio, "imu_gyro_y": vazio, "imu_gyro_z": np.radians(s["yaw_rate_dps"]),
        "imu_compass": np.radians(s["rumo_graus"]),
        "wp_x": bruto["wp_x"], "wp_y": bruto["wp_y"], "wp_road_id": bruto["wp_road_id"], "wp_lane_id": bruto["wp_lane_id"],
        "odom_m": s["odom_m"],
        "cloudiness": clima.get("cloudiness", np.nan), "precipitation": clima.get("precipitation", np.nan),
        "sun_altitude": clima.get("sun_altitude", np.nan),
        "n_collisions": bruto["n_collisions"], "n_lane_invasions": bruto["n_lane_invasions"],
        "cmd_manobra": s["manobra"], "cmd_throttle": controles["throttle"], "cmd_brake": controles["brake"],
        "cmd_steer": controles["steer"], "cmd_target_speed_kmh": vel_desejada_kmh if vel_desejada_kmh else np.nan,
        "t_unix": inicio.timestamp() + t, "hora_local": horas_locais(inicio, t),
        "rumo_graus": s["rumo_graus"], "speed_kmh": s["speed_kmh"],
        "acc_long": s["acc_long"], "acc_lat": s["acc_lat"], "acc_vert": s["acc_vert"],
        "jerk_long": s["jerk_long"], "jerk_lat": s["jerk_lat"], "yaw_rate_dps": s["yaw_rate_dps"],
        "curvatura": s["curvatura"], "grade_pct": s["grade_pct"], "alt_m": bruto["z"],
        "throttle_est": controles["throttle"], "brake_est": controles["brake"], "steer_est": controles["steer"],
        "manobra": s["manobra"], "sigma_pos_m": 0.0, "gps_hacc_m": np.nan, "fonte": "real",
    })
    return df.round({c: CASAS_DECIMAIS.get(c, 5) for c in df.select_dtypes("number").columns})


# ---------------------------------------------------------------------------------------------
# Coleta no CARLA
# ---------------------------------------------------------------------------------------------
class ColetorVoltas:
    """Mundo OpenDRIVE do campus, Traffic Manager e uma volta autônoma por semente."""

    def __init__(self, cfg_simulacao: dict[str, Any], manter_mundo: bool = False, sem_renderizacao: bool = False,
                 seguir_camera: bool = True):
        self.cc = cfg_simulacao.get("carla", {})
        self.cv = cfg_simulacao.get("veiculo", {})
        self.ccam = cfg_simulacao.get("camera", {})
        self.cvol = cfg_voltas(cfg_simulacao)
        self.manter_mundo = manter_mundo or not self.cc.get("carregar_mapa", True)
        self.sem_renderizacao = sem_renderizacao
        self.seguir_camera = seguir_camera and not sem_renderizacao
        self.passo_s = float(self.cc.get("passo_s", 0.05))
        self.porta_tm = int(self.cvol["porta_tm"])
        self.carla: Any = None
        self.client: Any = None
        self.world: Any = None
        self.mapa: Any = None
        self.tm: Any = None
        self.spectator: Any = None
        self.pontos: list[Any] = []
        self.clima: dict[str, float] = {}
        self._config_original: Any = None

    def conectar(self) -> None:
        try:
            import carla
        except ImportError:
            raise SystemExit("O pacote 'carla' não está instalado. Instale o cliente do CARLA 0.9.16 "
                             "(pip install carla==0.9.16, Python 3.12).") from None
        self.carla = carla
        host, porta = self.cc.get("host", "localhost"), int(self.cc.get("porta", 2000))
        self.client = carla.Client(host, porta)
        self.client.set_timeout(float(self.cc.get("timeout_s", 60.0)))
        try:
            versao = self.client.get_server_version()
        except RuntimeError as erro:
            raise SystemExit(f"O CARLA não respondeu em {host}:{porta} ({erro}). Abra o CarlaUE4 antes.") from None
        if versao != self.client.get_client_version():
            log.warning("Versões diferentes: CARLA %s e cliente Python %s.", versao, self.client.get_client_version())
        log.info("CARLA %s em %s:%d", versao, host, porta)
        self.world = self.client.get_world()

    def opendrive_atual(self) -> str:
        return self.world.get_map().to_opendrive()

    def preparar_mundo(self, texto_xodr: str | None) -> None:
        from mover.simulacao.replay_carla import parametros_opendrive

        if not self.manter_mundo:
            log.info("Gerando o mundo OpenDRIVE no CARLA (pode levar alguns segundos)...")
            self.world = self.client.generate_opendrive_world(texto_xodr, parametros_opendrive(self.carla, self.cc))
        self._config_original = self.world.get_settings()
        config = self.world.get_settings()
        config.synchronous_mode = True
        config.fixed_delta_seconds = self.passo_s
        if self.sem_renderizacao:
            config.no_rendering_mode = True
        self.world.apply_settings(config)
        # o TM precisa entrar em modo síncrono logo depois do mundo, no mesmo cliente que dá o tick
        self.tm = self.client.get_trafficmanager(self.porta_tm)
        self.tm.set_synchronous_mode(True)
        self.tm.set_hybrid_physics_mode(False)
        self.tm.set_osm_mode(True)  # via sem saída: o TM remove o veículo em vez de travar
        self._remover_sobras()
        self.mapa = self.world.get_map()
        self.pontos = self._pontos_de_partida()
        self.spectator = self.world.get_spectator()
        clima = self.world.get_weather()
        self.clima = {"cloudiness": float(clima.cloudiness), "precipitation": float(clima.precipitation),
                      "sun_altitude": float(clima.sun_altitude_angle)}
        self.world.tick()
        log.info("Mundo pronto: %d pontos de partida, Traffic Manager na porta %d.", len(self.pontos), self.porta_tm)

    def _remover_sobras(self) -> None:
        """Caminhões deixados por uma execução interrompida (mesmo role_name) saem antes de começar."""
        for ator in self.world.get_actors().filter("vehicle.*"):
            if ator.attributes.get("role_name") in (ROLE_NAME, self.cv.get("role_name", "mover_gemeo")):
                ator.destroy()

    def _pontos_de_partida(self) -> list[Any]:
        pontos = list(self.mapa.get_spawn_points())
        if pontos:
            return pontos
        carla = self.carla
        log.warning("O mapa não tem pontos de partida; usando waypoints das faixas de condução.")
        for wp in self.mapa.generate_waypoints(20.0):
            if not wp.is_junction:
                loc = wp.transform.location
                pontos.append(carla.Transform(carla.Location(x=loc.x, y=loc.y, z=loc.z + 0.5), wp.transform.rotation))
        if not pontos:
            raise SystemExit("O mapa não tem pontos de partida nem faixas de condução.")
        return pontos

    def _blueprint(self) -> Any:
        biblioteca = self.world.get_blueprint_library()
        nome = self.cv.get("blueprint", "vehicle.carlamotors.firetruck")
        try:
            bp = biblioteca.find(nome)
        except Exception:
            bp = (list(biblioteca.filter("vehicle.carlamotors.*")) or list(biblioteca.filter("vehicle.*")))[0]
            log.warning("Blueprint %s não encontrado; usando %s.", nome, bp.id)
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", ROLE_NAME)
        return bp

    def _sensores(self, ator: Any) -> tuple[list[Any], dict[str, int]]:
        contagem = {"colisoes": 0, "invasoes": 0}
        biblioteca = self.world.get_blueprint_library()
        sensores = []
        for nome, chave in (("sensor.other.collision", "colisoes"), ("sensor.other.lane_invasion", "invasoes")):
            sensor = self.world.spawn_actor(biblioteca.find(nome), self.carla.Transform(), attach_to=ator)

            def contar(_evento: Any, chave: str = chave) -> None:
                contagem[chave] += 1

            sensor.listen(contar)
            sensores.append(sensor)
        return sensores, contagem

    def _seguir(self, tf: Any) -> None:
        if not self.seguir_camera:
            return
        carla = self.carla
        r = math.radians(tf.rotation.yaw)
        d, h = float(self.ccam.get("distancia_m", 16)), float(self.ccam.get("altura_m", 8))
        loc = tf.location
        self.spectator.set_transform(carla.Transform(
            carla.Location(x=loc.x - d * math.cos(r), y=loc.y - d * math.sin(r), z=loc.z + h),
            carla.Rotation(pitch=float(self.ccam.get("inclinacao_graus", -20)), yaw=tf.rotation.yaw)))

    def gravar_volta(self, semente: int, duracao_s: float) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Uma volta autônoma com a semente dada; devolve as amostras brutas (COLUNAS_BRUTAS) e os metadados."""
        rng = random.Random(semente)
        self.tm.set_random_device_seed(int(semente))
        vel_min, vel_max = (float(v) for v in self.cvol["velocidade_kmh"])
        vel = round(rng.uniform(vel_min, vel_max), 1)
        ordem = list(range(len(self.pontos)))
        rng.shuffle(ordem)
        bp = self._blueprint()
        ator, partida = None, None
        for i in ordem[:20]:
            ator = self.world.try_spawn_actor(bp, self.pontos[i])
            if ator is not None:
                partida = i
                break
        if ator is None:
            raise RuntimeError("nenhum ponto de partida livre para o caminhão")
        sensores, contagem = self._sensores(ator)
        linhas: list[tuple] = []
        motivo, parado = "duracao", 0
        try:
            for _ in range(int(round(float(self.cvol["aquecimento_s"]) / self.passo_s))):
                self.world.tick()  # o caminhão nasce um pouco acima do asfalto e assenta antes de partir
            ator.set_autopilot(True, self.porta_tm)
            self.tm.ignore_lights_percentage(ator, 100.0)
            self.tm.ignore_signs_percentage(ator, 100.0)
            self.tm.auto_lane_change(ator, False)
            self.tm.random_left_lanechange_percentage(ator, 0.0)
            self.tm.random_right_lanechange_percentage(ator, 0.0)
            self.tm.set_desired_speed(ator, vel)
            max_parado = int(round(float(self.cvol["max_parado_s"]) / self.passo_s))
            for _ in range(int(round(duracao_s / self.passo_s))):
                self.world.tick()
                snap = self.world.get_snapshot()
                estado = snap.find(ator.id)
                if estado is None:
                    motivo = "removido pelo Traffic Manager"
                    break
                tf, vel3, acc3 = estado.get_transform(), estado.get_velocity(), estado.get_acceleration()
                ctrl = ator.get_control()
                wp = self.mapa.get_waypoint(tf.location)
                linhas.append((
                    snap.timestamp.elapsed_seconds, tf.location.x, tf.location.y, tf.location.z,
                    tf.rotation.yaw, tf.rotation.pitch, tf.rotation.roll, vel3.x, vel3.y, vel3.z,
                    acc3.x, acc3.y, acc3.z, ctrl.throttle, ctrl.brake, ctrl.steer, ctrl.gear,
                    wp.transform.location.x if wp else np.nan, wp.transform.location.y if wp else np.nan,
                    wp.road_id if wp else -1, wp.lane_id if wp else 0, contagem["colisoes"], contagem["invasoes"],
                ))
                self._seguir(tf)
                parado = parado + 1 if math.hypot(vel3.x, vel3.y) < 0.1 else 0
                if parado >= max_parado:
                    motivo = "parado"
                    break
        finally:
            self._remover(ator, sensores)
        p = self.pontos[partida]
        meta = {"semente": int(semente), "velocidade_desejada_kmh": vel, "encerrada_por": motivo,
                "partida": {"indice": partida, "x": round(p.location.x, 2), "y": round(p.location.y, 2),
                            "z": round(p.location.z, 2), "yaw": round(p.rotation.yaw, 2)},
                "colisoes": contagem["colisoes"], "invasoes_faixa": contagem["invasoes"],
                "passo_s": self.passo_s, "porta_tm": self.porta_tm}
        return pd.DataFrame(linhas, columns=list(COLUNAS_BRUTAS)), meta

    def _remover(self, ator: Any, sensores: list[Any]) -> None:
        for sensor in sensores:
            try:
                sensor.stop()
                sensor.destroy()
            except RuntimeError:
                pass
        if ator.is_alive:
            try:
                ator.set_autopilot(False, self.porta_tm)
                ator.destroy()
            except RuntimeError:
                pass
        self.world.tick()

    def encerrar(self) -> None:
        """Tira o TM e o mundo do modo síncrono (senão o simulador fica parado). Idempotente."""
        if self.tm is not None:
            try:
                self.tm.set_synchronous_mode(False)
            except RuntimeError:
                pass
            self.tm = None
        if self.world is not None and self._config_original is not None:
            try:
                self.world.apply_settings(self._config_original)
            except RuntimeError as erro:
                log.warning("Não foi possível restaurar as configurações do CARLA: %s", erro)
            self._config_original = None


def resumo_da_volta(tel: pd.DataFrame, limiares: dict[str, Any] | None) -> dict[str, Any]:
    resumo = {"duracao_s": round(float(tel["sim_time"].iloc[-1]) + 0.05, 2) if len(tel) else 0.0,
              "distancia_m": round(float(tel["odom_m"].iloc[-1]), 1) if len(tel) else 0.0,
              "vel_media_kmh": round(float(tel["speed_kmh"].mean()), 1) if len(tel) else 0.0,
              "vel_max_kmh": round(float(tel["speed_kmh"].max()), 1) if len(tel) else 0.0,
              "jerk_max_abs_mps3": round(float(tel["jerk_long"].abs().max()), 2) if len(tel) else 0.0}
    if limiares and len(tel):
        from mover.agentes.especialistas import EspecialistaJerk

        eventos = EspecialistaJerk(limiares).detectar(tel, -math.inf, math.inf)
        resumo["eventos_jerk"] = len(eventos)
        resumo["eventos_criticos"] = sum(e.nivel == "critico" for e in eventos)
    return resumo


def salvar_volta(pasta: Path, nome: str, tel: pd.DataFrame, meta: dict[str, Any]) -> Path:
    destino = pasta / nome
    if destino.exists():
        log.warning("%s já existe e será sobrescrita.", destino)
    destino.mkdir(parents=True, exist_ok=True)
    tel.to_csv(destino / ARQUIVO_TELEMETRIA, index=False, encoding="utf-8", lineterminator="\n")
    (destino / ARQUIVO_VOLTA).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return destino


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fase X: voltas autônomas do caminhão no CARLA (Traffic Manager).")
    parser.add_argument("--config", default="config/simulacao.yaml", help="YAML da simulação (seção voltas)")
    parser.add_argument("--config-agentes", default="config/agentes.yaml", help="limiares de jerk para o resumo")
    parser.add_argument("--config-tratamento", default="config/tratamento.yaml", help="filtros e manobras")
    parser.add_argument("--mapa", help="arquivo .xodr do mundo a gerar (padrão: YAML)")
    parser.add_argument("--voltas", type=int, help="quantas voltas gravar (padrão: YAML)")
    parser.add_argument("--semente", type=int, help="semente da 1ª tentativa; as seguintes somam 1 (padrão: YAML)")
    parser.add_argument("--duracao", type=float, help="segundos por volta (padrão: YAML)")
    parser.add_argument("--pasta", help="pasta das voltas (padrão: YAML)")
    parser.add_argument("--manter-mundo", action="store_true", help="não gera o mundo OpenDRIVE; usa o que está aberto")
    parser.add_argument("--sem-renderizacao", action="store_true", help="no_rendering_mode: mais rápido, sem imagem")
    parser.add_argument("--sem-camera", action="store_true", help="não leva a câmera do simulador atrás do caminhão")
    args = parser.parse_args(argv)

    from mover.servidor.rodar_servidor import configurar_logs

    configurar_logs()
    cfg = carregar_yaml(args.config)
    cvol = cfg_voltas(cfg)
    quantidade = int(args.voltas if args.voltas is not None else cvol["quantidade"])
    semente0 = int(args.semente if args.semente is not None else cvol["semente_inicial"])
    duracao = float(args.duracao or cvol["duracao_s"])
    pasta = caminho(args.pasta or cvol["pasta"])
    cfg_trat = carregar_yaml(args.config_tratamento)
    limiares = carregar_yaml(args.config_agentes).get("limiares")
    geo = GeoDoMapa.carregar(cfg)

    coletor = ColetorVoltas(cfg, args.manter_mundo, args.sem_renderizacao, seguir_camera=not args.sem_camera)
    texto_xodr, nome_mapa = None, "mundo já aberto no CARLA"
    if not coletor.manter_mundo:
        arquivo = caminho(args.mapa or cfg["mapa"]["arquivo"])
        if not arquivo.exists():
            raise SystemExit(f"Mapa não encontrado: {arquivo}")
        texto_xodr, nome_mapa = arquivo.read_text(encoding="utf-8"), arquivo.name
    coletor.conectar()
    salvas, tentativa, codigo = 0, 0, 0
    try:
        coletor.preparar_mundo(texto_xodr)
        while salvas < quantidade and tentativa < 2 * quantidade + 5:
            semente = semente0 + tentativa
            tentativa += 1
            inicio = datetime.now().astimezone()
            try:
                bruto, meta = coletor.gravar_volta(semente, duracao)
            except RuntimeError as erro:
                log.warning("Semente %d: %s; tentando a próxima.", semente, erro)
                continue
            if len(bruto) * coletor.passo_s < float(cvol["duracao_min_s"]):
                log.warning("Semente %d: volta de %.1f s (%s) descartada.", semente, len(bruto) * coletor.passo_s,
                            meta["encerrada_por"])
                continue
            tel = telemetria_carla(bruto, cfg_trat, inicio, geo, meta["velocidade_desejada_kmh"], coletor.clima)
            salvas += 1
            meta.update(volta=f"volta_{salvas:03d}", gerado_em=inicio.isoformat(timespec="seconds"),
                        mapa=nome_mapa, **resumo_da_volta(tel, limiares))
            destino = salvar_volta(pasta, meta["volta"], tel, meta)
            log.info("%s (semente %d): %.0f s, %.0f m, %.0f km/h desejados, %s eventos de jerk -> %s",
                     meta["volta"], semente, meta["duracao_s"], meta["distancia_m"], meta["velocidade_desejada_kmh"],
                     meta.get("eventos_jerk", "?"), destino)
    except KeyboardInterrupt:
        log.warning("Interrompido (Ctrl+C).")
        codigo = 130
    finally:
        coletor.encerrar()
    log.info("%d volta(s) gravada(s) em %s (%d tentativa(s)).", salvas, pasta, tentativa)
    if salvas < quantidade and codigo == 0:
        codigo = 1
    return codigo


if __name__ == "__main__":
    sys.exit(main())
