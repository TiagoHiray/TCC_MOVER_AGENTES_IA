# coleta_carla.py (v7)

import argparse
import csv
import json
import queue
import random
import signal
import sys
import time
from pathlib import Path

import carla
import cv2
import numpy as np
import math


CONFIG = {
    "host": "localhost",
    "port": 2000,
    "town": "Town03",
    "fixed_delta_seconds": 0.05,
    "vehicle_filter": "vehicle.carlamotors.firetruck",
    "camera_width": 800,
    "camera_height": 600,
    "camera_fov": 90,
    "camera_x": 4.0,               # deslocamento para frente (para fora da cabine do caminhao)
    "camera_z": 3.0,               # altura da camera
    "lidar_channels": 16,
    "lidar_range": 30.0,
    "lidar_points_per_second": 100000,
    "lidar_rotation_frequency": 20,
    "gnss_noise_lat_stddev": 1e-5,
    "gnss_noise_lon_stddev": 1e-5,
    "weather": carla.WeatherParameters.ClearNoon,
    "warmup_ticks": 30,
    "n_vehicles": 30,
    "n_pedestrians": 20,
    "try_collision_sensor": True,
    "try_lane_invasion_sensor": True,
    "ticks_between_event_sensors": 10,
    # NOVO
    "save_lidar_npy": True,        # se quiser economizar ainda mais espaço, deixe False
    "video_fps": 20,               # 1/0.05 = 20, casa com fixed_delta
    "video_codec": "mp4v",
    "hud_enabled": True,
    "seg_enabled": True,           # camera de segmentacao semantica (video lado a lado)
    # DADOS EXTERNOS DO CAMINHAO (o CARLA recebe dados e simula o estado do ego)
    "external_data_enabled": True,   # inicia com dados mockados e devolve o controle ao autopilot
    "data_control_mode": "control",  # "control" (throttle/brake/steer) ou "kinematic" (impoe velocidade)
    "data_update_interval_s": 5.0,   # intervalo entre novos pacotes de dados do caminhao
    "run_seed": None,                # semente global da run (spawns, cores, TM) p/ reprodutibilidade
    "data_seed": None,               # semente dos dados mockados (default = run_seed)
    "dados_iniciais": {              # pacote aplicado ao caminhao em t=0
        "manobra": "acelerar",
        "throttle": 0.6, "brake": 0.0, "steer": 0.0,
        "target_speed_kmh": 40.0, "hand_brake": False, "reverse": False,
    },
}


# Classes CityScapes usadas na legenda da segmentacao semantica (nome, RGB)
SEG_LEGEND = [
    ("Road", (128, 64, 128)),
    ("RoadLine", (157, 234, 50)),
    ("SideWalk", (244, 35, 232)),
    ("Vehicle", (0, 0, 142)),
    ("Pedestrian", (220, 20, 60)),
    ("Building", (70, 70, 70)),
    ("Vegetation", (107, 142, 35)),
    ("Pole", (153, 153, 153)),
    ("TrafficSign", (220, 220, 0)),
    ("TrafficLight", (250, 170, 30)),
]


class MiniMap:
    def __init__(self, world_map, size=200, margin=10):
        self.size = size
        self.margin = margin
        wps = world_map.generate_waypoints(5.0)
        xs = [w.transform.location.x for w in wps]
        ys = [w.transform.location.y for w in wps]
        self.min_x, self.max_x = min(xs), max(xs)
        self.min_y, self.max_y = min(ys), max(ys)
        rng = max(self.max_x - self.min_x, self.max_y - self.min_y)
        self.rng = rng if rng > 0 else 1.0

        # Renderiza a malha viária uma vez
        self.bg = np.zeros((size, size, 3), dtype=np.uint8)
        for w in wps:
            px, py = self._w2p(w.transform.location.x, w.transform.location.y)
            cv2.circle(self.bg, (px, py), 1, (90, 90, 90), -1)
        # Canvas persistente: fundo + rastro acumulado (desenhado incrementalmente)
        self.canvas = self.bg.copy()
        self._last_trail_pt = None

    def _w2p(self, x, y):
        px = int((x - self.min_x) / self.rng * (self.size - 1))
        py = int((y - self.min_y) / self.rng * (self.size - 1))
        py = self.size - 1 - py  # inverte y (tela vs mundo)
        return px, py

    def draw(self, frame_bgr, ego_x, ego_y, yaw_deg, npc_locations=None):
        # Desenha apenas o novo segmento do rastro no canvas persistente
        cur = self._w2p(ego_x, ego_y)
        if self._last_trail_pt is not None:
            cv2.line(self.canvas, self._last_trail_pt, cur, (0, 200, 255), 1)
        self._last_trail_pt = cur

        canvas = self.canvas.copy()

        # NPCs
        if npc_locations:
            for nx, ny in npc_locations:
                px, py = self._w2p(nx, ny)
                cv2.circle(canvas, (px, py), 2, (180, 180, 180), -1)

        # Ego + seta de heading
        epx, epy = self._w2p(ego_x, ego_y)
        cv2.circle(canvas, (epx, epy), 5, (0, 0, 255), -1)
        rad = math.radians(yaw_deg)
        dx = int(10 * math.cos(rad))
        dy = -int(10 * math.sin(rad))  # y invertido
        cv2.arrowedLine(canvas, (epx, epy), (epx + dx, epy + dy),
                        (0, 0, 255), 2, tipLength=0.4)

        # Cola no canto inferior direito
        h, w = frame_bgr.shape[:2]
        x0 = w - self.size - self.margin
        y0 = h - self.size - self.margin
        cv2.rectangle(frame_bgr, (x0 - 2, y0 - 2),
                      (x0 + self.size + 1, y0 + self.size + 1),
                      (255, 255, 255), 1)
        frame_bgr[y0:y0 + self.size, x0:x0 + self.size] = canvas
        cv2.putText(frame_bgr, "MINIMAP", (x0 + 4, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


def make_dirs(output_root: Path):
    if CONFIG["save_lidar_npy"]:
        (output_root / "lidar").mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)


def save_metadata(output_root: Path, cfg: dict):
    meta = {k: (str(v) if not isinstance(v, (int, float, str, bool)) else v)
            for k, v in cfg.items()}
    meta["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(output_root / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


# ---------- HUD ----------
def desenhar_hud(frame_bgr, info: dict):
    """Desenha overlay de telemetria por cima do frame BGR (in-place)."""
    h, w = frame_bgr.shape[:2]

    # Fundo semi-transparente (esquerdo)
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (430, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame_bgr, 0.55, 0, frame_bgr)

    # Fundo direito (eventos)
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (w - 280, 0), (w, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame_bgr, 0.55, 0, frame_bgr)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.45
    th = 1
    color = (0, 255, 0)
    color_warn = (0, 200, 255)
    color_bad = (0, 0, 255)

    linhas = [
        f"Frame: {info['frame']}  t={info['sim_time']:.2f}s",
        f"Speed: {info['speed_mps']*3.6:6.2f} km/h  ({info['speed_mps']:.2f} m/s)",
        f"Gear : {info['gear']}",
        f"Throttle: {info['throttle']:.2f}",
        f"Brake   : {info['brake']:.2f}",
        f"Steer   : {info['steer']:+.2f}",
        "",
        f"Pos: x={info['x']:7.1f} y={info['y']:7.1f} z={info['z']:5.1f}",
        f"Yaw: {info['yaw']:+6.1f}  Pitch: {info['pitch']:+5.1f}  Roll: {info['roll']:+5.1f}",
        "",
        f"Acc (veh): x={info['acc_x']:+5.2f} y={info['acc_y']:+5.2f} z={info['acc_z']:+5.2f}",
        f"IMU acc : x={info['imu_acc_x']:+5.2f} y={info['imu_acc_y']:+5.2f} z={info['imu_acc_z']:+5.2f}",
        f"IMU gyro: x={info['imu_gyro_x']:+5.2f} y={info['imu_gyro_y']:+5.2f} z={info['imu_gyro_z']:+5.2f}",
        f"Compass : {np.degrees(info['imu_compass']):6.1f} deg",
        "",
        f"GNSS lat: {info['gnss_lat']:.6f}",
        f"GNSS lon: {info['gnss_lon']:.6f}",
        f"GNSS alt: {info['gnss_alt']:.2f} m",
        "",
        f"WP: ({info['wp_x']:.1f}, {info['wp_y']:.1f}) road={info['wp_road_id']} lane={info['wp_lane_id']}",
        f"Odom: {info['odom_m']:.1f} m",
        f"Weather: cl={info['cloudiness']:.0f} prec={info['precipitation']:.0f} sun={info['sun_altitude']:.0f}",
        "",
        f"CMD [{info.get('cmd_manobra','-')}]",
        f"  thr={info.get('cmd_throttle',0):.2f} brk={info.get('cmd_brake',0):.2f} "
        f"steer={info.get('cmd_steer',0):+.2f} v*={info.get('cmd_target_speed_kmh',0):.0f}km/h",
    ]

    y = 20
    for ln in linhas:
        cv2.putText(frame_bgr, ln, (10, y), font, fs, color, th, cv2.LINE_AA)
        y += 17

    # Painel direito: eventos
    cv2.putText(frame_bgr, "EVENTOS", (w - 270, 20), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame_bgr, f"Colisoes: {info['n_collisions']}", (w - 270, 45),
                font, fs, color_bad if info['n_collisions'] > 0 else color, th, cv2.LINE_AA)
    cv2.putText(frame_bgr, f"Lane invasions: {info['n_lane_invasions']}", (w - 270, 65),
                font, fs, color_warn if info['n_lane_invasions'] > 0 else color, th, cv2.LINE_AA)
    if info["last_event"]:
        cv2.putText(frame_bgr, f"Ult: {info['last_event'][:30]}", (w - 270, 90),
                    font, 0.4, color_warn, th, cv2.LINE_AA)

    return frame_bgr


# ---------- LEGENDA SEGMENTACAO ----------
def desenhar_legenda_seg(frame_bgr):
    """Desenha titulo e legenda das classes CityScapes no frame de segmentacao (in-place).
    Titulo no topo-esquerdo, legenda no canto inferior esquerdo."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.4
    th = 1
    line_h = 18
    sw = 14  # tamanho do quadradinho de cor
    h = frame_bgr.shape[0]

    # Titulo (topo-esquerdo)
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (330, 30), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame_bgr, 0.55, 0, frame_bgr)
    cv2.putText(frame_bgr, "SEGMENTACAO SEMANTICA", (10, 20),
                font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # Legenda (canto inferior esquerdo)
    n = len(SEG_LEGEND)
    box_w = 150
    box_h = n * line_h + 8
    x0 = 10
    y0 = h - box_h - 10

    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (x0 - 4, y0 - 4), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame_bgr, 0.5, 0, frame_bgr)

    y = y0 + 12
    for name, rgb in SEG_LEGEND:
        bgr = (rgb[2], rgb[1], rgb[0])
        cv2.rectangle(frame_bgr, (x0, y - sw + 2), (x0 + sw, y + 2), bgr, -1)
        cv2.rectangle(frame_bgr, (x0, y - sw + 2), (x0 + sw, y + 2), (255, 255, 255), 1)
        cv2.putText(frame_bgr, name, (x0 + sw + 6, y), font, fs, (255, 255, 255), th, cv2.LINE_AA)
        y += line_h
    return frame_bgr


# ---------- FONTE DE DADOS DO CAMINHAO ----------
class GeradorDadosCaminhao:
    """Gera pacotes ficticios de telemetria/comando de um caminhao real.

    Simula a chegada continua de dados externos: a cada 'interval_s' um novo
    conjunto de comandos (acelerador, freio, angulo do volante, velocidade
    alvo) e sorteado; entre atualizacoes o ultimo pacote e mantido. E aqui que
    os dados de um caminhao real entrariam no lugar do sorteio ficticio.
    """

    # (nome, throttle, brake, steer_max, target_speed_kmh)
    MANOBRAS = [
        ("acelerar", 0.75, 0.0, 0.00, 55.0),
        ("cruzeiro", 0.45, 0.0, 0.00, 40.0),
        ("frear", 0.00, 0.6, 0.00, 15.0),
        ("curva_dir", 0.40, 0.0, 0.25, 25.0),
        ("curva_esq", 0.40, 0.0, -0.25, 25.0),
        ("parar", 0.00, 1.0, 0.00, 0.0),
    ]

    def __init__(self, interval_s=5.0, seed=None, dados_iniciais=None):
        self.interval_s = interval_s
        self.rng = random.Random(seed)
        self._pacote = dict(dados_iniciais) if dados_iniciais else self._novo_pacote()
        self._proximo_update = interval_s   # proximo sorteio apos o 1o intervalo
        self._emitido = False

    def _novo_pacote(self):
        nome, thr, brk, steer_max, vel = self.rng.choice(self.MANOBRAS)
        steer = steer_max
        if steer_max != 0.0:
            steer = round(self.rng.uniform(steer_max * 0.5, steer_max), 3)
        return {
            "manobra": nome,
            "throttle": round(thr * self.rng.uniform(0.8, 1.0), 3),
            "brake": brk,
            "steer": steer,
            "target_speed_kmh": vel,
            "hand_brake": False,
            "reverse": False,
        }

    def proximo(self, sim_time):
        """Retorna (pacote, novo) para o instante 'sim_time'.

        'novo' indica que um pacote inedito chegou agora: a condicao inicial
        em t=0 ou um novo dado a cada 'interval_s'.
        """
        if not self._emitido:
            self._emitido = True
            return dict(self._pacote), True
        if sim_time >= self._proximo_update:
            self._pacote = self._novo_pacote()
            self._proximo_update = sim_time + self.interval_s
            return dict(self._pacote), True
        return dict(self._pacote), False


# ---------- COLETOR ----------
class Coletor:
    def __init__(self, cfg, output_root, duration_s):
        self.cfg = cfg
        self.output_root = output_root
        self.duration_s = duration_s
        self.actors = []
        self.npc_vehicles = []
        self.walkers = []
        self.walker_controllers = []
        self.sensor_queues = {}
        self.events = []
        self.n_collisions = 0
        self.n_lane_invasions = 0
        self.has_collision = False
        self.has_lane_invasion = False
        self.video_writer = None
        self.event_log_file = None
        self.last_event_str = ""
        self.odom_m = 0.0
        self._last_pos = None
        self.finalizing = False
        self.minimap = None
        self.gerador = None
        self._pacote_atual = {}
        self._historico_mock = []
        self._reenable_autopilot = False
        self.spawn_transform = None
        self.forced_spawn = None


    def conectar(self):
        client = carla.Client(self.cfg["host"], self.cfg["port"])
        client.set_timeout(30.0)
        self.client = client
        self.world = client.load_world(self.cfg["town"])
        self.world.set_weather(self.cfg["weather"])

        settings = self.world.get_settings()
        self.original_settings = settings
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.cfg["fixed_delta_seconds"]
        self.world.apply_settings(settings)

        self.tm = client.get_trafficmanager()
        self.tm.set_synchronous_mode(True)
        self.tm.set_global_distance_to_leading_vehicle(2.5)
        seed = self.cfg.get("run_seed")
        if seed is not None:
            self.tm.set_random_device_seed(int(seed))
            try:
                self.world.set_pedestrians_seed(int(seed))
            except Exception:
                pass
        self.world.tick()
        self.map = self.world.get_map()
        self.minimap = MiniMap(self.map, size=200, margin=10)
        print("[OK] Minimap inicializado")
        print(f"[OK] Conectado, mapa: {self.map.name}")

    def spawnar_trafego(self):
        n = self.cfg["n_vehicles"]
        if n <= 0:
            return
        bp_lib = self.world.get_blueprint_library()
        vehicle_bps = bp_lib.filter("vehicle.*")
        vehicle_bps = [bp for bp in vehicle_bps if int(bp.get_attribute("number_of_wheels")) == 4]
        spawn_points = self.map.get_spawn_points()
        random.shuffle(spawn_points)
        spawned = 0
        for spawn in spawn_points:
            if spawned >= n:
                break
            bp = random.choice(vehicle_bps)
            if bp.has_attribute("color"):
                bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
            try:
                npc = self.world.spawn_actor(bp, spawn)
                npc.set_autopilot(True, self.tm.get_port())
                self.npc_vehicles.append(npc)
                self.actors.append(npc)
                spawned += 1
            except RuntimeError:
                continue
        self.world.tick()
        print(f"[OK] {spawned}/{n} veiculos NPC spawnados")

    def spawnar_pedestres(self):
        n = self.cfg["n_pedestrians"]
        if n <= 0:
            return
        bp_lib = self.world.get_blueprint_library()
        walker_bps = bp_lib.filter("walker.pedestrian.*")
        controller_bp = bp_lib.find("controller.ai.walker")
        spawned = 0
        attempts = 0
        max_attempts = n * 5
        while spawned < n and attempts < max_attempts:
            attempts += 1
            loc = self.world.get_random_location_from_navigation()
            if loc is None:
                continue
            spawn_tf = carla.Transform(loc)
            walker_bp = random.choice(walker_bps)
            if walker_bp.has_attribute("is_invincible"):
                walker_bp.set_attribute("is_invincible", "false")
            try:
                walker = self.world.spawn_actor(walker_bp, spawn_tf)
            except RuntimeError:
                continue
            self.world.tick()
            try:
                controller = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
            except RuntimeError:
                walker.destroy()
                continue
            self.world.tick()
            controller.start()
            controller.go_to_location(self.world.get_random_location_from_navigation())
            controller.set_max_speed(1.4)
            self.walkers.append(walker)
            self.walker_controllers.append(controller)
            self.actors.extend([walker, controller])
            spawned += 1
        print(f"[OK] {spawned}/{n} pedestres spawnados")

    def spawnar_caminhao(self):
        bp_lib = self.world.get_blueprint_library()
        veh_bp = bp_lib.filter(self.cfg["vehicle_filter"])[0]
        self.vehicle = None
        if self.forced_spawn is not None:
            try:
                self.vehicle = self.world.spawn_actor(veh_bp, self.forced_spawn)
            except RuntimeError:
                print("[WARN] Spawn forcado ocupado; usando spawn aleatorio")
        if self.vehicle is None:
            spawn_points = self.map.get_spawn_points()
            for spawn in random.sample(spawn_points, len(spawn_points)):
                try:
                    self.vehicle = self.world.spawn_actor(veh_bp, spawn)
                    break
                except RuntimeError:
                    continue
        self.spawn_transform = self.vehicle.get_transform()
        self.actors.append(self.vehicle)

        if self.cfg["external_data_enabled"]:
            self.gerador = GeradorDadosCaminhao(
                interval_s=self.cfg["data_update_interval_s"],
                seed=self.cfg.get("data_seed"),
                dados_iniciais=self.cfg.get("dados_iniciais"),
            )
            pacote, _ = self.gerador.proximo(0.0)   # condicao inicial mockada
            self._pacote_atual = pacote
            self._historico_mock.append({"sim_time": 0.0, **pacote})
            self._injetar_mock(pacote)              # aplica e agenda a volta do autopilot
            print(f"[OK] Ego inicia com dados mockados (modo={self.cfg['data_control_mode']}, "
                  f"novo pacote a cada {self.cfg['data_update_interval_s']}s); autopilot assume em seguida")
        else:
            self.vehicle.set_autopilot(True, self.tm.get_port())
        self.world.tick()
        print(f"[OK] Caminhao (ego) spawnado em {self.vehicle.get_location()}")

    def _injetar_mock(self, pacote):
        """Impoe o pacote mockado ao caminhao e devolve o controle ao autopilot no tick seguinte."""
        self.vehicle.set_autopilot(False, self.tm.get_port())
        self.aplicar_dados_caminhao(pacote)
        self._reenable_autopilot = True

    def salvar_condicoes_iniciais(self):
        """Salva as condicoes iniciais da run (spawn, seeds, mapa) para reproduzir cenarios."""
        tf = self.spawn_transform
        cond = {
            "town": self.cfg["town"],
            "run_seed": self.cfg.get("run_seed"),
            "data_seed": self.cfg.get("data_seed"),
            "weather": str(self.cfg["weather"]),
            "data_control_mode": self.cfg["data_control_mode"],
            "data_update_interval_s": self.cfg["data_update_interval_s"],
            "n_vehicles": self.cfg["n_vehicles"],
            "n_pedestrians": self.cfg["n_pedestrians"],
            "dados_iniciais": self.cfg.get("dados_iniciais"),
            "ego_spawn": {
                "x": tf.location.x, "y": tf.location.y, "z": tf.location.z,
                "pitch": tf.rotation.pitch, "yaw": tf.rotation.yaw, "roll": tf.rotation.roll,
            } if tf else None,
        }
        with open(self.output_root / "condicoes_iniciais.json", "w") as f:
            json.dump(cond, f, indent=2)
        print("[OK] Condicoes iniciais salvas em condicoes_iniciais.json")

    def aplicar_dados_caminhao(self, pacote):
        """Aplica um pacote de dados externos ao estado do caminhao no CARLA."""
        if self.cfg["data_control_mode"] == "kinematic" and pacote.get("target_speed_kmh") is not None:
            # Impoe a velocidade no sentido de avanco; direcao ainda pelo volante
            fwd = self.vehicle.get_transform().get_forward_vector()
            v = float(pacote["target_speed_kmh"]) / 3.6
            self.vehicle.set_target_velocity(carla.Vector3D(fwd.x * v, fwd.y * v, fwd.z * v))
            self.vehicle.apply_control(carla.VehicleControl(
                steer=float(pacote.get("steer", 0.0)),
                hand_brake=bool(pacote.get("hand_brake", False)),
            ))
        else:
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=float(pacote.get("throttle", 0.0)),
                steer=float(pacote.get("steer", 0.0)),
                brake=float(pacote.get("brake", 0.0)),
                hand_brake=bool(pacote.get("hand_brake", False)),
                reverse=bool(pacote.get("reverse", False)),
                manual_gear_shift=False,
            ))

    def _add_sensor_safe(self, bp, transform, name):
        sensor = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)
        q = queue.Queue()
        sensor.listen(q.put)
        self.sensor_queues[name] = q
        self.actors.append(sensor)
        self.world.tick()
        print(f"  + {name} anexado")
        return sensor

    def _try_add_event_sensor(self, blueprint_name, name, callback):
        for _ in range(self.cfg["ticks_between_event_sensors"]):
            self.world.tick()
        try:
            bp_lib = self.world.get_blueprint_library()
            bp = bp_lib.find(blueprint_name)
            sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
            sensor.listen(callback)
            self.actors.append(sensor)
            self.world.tick()
            print(f"  + {name} anexado (com workaround)")
            return True
        except Exception as e:
            print(f"  ! {name} FALHOU ({type(e).__name__}): omitido")
            return False

    def _log_event(self, ev_dict):
        if getattr(self, "finalizing", False):
            return  # ignora eventos que chegam depois do fim
        self.events.append(ev_dict)
        if ev_dict["tipo"] == "collision":
            self.n_collisions += 1
        elif ev_dict["tipo"] == "lane_invasion":
            self.n_lane_invasions += 1
        line = f"[frame {ev_dict['frame']}] {ev_dict['tipo']}: {ev_dict.get('outro','')}"
        self.last_event_str = f"{ev_dict['tipo']}: {ev_dict.get('outro','')[:25]}"
        try:
            if self.event_log_file and not self.event_log_file.closed:
                self.event_log_file.write(line + "\n")
                self.event_log_file.flush()
        except (ValueError, OSError):
            pass
        print(f"  [EVENTO] {line}")


    def anexar_sensores(self):
        bp_lib = self.world.get_blueprint_library()
        print("[INFO] Anexando sensores principais...")

        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self.cfg["camera_width"]))
        cam_bp.set_attribute("image_size_y", str(self.cfg["camera_height"]))
        cam_bp.set_attribute("fov", str(self.cfg["camera_fov"]))
        self._add_sensor_safe(cam_bp, carla.Transform(carla.Location(x=self.cfg["camera_x"], z=self.cfg["camera_z"])), "camera")

        if self.cfg["seg_enabled"]:
            seg_bp = bp_lib.find("sensor.camera.semantic_segmentation")
            seg_bp.set_attribute("image_size_x", str(self.cfg["camera_width"]))
            seg_bp.set_attribute("image_size_y", str(self.cfg["camera_height"]))
            seg_bp.set_attribute("fov", str(self.cfg["camera_fov"]))
            self._add_sensor_safe(seg_bp, carla.Transform(carla.Location(x=self.cfg["camera_x"], z=self.cfg["camera_z"])), "seg")

        lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels", str(self.cfg["lidar_channels"]))
        lidar_bp.set_attribute("range", str(self.cfg["lidar_range"]))
        lidar_bp.set_attribute("points_per_second", str(self.cfg["lidar_points_per_second"]))
        lidar_bp.set_attribute("rotation_frequency", str(self.cfg["lidar_rotation_frequency"]))
        self._add_sensor_safe(lidar_bp, carla.Transform(carla.Location(x=0.0, z=2.8)), "lidar")

        gnss_bp = bp_lib.find("sensor.other.gnss")
        gnss_bp.set_attribute("noise_lat_stddev", str(self.cfg["gnss_noise_lat_stddev"]))
        gnss_bp.set_attribute("noise_lon_stddev", str(self.cfg["gnss_noise_lon_stddev"]))
        self._add_sensor_safe(gnss_bp, carla.Transform(carla.Location(z=2.8)), "gnss")

        imu_bp = bp_lib.find("sensor.other.imu")
        self._add_sensor_safe(imu_bp, carla.Transform(carla.Location(z=2.8)), "imu")

        print("[INFO] Tentando anexar sensores de evento...")
        if self.cfg["try_collision_sensor"]:
            self.has_collision = self._try_add_event_sensor(
                "sensor.other.collision", "collision",
                lambda evt: self._log_event({
                    "frame": evt.frame, "tipo": "collision",
                    "outro": evt.other_actor.type_id,
                })
            )
        if self.cfg["try_lane_invasion_sensor"]:
            self.has_lane_invasion = self._try_add_event_sensor(
                "sensor.other.lane_invasion", "lane_invasion",
                lambda evt: self._log_event({
                    "frame": evt.frame, "tipo": "lane_invasion",
                    "outro": str([m.type for m in evt.crossed_lane_markings]),
                })
            )

        print("[OK] Sensores ATIVOS: camera, lidar, gnss, imu"
              + (", collision" if self.has_collision else "")
              + (", lane_invasion" if self.has_lane_invasion else ""))

    def _drenar_sensor(self, name, frame_alvo, timeout=5.0):
        q = self.sensor_queues[name]
        while True:
            data = q.get(timeout=timeout)
            if data.frame == frame_alvo:
                return data

    def warmup(self):
        n = self.cfg["warmup_ticks"]
        print(f"[INFO] Warm-up: {n} ticks...")
        for _ in range(n):
            self.world.tick()
            for q in self.sensor_queues.values():
                while not q.empty():
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break

    def _init_video(self):
        fourcc = cv2.VideoWriter_fourcc(*self.cfg["video_codec"])
        path = self.output_root / "video.mp4"
        # Se a segmentacao estiver ativa, o video sai lado a lado (largura dobrada)
        w = self.cfg["camera_width"] * (2 if self.cfg["seg_enabled"] else 1)
        self.video_writer = cv2.VideoWriter(
            str(path),
            fourcc,
            self.cfg["video_fps"],
            (w, self.cfg["camera_height"]),
        )
        if not self.video_writer.isOpened():
            raise RuntimeError(f"Nao consegui abrir VideoWriter em {path}")
        # Buffer reutilizado a cada frame para o video lado a lado (evita realocacao)
        self._combined = (np.empty((self.cfg["camera_height"], w, 3), dtype=np.uint8)
                          if self.cfg["seg_enabled"] else None)
        print(f"[OK] VideoWriter: {path} @ {self.cfg['video_fps']} fps ({w}x{self.cfg['camera_height']})")

    def rodar(self):
        # Telemetria CSV
        tele_path = self.output_root / "telemetria.csv"
        tele_file = open(tele_path, "w", newline="")
        cabecalho = [
            "frame", "sim_time",
            "x", "y", "z", "yaw", "pitch", "roll",
            "speed_mps", "throttle", "brake", "steer", "gear",
            "acc_x", "acc_y", "acc_z",
            "gnss_lat", "gnss_lon", "gnss_alt",
            "imu_acc_x", "imu_acc_y", "imu_acc_z",
            "imu_gyro_x", "imu_gyro_y", "imu_gyro_z",
            "imu_compass",
            # NOVOS
            "wp_x", "wp_y", "wp_road_id", "wp_lane_id",
            "odom_m",
            "cloudiness", "precipitation", "sun_altitude",
            "n_collisions", "n_lane_invasions",
            # DADOS EXTERNOS (comando aplicado ao caminhao)
            "cmd_manobra", "cmd_throttle", "cmd_brake", "cmd_steer", "cmd_target_speed_kmh",
        ]
        tele_writer = csv.DictWriter(tele_file, fieldnames=cabecalho)
        tele_writer.writeheader()

        # Log de eventos em texto
        self.event_log_file = open(self.output_root / "eventos.log", "w")
        self.event_log_file.write(f"# Iniciado em {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.event_log_file.flush()

        # Video
        self._init_video()

        ticks_total = int(self.duration_s / self.cfg["fixed_delta_seconds"])
        print(f"[INFO] Coletando {ticks_total} ticks (~{self.duration_s}s)")

        for i in range(ticks_total):
            # Dados mockados do caminhao: ao chegar um pacote novo, impoe a condicao e
            # devolve o controle ao autopilot (que otimiza a decisao do motorista).
            if self.cfg["external_data_enabled"] and self.gerador is not None:
                sim_time = i * self.cfg["fixed_delta_seconds"]
                pacote, novo = self.gerador.proximo(sim_time)
                if novo:
                    self._pacote_atual = pacote
                    self._historico_mock.append({"sim_time": sim_time, **pacote})
                    self._injetar_mock(pacote)
                elif self._reenable_autopilot:
                    self.vehicle.set_autopilot(True, self.tm.get_port())
                    self._reenable_autopilot = False

            frame = self.world.tick()

            try:
                img = self._drenar_sensor("camera", frame)
                lidar = self._drenar_sensor("lidar", frame)
                gnss = self._drenar_sensor("gnss", frame)
                imu = self._drenar_sensor("imu", frame)
                seg = self._drenar_sensor("seg", frame) if self.cfg["seg_enabled"] else None
            except queue.Empty:
                print(f"[WARN] Sensor nao respondeu no frame {frame}")
                continue

            # Lidar opcional
            if self.cfg["save_lidar_npy"]:
                pts = np.frombuffer(lidar.raw_data, dtype=np.float32).reshape(-1, 4)
                np.save(self.output_root / "lidar" / f"{frame:06d}.npy", pts)

            # Telemetria
            tf = self.vehicle.get_transform()
            vel = self.vehicle.get_velocity()
            acc = self.vehicle.get_acceleration()
            ctrl = self.vehicle.get_control()
            speed = (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5

            # Odometria (integral da posicao)
            cur_pos = (tf.location.x, tf.location.y)
            if self._last_pos is not None:
                dx = cur_pos[0] - self._last_pos[0]
                dy = cur_pos[1] - self._last_pos[1]
                self.odom_m += (dx*dx + dy*dy) ** 0.5
            self._last_pos = cur_pos

            # Waypoint do planner
            wp = self.map.get_waypoint(tf.location, project_to_road=True)
            wp_x = wp.transform.location.x if wp else 0.0
            wp_y = wp.transform.location.y if wp else 0.0
            wp_road = wp.road_id if wp else -1
            wp_lane = wp.lane_id if wp else -1

            # Clima
            wt = self.world.get_weather()

            n_coll = self.n_collisions
            n_lane = self.n_lane_invasions

            row = {
                "frame": frame,
                "sim_time": i * self.cfg["fixed_delta_seconds"],
                "x": tf.location.x, "y": tf.location.y, "z": tf.location.z,
                "yaw": tf.rotation.yaw, "pitch": tf.rotation.pitch, "roll": tf.rotation.roll,
                "speed_mps": speed,
                "throttle": ctrl.throttle, "brake": ctrl.brake,
                "steer": ctrl.steer, "gear": ctrl.gear,
                "acc_x": acc.x, "acc_y": acc.y, "acc_z": acc.z,
                "gnss_lat": gnss.latitude, "gnss_lon": gnss.longitude, "gnss_alt": gnss.altitude,
                "imu_acc_x": imu.accelerometer.x, "imu_acc_y": imu.accelerometer.y,
                "imu_acc_z": imu.accelerometer.z,
                "imu_gyro_x": imu.gyroscope.x, "imu_gyro_y": imu.gyroscope.y,
                "imu_gyro_z": imu.gyroscope.z,
                "imu_compass": imu.compass,
                "wp_x": wp_x, "wp_y": wp_y, "wp_road_id": wp_road, "wp_lane_id": wp_lane,
                "odom_m": self.odom_m,
                "cloudiness": wt.cloudiness,
                "precipitation": wt.precipitation,
                "sun_altitude": wt.sun_altitude_angle,
                "n_collisions": n_coll,
                "n_lane_invasions": n_lane,
                "cmd_manobra": self._pacote_atual.get("manobra", ""),
                "cmd_throttle": self._pacote_atual.get("throttle", 0.0),
                "cmd_brake": self._pacote_atual.get("brake", 0.0),
                "cmd_steer": self._pacote_atual.get("steer", 0.0),
                "cmd_target_speed_kmh": self._pacote_atual.get("target_speed_kmh", 0.0),
            }
            tele_writer.writerow(row)

            # Frame BGR para video
            arr = np.frombuffer(img.raw_data, dtype=np.uint8)
            arr = arr.reshape((img.height, img.width, 4))   # BGRA
            frame_bgr = arr[:, :, :3].copy()                # BGR

            if self.cfg["hud_enabled"]:
                hud_info = dict(row)
                hud_info["last_event"] = self.last_event_str
                desenhar_hud(frame_bgr, hud_info)
                # Posicoes dos NPCs pro minimapa
                npc_locs = []
                for npc in self.npc_vehicles:
                    try:
                        loc = npc.get_location()
                        npc_locs.append((loc.x, loc.y))
                    except Exception:
                        pass

                self.minimap.draw(frame_bgr, tf.location.x, tf.location.y,
                                tf.rotation.yaw, npc_locs)


            # Segmentacao semantica: converte para paleta CityScapes e monta lado a lado
            if seg is not None:
                seg.convert(carla.ColorConverter.CityScapesPalette)
                seg_arr = np.frombuffer(seg.raw_data, dtype=np.uint8)
                seg_arr = seg_arr.reshape((seg.height, seg.width, 4))
                cam_w = self.cfg["camera_width"]
                self._combined[:, :cam_w] = frame_bgr
                self._combined[:, cam_w:] = seg_arr[:, :, :3]
                desenhar_legenda_seg(self._combined[:, cam_w:])  # slice gravavel
                self.video_writer.write(self._combined)
            else:
                self.video_writer.write(frame_bgr)

            if i % 40 == 0:
                print(f"  tick {i}/{ticks_total}  v={speed*3.6:.1f}km/h  "
                      f"pos=({tf.location.x:.1f},{tf.location.y:.1f})  "
                      f"odom={self.odom_m:.1f}m  eventos={len(self.events)}")

        # Para os sensores de evento ANTES de fechar tudo
        self.finalizing = True
        for ator in self.actors:
            try:
                if hasattr(ator, "type_id") and ator.type_id.startswith("sensor."):
                    ator.stop()
            except Exception:
                pass
        time.sleep(0.1)  # da tempo dos callbacks pendentes drenarem
        tele_file.close()

        if self.video_writer is not None:
            self.video_writer.release()
            print(f"[OK] Video salvo em {self.output_root / 'video.mp4'}")

        if self.event_log_file is not None:
            self.event_log_file.write(f"# Finalizado em {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.event_log_file.write(f"# Total de eventos: {len(self.events)}\n")
            self.event_log_file.close()

        # CSV de eventos (para o dashboard.py continuar funcionando)
        if self.events:
            with open(self.output_root / "eventos.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["frame", "tipo", "outro"])
                w.writeheader()
                for ev in self.events:
                    w.writerow(ev)
            print(f"[OK] {len(self.events)} eventos em eventos.csv / eventos.log")
        else:
            # cria arquivo vazio pra dashboard nao reclamar
            with open(self.output_root / "eventos.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["frame", "tipo", "outro"])
                w.writeheader()

        # Historico completo dos dados mockados (como se fossem os dados reais do caminhao)
        if self._historico_mock:
            campos = ["sim_time", "manobra", "throttle", "brake", "steer",
                      "target_speed_kmh", "hand_brake", "reverse"]
            with open(self.output_root / "dados_mock.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=campos)
                w.writeheader()
                for pkt in self._historico_mock:
                    w.writerow({k: pkt.get(k, "") for k in campos})
            print(f"[OK] {len(self._historico_mock)} pacotes mockados em dados_mock.csv")

        print(f"[OK] Coleta finalizada. Dados em: {self.output_root}")


    def encerrar(self):
        print("[INFO] Encerrando — destruindo atores")
        try:
            if self.video_writer is not None:
                self.video_writer.release()
        except Exception:
            pass
        try:
            if self.event_log_file is not None and not self.event_log_file.closed:
                self.event_log_file.close()
        except Exception:
            pass
        for ctrl in self.walker_controllers:
            try:
                ctrl.stop()
            except Exception:
                pass
        try:
            self.world.apply_settings(self.original_settings)
            self.tm.set_synchronous_mode(False)
        except Exception:
            pass
        for ator in self.actors:
            try:
                ator.destroy()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--output", type=str, default="./dataset/run_001")
    parser.add_argument("--town", type=str, default=CONFIG["town"])
    parser.add_argument("--vehicles", type=int, default=CONFIG["n_vehicles"])
    parser.add_argument("--pedestrians", type=int, default=CONFIG["n_pedestrians"])
    parser.add_argument("--no-hud", action="store_true", help="Desativa overlay no video")
    parser.add_argument("--no-lidar", action="store_true", help="Nao salva .npy do LiDAR (economiza disco)")
    parser.add_argument("--no-seg", action="store_true", help="Desativa camera de segmentacao semantica (video lado a lado)")
    parser.add_argument("--autopilot", action="store_true", help="Usa o autopilot no ego em vez de dados externos")
    parser.add_argument("--data-interval", type=float, default=CONFIG["data_update_interval_s"],
                        help="Intervalo (s) entre novos pacotes de dados do caminhao")
    parser.add_argument("--data-mode", choices=["control", "kinematic"], default=CONFIG["data_control_mode"],
                        help="control: throttle/brake/steer; kinematic: impoe velocidade")
    parser.add_argument("--seed", type=int, default=None, help="Semente global da run (reprodutibilidade)")
    parser.add_argument("--replay", type=str, default=None,
                        help="Caminho de um condicoes_iniciais.json para reproduzir a run")
    args = parser.parse_args()

    CONFIG["town"] = args.town
    CONFIG["n_vehicles"] = args.vehicles
    CONFIG["n_pedestrians"] = args.pedestrians
    if args.no_hud:
        CONFIG["hud_enabled"] = False
    if args.no_lidar:
        CONFIG["save_lidar_npy"] = False
    if args.no_seg:
        CONFIG["seg_enabled"] = False
    if args.autopilot:
        CONFIG["external_data_enabled"] = False
    CONFIG["data_update_interval_s"] = args.data_interval
    CONFIG["data_control_mode"] = args.data_mode

    # Replay: reproduz uma run a partir das condicoes iniciais salvas
    forced_spawn = None
    if args.replay:
        with open(args.replay) as f:
            cond = json.load(f)
        CONFIG["town"] = cond.get("town", CONFIG["town"])
        CONFIG["run_seed"] = cond.get("run_seed")
        CONFIG["data_seed"] = cond.get("data_seed")
        CONFIG["data_control_mode"] = cond.get("data_control_mode", CONFIG["data_control_mode"])
        CONFIG["data_update_interval_s"] = cond.get("data_update_interval_s", CONFIG["data_update_interval_s"])
        if cond.get("dados_iniciais"):
            CONFIG["dados_iniciais"] = cond["dados_iniciais"]
        sp = cond.get("ego_spawn")
        if sp:
            forced_spawn = carla.Transform(
                carla.Location(x=sp["x"], y=sp["y"], z=sp["z"]),
                carla.Rotation(pitch=sp["pitch"], yaw=sp["yaw"], roll=sp["roll"]),
            )
        print(f"[OK] Replay de {args.replay}")

    # Semente global fixa a reprodutibilidade (spawns, cores, dados mock, traffic manager)
    if CONFIG.get("run_seed") is None:
        CONFIG["run_seed"] = args.seed if args.seed is not None else int(time.time())
    if CONFIG.get("data_seed") is None:
        CONFIG["data_seed"] = CONFIG["run_seed"]
    random.seed(CONFIG["run_seed"])
    print(f"[INFO] run_seed={CONFIG['run_seed']}  data_seed={CONFIG['data_seed']}")

    output_root = Path(args.output)
    make_dirs(output_root)
    save_metadata(output_root, CONFIG)

    coletor = Coletor(CONFIG, output_root, args.duration)
    coletor.forced_spawn = forced_spawn

    def handler(sig, frame):
        coletor.encerrar()
        sys.exit(0)
    signal.signal(signal.SIGINT, handler)

    try:
        coletor.conectar()
        coletor.spawnar_trafego()
        coletor.spawnar_pedestres()
        coletor.spawnar_caminhao()
        coletor.salvar_condicoes_iniciais()
        coletor.anexar_sensores()
        coletor.warmup()
        coletor.rodar()
    finally:
        coletor.encerrar()


if __name__ == "__main__":
    main()
