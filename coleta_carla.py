# coleta_carla.py (v7 - rota fixa, anti-colisao, emergency brake, sensores estendidos)

import argparse
import csv
import json
import math
import queue
import random
import signal
import sys
import time
import traceback
from pathlib import Path

# ---------- Path dos agents do CARLA ----------
CARLA_AGENTS_PATH = r"C:\CARLA_0.9.16\PythonAPI\carla"
if CARLA_AGENTS_PATH not in sys.path:
    sys.path.append(CARLA_AGENTS_PATH)

import carla
import cv2
import numpy as np

from agents.navigation.behavior_agent import BehaviorAgent


CONFIG = {
    "host": "localhost",
    "port": 2000,
    "town": "Town03",
    "fixed_delta_seconds": 0.05,
    "vehicle_filter": "vehicle.carlamotors.european_hgv",

    # ----- Camera -----
    "camera_width": 1280,
    "camera_height": 720,
    "camera_fov": 90,

    # ----- LiDAR -----
    "lidar_channels": 16,
    "lidar_range": 30.0,
    "lidar_points_per_second": 100000,
    "lidar_rotation_frequency": 20,

    # ----- GNSS / IMU -----
    "gnss_noise_lat_stddev": 1e-5,
    "gnss_noise_lon_stddev": 1e-5,

    # ----- Mundo -----
    "weather": carla.WeatherParameters.ClearNoon,
    "warmup_ticks": 30,
    "n_vehicles": 30,
    "n_pedestrians": 20,

    # ----- Sensores de evento -----
    "try_collision_sensor": True,
    "try_lane_invasion_sensor": True,
    "try_obstacle_sensor": True,
    "ticks_between_event_sensors": 10,

    # ----- Sensores opcionais (extras do diagrama) -----
    "enable_semantic_camera": True,    # True = adiciona stream + grava .npy
    "enable_radar": False,              # True = adiciona radar frontal

    # ----- Saida -----
    "save_lidar_npy": True,
    "save_semantic_npy": True,
    "video_fps": 20,
    "video_codec": "mp4v",
    "hud_enabled": True,

    # ----- Agente -----
    "agent_behavior": "cautious",
    "agent_target_speed_kmh": 50,
    "agent_braking_distance": 20,
    "agent_safety_time": 8.0,
    "min_destination_distance": 50.0,

    # ----- Rota fixa -----
    "rota_n_pontos": 6,                 # quantos waypoints alem do start
    "rota_dist_min": 80.0,              # distancia minima entre pontos da rota
    "rota_dist_max": 250.0,

    # ----- Speed limiter em curvas -----
    "curva_lookahead_wps": 15,          # quantos waypoints olhar a frente
    "curva_fator_leve": 0.75,           # 5-20 graus
    "curva_fator_medio": 0.50,          # 20-40 graus
    "curva_fator_forte": 0.30,          # 40+ graus

    # ----- Rate limiter (suaviza freadas/aceleracoes) -----
    "max_throttle_rate": 0.05,
    "max_brake_rate": 0.08,

    # ----- Emergency brake (anti-atropelamento) -----
    "ebrake_min_radius": 8.0,           # raio minimo de busca
    "ebrake_time_horizon": 2.0,         # segundos de antecipacao
    "ebrake_corridor_width": 3.0,       # largura do corredor a frente
    "ebrake_log_cooldown_frames": 40,

    # ----- Stuck detector -----
    "stuck_threshold_ticks": 100,
    "stuck_action": "respawn",

    # ----- Collision dedupe -----
    "collision_cooldown_frames": 20,
    "obstacle_log_cooldown_frames": 20,
}


# ============================================================
#              PALETA SEMANTICA (CityScapes / CARLA)
# ============================================================
# Cores RGB oficiais — CARLA 0.9.16 (sensor.camera.semantic_segmentation)
CITYSCAPES_PALETTE_RGB = np.array([
    (  0,   0,   0),   # 0  Unlabeled
    (128,  64, 128),   # 1  Road
    (244,  35, 232),   # 2  Sidewalk
    ( 70,  70,  70),   # 3  Building
    (102, 102, 156),   # 4  Wall
    (190, 153, 153),   # 5  Fence
    (153, 153, 153),   # 6  Pole
    (250, 170,  30),   # 7  TrafficLight
    (220, 220,   0),   # 8  TrafficSign
    (107, 142,  35),   # 9  Vegetation
    (152, 251, 152),   # 10 Terrain
    ( 70, 130, 180),   # 11 Sky
    (220,  20,  60),   # 12 Pedestrian
    (255,   0,   0),   # 13 Rider
    (  0,   0, 142),   # 14 Car
    (  0,   0,  70),   # 15 Truck
    (  0,  60, 100),   # 16 Bus
    (  0,  80, 100),   # 17 Train
    (  0,   0, 230),   # 18 Motorcycle
    (119,  11,  32),   # 19 Bicycle
    (110, 190, 160),   # 20 Static
    (170, 120,  50),   # 21 Dynamic
    ( 55,  90,  80),   # 22 Other
    ( 45,  60, 150),   # 23 Water
    (157, 234,  50),   # 24 RoadLine
    ( 81,   0,  81),   # 25 Ground
    (150, 100, 100),   # 26 Bridge
    (230, 150, 140),   # 27 RailTrack
    (180, 165, 180),   # 28 GuardRail
], dtype=np.uint8)
# OpenCV usa BGR
CITYSCAPES_PALETTE_BGR = CITYSCAPES_PALETTE_RGB[:, ::-1].copy()

# Classes mostradas na legenda (filtradas para nao poluir)
LEGENDA_CLASSES = [
    (1,  "Road"),
    (24, "RoadLine"),
    (2,  "Sidewalk"),
    (3,  "Building"),
    (14, "Car"),
    (15, "Truck"),
    (12, "Pedestrian"),
    (9,  "Vegetation"),
    (6,  "Pole"),
    (7,  "TrafficLight"),
    (8,  "TrafficSign"),
    (5,  "Fence"),
    (10, "Terrain"),
    (11, "Sky"),
]



# ============================================================
#                          MINIMAPA
# ============================================================
class MiniMap:
    def __init__(self, world_map, size=220, margin=10):
        self.size = size
        self.margin = margin
        wps = world_map.generate_waypoints(5.0)
        xs = [w.transform.location.x for w in wps]
        ys = [w.transform.location.y for w in wps]
        self.min_x, self.max_x = min(xs), max(xs)
        self.min_y, self.max_y = min(ys), max(ys)
        rng = max(self.max_x - self.min_x, self.max_y - self.min_y)
        self.rng = rng if rng > 0 else 1.0

        self.bg = np.zeros((size, size, 3), dtype=np.uint8)
        for w in wps:
            px, py = self._w2p(w.transform.location.x, w.transform.location.y)
            cv2.circle(self.bg, (px, py), 1, (90, 90, 90), -1)
        self.trail = []

    def _w2p(self, x, y):
        px = int((x - self.min_x) / self.rng * (self.size - 1))
        py = int((y - self.min_y) / self.rng * (self.size - 1))
        py = self.size - 1 - py
        return px, py

    def draw(self, frame_bgr, ego_x, ego_y, yaw_deg,
             npc_locations=None, route_points=None, route_idx=None):
        self.trail.append((ego_x, ego_y))
        if len(self.trail) > 800:
            self.trail = self.trail[-800:]

        canvas = self.bg.copy()

        # Rota fixa (linha magenta)
        if route_points:
            for i in range(1, len(route_points)):
                p1 = self._w2p(route_points[i - 1].x, route_points[i - 1].y)
                p2 = self._w2p(route_points[i].x, route_points[i].y)
                cv2.line(canvas, p1, p2, (255, 0, 255), 1)
            for i, p in enumerate(route_points):
                px, py = self._w2p(p.x, p.y)
                if route_idx is not None and i == route_idx:
                    cv2.circle(canvas, (px, py), 5, (0, 255, 255), -1)  # alvo atual amarelo
                else:
                    cv2.circle(canvas, (px, py), 3, (255, 0, 255), -1)

        # Rastro (laranja)
        for i in range(1, len(self.trail)):
            p1 = self._w2p(*self.trail[i - 1])
            p2 = self._w2p(*self.trail[i])
            cv2.line(canvas, p1, p2, (0, 200, 255), 1)

        # NPCs (cinza)
        if npc_locations:
            for nx, ny in npc_locations:
                px, py = self._w2p(nx, ny)
                cv2.circle(canvas, (px, py), 2, (180, 180, 180), -1)

        # Ego (vermelho com seta)
        epx, epy = self._w2p(ego_x, ego_y)
        cv2.circle(canvas, (epx, epy), 5, (0, 0, 255), -1)
        rad = math.radians(yaw_deg)
        dx = int(10 * math.cos(rad))
        dy = -int(10 * math.sin(rad))
        cv2.arrowedLine(canvas, (epx, epy), (epx + dx, epy + dy),
                        (0, 0, 255), 2, tipLength=0.4)

        h, w = frame_bgr.shape[:2]
        x0 = w - self.size - self.margin
        y0 = h - self.size - self.margin
        cv2.rectangle(frame_bgr, (x0 - 2, y0 - 2),
                      (x0 + self.size + 1, y0 + self.size + 1),
                      (255, 255, 255), 1)
        frame_bgr[y0:y0 + self.size, x0:x0 + self.size] = canvas
        cv2.putText(frame_bgr, "MINIMAP", (x0 + 4, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


# ============================================================
#                          UTILS
# ============================================================
def make_dirs(output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    if CONFIG["save_lidar_npy"]:
        (output_root / "lidar").mkdir(parents=True, exist_ok=True)
    if CONFIG["enable_semantic_camera"] and CONFIG["save_semantic_npy"]:
        (output_root / "semantic").mkdir(parents=True, exist_ok=True)


def save_metadata(output_root: Path, cfg: dict):
    meta = {k: (str(v) if not isinstance(v, (int, float, str, bool)) else v)
            for k, v in cfg.items()}
    meta["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(output_root / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def desenhar_hud(frame_bgr, info: dict):
    h, w = frame_bgr.shape[:2]

    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (440, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame_bgr, 0.55, 0, frame_bgr)

    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (w - 290, 0), (w, 180), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame_bgr, 0.55, 0, frame_bgr)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.45
    th = 1
    color = (0, 255, 0)
    color_warn = (0, 200, 255)
    color_bad = (0, 0, 255)

    linhas = [
        f"Frame: {info['frame']}  t={info['sim_time']:.2f}s",
        f"Speed   : {info['speed_mps']*3.6:6.2f} km/h  ({info['speed_mps']:.2f} m/s)",
        f"Target  : {info['target_speed_kmh']:.1f} km/h",
        f"Gear    : {info['gear']}",
        f"Throttle: {info['throttle']:.2f}",
        f"Brake   : {info['brake']:.2f}{' [EMERG]' if info['emerg_brake'] else ''}",
        f"Steer   : {info['steer']:+.2f}",
        "",
        f"Pos: x={info['x']:7.1f} y={info['y']:7.1f} z={info['z']:5.1f}",
        f"Yaw: {info['yaw']:+6.1f}  Pitch: {info['pitch']:+5.1f}  Roll: {info['roll']:+5.1f}",
        "",
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
        f"Rota: {info['route_idx']}/{info['route_total']}",
        f"Weather: cl={info['cloudiness']:.0f} prec={info['precipitation']:.0f} sun={info['sun_altitude']:.0f}",
    ]

    y = 18
    for ln in linhas:
        cv2.putText(frame_bgr, ln, (10, y), font, fs, color, th, cv2.LINE_AA)
        y += 17

    cv2.putText(frame_bgr, "EVENTOS", (w - 280, 20), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame_bgr, f"Colisoes      : {info['n_collisions']}", (w - 280, 45),
                font, fs, color_bad if info['n_collisions'] > 0 else color, th, cv2.LINE_AA)
    cv2.putText(frame_bgr, f"Lane invasions: {info['n_lane_invasions']}", (w - 280, 65),
                font, fs, color_warn if info['n_lane_invasions'] > 0 else color, th, cv2.LINE_AA)
    cv2.putText(frame_bgr, f"Stucks        : {info['n_stucks']}", (w - 280, 85),
                font, fs, color_warn if info['n_stucks'] > 0 else color, th, cv2.LINE_AA)
    cv2.putText(frame_bgr, f"Emerg brakes  : {info['n_emerg']}", (w - 280, 105),
                font, fs, color_warn if info['n_emerg'] > 0 else color, th, cv2.LINE_AA)
    if info["last_event"]:
        cv2.putText(frame_bgr, f"Ult: {info['last_event'][:32]}", (w - 280, 130),
                    font, 0.4, color_warn, th, cv2.LINE_AA)

    return frame_bgr


def colorir_semantic(label_img: np.ndarray) -> np.ndarray:
    """Converte mapa de labels (HxW, uint8) em imagem BGR colorida."""
    labels = np.clip(label_img, 0, len(CITYSCAPES_PALETTE_BGR) - 1)
    return CITYSCAPES_PALETTE_BGR[labels]


def desenhar_legenda_semantic(img_bgr: np.ndarray):
    """Desenha titulo e legenda de classes sobre a imagem semantica."""
    h, w = img_bgr.shape[:2]

    # Titulo no topo
    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, 32), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img_bgr, 0.45, 0, img_bgr)
    cv2.putText(img_bgr, "SEMANTIC SEGMENTATION",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA)

    # Caixa de legenda (canto inferior esquerdo)
    line_h = 18
    box_w = 200
    box_h = len(LEGENDA_CLASSES) * line_h + 26
    x0 = 10
    y0 = h - box_h - 10

    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img_bgr, 0.4, 0, img_bgr)
    cv2.rectangle(img_bgr, (x0, y0), (x0 + box_w, y0 + box_h),
                  (255, 255, 255), 1)

    cv2.putText(img_bgr, "LEGENDA", (x0 + 8, y0 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)

    y = y0 + 30
    for class_id, name in LEGENDA_CLASSES:
        color = CITYSCAPES_PALETTE_BGR[class_id]
        ctup = (int(color[0]), int(color[1]), int(color[2]))
        cv2.rectangle(img_bgr, (x0 + 8, y), (x0 + 24, y + 12), ctup, -1)
        cv2.rectangle(img_bgr, (x0 + 8, y), (x0 + 24, y + 12),
                      (255, 255, 255), 1)
        cv2.putText(img_bgr, f"{class_id:2d}  {name}",
                    (x0 + 32, y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += line_h


# ============================================================
#                         COLETOR
# ============================================================
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
        self.has_collision = False
        self.has_lane_invasion = False
        self.has_obstacle = False
        self.video_writer = None
        self.event_log_file = None
        self.last_event_str = ""
        self.odom_m = 0.0
        self._last_pos = None
        self.finalizing = False
        self.minimap = None
        self.agent = None
        # collision dedupe
        self.last_collision_frame = -999
        self.last_collision_actor = None
        # obstacle dedupe
        self._obs_last_frame = -999
        # emergency brake
        self._emerg_log_frame = -999
        self.emerg_active = False
        # rate limiter
        self._last_throttle = 0.0
        self._last_brake = 0.0
        # stuck detector
        self.stuck_counter = 0
        # rota
        self.rota = []
        self.rota_idx = 0
        self.rota_completa = False
        # speed alvo dinamico (debug/HUD)
        self.target_speed_atual = self.cfg["agent_target_speed_kmh"]

    # --------------------------------------------------------
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
        self.world.tick()
        self.map = self.world.get_map()
        self.minimap = MiniMap(self.map, size=220, margin=10)
        print("[OK] Minimap inicializado")
        print(f"[OK] Conectado, mapa: {self.map.name}")

    # --------------------------------------------------------
    def spawnar_trafego(self):
        n = self.cfg["n_vehicles"]
        if n <= 0:
            return
        bp_lib = self.world.get_blueprint_library()
        vehicle_bps = bp_lib.filter("vehicle.*")
        vehicle_bps = [bp for bp in vehicle_bps
                       if int(bp.get_attribute("number_of_wheels")) == 4]
        spawn_points = self.map.get_spawn_points()
        random.shuffle(spawn_points)
        spawned = 0
        for spawn in spawn_points:
            if spawned >= n:
                break
            bp = random.choice(vehicle_bps)
            if bp.has_attribute("color"):
                bp.set_attribute("color",
                                 random.choice(bp.get_attribute("color").recommended_values))
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

    # --------------------------------------------------------
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
                controller = self.world.spawn_actor(controller_bp,
                                                    carla.Transform(),
                                                    attach_to=walker)
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

    # --------------------------------------------------------
    def _gerar_rota_fixa(self, n_pontos=None):
        """Gera lista [start, wp1, wp2, ..., end] com waypoints distantes entre si."""
        if n_pontos is None:
            n_pontos = self.cfg["rota_n_pontos"]
        spawns = self.map.get_spawn_points()
        random.shuffle(spawns)
        start = self.vehicle.get_location()
        rota = [start]
        cur = start
        dmin = self.cfg["rota_dist_min"]
        dmax = self.cfg["rota_dist_max"]
        for _ in range(n_pontos):
            candidatos = [s.location for s in spawns
                          if dmin < s.location.distance(cur) < dmax]
            if not candidatos:
                # fallback: pega qualquer um a >50m
                candidatos = [s.location for s in spawns
                              if s.location.distance(cur) > 50]
            if not candidatos:
                candidatos = [s.location for s in spawns]
            nxt = random.choice(candidatos)
            rota.append(nxt)
            cur = nxt
        return rota

    # --------------------------------------------------------
    def spawnar_caminhao(self):
        bp_lib = self.world.get_blueprint_library()
        veh_bp = bp_lib.filter(self.cfg["vehicle_filter"])[0]
        spawn_points = self.map.get_spawn_points()
        for spawn in random.sample(spawn_points, len(spawn_points)):
            try:
                self.vehicle = self.world.spawn_actor(veh_bp, spawn)
                break
            except RuntimeError:
                continue
        self.actors.append(self.vehicle)
        self.world.tick()
        print(f"[OK] Caminhao spawnado em {self.vehicle.get_location()}")

        # ========= AGENTE =========
        self.agent = BehaviorAgent(self.vehicle,
                                   behavior=self.cfg["agent_behavior"])
        target_kmh = self.cfg["agent_target_speed_kmh"]
        self.agent.set_target_speed(target_kmh)
        self.agent._behavior.max_speed = target_kmh
        self.agent._behavior.braking_distance = self.cfg["agent_braking_distance"]
        self.agent._behavior.safety_time = self.cfg["agent_safety_time"]
        try:
            self.agent.ignore_traffic_lights(False)
            self.agent.ignore_stop_signs(False)
            self.agent.ignore_vehicles(False)
        except Exception:
            pass

        # ========= ROTA FIXA =========
        self.rota = self._gerar_rota_fixa()
        self.rota_idx = 1
        self.agent.set_destination(self.rota[self.rota_idx])
        print(f"[OK] Agente '{self.cfg['agent_behavior']}' "
              f"@ {target_kmh}km/h, rota com {len(self.rota)} pontos")
        print(f"[OK] Indo para waypoint {self.rota_idx}/{len(self.rota)-1}: "
              f"{self.rota[self.rota_idx]}")

        # Salva rota em JSON
        with open(self.output_root / "rota.json", "w") as f:
            json.dump([{"x": p.x, "y": p.y, "z": p.z} for p in self.rota],
                      f, indent=2)

    # --------------------------------------------------------
    def _add_sensor_safe(self, bp, transform, name):
        sensor = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)
        q = queue.Queue()
        sensor.listen(q.put)
        self.sensor_queues[name] = q
        self.actors.append(sensor)
        self.world.tick()
        print(f"  + {name} anexado")
        return sensor

    def _try_add_event_sensor(self, blueprint_name, name, callback,
                            attribs=None, transform=None):
        for _ in range(self.cfg["ticks_between_event_sensors"]):
            self.world.tick()
        try:
            bp_lib = self.world.get_blueprint_library()
            bp = bp_lib.find(blueprint_name)
            if attribs:
                for k, v in attribs.items():
                    bp.set_attribute(k, str(v))
            if transform is None:
                transform = carla.Transform(carla.Location(x=2.5, z=1.0))
            sensor = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)
            sensor.listen(callback)
            self.actors.append(sensor)
            self.world.tick()
            print(f"  + {name} anexado")
            return True
        except Exception as e:
            print(f"  ! {name} FALHOU ({type(e).__name__}): omitido")
            return False


    def _log_event(self, ev_dict):
        if self.finalizing:
            return
        self.events.append(ev_dict)
        line = f"[frame {ev_dict['frame']}] {ev_dict['tipo']}: {ev_dict.get('outro','')}"
        self.last_event_str = f"{ev_dict['tipo']}: {ev_dict.get('outro','')[:25]}"
        try:
            if self.event_log_file and not self.event_log_file.closed:
                self.event_log_file.write(line + "\n")
                self.event_log_file.flush()
        except (ValueError, OSError):
            pass
        print(f"  [EVENTO] {line}")

    def _on_collision(self, evt):
        if self.finalizing:
            return
        other_id = evt.other_actor.type_id
        if (evt.frame - self.last_collision_frame < self.cfg["collision_cooldown_frames"]
                and other_id == self.last_collision_actor):
            return
        self.last_collision_frame = evt.frame
        self.last_collision_actor = other_id
        self._log_event({
            "frame": evt.frame, "tipo": "collision", "outro": other_id,
        })

    def _on_lane_invasion(self, evt):
        if self.finalizing:
            return
        self._log_event({
            "frame": evt.frame, "tipo": "lane_invasion",
            "outro": str([m.type for m in evt.crossed_lane_markings]),
        })

    def _on_obstacle(self, evt):
        if self.finalizing:
            return
        tid = evt.other_actor.type_id
        # so loga walker/vehicle (filtra postes, pista, etc)
        if not ("walker" in tid or "vehicle" in tid):
            return
        if evt.frame - self._obs_last_frame < self.cfg["obstacle_log_cooldown_frames"]:
            return
        self._obs_last_frame = evt.frame
        self._log_event({
            "frame": evt.frame, "tipo": "obstacle_detected",
            "outro": f"{tid} a {evt.distance:.1f}m",
        })

   # --------------------------------------------------------
    def anexar_sensores(self):
        bp_lib = self.world.get_blueprint_library()
        print("[INFO] Anexando sensores principais...")

        # ===== Posicionamento dinamico baseado na bounding box do veiculo =====
        bb = self.vehicle.bounding_box
        cam_x = bb.extent.x + 0.5                       # ~0.5m a frente do para-choque
        cam_z = bb.location.z + bb.extent.z + 0.3       # ~0.3m acima do teto
        lidar_z = bb.location.z + bb.extent.z + 0.5     # ~0.5m acima do teto
        print(f"[INFO] Bounding box: extent=({bb.extent.x:.2f}, {bb.extent.y:.2f}, "
            f"{bb.extent.z:.2f})  location.z={bb.location.z:.2f}")
        print(f"[INFO] Camera pose : x={cam_x:.2f}  z={cam_z:.2f}  pitch=-8 deg")
        print(f"[INFO] LiDAR pose  : x=0.00  z={lidar_z:.2f}")

        cam_transform = carla.Transform(
            carla.Location(x=cam_x, z=cam_z),
            carla.Rotation(pitch=-8.0),
        )
        lidar_transform = carla.Transform(carla.Location(x=0.0, z=lidar_z))
        gnss_imu_transform = carla.Transform(carla.Location(z=lidar_z))
        event_transform = carla.Transform(carla.Location(x=cam_x, z=1.0))

        # ===== Camera RGB =====
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self.cfg["camera_width"]))
        cam_bp.set_attribute("image_size_y", str(self.cfg["camera_height"]))
        cam_bp.set_attribute("fov", str(self.cfg["camera_fov"]))
        self._add_sensor_safe(cam_bp, cam_transform, "camera")

        # ===== LiDAR =====
        lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels", str(self.cfg["lidar_channels"]))
        lidar_bp.set_attribute("range", str(self.cfg["lidar_range"]))
        lidar_bp.set_attribute("points_per_second",
                            str(self.cfg["lidar_points_per_second"]))
        lidar_bp.set_attribute("rotation_frequency",
                            str(self.cfg["lidar_rotation_frequency"]))
        self._add_sensor_safe(lidar_bp, lidar_transform, "lidar")

        # ===== GNSS =====
        gnss_bp = bp_lib.find("sensor.other.gnss")
        gnss_bp.set_attribute("noise_lat_stddev",
                            str(self.cfg["gnss_noise_lat_stddev"]))
        gnss_bp.set_attribute("noise_lon_stddev",
                            str(self.cfg["gnss_noise_lon_stddev"]))
        self._add_sensor_safe(gnss_bp, gnss_imu_transform, "gnss")

        # ===== IMU =====
        imu_bp = bp_lib.find("sensor.other.imu")
        self._add_sensor_safe(imu_bp, gnss_imu_transform, "imu")

        # ===== Camera semantica (mesma pose da RGB para alinhamento pixel-a-pixel) =====
        if self.cfg["enable_semantic_camera"]:
            sem_bp = bp_lib.find("sensor.camera.semantic_segmentation")
            sem_bp.set_attribute("image_size_x", str(self.cfg["camera_width"]))
            sem_bp.set_attribute("image_size_y", str(self.cfg["camera_height"]))
            sem_bp.set_attribute("fov", str(self.cfg["camera_fov"]))
            self._add_sensor_safe(sem_bp, cam_transform, "semantic")

        # ===== Radar (opcional) =====
        if self.cfg["enable_radar"]:
            radar_bp = bp_lib.find("sensor.other.radar")
            radar_bp.set_attribute("horizontal_fov", "30")
            radar_bp.set_attribute("vertical_fov", "10")
            radar_bp.set_attribute("range", "50")
            radar_transform = carla.Transform(carla.Location(x=cam_x, z=1.0))
            self._add_sensor_safe(radar_bp, radar_transform, "radar")

        # ===== Sensores de evento =====
        print("[INFO] Tentando anexar sensores de evento...")
        if self.cfg["try_collision_sensor"]:
            self.has_collision = self._try_add_event_sensor(
                "sensor.other.collision", "collision", self._on_collision,
                transform=event_transform,
            )
        if self.cfg["try_lane_invasion_sensor"]:
            self.has_lane_invasion = self._try_add_event_sensor(
                "sensor.other.lane_invasion", "lane_invasion", self._on_lane_invasion,
                transform=event_transform,
            )
        if self.cfg["try_obstacle_sensor"]:
            self.has_obstacle = self._try_add_event_sensor(
                "sensor.other.obstacle", "obstacle", self._on_obstacle,
                attribs={"distance": "15", "hit_radius": "1.5",
                        "only_dynamics": "true"},
                transform=event_transform,
            )

        ativos = ["camera", "lidar", "gnss", "imu"]
        if self.cfg["enable_semantic_camera"]: ativos.append("semantic")
        if self.cfg["enable_radar"]: ativos.append("radar")
        if self.has_collision: ativos.append("collision")
        if self.has_lane_invasion: ativos.append("lane_invasion")
        if self.has_obstacle: ativos.append("obstacle")
        print(f"[OK] Sensores ATIVOS: {', '.join(ativos)}")


    # --------------------------------------------------------
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

        width = self.cfg["camera_width"]
        if self.cfg["enable_semantic_camera"]:
            width *= 2  # split view: RGB | SEMANTIC

        self.video_writer = cv2.VideoWriter(
            str(path), fourcc, self.cfg["video_fps"],
            (width, self.cfg["camera_height"]),
        )
        if not self.video_writer.isOpened():
            raise RuntimeError(f"Nao consegui abrir VideoWriter em {path}")
        print(f"[OK] VideoWriter: {path} @ {self.cfg['video_fps']} fps  "
            f"({width}x{self.cfg['camera_height']})")


    # --------------------------------------------------------
    def _speed_limit_curva(self, base_kmh):
        """Reduz velocidade alvo se ha curva forte nos proximos waypoints."""
        try:
            queue_wps = list(self.agent._local_planner._waypoints_queue)
            n = min(self.cfg["curva_lookahead_wps"], len(queue_wps))
            if n < 3:
                return base_kmh
            wps = [w for w, _ in queue_wps[:n]]
            yaw_ini = wps[0].transform.rotation.yaw
            yaw_fim = wps[-1].transform.rotation.yaw
            delta = abs((yaw_fim - yaw_ini + 180) % 360 - 180)
            if delta < 5:
                fator = 1.0
            elif delta < 20:
                fator = self.cfg["curva_fator_leve"]
            elif delta < 40:
                fator = self.cfg["curva_fator_medio"]
            else:
                fator = self.cfg["curva_fator_forte"]
            return base_kmh * fator
        except Exception:
            return base_kmh

    # --------------------------------------------------------
    def _emergency_brake_pedestres(self):
        """Verifica pedestres em rota de colisao. Retorna (bool, dist)."""
        ego_loc = self.vehicle.get_location()
        ego_tf = self.vehicle.get_transform()
        fwd = ego_tf.get_forward_vector()
        speed = self.vehicle.get_velocity().length()
        raio = max(self.cfg["ebrake_min_radius"],
                   speed * self.cfg["ebrake_time_horizon"])
        corredor = self.cfg["ebrake_corridor_width"]
        for w in self.walkers:
            try:
                wloc = w.get_location()
                dx = wloc.x - ego_loc.x
                dy = wloc.y - ego_loc.y
                dist = (dx*dx + dy*dy) ** 0.5
                if dist > raio:
                    continue
                dot = dx * fwd.x + dy * fwd.y
                if dot < 0:
                    continue  # pedestre atras
                cross = abs(dx * fwd.y - dy * fwd.x)
                if cross < corredor:
                    return True, dist
            except Exception:
                continue
        return False, None

    # --------------------------------------------------------
    def _suavizar_controle(self, control):
        """Rate limiter no throttle e brake."""
        dt_thr = control.throttle - self._last_throttle
        if abs(dt_thr) > self.cfg["max_throttle_rate"]:
            control.throttle = self._last_throttle + math.copysign(
                self.cfg["max_throttle_rate"], dt_thr)
        dt_brk = control.brake - self._last_brake
        if abs(dt_brk) > self.cfg["max_brake_rate"]:
            control.brake = self._last_brake + math.copysign(
                self.cfg["max_brake_rate"], dt_brk)
        control.throttle = max(0.0, min(1.0, control.throttle))
        control.brake = max(0.0, min(1.0, control.brake))
        self._last_throttle = control.throttle
        self._last_brake = control.brake
        return control

    # --------------------------------------------------------
    def _tratar_stuck(self, frame, speed, ctrl):
        if speed < 0.3 and ctrl.throttle > 0.3:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        if self.stuck_counter >= self.cfg["stuck_threshold_ticks"]:
            print(f"[STUCK] Preso por {self.stuck_counter} ticks. "
                  f"Acao: {self.cfg['stuck_action']}")
            self._log_event({
                "frame": frame, "tipo": "stuck",
                "outro": f"acao={self.cfg['stuck_action']}"
            })
            if self.cfg["stuck_action"] == "respawn":
                spawns = self.map.get_spawn_points()
                random.shuffle(spawns)
                for sp in spawns:
                    try:
                        self.vehicle.set_transform(sp)
                        self.vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
                        self.stuck_counter = 0
                        # remarca destino atual da rota
                        if self.rota_idx < len(self.rota):
                            self.agent.set_destination(self.rota[self.rota_idx])
                        print(f"[STUCK] Respawn em {sp.location}, "
                              f"retomando rota wp {self.rota_idx}")
                        break
                    except Exception:
                        continue
                return False
            elif self.cfg["stuck_action"] == "abort":
                return True
        return False

    # --------------------------------------------------------
    def rodar(self):
        tele_path = self.output_root / "telemetria.csv"
        tele_file = open(tele_path, "w", newline="")
        cabecalho = [
            "frame", "sim_time",
            "x", "y", "z", "yaw", "pitch", "roll",
            "speed_mps", "target_speed_kmh",
            "throttle", "brake", "steer", "gear",
            "acc_x", "acc_y", "acc_z",
            "gnss_lat", "gnss_lon", "gnss_alt",
            "imu_acc_x", "imu_acc_y", "imu_acc_z",
            "imu_gyro_x", "imu_gyro_y", "imu_gyro_z",
            "imu_compass",
            "wp_x", "wp_y", "wp_road_id", "wp_lane_id",
            "odom_m",
            "route_idx", "route_total",
            "cloudiness", "precipitation", "sun_altitude",
            "n_collisions", "n_lane_invasions", "n_stucks", "n_emerg",
            "emerg_brake",
        ]
        tele_writer = csv.DictWriter(tele_file, fieldnames=cabecalho)
        tele_writer.writeheader()

        self.event_log_file = open(self.output_root / "eventos.log", "w")
        self.event_log_file.write(f"# Iniciado em {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.event_log_file.flush()

        self._init_video()

        ticks_total = int(self.duration_s / self.cfg["fixed_delta_seconds"])
        print(f"[INFO] Coletando ate {ticks_total} ticks (~{self.duration_s}s)")

        for i in range(ticks_total):
            frame = self.world.tick()

            # ===== AGENTE COM CONTROLE INTELIGENTE =====
            # 1. Speed limit por curvatura
            v_alvo = self._speed_limit_curva(self.cfg["agent_target_speed_kmh"])
            self.target_speed_atual = v_alvo
            self.agent.set_target_speed(v_alvo)

            # 2. run_step
            try:
                control = self.agent.run_step()
            except Exception as e:
                print(f"[WARN] agent.run_step falhou no frame {frame}: {e}")
                control = carla.VehicleControl(brake=1.0)

            # 3. Suavizar (rate limiter)
            control = self._suavizar_controle(control)

            # 4. Emergency brake (sobrescreve, ignora rate limit)
            self.emerg_active = False
            emerg, dist_ped = self._emergency_brake_pedestres()
            if emerg:
                control.throttle = 0.0
                control.brake = 1.0
                self._last_brake = 1.0
                self._last_throttle = 0.0
                self.emerg_active = True
                if frame - self._emerg_log_frame > self.cfg["ebrake_log_cooldown_frames"]:
                    self._log_event({
                        "frame": frame, "tipo": "emergency_brake",
                        "outro": f"pedestre a {dist_ped:.1f}m",
                    })
                    self._emerg_log_frame = frame

            self.vehicle.apply_control(control)

            # 5. Avanco de rota
            if self.agent.done():
                self.rota_idx += 1
                if self.rota_idx >= len(self.rota):
                    print("[INFO] ROTA COMPLETA! Encerrando coleta cedo.")
                    self._log_event({
                        "frame": frame, "tipo": "route_complete", "outro": ""
                    })
                    self.rota_completa = True
                    # processa o frame atual antes de sair
                else:
                    try:
                        self.agent.set_destination(self.rota[self.rota_idx])
                        print(f"[INFO] Waypoint {self.rota_idx}/{len(self.rota)-1} setado")
                    except Exception as e:
                        print(f"[WARN] set_destination falhou: {e}")

            # ===== SENSORES =====
            try:
                img = self._drenar_sensor("camera", frame)
                lidar = self._drenar_sensor("lidar", frame)
                gnss = self._drenar_sensor("gnss", frame)
                imu = self._drenar_sensor("imu", frame)
                semantic = None
                if self.cfg["enable_semantic_camera"]:
                    semantic = self._drenar_sensor("semantic", frame)
            except queue.Empty:
                print(f"[WARN] Sensor nao respondeu no frame {frame}")
                if self.rota_completa:
                    break
                continue

            if self.cfg["save_lidar_npy"]:
                pts = np.frombuffer(lidar.raw_data,
                                    dtype=np.float32).reshape(-1, 4)
                np.save(self.output_root / "lidar" / f"{frame:06d}.npy", pts)

            tf = self.vehicle.get_transform()
            vel = self.vehicle.get_velocity()
            acc = self.vehicle.get_acceleration()
            ctrl = self.vehicle.get_control()
            speed = (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5

            # Odometria
            cur_pos = (tf.location.x, tf.location.y)
            if self._last_pos is not None:
                dx = cur_pos[0] - self._last_pos[0]
                dy = cur_pos[1] - self._last_pos[1]
                self.odom_m += (dx*dx + dy*dy) ** 0.5
            self._last_pos = cur_pos

            # Stuck
            abortar = self._tratar_stuck(frame, speed, ctrl)
            if abortar:
                print("[STUCK] Abortando coleta")
                break

            # Waypoint
            wp = self.map.get_waypoint(tf.location, project_to_road=True)
            wp_x = wp.transform.location.x if wp else 0.0
            wp_y = wp.transform.location.y if wp else 0.0
            wp_road = wp.road_id if wp else -1
            wp_lane = wp.lane_id if wp else -1

            wt = self.world.get_weather()

            n_coll = sum(1 for e in self.events if e["tipo"] == "collision")
            n_lane = sum(1 for e in self.events if e["tipo"] == "lane_invasion")
            n_stuck = sum(1 for e in self.events if e["tipo"] == "stuck")
            n_emerg = sum(1 for e in self.events if e["tipo"] == "emergency_brake")

            row = {
                "frame": frame,
                "sim_time": i * self.cfg["fixed_delta_seconds"],
                "x": tf.location.x, "y": tf.location.y, "z": tf.location.z,
                "yaw": tf.rotation.yaw, "pitch": tf.rotation.pitch, "roll": tf.rotation.roll,
                "speed_mps": speed,
                "target_speed_kmh": self.target_speed_atual,
                "throttle": ctrl.throttle, "brake": ctrl.brake,
                "steer": ctrl.steer, "gear": ctrl.gear,
                "acc_x": acc.x, "acc_y": acc.y, "acc_z": acc.z,
                "gnss_lat": gnss.latitude, "gnss_lon": gnss.longitude, "gnss_alt": gnss.altitude,
                "imu_acc_x": imu.accelerometer.x, "imu_acc_y": imu.accelerometer.y,
                "imu_acc_z": imu.accelerometer.z,
                "imu_gyro_x": imu.gyroscope.x, "imu_gyro_y": imu.gyroscope.y,
                "imu_gyro_z": imu.gyroscope.z,
                "imu_compass": imu.compass,
                "wp_x": wp_x, "wp_y": wp_y,
                "wp_road_id": wp_road, "wp_lane_id": wp_lane,
                "odom_m": self.odom_m,
                "route_idx": self.rota_idx, "route_total": len(self.rota) - 1,
                "cloudiness": wt.cloudiness,
                "precipitation": wt.precipitation,
                "sun_altitude": wt.sun_altitude_angle,
                "n_collisions": n_coll,
                "n_lane_invasions": n_lane,
                "n_stucks": n_stuck,
                "n_emerg": n_emerg,
                "emerg_brake": int(self.emerg_active),
            }
            tele_writer.writerow(row)

            # ===== Painel RGB (esquerda) =====
            arr = np.frombuffer(img.raw_data, dtype=np.uint8)
            arr = arr.reshape((img.height, img.width, 4))
            frame_bgr = arr[:, :, :3].copy()

            if self.cfg["hud_enabled"]:
                try:
                    hud_info = dict(row)
                    hud_info["last_event"] = self.last_event_str
                    hud_info.setdefault("target_speed", row.get("target_speed_kmh", 0.0))
                    desenhar_hud(frame_bgr, hud_info)

                    npc_locs = []
                    for npc in self.npc_vehicles:
                        try:
                            loc = npc.get_location()
                            npc_locs.append((loc.x, loc.y))
                        except Exception:
                            pass
                    self.minimap.draw(frame_bgr, tf.location.x, tf.location.y,
                                    tf.rotation.yaw,
                                    npc_locations=npc_locs,
                                    route_points=self.rota,
                                    route_idx=self.rota_idx)
                except Exception as e:
                    print(f"[WARN] HUD/minimap falhou no frame {frame}: {e}")

            # ===== Painel SEMANTIC (direita) =====
            if semantic is not None:
                sem_arr = np.frombuffer(semantic.raw_data, dtype=np.uint8)
                sem_arr = sem_arr.reshape((semantic.height, semantic.width, 4))
                labels = sem_arr[:, :, 2]  # canal R = class id

                # Salva labels brutos
                if self.cfg["save_semantic_npy"]:
                    np.save(self.output_root / "semantic" / f"{frame:06d}.npy", labels)

                # Coloriza + legenda
                sem_bgr = colorir_semantic(labels)
                desenhar_legenda_semantic(sem_bgr)

                # Garante mesma altura/largura do painel RGB
                if sem_bgr.shape[:2] != frame_bgr.shape[:2]:
                    sem_bgr = cv2.resize(sem_bgr,
                                        (frame_bgr.shape[1], frame_bgr.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)

                # Linha divisoria branca entre painéis
                final_frame = np.hstack([frame_bgr, sem_bgr])
                cv2.line(final_frame,
                        (frame_bgr.shape[1], 0),
                        (frame_bgr.shape[1], frame_bgr.shape[0]),
                        (255, 255, 255), 1)
            else:
                final_frame = frame_bgr

            self.video_writer.write(final_frame)


            if i % 40 == 0:
                print(f"  tick {i}/{ticks_total}  v={speed*3.6:5.1f}km/h  "
                      f"alvo={self.target_speed_atual:4.1f}  "
                      f"pos=({tf.location.x:6.1f},{tf.location.y:6.1f})  "
                      f"odom={self.odom_m:6.1f}m  "
                      f"rota={self.rota_idx}/{len(self.rota)-1}  "
                      f"coll={n_coll} lane={n_lane} emerg={n_emerg}")

            if self.rota_completa:
                break

        # ---- Finalização ----
        self.finalizing = True
        for ator in self.actors:
            try:
                if hasattr(ator, "type_id") and ator.type_id.startswith("sensor."):
                    ator.stop()
            except Exception:
                pass
        time.sleep(0.1)

        tele_file.close()

        if self.video_writer is not None:
            self.video_writer.release()
            print(f"[OK] Video salvo em {self.output_root / 'video.mp4'}")

        if self.event_log_file is not None:
            self.event_log_file.write(f"# Finalizado em {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.event_log_file.write(f"# Total de eventos: {len(self.events)}\n")
            self.event_log_file.close()

        if self.events:
            with open(self.output_root / "eventos.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["frame", "tipo", "outro"])
                w.writeheader()
                for ev in self.events:
                    w.writerow(ev)
            print(f"[OK] {len(self.events)} eventos em eventos.csv / eventos.log")
        else:
            with open(self.output_root / "eventos.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["frame", "tipo", "outro"])
                w.writeheader()

        # Sumario
        print("\n=== SUMARIO ===")
        print(f"Rota: {'COMPLETA' if self.rota_completa else 'INCOMPLETA'} "
              f"({self.rota_idx}/{len(self.rota)-1} waypoints)")
        print(f"Distancia percorrida: {self.odom_m:.1f} m")
        print(f"Colisoes: {sum(1 for e in self.events if e['tipo'] == 'collision')}")
        print(f"Lane invasions: {sum(1 for e in self.events if e['tipo'] == 'lane_invasion')}")
        print(f"Stucks: {sum(1 for e in self.events if e['tipo'] == 'stuck')}")
        print(f"Emergency brakes: {sum(1 for e in self.events if e['tipo'] == 'emergency_brake')}")
        print(f"[OK] Coleta finalizada. Dados em: {self.output_root}")

    # --------------------------------------------------------
    def encerrar(self):
        print("[INFO] Encerrando — destruindo atores")
        self.finalizing = True
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


# ============================================================
#                            MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=120.0,
                        help="Duracao maxima em segundos (encerra antes se rota completa)")
    parser.add_argument("--output", type=str, default="./dataset/run_001")
    parser.add_argument("--town", type=str, default=CONFIG["town"])
    parser.add_argument("--vehicles", type=int, default=CONFIG["n_vehicles"])
    parser.add_argument("--pedestrians", type=int, default=CONFIG["n_pedestrians"])
    parser.add_argument("--target-speed", type=int,
                        default=CONFIG["agent_target_speed_kmh"],
                        help="Velocidade alvo do agente em km/h")
    parser.add_argument("--behavior", type=str, default=CONFIG["agent_behavior"],
                        choices=["cautious", "normal", "aggressive"])
    parser.add_argument("--rota-pontos", type=int, default=CONFIG["rota_n_pontos"],
                        help="Numero de waypoints da rota fixa")
    parser.add_argument("--no-hud", action="store_true",
                        help="Desativa overlay no video")
    parser.add_argument("--no-lidar", action="store_true",
                        help="Nao salva .npy do LiDAR")
    parser.add_argument("--semantic", action="store_true",
                        help="Adiciona camera de segmentacao semantica")
    parser.add_argument("--radar", action="store_true",
                        help="Adiciona radar frontal")
    args = parser.parse_args()

    CONFIG["town"] = args.town
    CONFIG["n_vehicles"] = args.vehicles
    CONFIG["n_pedestrians"] = args.pedestrians
    CONFIG["agent_target_speed_kmh"] = args.target_speed
    CONFIG["agent_behavior"] = args.behavior
    CONFIG["rota_n_pontos"] = args.rota_pontos
    if args.no_hud:
        CONFIG["hud_enabled"] = False
    if args.no_lidar:
        CONFIG["save_lidar_npy"] = False
    if args.semantic:
        CONFIG["enable_semantic_camera"] = True
        CONFIG["save_semantic_npy"] = True
    if args.radar:
        CONFIG["enable_radar"] = True

    output_root = Path(args.output)
    make_dirs(output_root)
    save_metadata(output_root, CONFIG)

    coletor = Coletor(CONFIG, output_root, args.duration)

    def handler(sig, frame):
        coletor.encerrar()
        sys.exit(0)
    signal.signal(signal.SIGINT, handler)

    try:
        coletor.conectar()
        coletor.spawnar_trafego()
        coletor.spawnar_pedestres()
        coletor.spawnar_caminhao()
        coletor.anexar_sensores()
        coletor.warmup()
        coletor.rodar()
    except Exception:
        print("[ERRO] Excecao na coleta:")
        traceback.print_exc()
    finally:
        coletor.encerrar()


if __name__ == "__main__":
    main()
