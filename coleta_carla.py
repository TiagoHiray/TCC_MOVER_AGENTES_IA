# coleta_carla.py (v5)

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
}


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
        self.trail = []

    def _w2p(self, x, y):
        px = int((x - self.min_x) / self.rng * (self.size - 1))
        py = int((y - self.min_y) / self.rng * (self.size - 1))
        py = self.size - 1 - py  # inverte y (tela vs mundo)
        return px, py

    def draw(self, frame_bgr, ego_x, ego_y, yaw_deg, npc_locations=None):
        self.trail.append((ego_x, ego_y))
        if len(self.trail) > 600:
            self.trail = self.trail[-600:]

        canvas = self.bg.copy()

        # Rastro
        for i in range(1, len(self.trail)):
            p1 = self._w2p(*self.trail[i - 1])
            p2 = self._w2p(*self.trail[i])
            cv2.line(canvas, p1, p2, (0, 200, 255), 1)

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
        self.has_collision = False
        self.has_lane_invasion = False
        self.video_writer = None
        self.event_log_file = None
        self.last_event_str = ""
        self.odom_m = 0.0
        self._last_pos = None
        self.finalizing = False
        self.minimap = None


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
        spawn_points = self.map.get_spawn_points()
        for spawn in random.sample(spawn_points, len(spawn_points)):
            try:
                self.vehicle = self.world.spawn_actor(veh_bp, spawn)
                break
            except RuntimeError:
                continue
        self.vehicle.set_autopilot(True, self.tm.get_port())
        self.actors.append(self.vehicle)
        self.world.tick()
        print(f"[OK] Caminhao (ego) spawnado em {self.vehicle.get_location()}")

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
        self._add_sensor_safe(cam_bp, carla.Transform(carla.Location(x=2.0, z=2.5)), "camera")

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
        self.video_writer = cv2.VideoWriter(
            str(path),
            fourcc,
            self.cfg["video_fps"],
            (self.cfg["camera_width"], self.cfg["camera_height"]),
        )
        if not self.video_writer.isOpened():
            raise RuntimeError(f"Nao consegui abrir VideoWriter em {path}")
        print(f"[OK] VideoWriter: {path} @ {self.cfg['video_fps']} fps")

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
            frame = self.world.tick()

            try:
                img = self._drenar_sensor("camera", frame)
                lidar = self._drenar_sensor("lidar", frame)
                gnss = self._drenar_sensor("gnss", frame)
                imu = self._drenar_sensor("imu", frame)
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

            n_coll = sum(1 for e in self.events if e["tipo"] == "collision")
            n_lane = sum(1 for e in self.events if e["tipo"] == "lane_invasion")

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
    args = parser.parse_args()

    CONFIG["town"] = args.town
    CONFIG["n_vehicles"] = args.vehicles
    CONFIG["n_pedestrians"] = args.pedestrians
    if args.no_hud:
        CONFIG["hud_enabled"] = False
    if args.no_lidar:
        CONFIG["save_lidar_npy"] = False

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
    finally:
        coletor.encerrar()


if __name__ == "__main__":
    main()
