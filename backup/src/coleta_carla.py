#coleta_carla.py

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
import numpy as np


CONFIG = {
    "host": "localhost",
    "port": 2000,
    "town": "Town03",  # Town03 tem mais cruzamentos = mais ação
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
    # NOVO: tráfego
    "n_vehicles": 30,
    "n_pedestrians": 20,
    # NOVO: tentar reativar sensores de evento
    "try_collision_sensor": True,
    "try_lane_invasion_sensor": True,
    "ticks_between_event_sensors": 10,  # quantos ticks de espera antes de anexar event sensor
}


def make_dirs(output_root: Path):
    (output_root / "camera").mkdir(parents=True, exist_ok=True)
    (output_root / "lidar").mkdir(parents=True, exist_ok=True)


def save_metadata(output_root: Path, cfg: dict):
    meta = {k: (str(v) if not isinstance(v, (int, float, str, bool)) else v)
            for k, v in cfg.items()}
    meta["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(output_root / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


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
        # tracking de quais sensores de evento conseguiram ser anexados
        self.has_collision = False
        self.has_lane_invasion = False

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
        print(f"[OK] Conectado, mapa: {self.world.get_map().name}")

    def spawnar_trafego(self):
        """Spawna veículos NPC com autopilot."""
        n = self.cfg["n_vehicles"]
        if n <= 0:
            return
        bp_lib = self.world.get_blueprint_library()
        # Filtra blueprints de veículos comuns (ignora bicicletas/motos pra simplificar)
        vehicle_bps = bp_lib.filter("vehicle.*")
        vehicle_bps = [bp for bp in vehicle_bps if int(bp.get_attribute("number_of_wheels")) == 4]
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        spawned = 0
        for spawn in spawn_points:
            if spawned >= n:
                break
            bp = random.choice(vehicle_bps)
            if bp.has_attribute("color"):
                color = random.choice(bp.get_attribute("color").recommended_values)
                bp.set_attribute("color", color)
            try:
                npc = self.world.spawn_actor(bp, spawn)
                npc.set_autopilot(True, self.tm.get_port())
                self.npc_vehicles.append(npc)
                self.actors.append(npc)
                spawned += 1
            except RuntimeError:
                continue  # spawn point ocupado, tenta o próximo
        self.world.tick()
        print(f"[OK] {spawned}/{n} veículos NPC spawnados")

    def spawnar_pedestres(self):
        """Spawna pedestres com IA de caminhada."""
        n = self.cfg["n_pedestrians"]
        if n <= 0:
            return
        bp_lib = self.world.get_blueprint_library()
        walker_bps = bp_lib.filter("walker.pedestrian.*")
        controller_bp = bp_lib.find("controller.ai.walker")

        spawned = 0
        spawn_attempts = 0
        max_attempts = n * 5  # tenta até 5x mais que o desejado
        while spawned < n and spawn_attempts < max_attempts:
            spawn_attempts += 1
            # Acha posição navegável aleatória
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
            controller.set_max_speed(1.4)  # ~5 km/h, velocidade humana

            self.walkers.append(walker)
            self.walker_controllers.append(controller)
            self.actors.extend([walker, controller])
            spawned += 1
        print(f"[OK] {spawned}/{n} pedestres spawnados")

    def spawnar_caminhao(self):
        """Spawna o caminhão principal (ego vehicle)."""
        bp_lib = self.world.get_blueprint_library()
        veh_bp = bp_lib.filter(self.cfg["vehicle_filter"])[0]
        spawn_points = self.world.get_map().get_spawn_points()
        # Tenta vários spawn points caso o primeiro esteja ocupado por NPC
        for spawn in random.sample(spawn_points, len(spawn_points)):
            try:
                self.vehicle = self.world.spawn_actor(veh_bp, spawn)
                break
            except RuntimeError:
                continue
        self.vehicle.set_autopilot(True, self.tm.get_port())
        self.actors.append(self.vehicle)
        self.world.tick()
        print(f"[OK] Caminhão (ego) spawnado em {self.vehicle.get_location()}")

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
        """Tenta anexar um sensor de evento (collision/lane_invasion) com workaround.
        Se der bug, retorna False e segue."""
        # Espera vários ticks pra "drenar" qualquer streaming pendente
        for _ in range(self.cfg["ticks_between_event_sensors"]):
            self.world.tick()

        try:
            bp_lib = self.world.get_blueprint_library()
            bp = bp_lib.find(blueprint_name)
            sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
            sensor.listen(callback)
            self.actors.append(sensor)
            # se sobreviveu até aqui, tenta um tick extra pra confirmar
            self.world.tick()
            print(f"  + {name} anexado (com workaround)")
            return True
        except Exception as e:
            print(f"  ! {name} FALHOU ({type(e).__name__}): será omitido")
            return False

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

        # Sensores de evento — tenta anexar com workaround
        print("[INFO] Tentando anexar sensores de evento (podem falhar por bug do 0.9.16)...")

        if self.cfg["try_collision_sensor"]:
            self.has_collision = self._try_add_event_sensor(
                "sensor.other.collision",
                "collision",
                lambda evt: self.events.append({
                    "frame": evt.frame, "tipo": "collision",
                    "outro": evt.other_actor.type_id,
                })
            )

        if self.cfg["try_lane_invasion_sensor"]:
            self.has_lane_invasion = self._try_add_event_sensor(
                "sensor.other.lane_invasion",
                "lane_invasion",
                lambda evt: self.events.append({
                    "frame": evt.frame, "tipo": "lane_invasion",
                    "outro": str([m.type for m in evt.crossed_lane_markings]),
                })
            )

        print("[OK] Setup de sensores finalizado")
        print(f"     Sensores ATIVOS: camera, lidar, gnss, imu"
              + (", collision" if self.has_collision else "")
              + (", lane_invasion" if self.has_lane_invasion else ""))
        if not self.has_collision:
            print(f"     Sensor collision OMITIDO (bug do streaming) — colisão será inferida da telemetria")
        if not self.has_lane_invasion:
            print(f"     Sensor lane_invasion OMITIDO (bug do streaming) — invasão será inferida pós-coleta")

    def _drenar_sensor(self, name, frame_alvo, timeout=5.0):
        q = self.sensor_queues[name]
        while True:
            data = q.get(timeout=timeout)
            if data.frame == frame_alvo:
                return data

    def warmup(self):
        n = self.cfg["warmup_ticks"]
        print(f"[INFO] Warm-up: {n} ticks (NPCs e pedestres começam a se mover)...")
        for _ in range(n):
            self.world.tick()
            for q in self.sensor_queues.values():
                while not q.empty():
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        break

    def rodar(self):
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
        ]
        tele_writer = csv.DictWriter(tele_file, fieldnames=cabecalho)
        tele_writer.writeheader()

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
                print(f"[WARN] Sensor não respondeu no frame {frame}")
                continue

            img.save_to_disk(str(self.output_root / "camera" / f"{frame:06d}.png"))

            pts = np.frombuffer(lidar.raw_data, dtype=np.float32).reshape(-1, 4)
            np.save(self.output_root / "lidar" / f"{frame:06d}.npy", pts)

            tf = self.vehicle.get_transform()
            vel = self.vehicle.get_velocity()
            acc = self.vehicle.get_acceleration()
            ctrl = self.vehicle.get_control()
            speed = (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5

            tele_writer.writerow({
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
            })

            if i % 40 == 0:
                print(f"  tick {i}/{ticks_total}  speed={speed:.2f}m/s  pos=({tf.location.x:.1f},{tf.location.y:.1f})  eventos={len(self.events)}")

        tele_file.close()

        # Salva eventos (se algum sensor de evento foi anexado)
        if self.events:
            with open(self.output_root / "eventos.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["frame", "tipo", "outro"])
                w.writeheader()
                for ev in self.events:
                    w.writerow(ev)
            print(f"[OK] {len(self.events)} eventos registrados em eventos.csv")

        print(f"[OK] Coleta finalizada. Dados em: {self.output_root}")

    def encerrar(self):
        print("[INFO] Encerrando — destruindo atores")
        # Para os controllers dos walkers primeiro
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
    args = parser.parse_args()

    CONFIG["town"] = args.town
    CONFIG["n_vehicles"] = args.vehicles
    CONFIG["n_pedestrians"] = args.pedestrians

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