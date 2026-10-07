from pathlib import Path

import carla

# Distancia (m) entre barreiras ao longo da via
PASSO_BARREIRA = 1.5
# Altura da parede embutida do CARLA: 0 = desligada (ela cria parede tambem entre as pistas)
WALL_HEIGHT = 0.0

client = carla.Client('localhost', 2000)
client.set_timeout(120.0)

with open(Path(__file__).resolve().parent / 'mapa_final_3d.xodr', 'r', encoding='utf-8') as f:
    xodr_content = f.read()

params = carla.OpendriveGenerationParameters(
    vertex_distance=0.5,
    max_road_length=50.0,
    wall_height=WALL_HEIGHT,
    additional_width=0.6,
    smooth_junctions=True,
    enable_mesh_visibility=True,
    enable_pedestrian_navigation=True
)

world = client.generate_opendrive_world(xodr_content, params)
print("Mapa 3D (mapa_final_3d.xodr) carregado com sucesso no CARLA.")

# --- Barreiras apenas nas bordas externas das vias ---
carla_map = world.get_map()
bp_lib = world.get_blueprint_library()
candidatos = bp_lib.filter('static.prop.streetbarrier')
if not candidatos:
    candidatos = bp_lib.filter('static.prop.*barrier*')
barreira_bp = candidatos[0]
print(f"Blueprint de barreira: {barreira_bp.id}")

comandos = []
for wp in carla_map.generate_waypoints(PASSO_BARREIRA):
    if wp.lane_type != carla.LaneType.Driving:
        continue
    t = wp.transform
    lateral = t.get_right_vector()
    meia = wp.lane_width / 2.0
    for lado in (1.0, -1.0):
        teste = t.location + lateral * (lado * (meia + 0.5))
        if carla_map.get_waypoint(teste, project_to_road=False,
                                  lane_type=carla.LaneType.Driving) is not None:
            continue  # ha outra pista ao lado: nao e borda
        pos = t.location + lateral * (lado * (meia + 0.25))
        comandos.append(carla.command.SpawnActor(
            barreira_bp, carla.Transform(pos, carla.Rotation(yaw=t.rotation.yaw))))

resultados = client.apply_batch_sync(comandos, True)
ok = sum(1 for r in resultados if not r.error)
print(f"Barreiras nas bordas: {ok} criadas, {len(resultados) - ok} falharam.")
