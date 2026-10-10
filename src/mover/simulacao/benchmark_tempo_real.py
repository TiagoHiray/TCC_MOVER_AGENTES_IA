"""Mede quantas vezes mais rápido que o tempo real o CARLA simula (ego sozinho, modo síncrono).

Responde à pergunta do gêmeo: cabe simular H segundos à frente dentro do ciclo de 1 s?
Não carrega mapa (usa o mundo já aberto) e restaura as configurações no fim.

Uso (na VM, com o CarlaUE4 aberto; ambiente com Python 3.12 + carla==0.9.16):
    python -m mover.simulacao.benchmark_tempo_real
    python -m mover.simulacao.benchmark_tempo_real --horizonte 10 --repeticoes 5 --com-renderizacao
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # execução direta
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mover.config import caminho, carregar_yaml


def medir(world, carla, ator, passos: int, ticks_reposicao: int) -> tuple[float, float]:
    """Devolve (s de parede para reposicionar + estabilizar, s de parede para simular `passos`)."""
    inicio_pose = ator.get_transform()
    t0 = time.perf_counter()
    ator.set_transform(inicio_pose)
    ator.set_target_velocity(carla.Vector3D(5.0, 0.0, 0.0))
    for _ in range(ticks_reposicao):
        world.tick()
    t1 = time.perf_counter()
    ator.apply_control(carla.VehicleControl(throttle=0.4, steer=0.0))
    for _ in range(passos):
        world.tick()
    return t1 - t0, time.perf_counter() - t1


def main(argv: list[str] | None = None) -> None:
    cfg = carregar_yaml("config/simulacao.yaml") if caminho("config/simulacao.yaml").exists() else {}
    cc = cfg.get("carla", {})
    parser = argparse.ArgumentParser(description="Fator de tempo real do CARLA (ego sozinho).")
    parser.add_argument("--host", default=cc.get("host", "localhost"))
    parser.add_argument("--porta", type=int, default=int(cc.get("porta", 2000)))
    parser.add_argument("--passo", type=float, default=float(cc.get("passo_s", 0.05)))
    parser.add_argument("--horizonte", type=float, default=10.0, help="segundos simulados por medição")
    parser.add_argument("--repeticoes", type=int, default=5)
    parser.add_argument("--ticks-reposicao", type=int, default=3, help="ticks de estabilização após teleporte")
    parser.add_argument("--blueprint", default=cfg.get("veiculo", {}).get("blueprint", "vehicle.carlamotors.firetruck"))
    parser.add_argument("--com-renderizacao", action="store_true", help="não liga o no_rendering_mode")
    args = parser.parse_args(argv)

    try:
        import carla
    except ImportError:
        raise SystemExit("Instale o cliente do CARLA 0.9.16 (Python 3.12): pip install carla==0.9.16") from None

    client = carla.Client(args.host, args.porta)
    client.set_timeout(float(cc.get("timeout_s", 60.0)))
    world = client.get_world()
    original = world.get_settings()
    ator = None
    try:
        config = world.get_settings()
        config.synchronous_mode = True
        config.fixed_delta_seconds = args.passo
        config.no_rendering_mode = not args.com_renderizacao
        world.apply_settings(config)

        bp = world.get_blueprint_library().find(args.blueprint)
        for ponto in world.get_map().get_spawn_points() or [carla.Transform(carla.Location(z=2.0))]:
            ator = world.try_spawn_actor(bp, ponto)
            if ator is not None:
                break
        if ator is None:
            raise SystemExit("Não foi possível criar o veículo (sem spawn points livres?).")
        world.tick()

        passos = int(round(args.horizonte / args.passo))
        medicoes = [medir(world, carla, ator, passos, args.ticks_reposicao) for _ in range(args.repeticoes)]
        reposicao = [m[0] for m in medicoes]
        simulacao = [m[1] for m in medicoes]
        sim_med = statistics.median(simulacao)
        ciclo = statistics.median(r + s for r, s in medicoes)
        print(f"CARLA {client.get_server_version()} | mapa {world.get_map().name} | "
              f"renderização {'ligada' if args.com_renderizacao else 'desligada'} | passo {args.passo} s")
        print(f"Horizonte {args.horizonte:.1f} s ({passos} ticks), {args.repeticoes} repetições:")
        print(f"  simulação: mediana {sim_med:.3f} s de parede -> {args.horizonte / sim_med:.1f}x o tempo real "
              f"({1000 * sim_med / passos:.2f} ms/tick)")
        print(f"  reposicionar + {args.ticks_reposicao} ticks: mediana {statistics.median(reposicao) * 1000:.0f} ms")
        print(f"  ciclo completo (reposicionar + horizonte): {ciclo:.3f} s "
              f"-> {'cabe' if ciclo < 1.0 else 'NÃO cabe'} num ciclo de 1 s")
    finally:
        if ator is not None:
            ator.destroy()
        world.apply_settings(original)


if __name__ == "__main__":
    main()
