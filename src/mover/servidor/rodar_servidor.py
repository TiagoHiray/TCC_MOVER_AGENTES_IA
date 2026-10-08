"""Sobe o servidor da camada agêntica (FastAPI + WebSocket) e a página dos 4 painéis.

Uso, a partir da pasta src/ do repositório:
    python -m mover.servidor.rodar_servidor                      # 127.0.0.1:8000 (config/simulacao.yaml)
    python -m mover.servidor.rodar_servidor --host 0.0.0.0 --porta 8001

Com o servidor no ar, rode o replay em outro terminal (python -m mover.simulacao.replay_carla) e
abra a página em http://127.0.0.1:8000/ (simulação, dashboard, log e chat). A documentação
interativa das rotas fica em http://127.0.0.1:8000/docs e o log ao vivo em ws://127.0.0.1:8000/ws.
O provedor de LLM de cada volta é escolhido pelo replay ao abrir a sessão (--provedor/--modelo);
sem isso vale o .env ou o config/agentes.yaml.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # execução direta: python src/mover/servidor/rodar_servidor.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mover.config import caminho, carregar_yaml

log = logging.getLogger("mover.servidor")

TIMEOUT_ENCERRAMENTO_S = 2  # depois disso o uvicorn fecha à força as conexões abertas da página


def configurar_logs(depurar: bool = False) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # terminais sem UTF-8 (Windows) não quebram com "³"
    logging.basicConfig(level=logging.DEBUG if depurar else logging.INFO, format="%(levelname)-7s %(message)s")
    for ruidoso in ("httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)


def carregar_configs(config_simulacao: str = "config/simulacao.yaml",
                     config_agentes: str = "config/agentes.yaml") -> tuple[dict[str, Any], dict[str, Any]]:
    cfg_sim = carregar_yaml(config_simulacao) if caminho(config_simulacao).exists() else {}
    return carregar_yaml(config_agentes), cfg_sim


def iniciar_em_thread(app: Any, host: str, porta: int, timeout_s: float = 30.0):
    """Roda o uvicorn numa thread daemon (usado pelo replay com --iniciar-servidor).

    Devolve (servidor, thread). Para parar: parar_servidor(servidor, thread).
    """
    import uvicorn

    # a página deixa o WebSocket (e a transmissão da câmera) abertos: sem o limite, o
    # encerramento esperaria o navegador fechar a aba
    config = uvicorn.Config(app, host=host, port=porta, log_level="warning", lifespan="on",
                            timeout_graceful_shutdown=TIMEOUT_ENCERRAMENTO_S)
    servidor = uvicorn.Server(config)
    thread = threading.Thread(target=servidor.run, name="servidor_agentes", daemon=True)
    thread.start()
    limite = time.monotonic() + timeout_s
    while not servidor.started:
        if not thread.is_alive():
            raise RuntimeError(f"o servidor não subiu em {host}:{porta} (porta ocupada?)")
        if time.monotonic() > limite:
            servidor.should_exit = True
            raise TimeoutError(f"o servidor não respondeu em {timeout_s:.0f} s")
        time.sleep(0.05)
    log.info("Servidor da camada agêntica em http://%s:%d (painel em http://%s:%d/, log ao vivo em ws://%s:%d/ws)",
             host, porta, host, porta, host, porta)
    return servidor, thread


def parar_servidor(servidor: Any, thread: threading.Thread, timeout_s: float = 15.0) -> None:
    servidor.should_exit = True
    thread.join(timeout=timeout_s)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Servidor da camada agêntica (FastAPI + WebSocket).")
    parser.add_argument("--config", default="config/simulacao.yaml", help="YAML da simulação (seção servidor)")
    parser.add_argument("--config-agentes", default="config/agentes.yaml", help="YAML da camada agêntica")
    parser.add_argument("--host", help="endereço (padrão: YAML, 127.0.0.1)")
    parser.add_argument("--porta", type=int, help="porta (padrão: YAML, 8000)")
    parser.add_argument("--depurar", action="store_true", help="mostra os prompts e as respostas do Supervisor")
    args = parser.parse_args(argv)

    configurar_logs(args.depurar)
    import uvicorn

    from mover.servidor.app import criar_app

    cfg_agentes, cfg_sim = carregar_configs(args.config, args.config_agentes)
    cs = cfg_sim.get("servidor", {})
    host = args.host or cs.get("host", "127.0.0.1")
    porta = int(args.porta or cs.get("porta", 8000))
    app = criar_app(cfg_agentes, cfg_sim)
    log.info("Painel: http://%s:%d/ | documentação das rotas: http://%s:%d/docs | log ao vivo: ws://%s:%d/ws",
             host, porta, host, porta, host, porta)
    uvicorn.run(app, host=host, port=porta, log_level="info", timeout_graceful_shutdown=TIMEOUT_ENCERRAMENTO_S)


if __name__ == "__main__":
    main()
