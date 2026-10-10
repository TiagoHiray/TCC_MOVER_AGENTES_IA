"""Servidor só de ingestão do Sensor Logger (sem agentes, CARLA, numpy ou scipy).

Para a máquina de campo, que só recebe o HTTP Push do celular. Uso, a partir de src/:
    python -m mover.ingestao.rodar_ingestao                  # 0.0.0.0:8000
    python -m mover.ingestao.rodar_ingestao --porta 8001
"""

from __future__ import annotations

import argparse
import logging
import socket
import sys
from pathlib import Path

if __package__ in (None, ""):  # execução direta: python src/mover/ingestao/rodar_ingestao.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mover.config import caminho, carregar_yaml

log = logging.getLogger("mover.ingestao")


def ips_locais() -> list[str]:
    ips: list[str] = []
    try:
        # IP da rota padrão (nenhum pacote é enviado em UDP connect)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ips.append(s.getsockname()[0])
    except OSError:
        pass
    try:
        ips += [i[4][0] for i in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)]
    except OSError:
        pass
    unicos = [ip for ip in dict.fromkeys(ips) if not ip.startswith("127.")]
    return unicos or ["127.0.0.1"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Servidor só de ingestão do Sensor Logger.")
    parser.add_argument("--config", default="config/simulacao.yaml", help="YAML com a seção ingestao")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--porta", type=int, default=8000)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    import uvicorn
    from fastapi import FastAPI

    from mover.ingestao.rotas import instalar_ingestao

    cfg = carregar_yaml(args.config) if caminho(args.config).exists() else {}
    log.info("Use o IPv4 da placa Wi-Fi conectada ao celular (ipconfig), não o da VPN.")
    app = FastAPI(title="MOVER - ingestão do Sensor Logger")
    instalar_ingestao(app, cfg)

    @app.get("/saude")
    async def saude() -> dict[str, bool]:
        return {"ok": True}

    for ip in ips_locais():
        log.info("Push URL: http://%s:%d/ingestao/sensorlogger", ip, args.porta)
    log.info("Status:   http://127.0.0.1:%d/ingestao/status", args.porta)
    # h11/asyncio puros: não dependem de httptools/uvloop compilados (bloqueáveis por política do Windows)
    uvicorn.run(app, host=args.host, port=args.porta, http="h11", loop="asyncio", log_level="info")


if __name__ == "__main__":
    main()
