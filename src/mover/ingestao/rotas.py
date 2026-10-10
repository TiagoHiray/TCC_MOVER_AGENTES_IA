"""Rotas da ingestão ao vivo do Sensor Logger.

    POST /ingestao/sensorlogger[?token=...]  -> Push URL do app (Settings > Data Streaming > HTTP Push)
    GET  /ingestao/status[?token=...]        -> amostras, taxa e última leitura por sensor

O token é opcional: MOVER_INGESTAO_TOKEN no .env (ou `ingestao.token` no YAML, usado pelos testes).
"""

from __future__ import annotations

import asyncio
import hmac
import json
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from mover.agentes.llm import carregar_env
from mover.config import caminho
from mover.ingestao.armazem import ArmazemIngestao, MensagemInvalida


def instalar_ingestao(app: FastAPI, cfg_simulacao: dict[str, Any]) -> ArmazemIngestao:
    cfg = cfg_simulacao.get("ingestao", {})
    armazem = ArmazemIngestao(caminho(cfg.get("pasta", "data/ingestao")))
    max_bytes = int(cfg.get("max_corpo_bytes", 5_000_000))
    if "token" in cfg:
        token = cfg["token"] or None
    else:
        carregar_env()
        token = os.getenv("MOVER_INGESTAO_TOKEN") or None
    app.state.ingestao = armazem

    def autorizar(recebido: str | None) -> None:
        if token and not hmac.compare_digest(recebido or "", token):
            raise HTTPException(401, "token inválido")

    @app.post("/ingestao/sensorlogger")
    async def receber(request: Request, token: str | None = None) -> dict[str, Any]:
        autorizar(token)
        try:
            declarado = int(request.headers.get("content-length") or 0)
        except ValueError:
            raise HTTPException(400, "content-length inválido") from None
        if declarado > max_bytes:
            raise HTTPException(413, "corpo grande demais")
        corpo = await request.body()
        if not corpo or len(corpo) > max_bytes:
            raise HTTPException(413 if corpo else 400, "corpo vazio ou grande demais")
        try:
            msg = json.loads(corpo)
        except (ValueError, UnicodeDecodeError):
            raise HTTPException(400, "JSON inválido") from None
        if not isinstance(msg, dict):
            raise HTTPException(400, "esperado um objeto JSON")
        try:
            n = await asyncio.to_thread(armazem.registrar, msg)
        except MensagemInvalida as erro:
            raise HTTPException(400, str(erro)) from None
        return {"ok": True, "leituras": n}

    @app.get("/ingestao/status")
    async def status(token: str | None = None) -> dict[str, Any]:
        autorizar(token)
        return armazem.status()

    return armazem
