"""Aplicação FastAPI da camada agêntica (uma sessão por vez) e da interface de 4 painéis.

Rotas da camada agêntica:
    GET    /saude                 -> {"ok": true, "sessao": {...} | null}
    POST   /sessao                -> cria a sessão e analisa o bloco 0 (corpo: injetar_eventos, provedor, modelo, sem_ml)
    GET    /sessao                -> dados da sessão atual
    POST   /sessao/blocos/{k}     -> o caminhão entrou no bloco k: espera a análise, publica e libera o k+1 (409 fora de ordem)
    POST   /sessao/estado         -> pose do caminhão (sim_time, x, y, yaw...); vira mensagem "estado" com a telemetria
    GET    /sessao/log?desde=N    -> entradas publicadas a partir da N-ésima
    DELETE /sessao                -> encerra a sessão e devolve o resumo
    WS     /ws                    -> histórico ao conectar e, depois, as mensagens ao vivo (ver sessao.py)

Rotas da interface (página, câmera do CARLA, cena 2D e chat): ver mover/interface/rotas.py.
Rotas da ingestão ao vivo do Sensor Logger (/ingestao/...): ver mover/ingestao/rotas.py.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

from mover.agentes.supervisor import Supervisor
from mover.config import caminho, carregar_yaml
from mover.ingestao.rotas import instalar_ingestao
from mover.interface.rotas import instalar_interface
from mover.servidor.sessao import (BlocoForaDeOrdem, BlocoInvalido, Difusor, OpcoesSessao, SessaoAgentes, SessaoEncerrada,
                                   para_json)

log = logging.getLogger("mover.servidor")


class OpcoesSessaoModelo(BaseModel):
    injetar_eventos: bool = False
    provedor: str | None = None
    modelo: str | None = None
    sem_ml: bool = False
    fator_tempo: float = 1.0
    tempo_real: bool = True


class EstadoModelo(BaseModel):
    """Pose enviada pelo replay; campos extras (x, y, yaw, z, correcao_m...) são repassados."""

    model_config = ConfigDict(extra="allow")
    sim_time: float


def criar_app(cfg_agentes: dict[str, Any] | None = None, cfg_simulacao: dict[str, Any] | None = None,
              telemetria: pd.DataFrame | None = None,
              fabrica_supervisor: Callable[[OpcoesSessao], Supervisor] | None = None,
              sem_ml: bool = False, fabrica_llm_chat: Callable[[OpcoesSessao | None], Any] | None = None) -> FastAPI:
    """Monta o app. `telemetria`, `fabrica_supervisor`, `sem_ml` e `fabrica_llm_chat` existem para os testes."""
    cfg_agentes = cfg_agentes or carregar_yaml("config/agentes.yaml")
    cfg_simulacao = cfg_simulacao or (carregar_yaml("config/simulacao.yaml") if caminho("config/simulacao.yaml").exists() else {})
    pasta_sessoes = caminho(cfg_simulacao.get("servidor", {}).get("pasta_sessoes", "data/agentes/sessoes"))
    difusor = Difusor()
    estado: dict[str, SessaoAgentes | None] = {"sessao": None}
    trava_sessao = asyncio.Lock()

    @asynccontextmanager
    async def ciclo_de_vida(app: FastAPI):
        difusor.ligar(asyncio.get_running_loop())
        yield
        sessao = estado["sessao"]
        if sessao is not None and not sessao.encerrada:
            await asyncio.to_thread(sessao.encerrar)

    app = FastAPI(title="MOVER - camada agêntica", version="1.0", lifespan=ciclo_de_vida)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.state.difusor = difusor
    app.state.estado = estado
    instalar_interface(app, estado, cfg_agentes, cfg_simulacao, fabrica_llm_chat)
    instalar_ingestao(app, cfg_simulacao)

    def sessao_atual() -> SessaoAgentes:
        sessao = estado["sessao"]
        if sessao is None:
            raise HTTPException(404, "nenhuma sessão; crie uma com POST /sessao")
        return sessao

    @app.get("/saude")
    async def saude() -> dict[str, Any]:
        sessao = estado["sessao"]
        return {"ok": True, "sessao": sessao.info() if sessao else None, "clientes_ws": difusor.n_clientes}

    @app.post("/sessao")
    async def criar_sessao(opcoes: OpcoesSessaoModelo) -> dict[str, Any]:
        async with trava_sessao:
            anterior = estado["sessao"]
            if anterior is not None and not anterior.encerrada:
                log.info("Encerrando a sessão %s para abrir outra.", anterior.id)
                await asyncio.to_thread(anterior.encerrar)
            op = OpcoesSessao(**opcoes.model_dump())

            def montar() -> SessaoAgentes:
                supervisor = fabrica_supervisor(op) if fabrica_supervisor else None
                sessao = SessaoAgentes(cfg_agentes, op, difusor, pasta_sessoes, telemetria=telemetria,
                                       supervisor=supervisor, sem_ml_forcado=sem_ml)
                sessao.preparar()
                return sessao

            sessao = await asyncio.to_thread(montar)
            estado["sessao"] = sessao
            log.info("Sessão %s: %d blocos de %.0f s, LLM %s", sessao.id, sessao.camada.n_blocos,
                     sessao.camada.duracao_bloco, sessao.info()["llm"])
            # a previsão do bloco 0 (analisado antes da partida) vai junto: a página zera tudo ao trocar de sessão
            difusor.publicar({"tipo": "sessao", "evento": "iniciada", "sessao": sessao.info(),
                              "previsao": sessao.previsao_atual()})
            return sessao.info()

    @app.get("/sessao")
    async def ver_sessao() -> dict[str, Any]:
        return sessao_atual().info()

    @app.post("/sessao/blocos/{k}")
    async def entrar_no_bloco(k: int) -> dict[str, Any]:
        sessao = sessao_atual()
        try:
            resultado = await asyncio.to_thread(sessao.entrar_no_bloco, k)
        except BlocoInvalido as erro:
            raise HTTPException(404, {"mensagem": str(erro)}) from None
        except BlocoForaDeOrdem as erro:
            raise HTTPException(409, {"mensagem": str(erro), "proximo_bloco": erro.esperado}) from None
        except SessaoEncerrada as erro:
            raise HTTPException(409, {"mensagem": str(erro)}) from None
        return para_json({"bloco": k, "entradas": resultado.entradas, "espera_s": round(resultado.espera_s, 3),
                          "latencia_s": round(resultado.latencia_s, 3), "proximo_bloco": sessao.proximo_bloco})

    @app.post("/sessao/estado", status_code=204)
    async def receber_estado(estado_caminhao: EstadoModelo) -> None:
        sessao = sessao_atual()
        extra = {k: v for k, v in estado_caminhao.model_dump().items() if k != "sim_time"}
        difusor.publicar({"tipo": "estado", **sessao.estado(estado_caminhao.sim_time, extra)})

    @app.get("/sessao/log")
    async def ver_log(desde: int = 0) -> dict[str, Any]:
        sessao = sessao_atual()
        return {"sessao": sessao.id, "total": len(sessao.entradas), "entradas": sessao.entradas[max(0, desde):]}

    @app.delete("/sessao")
    async def encerrar_sessao() -> dict[str, Any]:
        sessao = sessao_atual()
        resumo = await asyncio.to_thread(sessao.encerrar)
        return {"sessao": sessao.info(), "resumo": resumo}

    @app.websocket("/ws")
    async def websocket(ws: WebSocket) -> None:
        await ws.accept()
        fila = difusor.inscrever()
        try:
            sessao = estado["sessao"]
            historico = sessao.historico() if sessao else {"sessao": None, "entradas": [], "blocos": [],
                                                           "telemetria": None, "previsao": None, "estado": None}
            await ws.send_json({"tipo": "historico", **historico})

            async def enviar() -> None:
                while True:
                    await ws.send_json(await fila.get())

            async def receber() -> None:  # só para perceber quando o cliente fecha
                while True:
                    await ws.receive_text()

            tarefas = {asyncio.create_task(enviar()), asyncio.create_task(receber())}
            _, pendentes = await asyncio.wait(tarefas, return_when=asyncio.FIRST_COMPLETED)
            for tarefa in pendentes:
                tarefa.cancel()
            for tarefa in tarefas:  # recolhe também a que terminou (senão o asyncio loga "never retrieved")
                try:
                    await tarefa
                except (asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
                    pass  # a página foi fechada ou recarregada: o normal
                except Exception:
                    log.exception("Erro no WebSocket /ws")
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            difusor.cancelar(fila)

    return app


__all__ = ["criar_app", "Path"]
