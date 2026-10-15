"""Aplicação FastAPI do gêmeo digital (uma sessão por vez) e da interface de 4 painéis.

Rotas da corrida autônoma (quem chama é o piloto, mover.autonomo.dirigir):
    GET    /saude                 -> {"ok": true, "sessao": {...} | null}
    POST   /sessao                -> abre a sessão (corpo: modo, voltas, simulador, provedor, modelo, linha_de_base...)
    GET    /sessao                -> dados da sessão atual
    POST   /sessao/telemetria     -> lote de linhas de telemetria medida ({"linhas": [...]}); vira mensagem "telemetria"
    POST   /sessao/decisoes       -> uma decisão da camada agêntica (entrada "intervencao"); vira "entrada" e "previsao"
    GET    /sessao/log?desde=N    -> entradas publicadas a partir da N-ésima
    DELETE /sessao                -> encerra a sessão (corpo opcional: {"resumo": {...}} do piloto) e devolve o resumo
    GET    /benchmark/ultimo      -> resumo do benchmark mais recente (data/benchmark/<id>/resumo.json)
    WS     /ws                    -> histórico ao conectar e, depois, as mensagens ao vivo (ver sessao.py)

Rotas da camada agêntica (o caminhão manda o plano e recebe as alterações por HTTP; ver
mover/agentes/servico.py e mover/autonomo/canal_agentes.py):
    POST   /agentes/sessao        -> abre a sessão dos agentes (voltas, LLM, ML); monta o Decisor
    POST   /agentes/plano         -> plano dos próximos 10 s + estado do governador; 202, ou as mensagens com "esperar"
    GET    /agentes/decisoes?desde=N -> mensagens das decisões (regra, final, vazia, erro) a partir da N-ésima
    DELETE /agentes/sessao        -> encerra a sessão dos agentes

Rotas da interface (página, câmera do CARLA, cena 2D e chat): ver mover/interface/rotas.py.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from fastapi import Body, FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from mover.agentes.servico import ServicoAgentes, criar_llm_decisoes
from mover.agentes.supervisor import Supervisor
from mover.config import caminho, carregar_yaml
from mover.interface.rotas import instalar_interface
from mover.servidor.sessao import Difusor, OpcoesSessao, SessaoAutonoma, SessaoEncerrada, para_json

log = logging.getLogger("mover.servidor")


class OpcoesSessaoModelo(BaseModel):
    modo: str = Field("agentes", pattern="^(agentes|bruto)$")
    voltas: int = Field(1, ge=1, le=1000)
    simulador: str = "substituto"
    provedor: str | None = None
    modelo: str | None = None
    llm_decisoes: str | None = None
    fator_tempo: float = 1.0
    tempo_real: bool = True
    duracao_bloco_s: float = 10.0
    comprimento_volta_m: float = 0.0
    duracao_prevista_s: float = 0.0
    linha_de_base: dict[str, Any] | None = None
    combustivel: str = "diesel"
    preco_r_l: float = 6.0
    co2_kg_l: float = 2.68
    logs_medidos: bool = True


class LoteModelo(BaseModel):
    linhas: list[dict[str, Any]]


class DecisaoModelo(BaseModel):
    """Entrada da decisão (Decisao.para_entrada()); os campos extras são guardados como vieram."""

    model_config = ConfigDict(extra="allow")
    tipo: str = "intervencao"
    texto: str
    opcao: str


class SessaoAgentesModelo(BaseModel):
    voltas: int = Field(1, ge=1, le=1000)
    usar_llm: bool = True
    usar_ml: bool = True
    provedor: str | None = None
    modelo: str | None = None
    comprimento_volta_m: float | None = Field(None, ge=0.0)  # confere se os dois lados têm o mesmo plano de missão


class PlanoAgentesModelo(BaseModel):
    plano: dict[str, Any]          # PlanoFuturo.para_json()
    governador: dict[str, Any]     # Governador.exportar_estado()
    esperar: bool = False


def ultimo_benchmark(pasta: Path) -> dict[str, Any] | None:
    resumos = sorted(pasta.glob("*/resumo.json"), key=lambda p: p.stat().st_mtime) if pasta.exists() else []
    if not resumos:
        return None
    with resumos[-1].open(encoding="utf-8") as f:
        dados = json.load(f)
    dados["pasta"] = str(resumos[-1].parent)
    return dados


def criar_app(cfg_agentes: dict[str, Any] | None = None, cfg_simulacao: dict[str, Any] | None = None,
              fabrica_supervisor: Callable[[OpcoesSessao], Supervisor] | None = None,
              fabrica_llm_chat: Callable[[Any], Any] | None = None, cfg_autonomo: dict[str, Any] | None = None,
              montagem: Any = None) -> FastAPI:
    """Monta o app. `fabrica_supervisor` e `fabrica_llm_chat` existem para os testes.

    `montagem` (mover.autonomo.montagem.Montagem) dá o plano de missão e os parâmetros dos agentes;
    sem ela, o servidor monta a partir do config/autonomo.yaml na primeira sessão dos agentes.
    """
    cfg_agentes = cfg_agentes or carregar_yaml("config/agentes.yaml")
    cfg_simulacao = cfg_simulacao if cfg_simulacao is not None else (
        carregar_yaml("config/simulacao.yaml") if caminho("config/simulacao.yaml").exists() else {})
    cfg_autonomo = cfg_autonomo if cfg_autonomo is not None else (
        carregar_yaml("config/autonomo.yaml") if caminho("config/autonomo.yaml").exists() else {})
    pasta_sessoes = caminho(cfg_simulacao.get("servidor", {}).get("pasta_sessoes", "data/autonomo/sessoes"))
    pasta_benchmark = caminho((cfg_autonomo.get("benchmark") or {}).get("pasta", "data/benchmark"))
    difusor = Difusor()
    estado: dict[str, SessaoAutonoma | None] = {"sessao": None}
    agentes: dict[str, Any] = {"servico": None, "montagem": montagem}
    trava_sessao = asyncio.Lock()
    trava_agentes = asyncio.Lock()

    @asynccontextmanager
    async def ciclo_de_vida(app: FastAPI):
        difusor.ligar(asyncio.get_running_loop())
        yield
        if agentes["servico"] is not None:
            agentes["servico"].encerrar()
            agentes["servico"] = None
        sessao = estado["sessao"]
        if sessao is not None and not sessao.encerrada:
            await asyncio.to_thread(sessao.encerrar, None, 5.0)

    app = FastAPI(title="MOVER - gêmeo digital do caminhão autônomo", version="2.0", lifespan=ciclo_de_vida)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.state.difusor = difusor
    app.state.estado = estado
    instalar_interface(app, estado, cfg_agentes, cfg_simulacao, fabrica_llm_chat, cfg_autonomo)

    def sessao_atual() -> SessaoAutonoma:
        sessao = estado["sessao"]
        if sessao is None:
            raise HTTPException(404, "nenhuma sessão; crie uma com POST /sessao")
        return sessao

    @app.get("/saude")
    async def saude() -> dict[str, Any]:
        sessao = estado["sessao"]
        servico = agentes["servico"]
        return {"ok": True, "sessao": sessao.info() if sessao else None, "clientes_ws": difusor.n_clientes,
                "agentes": servico.info() if servico else None}

    @app.post("/sessao")
    async def criar_sessao(opcoes: OpcoesSessaoModelo) -> dict[str, Any]:
        async with trava_sessao:
            anterior = estado["sessao"]
            if anterior is not None and not anterior.encerrada:
                log.info("Encerrando a sessão %s para abrir outra.", anterior.id)
                await asyncio.to_thread(anterior.encerrar, None, 5.0)
            op = OpcoesSessao(**opcoes.model_dump())

            def montar() -> SessaoAutonoma:
                supervisor = fabrica_supervisor(op) if fabrica_supervisor else None
                return SessaoAutonoma(cfg_agentes, op, difusor, pasta_sessoes, supervisor=supervisor)

            sessao = await asyncio.to_thread(montar)
            estado["sessao"] = sessao
            log.info("Sessão %s: %s, %d volta(s), simulador %s, LLM das decisões %s, dos logs %s", sessao.id, op.modo,
                     op.voltas, op.simulador, op.llm_decisoes or "-", sessao.llm_logs)
            difusor.publicar({"tipo": "sessao", "evento": "iniciada", "sessao": sessao.info(),
                              "linha_de_base": op.linha_de_base})
            return sessao.info()

    @app.get("/sessao")
    async def ver_sessao() -> dict[str, Any]:
        return sessao_atual().info()

    @app.post("/sessao/telemetria", status_code=204)
    async def receber_telemetria(lote: LoteModelo) -> None:
        sessao = sessao_atual()
        try:
            sessao.receber_telemetria(lote.linhas)
        except SessaoEncerrada as erro:
            raise HTTPException(409, {"mensagem": str(erro)}) from None

    @app.post("/sessao/decisoes")
    async def receber_decisao(decisao: DecisaoModelo) -> dict[str, Any]:
        sessao = sessao_atual()
        try:
            entrada = sessao.receber_decisao(decisao.model_dump())
        except SessaoEncerrada as erro:
            raise HTTPException(409, {"mensagem": str(erro)}) from None
        return {"id": entrada["id"]}

    @app.get("/sessao/log")
    async def ver_log(desde: int = 0) -> dict[str, Any]:
        sessao = sessao_atual()
        return {"sessao": sessao.id, "total": len(sessao.entradas), "entradas": sessao.entradas[max(0, desde):]}

    @app.delete("/sessao")
    async def encerrar_sessao(corpo: dict[str, Any] | None = Body(None)) -> dict[str, Any]:
        sessao = sessao_atual()
        resumo = await asyncio.to_thread(sessao.encerrar, (corpo or {}).get("resumo"))
        return {"sessao": sessao.info(), "resumo": resumo}

    @app.get("/benchmark/ultimo")
    async def benchmark() -> dict[str, Any]:
        dados = await asyncio.to_thread(ultimo_benchmark, pasta_benchmark)
        if dados is None:
            raise HTTPException(404, {"mensagem": "nenhum benchmark ainda; rode python -m mover.autonomo.benchmark"})
        return para_json(dados)

    # ------------------------------------------------------------------ camada agêntica
    def servico_atual() -> ServicoAgentes:
        servico = agentes["servico"]
        if servico is None:
            raise HTTPException(404, {"mensagem": "nenhuma sessão dos agentes; abra com POST /agentes/sessao"})
        return servico

    def montar_servico(opcoes: SessaoAgentesModelo) -> tuple[ServicoAgentes, Any]:
        if agentes["montagem"] is None:
            from mover.autonomo.montagem import montar

            agentes["montagem"] = montar(cfg_autonomo or None)
        m = agentes["montagem"]
        if opcoes.comprimento_volta_m and abs(opcoes.comprimento_volta_m - m.plano.comprimento) > 1.0:
            raise ValueError(f"plano de missão diferente: o caminhão tem {opcoes.comprimento_volta_m:.0f} m por volta e "
                             f"o servidor dos agentes, {m.plano.comprimento:.0f} m")
        suavizador = m.suavizador() if opcoes.usar_ml else None  # treina o modelo de ML na primeira vez (~1-2 min)
        llm = (criar_llm_decisoes(cfg_agentes, opcoes.provedor, opcoes.modelo, float(m.cfg_decisao.get("prazo_llm_s", 6.0)))
               if opcoes.usar_llm else None)

        def governador(estado_gov: dict[str, Any]):
            return m.governador(str(estado_gov.get("modo", "agentes")), estado_gov.get("n_voltas")).restaurar_estado(estado_gov)

        return ServicoAgentes(m.decisor(opcoes.voltas, llm, suavizador), llm is not None, governador), m

    @app.post("/agentes/sessao")
    async def abrir_agentes(opcoes: SessaoAgentesModelo) -> dict[str, Any]:
        async with trava_agentes:
            if agentes["servico"] is not None:
                agentes["servico"].encerrar()
                agentes["servico"] = None
            try:
                servico, m = await asyncio.to_thread(montar_servico, opcoes)
            except ValueError as erro:
                raise HTTPException(409, {"mensagem": str(erro)}) from None
            except SystemExit as erro:  # sem o plano de missão no servidor
                raise HTTPException(503, {"mensagem": str(erro)}) from None
            agentes["servico"] = servico
            info = servico.info()
            log.info("Agentes: %d volta(s), LLM %s, ML %s.", opcoes.voltas, info["llm"] or "-", info["ml"] or "-")
            return {**info, "voltas": opcoes.voltas, "comprimento_volta_m": round(m.plano.comprimento, 1)}

    @app.post("/agentes/plano")
    async def receber_plano(pedido: PlanoAgentesModelo, resposta: Response) -> dict[str, Any]:
        servico = servico_atual()
        try:
            plano, governador = servico.ler(pedido.model_dump())
        except (KeyError, TypeError, ValueError, IndexError) as erro:
            raise HTTPException(422, {"mensagem": f"plano inválido ({type(erro).__name__}: {erro})"}) from None
        if pedido.esperar:
            return {"mensagens": para_json(await asyncio.to_thread(servico.processar_agora, plano, governador))}
        servico.enviar(plano, governador)
        resposta.status_code = 202
        return {"aceito": True, "bloco": plano.bloco}

    @app.get("/agentes/decisoes")
    async def ver_decisoes(desde: int = 0) -> dict[str, Any]:
        mensagens, total, ocupado = servico_atual().mensagens_desde(desde)
        return {"mensagens": para_json(mensagens), "total": total, "ocupado": ocupado}

    @app.delete("/agentes/sessao")
    async def encerrar_agentes() -> dict[str, Any]:
        async with trava_agentes:
            servico = servico_atual()
            info = servico.info()
            servico.encerrar()
            agentes["servico"] = None
            return info

    @app.websocket("/ws")
    async def websocket(ws: WebSocket) -> None:
        await ws.accept()
        fila = difusor.inscrever()
        try:
            sessao = estado["sessao"]
            historico = sessao.historico() if sessao else {"sessao": None, "entradas": [], "serie": None, "previsao": None,
                                                           "estado": None, "metricas": None, "linha_de_base": None}
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


__all__ = ["criar_app", "ultimo_benchmark"]
