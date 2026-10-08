"""Rotas da interface (Etapa 4): página dos 4 painéis, câmera do CARLA, cena 2D e chat sobre o log.

    GET  /                  -> página (interface/estatico/index.html)
    GET  /estatico/...      -> JS e CSS da página (sem build, sem CDN)
    PUT  /cena              -> o replay manda as vias do mapa e o trajeto alinhado (referencial do mapa)
    GET  /cena              -> a cena; sem PUT, é calculada uma vez a partir do config/simulacao.yaml
    POST /camera            -> o replay manda um quadro JPEG da câmera de perseguição (corpo image/jpeg)
    GET  /camera/info       -> quadros recebidos e idade do último (a página escolhe câmera ou mapa 2D)
    GET  /camera.jpg        -> último quadro
    GET  /camera.mjpg       -> transmissão MJPEG (multipart/x-mixed-replace), usada num <img>
    POST /chat              -> pergunta sobre o log; resposta curta citando as entradas (#id, tempo)
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import time
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import APIRouter, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from mover.interface.cena import cena_do_yaml
from mover.interface.chat import ServicoChat

log = logging.getLogger("mover.interface")

PASTA_ESTATICA = Path(__file__).resolve().parent / "estatico"
FRONTEIRA_MJPEG = "quadro"
LIMITE_QUADRO_BYTES = 4 * 1024 * 1024
CAMERA_ATIVA_ATE_S = 2.0      # sem quadro novo por mais que isso, a página volta ao mapa 2D
MJPEG_OCIOSO_MAX_S = 10.0     # a transmissão termina depois disso sem quadros (a página reabre quando voltarem)

# No Windows, o registro às vezes diz que .js é text/plain, e o navegador recusa módulos JS assim.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")


class EstaticosSemCache(StaticFiles):
    """Arquivos estáticos que o navegador sempre revalida (uma versão nova do JS aparece sem limpar o cache)."""

    async def get_response(self, path: str, scope: Any) -> Response:
        resposta = await super().get_response(path, scope)
        resposta.headers["Cache-Control"] = "no-cache"
        return resposta


class QuadroCamera:
    """Último quadro JPEG recebido. A tupla é trocada de uma vez, então ler e escrever é seguro."""

    def __init__(self) -> None:
        self.ultimo: tuple[int, bytes, float] = (0, b"", 0.0)  # (número, jpeg, time.monotonic())

    def guardar(self, jpeg: bytes) -> None:
        self.ultimo = (self.ultimo[0] + 1, jpeg, time.monotonic())

    def info(self) -> dict[str, Any]:
        numero, _, recebido_em = self.ultimo
        idade = round(time.monotonic() - recebido_em, 2) if numero else None
        return {"quadros": numero, "idade_s": idade, "ativa": idade is not None and idade < CAMERA_ATIVA_ATE_S}


class ViaModelo(BaseModel):
    via: str = ""
    largura_m: float = 3.2
    pontos: list[tuple[float, float]]


class TrajetoModelo(BaseModel):
    t: list[float]
    x: list[float]
    y: list[float]
    rumo_graus: list[float]

    @model_validator(mode="after")
    def mesmo_tamanho(self) -> "TrajetoModelo":
        if not len(self.t) == len(self.x) == len(self.y) == len(self.rumo_graus):
            raise ValueError("t, x, y e rumo_graus precisam ter o mesmo tamanho")
        return self


class CenaModelo(BaseModel):
    referencial: str = "mapa OpenDRIVE (x = leste, y = norte, metros)"
    origem: str = "replay"
    vias: list[ViaModelo]
    trajeto: TrajetoModelo
    limites: tuple[float, float, float, float]


class TurnoModelo(BaseModel):
    papel: Literal["operador", "assistente"]
    texto: str = Field(max_length=2000)


class PerguntaModelo(BaseModel):
    pergunta: str = Field(min_length=1, max_length=500)
    historico: list[TurnoModelo] = Field(default_factory=list)


def instalar_interface(app: FastAPI, estado: dict[str, Any], cfg_agentes: dict[str, Any], cfg_simulacao: dict[str, Any],
                       fabrica_llm_chat: Callable[[Any], Any] | None = None) -> None:
    """Pendura no app as rotas da interface e os arquivos estáticos."""
    camera = QuadroCamera()
    chat = ServicoChat(cfg_agentes, fabrica_llm_chat)
    cena: dict[str, Any] = {"dados": None, "erro": None, "tentou_yaml": False}
    trava_cena = asyncio.Lock()
    rotas = APIRouter()
    app.state.camera, app.state.chat, app.state.cena = camera, chat, cena

    @rotas.get("/", include_in_schema=False)
    async def pagina() -> FileResponse:
        return FileResponse(PASTA_ESTATICA / "index.html", headers={"Cache-Control": "no-cache"})

    # ------------------------------------------------------------------ cena 2D
    @rotas.put("/cena", status_code=204)
    async def receber_cena(dados: CenaModelo) -> None:
        cena["dados"], cena["erro"] = dados.model_dump(), None
        log.info("Cena recebida (%s): %d vias, %d poses.", dados.origem, len(dados.vias), len(dados.trajeto.t))

    @rotas.get("/cena")
    async def ver_cena() -> dict[str, Any]:
        async with trava_cena:
            if cena["dados"] is None and not cena["tentou_yaml"]:
                cena["tentou_yaml"] = True
                try:
                    cena["dados"] = await asyncio.to_thread(cena_do_yaml, cfg_simulacao)
                except (SystemExit, Exception) as erro:  # alinhar_volta sai com SystemExit se faltar o mapa
                    cena["erro"] = f"cena indisponível até o replay mandá-la ({type(erro).__name__}: {erro})"[:300]
                    log.info("%s", cena["erro"])
        if cena["dados"] is None:
            raise HTTPException(404, {"mensagem": cena["erro"] or "cena indisponível"})
        return cena["dados"]

    # ------------------------------------------------------------------ câmera do CARLA
    @rotas.post("/camera", status_code=204)
    async def receber_quadro(request: Request) -> None:
        if not request.headers.get("content-type", "").startswith("image/jpeg"):
            raise HTTPException(415, "envie o quadro como image/jpeg")
        corpo = await request.body()
        if len(corpo) > LIMITE_QUADRO_BYTES:
            raise HTTPException(413, "quadro grande demais")
        if not corpo.startswith(b"\xff\xd8"):
            raise HTTPException(400, "o corpo não é um JPEG")
        camera.guardar(corpo)

    @rotas.get("/camera/info")
    async def info_camera() -> dict[str, Any]:
        return camera.info()

    @rotas.get("/camera.jpg")
    async def quadro_atual() -> Response:
        numero, jpeg, _ = camera.ultimo
        if not numero:
            raise HTTPException(404, "nenhum quadro da câmera ainda")
        return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @rotas.get("/camera.mjpg")
    async def transmissao(max_quadros: int | None = None) -> StreamingResponse:
        """`max_quadros` encerra a transmissão depois de N quadros (testes e depuração)."""

        async def quadros():
            ultimo_enviado, enviados, parado_desde = -1, 0, time.monotonic()
            while True:
                numero, jpeg, _ = camera.ultimo
                if numero and numero != ultimo_enviado:
                    ultimo_enviado, parado_desde = numero, time.monotonic()
                    yield (f"--{FRONTEIRA_MJPEG}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n"
                           .encode("ascii") + jpeg + b"\r\n")
                    enviados += 1
                    if max_quadros and enviados >= max_quadros:
                        return
                elif time.monotonic() - parado_desde > MJPEG_OCIOSO_MAX_S:
                    return
                else:
                    await asyncio.sleep(0.04)  # até 25 verificações por segundo; a câmera manda ~10 fps

        return StreamingResponse(quadros(), media_type=f"multipart/x-mixed-replace; boundary={FRONTEIRA_MJPEG}",
                                 headers={"Cache-Control": "no-store", "Pragma": "no-cache"})

    # ------------------------------------------------------------------ chat
    @rotas.post("/chat")
    async def perguntar(pedido: PerguntaModelo) -> dict[str, Any]:
        historico = [t.model_dump() for t in pedido.historico]
        return await asyncio.to_thread(chat.responder, estado["sessao"], pedido.pergunta.strip(), historico)

    app.include_router(rotas)
    app.mount("/estatico", EstaticosSemCache(directory=PASTA_ESTATICA), name="estatico")
