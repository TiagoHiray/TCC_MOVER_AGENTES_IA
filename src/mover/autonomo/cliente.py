"""Cliente do servidor do painel, usado pelo piloto (dirigir.py).

A telemetria e as decisões vão por uma thread própria, numa fila (FilaEnvio): o laço de 20 Hz
nunca espera o servidor. A fila é longa (dezenas de segundos de lotes) e mantém a ordem; ela só
descarta lotes se o servidor ficar fora do ar. Os quadros da câmera do CARLA vão por outra fila,
curta (vale o quadro mais novo).
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

from mover.simulacao.fila_envio import FilaEnvio

log = logging.getLogger("mover.autonomo")


class ErroServidor(RuntimeError):
    pass


def json_seguro(valor: Any) -> Any:
    """NaN e infinito viram null: o JSON não os aceita e o lote inteiro seria recusado."""
    if isinstance(valor, float):
        return valor if math.isfinite(valor) else None
    if isinstance(valor, dict):
        return {k: json_seguro(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [json_seguro(v) for v in valor]
    return valor


def servidor_no_ar(url: str, timeout_s: float = 1.0) -> bool:
    import httpx

    try:
        return httpx.get(f"{url}/saude", timeout=timeout_s).status_code == 200
    except httpx.HTTPError:
        return False


class ClientePainel:
    """Fala com o servidor do painel (httpx.Client, ou o TestClient do FastAPI nos testes).

    Com `em_segundo_plano=False` (testes), cada envio é feito na hora.
    """

    def __init__(self, http: Any, http_envio: Any = None, http_camera: Any = None, em_segundo_plano: bool = True,
                 proprio: bool = False):
        self.http = http
        self._http_envio = http_envio or http
        self._http_camera = http_camera or http
        self._proprio = proprio
        self.lotes_enviados = 0
        self.decisoes_enviadas = 0
        self._falhas_seguidas = 0
        self._fila: FilaEnvio | None = FilaEnvio(self._enviar, "envio_painel", tamanho=200) if em_segundo_plano else None

    @classmethod
    def conectar(cls, url: str, prazo_s: float = 10.0) -> "ClientePainel":
        import httpx

        http = httpx.Client(base_url=url, timeout=httpx.Timeout(10.0, read=120.0))
        limite = time.monotonic() + prazo_s
        while True:
            try:
                http.get("/saude").raise_for_status()
                break
            except httpx.HTTPError as erro:
                if time.monotonic() > limite:
                    http.close()
                    raise ErroServidor(f"o servidor do painel não respondeu em {url} ({erro}).") from None
                time.sleep(0.5)
        log.info("Servidor do painel: %s", url)
        return cls(http, httpx.Client(base_url=url, timeout=httpx.Timeout(5.0)),
                   httpx.Client(base_url=url, timeout=httpx.Timeout(2.0)), proprio=True)

    @staticmethod
    def _conteudo(resposta: Any, acao: str) -> dict[str, Any]:
        if resposta.status_code >= 400:
            try:
                detalhe = resposta.json().get("detail")
            except ValueError:
                detalhe = resposta.text
            if isinstance(detalhe, dict):
                detalhe = detalhe.get("mensagem", detalhe)
            raise ErroServidor(f"{acao}: HTTP {resposta.status_code} - {detalhe}")
        return resposta.json() if resposta.content else {}

    # ------------------------------------------------------------------ sessão
    def enviar_cena(self, cena: dict[str, Any]) -> bool:
        try:
            self._conteudo(self.http.put("/cena", json=cena), "enviar a cena")
            return True
        except Exception as erro:  # a cena é só para a página: a corrida continua
            log.warning("Não foi possível mandar a cena à página (%s).", erro)
            return False

    def abrir_sessao(self, opcoes: dict[str, Any]) -> dict[str, Any]:
        return self._conteudo(self.http.post("/sessao", json=opcoes), "abrir a sessão")

    def encerrar_sessao(self, resumo: dict[str, Any] | None = None) -> dict[str, Any] | None:
        self.esvaziar()
        try:
            return self._conteudo(self.http.request("DELETE", "/sessao", json={"resumo": resumo}), "encerrar a sessão")
        except Exception as erro:
            log.warning("Não foi possível encerrar a sessão: %s", erro)
            return None

    # ------------------------------------------------------------------ corrida
    def enviar_telemetria(self, linhas: list[dict[str, Any]]) -> None:
        self._colocar(("telemetria", linhas))

    def enviar_decisao(self, entrada: dict[str, Any]) -> None:
        self._colocar(("decisao", entrada))

    def _colocar(self, item: tuple[str, Any]) -> None:
        if self._fila is None:
            self._enviar(item)
        else:
            self._fila.colocar(item)

    def _enviar(self, item: tuple[str, Any]) -> None:
        tipo, dados = item
        try:
            if tipo == "telemetria":
                resposta = self._http_envio.post("/sessao/telemetria", json={"linhas": json_seguro(dados)})
                self.lotes_enviados += resposta.status_code < 400
            else:
                resposta = self._http_envio.post("/sessao/decisoes", json=json_seguro(dados))
                self.decisoes_enviadas += resposta.status_code < 400
            if resposta.status_code >= 400:
                raise ErroServidor(f"HTTP {resposta.status_code}: {resposta.text[:200]}")
            self._falhas_seguidas = 0
        except Exception as erro:  # o painel é só para ver: a corrida continua
            self._falhas_seguidas += 1
            if self._falhas_seguidas == 1:
                log.warning("Falha ao mandar %s ao servidor do painel (%s); a corrida continua.", tipo, erro)

    def postar_quadro(self, jpeg: bytes) -> None:
        """Um quadro JPEG da câmera do CARLA (chamado pela thread da câmera)."""
        resposta = self._http_camera.post("/camera", content=jpeg, headers={"Content-Type": "image/jpeg"})
        if resposta.status_code >= 400:
            raise ErroServidor(f"HTTP {resposta.status_code} ao mandar o quadro da câmera")

    @property
    def descartados(self) -> int:
        return self._fila.descartados if self._fila is not None else 0

    def esvaziar(self) -> None:
        """Espera a fila de envio terminar (antes de encerrar a sessão)."""
        if self._fila is not None:
            self._fila.fechar(timeout_s=30.0)
            self._fila = FilaEnvio(self._enviar, "envio_painel", tamanho=200)

    def fechar(self) -> None:
        if self._fila is not None:
            self._fila.fechar()
            self._fila = None
        if self._proprio:
            for http in {id(h): h for h in (self.http, self._http_envio, self._http_camera)}.values():
                http.close()
            self._proprio = False


__all__ = ["ClientePainel", "ErroServidor", "servidor_no_ar"]
