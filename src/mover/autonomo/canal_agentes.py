"""Canal do caminhão com a camada agêntica: manda o plano dos próximos 10 s e recolhe as alterações.

- CanalHttp: o caminho da corrida (dirigir, benchmark). O caminhão fala com o servidor dos agentes
  por HTTP, como um caminhão real ligado a um servidor: POST /agentes/plano e GET /agentes/decisoes.
- CanalLocal: o mesmo serviço no próprio processo, sem rede (testes e uso como biblioteca).

Os dois entregam as mesmas mensagens (mover.agentes.servico); o laço da corrida (conducao.py) não
sabe qual está usando. No modo assíncrono, o caminhão só busca as mensagens enquanto espera uma
decisão, e no máximo a cada `intervalo_s` de relógio.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from mover.agentes.servico import ETAPAS_FINAIS, ServicoAgentes
from mover.autonomo.governador import Governador
from mover.autonomo.planejador import PlanoFuturo

log = logging.getLogger("mover.autonomo")


class ErroAgentes(RuntimeError):
    pass


class CanalAgentes:
    """Base: conta os planos em aberto (cada um termina com uma mensagem final, vazia ou de erro)."""

    def __init__(self) -> None:
        self._abertos = 0

    @property
    def ocupado(self) -> bool:
        return self._abertos > 0

    def pedir(self, plano: PlanoFuturo, governador: Governador, esperar: bool) -> list[dict[str, Any]]:
        """Manda o plano. Com `esperar`, devolve as mensagens; sem, elas chegam em recolher()."""
        if esperar:
            return self._pedir_e_esperar(plano, governador)
        self._enviar(plano, governador)
        self._abertos += 1
        return []

    def recolher(self, forcar: bool = False) -> list[dict[str, Any]]:
        if not self._abertos:
            return []
        mensagens = self._buscar(forcar)
        self._abertos = max(0, self._abertos - sum(m.get("etapa") in ETAPAS_FINAIS for m in mensagens))
        return mensagens

    def encerrar(self) -> None:
        pass

    def _pedir_e_esperar(self, plano: PlanoFuturo, governador: Governador) -> list[dict[str, Any]]:
        raise NotImplementedError

    def _enviar(self, plano: PlanoFuturo, governador: Governador) -> None:
        raise NotImplementedError

    def _buscar(self, forcar: bool) -> list[dict[str, Any]]:
        raise NotImplementedError


class CanalLocal(CanalAgentes):
    def __init__(self, servico: ServicoAgentes, proprio: bool = False):
        super().__init__()
        self.servico = servico
        self._proprio = proprio
        self._lidas = 0

    def _pedir_e_esperar(self, plano: PlanoFuturo, governador: Governador) -> list[dict[str, Any]]:
        return self.servico.processar(plano, governador.clonar())

    def _enviar(self, plano: PlanoFuturo, governador: Governador) -> None:
        self.servico.enviar(plano, governador.clonar())

    def _buscar(self, forcar: bool) -> list[dict[str, Any]]:
        mensagens, self._lidas, _ = self.servico.mensagens_desde(self._lidas)
        return mensagens

    def encerrar(self) -> None:
        if self._proprio:  # o Decisor é de quem o criou: aqui só a fila do serviço fecha
            self.servico.encerrar(com_decisor=False)
            self._proprio = False


class CanalHttp(CanalAgentes):
    """Fala com o servidor dos agentes (httpx.Client, ou o TestClient do FastAPI nos testes)."""

    def __init__(self, http: Any, intervalo_s: float = 0.1, proprio: bool = False):
        super().__init__()
        self.http = http
        self.intervalo_s = float(intervalo_s)
        self._proprio = proprio
        self._lidas = 0
        self._ultima_busca = 0.0
        self.requisicoes = 0

    @classmethod
    def conectar(cls, url: str, prazo_s: float = 10.0, intervalo_s: float = 0.1) -> "CanalHttp":
        import httpx

        # abrir a sessão pode treinar o modelo de ML (~1-2 min) e aquecer o LLM: leitura com prazo longo
        http = httpx.Client(base_url=url, timeout=httpx.Timeout(10.0, read=900.0))
        limite = time.monotonic() + prazo_s
        while True:
            try:
                http.get("/saude").raise_for_status()
                break
            except httpx.HTTPError as erro:
                if time.monotonic() > limite:
                    http.close()
                    raise ErroAgentes(f"o servidor dos agentes não respondeu em {url} ({erro}).") from None
                time.sleep(0.5)
        log.info("Servidor dos agentes: %s", url)
        return cls(http, intervalo_s, proprio=True)

    def _requisitar(self, metodo: str, rota: str, acao: str, **kwargs: Any) -> dict[str, Any]:
        self.requisicoes += 1
        try:
            resposta = self.http.request(metodo, rota, **kwargs)
        except Exception as erro:  # rede: httpx.HTTPError e afins
            raise ErroAgentes(f"{acao}: {type(erro).__name__}: {erro}") from None
        if resposta.status_code >= 400:
            try:
                detalhe = resposta.json().get("detail")
            except ValueError:
                detalhe = resposta.text[:200]
            raise ErroAgentes(f"{acao}: HTTP {resposta.status_code} - {detalhe}")
        return resposta.json() if resposta.content else {}

    def abrir(self, opcoes: dict[str, Any]) -> dict[str, Any]:
        """Abre a sessão dos agentes no servidor (ele monta o Decisor, o LLM e o modelo de ML)."""
        return self._requisitar("POST", "/agentes/sessao", "abrir a sessão dos agentes", json=opcoes)

    def _corpo(self, plano: PlanoFuturo, governador: Governador, esperar: bool) -> dict[str, Any]:
        return {"plano": plano.para_json(), "governador": governador.exportar_estado(), "esperar": esperar}

    def _pedir_e_esperar(self, plano: PlanoFuturo, governador: Governador) -> list[dict[str, Any]]:
        return self._requisitar("POST", "/agentes/plano", "mandar o plano aos agentes",
                                json=self._corpo(plano, governador, True)).get("mensagens", [])

    def _enviar(self, plano: PlanoFuturo, governador: Governador) -> None:
        self._requisitar("POST", "/agentes/plano", "mandar o plano aos agentes", json=self._corpo(plano, governador, False))

    def _buscar(self, forcar: bool) -> list[dict[str, Any]]:
        agora = time.monotonic()
        if not forcar and agora - self._ultima_busca < self.intervalo_s:
            return []
        self._ultima_busca = agora
        dados = self._requisitar("GET", "/agentes/decisoes", "buscar as decisões dos agentes", params={"desde": self._lidas})
        mensagens = dados.get("mensagens", [])
        self._lidas += len(mensagens)
        return mensagens

    def encerrar(self) -> None:
        try:
            self._requisitar("DELETE", "/agentes/sessao", "encerrar a sessão dos agentes")
        except ErroAgentes as erro:
            log.warning("%s", erro)
        if self._proprio:
            self.http.close()
            self._proprio = False


__all__ = ["CanalAgentes", "CanalHttp", "CanalLocal", "ErroAgentes"]
