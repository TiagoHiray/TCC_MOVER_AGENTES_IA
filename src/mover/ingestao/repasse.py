"""Repasse das mensagens recebidas para outra máquina (ex.: Notebook de campo -> VM do CARLA).

O envio roda em segundo plano: a resposta ao celular não espera a VM. Se a VM ficar fora do ar,
a fila guarda as mensagens mais novas e descarta as mais antigas.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from mover.simulacao.fila_envio import FilaEnvio

log = logging.getLogger("mover.ingestao")


class Repassador:
    def __init__(self, url: str, token: str | None = None, tamanho_fila: int = 120, timeout_s: float = 5.0,
                 cliente: Any = None):
        import httpx

        if not url.startswith(("http://", "https://")):
            raise ValueError("a URL de repasse deve começar com http:// ou https://")
        self.url = url
        self._cliente = cliente or httpx.Client(timeout=timeout_s, params={"token": token} if token else None)
        self._trava = threading.Lock()
        self.enviadas = 0
        self.falhas = 0
        self.ultimo_erro: str | None = None
        self._proximo_aviso = 0.0
        self._fila = FilaEnvio(self._enviar, "repasse_ingestao", tamanho_fila)

    def receber(self, msg: dict[str, Any]) -> None:
        self._fila.colocar(msg)

    def _enviar(self, msg: dict[str, Any]) -> None:
        import httpx

        try:
            r = self._cliente.post(self.url, json=msg)
            erro = None if r.status_code == 200 else f"HTTP {r.status_code}"
        except httpx.HTTPError as e:
            erro = type(e).__name__
        with self._trava:
            if erro is None:
                self.enviadas += 1
                return
            self.falhas += 1
            self.ultimo_erro = erro
        agora = time.monotonic()
        if agora >= self._proximo_aviso:
            self._proximo_aviso = agora + 10.0
            log.warning("Repasse para %s falhou (%s); %d falhas até agora.", self.url, erro, self.falhas)

    def status(self) -> dict[str, Any]:
        with self._trava:
            return {"url": self.url, "enviadas": self.enviadas, "falhas": self.falhas,
                    "descartadas": self._fila.descartados, "ultimo_erro": self.ultimo_erro}

    def fechar(self) -> None:
        self._fila.fechar()
        self._cliente.close()
