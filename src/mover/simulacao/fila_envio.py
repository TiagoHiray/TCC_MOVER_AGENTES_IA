"""Fila de envio em segundo plano: quem coloca um item nunca espera o destino.

Usada pelo replay para mandar ao servidor as poses do caminhão (~5 Hz) e os quadros da câmera do
painel (~10 fps). O laço de 20 Hz do replay não pode travar se o servidor ficar lento: a fila é
curta e, cheia, descarta o item mais antigo (para um painel ao vivo, o mais novo é o que importa).
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Callable

log = logging.getLogger("mover.simulacao")

_PARAR = object()


class FilaEnvio:
    def __init__(self, destino: Callable[[Any], None], nome: str, tamanho: int = 4):
        self.descartados = 0
        self._destino = destino
        self._fila: queue.Queue = queue.Queue(maxsize=max(1, int(tamanho)))
        self._thread: threading.Thread | None = threading.Thread(target=self._laco, name=nome, daemon=True)
        self._thread.start()

    def colocar(self, item: Any) -> None:
        while True:
            try:
                self._fila.put_nowait(item)
                return
            except queue.Full:  # destino lento: descarta o item mais antigo
                try:
                    self._fila.get_nowait()
                    self.descartados += 1
                except queue.Empty:
                    pass

    def _laco(self) -> None:
        while True:
            item = self._fila.get()
            if item is _PARAR:
                return
            try:
                self._destino(item)
            except Exception:  # o destino trata os próprios erros; isto só evita matar a thread
                log.exception("Falha no envio em segundo plano (%s).", threading.current_thread().name)

    def fechar(self, timeout_s: float = 5.0) -> None:
        """Para a thread depois dos itens que ainda couberem na fila (idempotente)."""
        if self._thread is None:
            return
        while True:  # garante lugar para o sinal de parada
            try:
                self._fila.put_nowait(_PARAR)
                break
            except queue.Full:
                try:
                    self._fila.get_nowait()
                except queue.Empty:
                    pass
        self._thread.join(timeout=timeout_s)
        self._thread = None
