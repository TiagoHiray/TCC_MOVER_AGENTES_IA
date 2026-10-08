"""Liberação antecipada dos blocos de 10 s.

Quando o caminhão entra no bloco k, o resultado do bloco k (analisado durante o bloco k-1)
é publicado e o bloco k+1 é liberado para a camada agêntica numa thread separada. Assim a
análise de cada bloco tem a duração de um bloco inteiro para ficar pronta. O bloco 0 é
analisado antes da partida (`preparar`).

Uma única thread de trabalho garante que os blocos sejam processados em ordem, porque a
camada agêntica guarda memória entre blocos (mesclagem de trechos estáveis).

O mesmo executor é usado pelo rodar_agentes.py (com ou sem tempo real) e pelo replay no CARLA.
O servidor também passa `ao_analisar`, chamado assim que a análise de um bloco termina (antes de
o caminhão entrar nele): é assim que a interface mostra o próximo bloco como previsão.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger("mover.agentes")


@dataclass
class ResultadoBloco:
    bloco: int
    entradas: list[dict[str, Any]]
    latencia_s: float                 # tempo de processamento do bloco
    espera_s: float = 0.0             # quanto o caminhão esperou pela análise ao entrar no bloco
    liberado_em: float = field(default=0.0, repr=False)


class ExecutorBlocos:
    def __init__(self, camada: Any, ao_publicar: Callable[[ResultadoBloco], None] | None = None,
                 tolerancia_espera_s: float = 0.05, avisar_atrasos: bool = True, fator_tempo: float = 1.0,
                 ao_analisar: Callable[[ResultadoBloco], None] | None = None):
        self.camada = camada
        self.n_blocos: int = camada.n_blocos
        self.duracao_bloco_s: float = camada.duracao_bloco
        self.ao_publicar = ao_publicar
        self.ao_analisar = ao_analisar        # chamado na thread de trabalho, ao fim de cada análise
        self.tolerancia_espera_s = tolerancia_espera_s
        self.avisar_atrasos = avisar_atrasos  # desligado no modo rápido, onde esperar é normal
        self.fator_tempo = fator_tempo        # 2 = simulação 2x mais rápida que o relógio
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="camada_agentica")
        self._futuros: dict[int, Future] = {}
        self.resultados: list[ResultadoBloco] = []

    # ------------------------------------------------------------------ controle
    def _liberar(self, k: int) -> None:
        if 0 <= k < self.n_blocos and k not in self._futuros:
            self._futuros[k] = self._pool.submit(self._processar, k, time.monotonic())

    def _processar(self, k: int, liberado_em: float) -> ResultadoBloco:
        inicio = time.monotonic()
        entradas = self.camada.processar_bloco(k)
        resultado = ResultadoBloco(k, entradas, time.monotonic() - inicio, liberado_em=liberado_em)
        # Avisa ainda dentro da tarefa, antes de o futuro terminar: quem espera o bloco (preparar,
        # entrar_no_bloco) já encontra a previsão registrada. Com add_done_callback o aviso podia
        # chegar depois de o result() voltar.
        if self.ao_analisar is not None:
            try:
                self.ao_analisar(resultado)
            except Exception:
                log.exception("Falha ao avisar o fim da análise do bloco %d.", k)
        return resultado

    def preparar(self) -> None:
        """Libera o bloco 0 e espera a análise dele (o caminhão só parte com o 1º bloco pronto)."""
        self._liberar(0)
        self._futuros[0].result()

    def entrar_no_bloco(self, k: int) -> ResultadoBloco:
        """Chamado quando o caminhão entra no bloco k: publica o bloco k e libera o k+1."""
        self._liberar(k)
        inicio_espera = time.monotonic()
        resultado: ResultadoBloco = self._futuros.pop(k).result()
        resultado.espera_s = time.monotonic() - inicio_espera
        self._liberar(k + 1)
        if self.avisar_atrasos and k > 0 and resultado.espera_s > self.tolerancia_espera_s:
            log.warning("Bloco %d: a análise atrasou %.2f s (processamento de %.2f s; o bloco dura %.1f s de relógio).",
                        k, resultado.espera_s, resultado.latencia_s, self.duracao_bloco_s / self.fator_tempo)
        self.resultados.append(resultado)
        if self.ao_publicar is not None:
            self.ao_publicar(resultado)
        return resultado

    def encerrar(self) -> None:
        for futuro in self._futuros.values():
            futuro.cancel()
        self._pool.shutdown(wait=True)

    # ------------------------------------------------------------------ execuções prontas
    def rodar(self, tempo_real: bool = False, fator_tempo: float = 1.0) -> list[ResultadoBloco]:
        """Percorre todos os blocos.

        Sem tempo real, o "caminhão" entra no bloco seguinte assim que o atual é publicado
        (útil para gerar o log rapidamente). Em tempo real, cada bloco dura duracao_bloco_s /
        fator_tempo segundos de relógio, como no replay.
        """
        self.avisar_atrasos = tempo_real
        self.fator_tempo = fator_tempo
        try:
            self.preparar()
            for k in range(self.n_blocos):
                partida = time.monotonic()
                self.entrar_no_bloco(k)
                if tempo_real:
                    restante = self.duracao_bloco_s / fator_tempo - (time.monotonic() - partida)
                    if restante > 0:
                        time.sleep(restante)
        finally:
            self.encerrar()
        return self.resultados
