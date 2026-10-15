"""Serviço da camada agêntica: recebe o plano que o caminhão manda e devolve as alterações.

O caminhão (CARLA ou substituto) e os agentes conversam por HTTP, como um caminhão real ligado a
um servidor (rotas em mover/servidor/app.py):
- POST /agentes/sessao       o caminhão abre a sessão (voltas, LLM, ML); o servidor monta o Decisor;
- POST /agentes/plano        o plano dos próximos 10 s (PlanoFuturo.para_json) e o estado do governador;
- GET  /agentes/decisoes     as alterações, em duas etapas: a da regra (na hora) e a do Supervisor (LLM).

Cada plano gera uma mensagem final ("final", "vazia" ou "erro"); antes dela pode vir a da regra.
Com "esperar", o servidor responde as mensagens na própria requisição (benchmark, largada).
"""

from __future__ import annotations

import concurrent.futures as futures
import logging
import threading
import time
from dataclasses import asdict
from typing import Any, Callable

from mover.agentes.decisao import Decisao, Decisor, RespostaDecisao
from mover.agentes.llm import MARCADOR_RASCUNHO, ClienteLLM, criar_llm
from mover.autonomo.governador import Governador
from mover.autonomo.planejador import PlanoFuturo

log = logging.getLogger("mover.agentes")

ETAPAS_FINAIS = ("final", "vazia", "erro")


def criar_llm_decisoes(cfg_agentes: dict[str, Any], provedor: str | None, modelo: str | None,
                       prazo_s: float) -> ClienteLLM | None:
    """LLM do Supervisor das decisões, já aquecido (a 1ª chamada do Ollama carrega o modelo na memória)."""
    try:
        llm = criar_llm(cfg_agentes.get("llm", {}), provedor, modelo)
    except Exception as erro:  # pacote ausente, chave faltando...
        log.warning("Sem LLM para as decisões (%s); os agentes decidem só pela regra.", erro)
        return None
    if llm.nome == "falso":
        return llm
    inicio = time.monotonic()
    try:
        llm.gerar("Responda só com o JSON do rascunho.",
                  f'{MARCADOR_RASCUNHO} {{"opcao": "suave", "texto": "teste"}}', RespostaDecisao)
        latencia = time.monotonic() - inicio
        log.info("LLM %s pronto (aquecimento em %.1f s).", llm.nome, latencia)
        if latencia > prazo_s:
            log.warning("O LLM levou %.1f s no aquecimento; acima do prazo de %.0f s, as decisões caem na regra.",
                        latencia, prazo_s)
    except Exception as erro:
        log.warning("O LLM %s não respondeu (%s: %s). Ligue o Ollama (scripts/preparar_ollama) ou use --sem-llm; "
                    "por enquanto as decisões usam a regra.", llm.nome, type(erro).__name__, erro)
    return llm


def mensagem_regra(decisao: Decisao, opcoes: dict[str, Any]) -> dict[str, Any]:
    intervencoes = decisao.intervencoes_para(decisao.recomendada, opcoes, "regra")
    return {"etapa": "regra", "bloco": decisao.bloco,
            "intervencoes": {chave: asdict(interv) for chave, interv in intervencoes.items()}}


def mensagem_final(decisao: Decisao, opcoes: dict[str, Any]) -> dict[str, Any]:
    trocas = {}
    if decisao.escolhida != decisao.recomendada:  # o Supervisor trocou a alteração da regra
        trocas = {chave: asdict(interv) for chave, interv in
                  decisao.intervencoes_para(decisao.escolhida, opcoes, decisao.origem).items()}
    return {"etapa": "final", "bloco": decisao.bloco, "escolhida": decisao.escolhida, "recomendada": decisao.recomendada,
            "origem": decisao.origem, "latencia_s": decisao.latencia_s, "texto": decisao.texto, "trocas": trocas,
            "chaves": [i["chave"] for i in decisao.instancias], "entrada": decisao.para_entrada()}


class ServicoAgentes:
    """O Decisor atrás de uma fila: um plano por vez, mensagens numeradas para o caminhão buscar."""

    def __init__(self, decisor: Decisor, usar_llm: bool = True,
                 governador: Callable[[dict[str, Any]], Governador] | None = None):
        self.decisor = decisor
        self.usar_llm = bool(usar_llm) and decisor.llm is not None
        self._governador = governador
        self.mensagens: list[dict[str, Any]] = []
        self.planos_recebidos = 0
        self._em_andamento = 0
        self._trava = threading.Lock()
        self._executor = futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="agentes")

    def processar(self, plano: PlanoFuturo, governador: Governador,
                  publicar: Callable[[dict[str, Any]], None] | None = None) -> list[dict[str, Any]]:
        """Regra (publicada assim que sai) e depois o Supervisor. O governador recebido não é alterado."""
        saida: list[dict[str, Any]] = []

        def emitir(mensagem: dict[str, Any]) -> None:
            saida.append(mensagem)
            if publicar is not None:
                publicar(mensagem)

        decisao = self.decisor.avaliar(plano, governador)
        if decisao is None:
            emitir({"etapa": "vazia", "bloco": plano.bloco if plano is not None else None})
            return saida
        emitir(mensagem_regra(decisao, self.decisor.opcoes))
        if self.usar_llm:
            self.decisor.supervisionar(decisao)
        emitir(mensagem_final(decisao, self.decisor.opcoes))
        return saida

    def enviar(self, plano: PlanoFuturo, governador: Governador) -> None:
        """Assíncrono: o plano entra na fila e as mensagens saem em mensagens_desde()."""
        with self._trava:
            self._em_andamento += 1
            self.planos_recebidos += 1
        self._executor.submit(self._trabalho, plano, governador)

    def _trabalho(self, plano: PlanoFuturo, governador: Governador) -> None:
        try:
            self.processar(plano, governador, self._publicar)
        except Exception as erro:  # o caminhão segue sem esta decisão
            log.exception("Erro na camada agêntica no bloco %s", plano.bloco)
            self._publicar({"etapa": "erro", "bloco": plano.bloco, "mensagem": f"{type(erro).__name__}: {erro}"})
        finally:
            with self._trava:
                self._em_andamento -= 1

    def _publicar(self, mensagem: dict[str, Any]) -> None:
        with self._trava:
            self.mensagens.append(mensagem)

    def mensagens_desde(self, desde: int) -> tuple[list[dict[str, Any]], int, bool]:
        with self._trava:
            return list(self.mensagens[max(0, desde):]), len(self.mensagens), self._em_andamento > 0

    def ler(self, dados: dict[str, Any]) -> tuple[PlanoFuturo, Governador]:
        """O pedido em JSON (corpo do POST /agentes/plano): o plano do caminhão e o estado do governador."""
        if self._governador is None:
            raise RuntimeError("serviço sem fábrica de governador: não recebe planos em JSON")
        return PlanoFuturo.de_json(dados["plano"]), self._governador(dados["governador"])

    def processar_agora(self, plano: PlanoFuturo, governador: Governador) -> list[dict[str, Any]]:
        """Síncrono (o caminhão espera a resposta): as mensagens não entram na fila de mensagens_desde()."""
        with self._trava:
            self.planos_recebidos += 1
        return self.processar(plano, governador)

    def info(self) -> dict[str, Any]:
        with self._trava:
            return {"llm": self.decisor.nome_llm if self.usar_llm else None, "ml": self.decisor.nome_ml,
                    "planos_recebidos": self.planos_recebidos, "mensagens": len(self.mensagens),
                    "ocupado": self._em_andamento > 0}

    def encerrar(self, com_decisor: bool = True) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        if com_decisor:
            self.decisor.encerrar()


__all__ = ["ETAPAS_FINAIS", "ServicoAgentes", "criar_llm_decisoes", "mensagem_final", "mensagem_regra"]
