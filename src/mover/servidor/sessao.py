"""Sessão da camada agêntica dentro do servidor e difusão das mensagens para os WebSockets.

Uma sessão = uma volta do replay. Ela monta a camada agêntica da Etapa 2 (mesmas funções do
rodar_agentes.py), analisa o bloco 0 antes da partida e, a cada aviso do replay de que o
caminhão entrou no bloco k, publica o resultado do bloco k e libera o k+1 (liberação
antecipada, ExecutorBlocos). Tudo o que é publicado vai para:
- os WebSockets conectados (log ao vivo para a interface);
- data/agentes/sessoes/sessao_<data>.jsonl (uma entrada por linha, como na Etapa 2).

Mensagens do WebSocket (campo "tipo"):
- historico: ao conectar; sessão atual, entradas e blocos já publicados, telemetria liberada,
             previsão do próximo bloco e último estado (para a página abrir no meio da volta);
- sessao:    evento "iniciada" ou "encerrada", com os dados da sessão (e o resumo, no fim);
- entrada:   uma entrada nova do log (log ou problema), no formato do JSONL da Etapa 2;
- bloco:     o caminhão entrou no bloco k (espera, latência da análise e telemetria do bloco);
- previsao:  a análise do próximo bloco terminou; entradas e telemetria dele, que o caminhão
             ainda não percorreu (a "previsão" mostrada na interface e usada pelo chat);
- estado:    pose e telemetria do caminhão (~5 Hz), para o dashboard.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mover.agentes.executor import ExecutorBlocos, ResultadoBloco
from mover.agentes.grafo import CamadaAgentica
from mover.agentes.rodar_agentes import (aplicar_eventos_sinteticos, carregar_especialista_ml, carregar_telemetria,
                                         criar_supervisor)
from mover.agentes.supervisor import Supervisor

log = logging.getLogger("mover.servidor")

# colunas da telemetria repassadas no estado do caminhão
COLUNAS_ESTADO = ("hora_local", "speed_kmh", "acc_long", "acc_lat", "acc_vert", "jerk_long", "yaw_rate_dps",
                  "grade_pct", "manobra", "throttle_est", "brake_est", "steer_est", "odom_m", "fonte")
# colunas dos gráficos do dashboard (telemetria dos blocos liberados, a 5 Hz)
COLUNAS_GRAFICO = ("speed_kmh", "acc_long")


def para_json(valor: Any) -> Any:
    """Converte tipos do numpy/pandas e NaN para JSON puro."""
    if isinstance(valor, dict):
        return {str(k): para_json(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [para_json(v) for v in valor]
    if isinstance(valor, np.generic):
        valor = valor.item()
    if isinstance(valor, float) and not math.isfinite(valor):
        return None
    return valor


class Difusor:
    """Entrega mensagens a todos os WebSockets conectados. Pode ser chamado de qualquer thread."""

    def __init__(self, tamanho_fila: int = 2000):
        self._filas: set[asyncio.Queue] = set()
        self._laco: asyncio.AbstractEventLoop | None = None
        self._tamanho = tamanho_fila

    def ligar(self, laco: asyncio.AbstractEventLoop) -> None:
        self._laco = laco

    def inscrever(self) -> asyncio.Queue:
        fila: asyncio.Queue = asyncio.Queue(maxsize=self._tamanho)
        self._filas.add(fila)
        return fila

    def cancelar(self, fila: asyncio.Queue) -> None:
        self._filas.discard(fila)

    @property
    def n_clientes(self) -> int:
        return len(self._filas)

    def _entregar(self, mensagem: dict[str, Any]) -> None:
        for fila in list(self._filas):
            if fila.full():  # cliente lento: descarta a mensagem mais antiga
                try:
                    fila.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            fila.put_nowait(mensagem)

    def publicar(self, mensagem: dict[str, Any]) -> None:
        if self._laco is None or self._laco.is_closed():
            return
        mensagem = para_json(mensagem)
        try:
            atual = asyncio.get_running_loop()
        except RuntimeError:
            atual = None
        if atual is self._laco:
            self._entregar(mensagem)
        else:
            self._laco.call_soon_threadsafe(self._entregar, mensagem)


@dataclass
class OpcoesSessao:
    injetar_eventos: bool = False
    provedor: str | None = None
    modelo: str | None = None
    sem_ml: bool = False
    fator_tempo: float = 1.0   # velocidade do replay (2 = duas vezes mais rápido), para os avisos de atraso
    tempo_real: bool = True    # False com --sem-espera: o caminhão não segue o relógio e esperar é normal
    volta: str | None = None   # volta da Fase X (data/voltas/<volta>) analisada nesta sessão


class BlocoForaDeOrdem(Exception):
    def __init__(self, esperado: int, recebido: int):
        super().__init__(f"bloco {recebido} fora de ordem; o próximo é o {esperado}")
        self.esperado, self.recebido = esperado, recebido


class BlocoInvalido(Exception):
    pass


class SessaoEncerrada(Exception):
    pass


class SessaoAgentes:
    """Uma volta do replay: camada agêntica + executor de blocos + log publicado."""

    def __init__(self, cfg_agentes: dict[str, Any], opcoes: OpcoesSessao, difusor: Difusor, pasta_sessoes: Path,
                 telemetria: pd.DataFrame | None = None, supervisor: Supervisor | None = None, sem_ml_forcado: bool = False):
        self.cfg = cfg_agentes
        self.opcoes = opcoes
        self.difusor = difusor
        if telemetria is None:
            telemetria = carregar_telemetria(cfg_agentes, None, opcoes.injetar_eventos)
        elif opcoes.injetar_eventos:
            telemetria = aplicar_eventos_sinteticos(cfg_agentes, telemetria)
        self.telemetria = telemetria.reset_index(drop=True)
        self.supervisor = supervisor or criar_supervisor(cfg_agentes, opcoes.provedor, opcoes.modelo)
        ml = carregar_especialista_ml(cfg_agentes, opcoes.sem_ml or sem_ml_forcado)
        self.camada = CamadaAgentica(cfg_agentes, self.telemetria, self.supervisor, ml)
        self.entradas: list[dict[str, Any]] = []
        self.blocos: list[dict[str, Any]] = []
        self.proximo_bloco = 0
        self.encerrada = False
        self.previsao: dict[str, Any] | None = None       # próximo bloco já analisado e ainda não publicado
        self.ultimo_estado: dict[str, Any] | None = None  # última pose recebida do replay
        self._trava = threading.Lock()
        self._trava_previsao = threading.Lock()
        self._t = self.telemetria["sim_time"].to_numpy(float)
        self.executor = ExecutorBlocos(self.camada, ao_publicar=self._ao_publicar, ao_analisar=self._ao_analisar,
                                       avisar_atrasos=opcoes.tempo_real, fator_tempo=max(float(opcoes.fator_tempo), 1e-6))
        self.criada_em = datetime.now()
        pasta_sessoes.mkdir(parents=True, exist_ok=True)
        base = self.criada_em.strftime("%Y%m%d_%H%M%S")
        self.id, n = base, 1
        while (pasta_sessoes / f"sessao_{self.id}.jsonl").exists():  # duas sessões no mesmo segundo
            n += 1
            self.id = f"{base}_{n}"
        self.arquivo = pasta_sessoes / f"sessao_{self.id}.jsonl"
        self._arquivo = self.arquivo.open("w", encoding="utf-8")

    # ------------------------------------------------------------------ informações
    def info(self) -> dict[str, Any]:
        llm = getattr(self.supervisor, "llm", None)
        previsao = self.previsao
        return {
            "id": self.id,
            "n_blocos": self.camada.n_blocos,
            "duracao_bloco_s": self.camada.duracao_bloco,
            "t0": self.camada.t0,
            "duracao_volta_s": round(float(self._t[-1] - self._t[0] + 1.0 / self.camada.taxa_hz), 2),
            "taxa_hz": self.camada.taxa_hz,
            "injetar_eventos": self.opcoes.injetar_eventos,
            "volta": self.opcoes.volta,
            "fator_tempo": self.opcoes.fator_tempo,
            "tempo_real": self.opcoes.tempo_real,
            "llm": getattr(llm, "nome", None) or "texto-modelo",
            "provedor": self.opcoes.provedor,
            "modelo": self.opcoes.modelo,
            "especialista_ml": self.camada.especialista_ml is not None,
            "proximo_bloco": self.proximo_bloco,
            "previsao_bloco": previsao["bloco"] if previsao else None,
            "n_entradas": len(self.entradas),
            "encerrada": self.encerrada,
            "arquivo": str(self.arquivo),
        }

    def previsao_atual(self) -> dict[str, Any] | None:
        with self._trava_previsao:
            return self.previsao

    def telemetria_resumida(self, t_ini: float, t_fim: float, passo: int = 4) -> dict[str, list[float | None]]:
        """Telemetria de [t_ini, t_fim) para os gráficos: média de cada `passo` quadros (5 Hz a 20 Hz)."""
        sel = np.flatnonzero((self._t >= t_ini - 1e-6) & (self._t < t_fim - 1e-6))
        if sel.size == 0:
            return {"t": [], **{c: [] for c in COLUNAS_GRAFICO if c in self.telemetria}}
        grupo = np.arange(sel.size) // max(1, int(passo))
        contagem = np.bincount(grupo)

        def media(valores: np.ndarray) -> list[float | None]:
            return para_json(np.round(np.bincount(grupo, weights=valores) / contagem, 2).tolist())

        saida = {"t": media(self._t[sel])}
        for coluna in COLUNAS_GRAFICO:
            if coluna in self.telemetria:
                saida[coluna] = media(self.telemetria[coluna].to_numpy(float)[sel])
        return saida

    def historico(self) -> dict[str, Any]:
        """Tudo o que uma página que conecta no meio da volta precisa para se montar."""
        fim = self.blocos[-1]["t_fim"] if self.blocos else self.camada.t0
        return para_json({"sessao": self.info(), "entradas": list(self.entradas), "blocos": list(self.blocos),
                          "telemetria": self.telemetria_resumida(self.camada.t0, fim),
                          "previsao": self.previsao_atual(), "estado": self.ultimo_estado})

    # ------------------------------------------------------------------ ciclo de vida
    def preparar(self) -> None:
        """Analisa o bloco 0 (bloqueante; o caminhão só parte com ele pronto)."""
        self.executor.preparar()

    def entrar_no_bloco(self, k: int) -> ResultadoBloco:
        """Bloqueante: espera a análise do bloco k, publica e libera o k+1."""
        with self._trava:
            if self.encerrada:
                raise SessaoEncerrada("a sessão já foi encerrada")
            if not 0 <= k < self.camada.n_blocos:
                raise BlocoInvalido(f"bloco {k} não existe; a volta tem os blocos 0 a {self.camada.n_blocos - 1}")
            if k != self.proximo_bloco:
                raise BlocoForaDeOrdem(self.proximo_bloco, k)
            resultado = self.executor.entrar_no_bloco(k)
            self.proximo_bloco = k + 1
            return resultado

    def _ao_publicar(self, resultado: ResultadoBloco) -> None:
        for entrada in resultado.entradas:
            entrada = para_json(entrada)
            self.entradas.append(entrada)
            self._arquivo.write(json.dumps(entrada, ensure_ascii=False) + "\n")
            self.difusor.publicar({"tipo": "entrada", "entrada": entrada})
        self._arquivo.flush()
        with self._trava_previsao:  # o bloco deixou de ser previsão (a do k+1 pode já ter chegado)
            if self.previsao is not None and self.previsao["bloco"] <= resultado.bloco:
                self.previsao = None
        t_ini, t_fim = self.camada.intervalo_do_bloco(resultado.bloco)
        bloco = {"bloco": resultado.bloco, "t_ini": t_ini, "t_fim": t_fim, "n_entradas": len(resultado.entradas),
                 "n_problemas": sum(e["tipo"] == "problema" for e in resultado.entradas),
                 "espera_s": round(resultado.espera_s, 3), "latencia_s": round(resultado.latencia_s, 3)}
        self.blocos.append(bloco)
        self.difusor.publicar({"tipo": "bloco", **bloco, "telemetria": self.telemetria_resumida(t_ini, t_fim)})

    def _ao_analisar(self, resultado: ResultadoBloco) -> None:
        """Na thread de trabalho: a análise de um bloco terminou e ele vira a previsão até ser publicado."""
        t_ini, t_fim = self.camada.intervalo_do_bloco(resultado.bloco)
        previsao = para_json({"bloco": resultado.bloco, "t_ini": t_ini, "t_fim": t_fim, "entradas": resultado.entradas,
                              "n_entradas": len(resultado.entradas),
                              "n_problemas": sum(e["tipo"] == "problema" for e in resultado.entradas),
                              "latencia_s": round(resultado.latencia_s, 3),
                              "telemetria": self.telemetria_resumida(t_ini, t_fim)})
        with self._trava_previsao:
            if resultado.bloco < self.proximo_bloco:  # o caminhão já entrou nele
                return
            self.previsao = previsao
        # com o id da sessão: a previsão do bloco 0 sai antes do aviso "iniciada" desta sessão
        self.difusor.publicar({"tipo": "previsao", "sessao": self.id, **previsao})

    def estado(self, sim_time: float, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Estado do caminhão no instante sim_time (linha mais próxima da telemetria da sessão)."""
        i = int(np.clip(np.searchsorted(self._t, sim_time), 0, len(self._t) - 1))
        if i > 0 and abs(self._t[i - 1] - sim_time) < abs(self._t[i] - sim_time):
            i -= 1
        linha = self.telemetria.iloc[i]
        estado = {"sim_time": float(self._t[i]), "bloco": int((self._t[i] - self.camada.t0) // self.camada.duracao_bloco),
                  "quadro": i, "progresso": round(i / max(1, len(self._t) - 1), 4)}
        estado.update({c: linha[c] for c in COLUNAS_ESTADO if c in linha.index})
        estado.update(extra or {})
        estado = para_json(estado)
        self.ultimo_estado = estado
        return estado

    def resumo(self) -> dict[str, Any]:
        problemas = [e for e in self.entradas if e["tipo"] == "problema"]
        return {
            "entradas": len(self.entradas),
            "logs": sum(e["tipo"] == "log" for e in self.entradas),
            "problemas": len(problemas),
            "problemas_por_nivel": dict(Counter(e["nivel"] for e in problemas)),
            "problemas_por_evento": dict(Counter(e["evento"] for e in problemas)),
            "texto_origem": dict(Counter(e["texto_origem"] for e in self.entradas)),
            "blocos_publicados": len(self.blocos),
            "maior_espera_s": max((b["espera_s"] for b in self.blocos[1:]), default=0.0),
            "latencia_media_s": round(float(np.mean([b["latencia_s"] for b in self.blocos])), 3) if self.blocos else 0.0,
        }

    def encerrar(self) -> dict[str, Any]:
        with self._trava:
            if not self.encerrada:
                self.encerrada = True
                self.executor.encerrar()
                self._arquivo.close()
                resumo = self.resumo()
                with self.arquivo.with_name(self.arquivo.stem + "_resumo.json").open("w", encoding="utf-8") as f:
                    json.dump(para_json(resumo), f, ensure_ascii=False, indent=2)
                self.difusor.publicar({"tipo": "sessao", "evento": "encerrada", "sessao": self.info(), "resumo": resumo})
        return self.resumo()
