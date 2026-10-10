"""Grafo da camada agêntica (LangGraph), com estado no estilo BDI.

    perceber -> especialista_jerk -> especialista_ml -> deliberar -> [supervisor] -> registrar

- crenças: fatos de cada janela de 5 s, eventos de jerk e avaliação do modelo de ML;
- desejos: registrar o log do bloco e, se houver eventos, apontar problemas;
- intenções: o plano de entradas do bloco. Cada entrada nova é escrita pelo Supervisor (LLM);
  um trecho que só repete o estado da entrada anterior vira uma linha curta de continuação,
  sem LLM.

Regras de mesclagem (config/agentes.yaml, seção blocos/estado):
- janelas seguidas com o mesmo estado (manobra predominante, faixa de velocidade, nível e
  atipicidade pelo ML) viram uma entrada só, até `mesclar_ate_s`;
- janelas com evento de jerk nunca são mescladas e cada evento gera uma entrada de problema,
  com o ajuste que o caminhão deve executar (ajustes.py);
- todo bloco tem pelo menos uma entrada de log.

A memória entre blocos guarda a última entrada de log (para continuar trechos estáveis), o
fim do último evento (para não repetir um evento na fronteira entre blocos) e o próximo id.
"""

from __future__ import annotations

import logging
import math
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from langgraph.graph import END, START, StateGraph

from mover.agentes import textos
from mover.agentes.ajustes import PlanejadorAjustes
from mover.agentes.especialistas import EspecialistaJerk, EspecialistaML, EventoJerk
from mover.agentes.fatos import ORDEM_NIVEL, calcular_fatos, estado_do_trecho
from mover.agentes.supervisor import Redacao, Supervisor

log = logging.getLogger("mover.agentes")

EPS = 1e-6


class EstadoAgentes(TypedDict, total=False):
    bloco: int
    t_ini: float
    t_fim: float
    dados: pd.DataFrame              # bloco + margem de contexto
    crencas: dict[str, Any]
    desejos: list[str]
    intencoes: list[dict[str, Any]]
    entradas: list[dict[str, Any]]
    memoria: dict[str, Any]


def memoria_inicial() -> dict[str, Any]:
    return {"proximo_id": 1, "ultimo_log": None, "ultimo_evento_fim": -math.inf}


class CamadaAgentica:
    """Processa a telemetria tratada bloco a bloco e devolve as entradas do log."""

    def __init__(self, cfg: dict[str, Any], telemetria: pd.DataFrame, supervisor: Supervisor,
                 especialista_ml: EspecialistaML | None = None):
        self.cfg = cfg
        self.taxa_hz = float(cfg["entrada"]["taxa_hz"])
        cb = cfg["blocos"]
        self.duracao_bloco = float(cb["duracao_s"])
        self.janela_s = float(cb["janela_s"])
        self.mesclar_ate_s = float(cb["mesclar_ate_s"])
        self.margem_s = float(cb["margem_eventos_s"])
        self.faixas = list(cfg["estado"]["faixas_velocidade_kmh"])
        self.reto_ate_graus = float(cfg["estado"].get("reto_ate_graus", 15))
        self.telemetria = telemetria.reset_index(drop=True)
        self._t = self.telemetria["sim_time"].to_numpy(dtype=float)
        self.t0 = float(self._t[0])
        self.especialista_jerk = EspecialistaJerk(cfg["limiares"])
        self.especialista_ml = especialista_ml
        self.planejador = PlanejadorAjustes(cfg.get("ajustes"))
        self.supervisor = supervisor
        self.memoria = memoria_inicial()
        self.grafo = self._montar_grafo()

    # ------------------------------------------------------------------ interface pública
    @property
    def n_blocos(self) -> int:
        duracao = self._t[-1] + 1.0 / self.taxa_hz - self.t0
        return max(1, math.ceil(duracao / self.duracao_bloco - EPS))

    def intervalo_do_bloco(self, indice: int) -> tuple[float, float]:
        t_ini = self.t0 + indice * self.duracao_bloco
        return t_ini, t_ini + self.duracao_bloco

    def processar_bloco(self, indice: int) -> list[dict[str, Any]]:
        """Roda o grafo no bloco `indice` (blocos devem ser processados em ordem)."""
        t_ini, t_fim = self.intervalo_do_bloco(indice)
        no_bloco = (self._t >= t_ini - self.margem_s - EPS) & (self._t < t_fim + self.margem_s - EPS)
        saida = self.grafo.invoke({
            "bloco": indice,
            "t_ini": t_ini,
            "t_fim": t_fim,
            "dados": self.telemetria.loc[no_bloco].reset_index(drop=True),
            "memoria": self.memoria,
        })
        self.memoria = saida["memoria"]
        return saida["entradas"]

    # ------------------------------------------------------------------ grafo
    def _montar_grafo(self):
        grafo = StateGraph(EstadoAgentes)
        grafo.add_node("perceber", self._perceber)
        grafo.add_node("especialista_jerk", self._especialista_jerk)
        grafo.add_node("especialista_ml", self._especialista_ml)
        grafo.add_node("deliberar", self._deliberar)
        grafo.add_node("supervisor", self._supervisor)
        grafo.add_node("registrar", self._registrar)
        grafo.add_edge(START, "perceber")
        grafo.add_edge("perceber", "especialista_jerk")
        grafo.add_edge("especialista_jerk", "especialista_ml")
        grafo.add_edge("especialista_ml", "deliberar")
        grafo.add_conditional_edges("deliberar", self._rota_apos_deliberar,
                                    {"supervisor": "supervisor", "registrar": "registrar"})
        grafo.add_edge("supervisor", "registrar")
        grafo.add_edge("registrar", END)
        return grafo.compile()

    def _perceber(self, estado: EstadoAgentes) -> dict[str, Any]:
        """Divide o bloco em janelas de `janela_s` e calcula os fatos de cada uma."""
        dados = estado["dados"]
        t = dados["sim_time"].to_numpy(dtype=float)
        janelas: list[dict[str, Any]] = []
        inicio, fim_bloco = estado["t_ini"], estado["t_fim"]
        while inicio < fim_bloco - EPS:
            fim = min(inicio + self.janela_s, fim_bloco)
            pos = np.flatnonzero((t >= inicio - EPS) & (t < fim - EPS))
            if pos.size:
                curta = pos.size < 0.5 * self.janela_s * self.taxa_hz
                if curta and janelas:  # sobra curta no fim do bloco: junta à janela anterior
                    janelas[-1]["linhas"] = (janelas[-1]["linhas"][0], int(pos[-1]) + 1)
                    janelas[-1]["t_fim"] = fim
                else:
                    janelas.append({"t_ini": inicio, "t_fim": fim, "linhas": (int(pos[0]), int(pos[-1]) + 1)})
            inicio = fim
        for janela in janelas:
            a, b = janela["linhas"]
            janela["fatos"] = calcular_fatos(dados.iloc[a:b], self.taxa_hz, self.reto_ate_graus)
            janela["eventos"] = []
        return {"crencas": {"janelas": janelas, "eventos": []}}

    def _especialista_jerk(self, estado: EstadoAgentes) -> dict[str, Any]:
        crencas = estado["crencas"]
        ja_reportado_ate = estado["memoria"]["ultimo_evento_fim"]
        eventos = [
            ev for ev in self.especialista_jerk.detectar(estado["dados"], estado["t_ini"], estado["t_fim"])
            if ev.t_ini > ja_reportado_ate + EPS  # o mesmo grupo de picos já saiu no bloco anterior
        ]
        janelas = []
        for janela in crencas["janelas"]:
            dela = [ev for ev in eventos if janela["t_ini"] - EPS <= ev.t_pico < janela["t_fim"] - EPS]
            nivel = max((ev.nivel for ev in dela), key=ORDEM_NIVEL.__getitem__, default="info")
            janelas.append(dict(janela, eventos=dela, fatos=dict(janela["fatos"], n_eventos=len(dela), nivel=nivel)))
        return {"crencas": dict(crencas, janelas=janelas, eventos=eventos)}

    def _especialista_ml(self, estado: EstadoAgentes) -> dict[str, Any]:
        crencas = estado["crencas"]
        if self.especialista_ml is None:
            return {"crencas": crencas}
        janelas = []
        for janela in crencas["janelas"]:
            a, b = janela["linhas"]
            avaliacao = self.especialista_ml.avaliar(estado["dados"].iloc[a:b])
            janelas.append(dict(janela, fatos=dict(janela["fatos"], **avaliacao)))
        return {"crencas": dict(crencas, janelas=janelas)}

    def _deliberar(self, estado: EstadoAgentes) -> dict[str, Any]:
        """Agrupa janelas com o mesmo estado e decide quais entradas o bloco vai ter."""
        crencas = estado["crencas"]
        grupos: list[dict[str, Any]] = []
        for janela in crencas["janelas"]:
            situacao = estado_do_trecho(janela["fatos"], self.faixas)
            ultimo = grupos[-1] if grupos else None
            if (ultimo is not None and ultimo["estado"] == situacao and situacao[2] == "info"
                    and janela["t_fim"] - ultimo["t_ini"] <= self.mesclar_ate_s + EPS):
                ultimo["janelas"].append(janela)
                ultimo["t_fim"] = janela["t_fim"]
            else:
                grupos.append({"estado": situacao, "janelas": [janela], "t_ini": janela["t_ini"], "t_fim": janela["t_fim"]})

        anterior = estado["memoria"]["ultimo_log"]
        intencoes: list[dict[str, Any]] = []
        for i, grupo in enumerate(grupos):
            continua = (
                i == 0 and anterior is not None
                and anterior["estado"] == grupo["estado"] and grupo["estado"][2] == "info"
                and abs(anterior["t_fim"] - grupo["t_ini"]) < EPS
                and grupo["t_fim"] - anterior["desde"] <= self.mesclar_ate_s + EPS
            )
            intencoes.append({
                "tipo": "log",
                "modo": "continuacao" if continua else "llm",
                "grupo": grupo,
                "desde": anterior["desde"] if continua else grupo["t_ini"],
            })
            for janela in grupo["janelas"]:
                intencoes.extend({"tipo": "problema", "modo": "llm", "evento": ev} for ev in janela["eventos"])

        desejos = ["registrar_log"] + (["apontar_problemas"] if crencas["eventos"] else [])
        return {"desejos": desejos, "intencoes": intencoes}

    @staticmethod
    def _rota_apos_deliberar(estado: EstadoAgentes) -> str:
        precisa = any(it["modo"] == "llm" for it in estado["intencoes"])
        return "supervisor" if precisa else "registrar"

    def _supervisor(self, estado: EstadoAgentes) -> dict[str, Any]:
        self.supervisor.iniciar_bloco()
        intencoes = []
        for it in estado["intencoes"]:
            if it["modo"] == "llm" and it["tipo"] == "log":
                fatos = self._fatos_do_grupo(estado["dados"], it["grupo"])
                eventos = [ev.fatos() for janela in it["grupo"]["janelas"] for ev in janela["eventos"]]
                it = dict(it, fatos=fatos, redacao=self.supervisor.escrever_log(fatos, eventos))
            elif it["modo"] == "llm":
                it = dict(it, redacao=self.supervisor.escrever_problema(it["evento"].fatos()))
            intencoes.append(it)
        return {"intencoes": intencoes}

    def _registrar(self, estado: EstadoAgentes) -> dict[str, Any]:
        memoria = dict(estado["memoria"])
        entradas = []
        for it in estado["intencoes"]:
            if it["tipo"] == "log":
                grupo = it["grupo"]
                if it["modo"] == "continuacao":
                    fatos = self._fatos_do_grupo(estado["dados"], grupo)
                    texto = textos.texto_continuacao(fatos, fatos["t_fim"] - it["desde"])
                    redacao = Redacao({"resumo": texto}, "modelo", observacao="continuação de trecho estável (sem LLM)")
                else:
                    fatos, redacao = it["fatos"], it["redacao"]
                entrada = self._entrada_log(estado["bloco"], grupo, fatos, redacao, it)
                memoria["ultimo_log"] = {"estado": grupo["estado"], "desde": it["desde"], "t_fim": grupo["t_fim"]}
            else:
                evento: EventoJerk = it["evento"]
                entrada = self._entrada_problema(estado["bloco"], evento, it["redacao"])
                memoria["ultimo_evento_fim"] = max(memoria["ultimo_evento_fim"], evento.t_fim)
            entradas.append({"id": memoria["proximo_id"], **entrada})
            memoria["proximo_id"] += 1
        return {"entradas": entradas, "memoria": memoria}

    # ------------------------------------------------------------------ auxiliares
    def _fatos_do_grupo(self, dados: pd.DataFrame, grupo: dict[str, Any]) -> dict[str, Any]:
        janelas = grupo["janelas"]
        if len(janelas) == 1:
            return janelas[0]["fatos"]
        a, b = janelas[0]["linhas"][0], janelas[-1]["linhas"][1]
        fatos = calcular_fatos(dados.iloc[a:b], self.taxa_hz, self.reto_ate_graus)
        fatos.update(n_eventos=0, nivel="info")
        if self.especialista_ml is not None:
            fatos.update(self.especialista_ml.avaliar(dados.iloc[a:b]))
        return fatos

    def _metadados_redacao(self, redacao: Redacao) -> dict[str, Any]:
        return {
            "autor": "supervisor",
            "texto_origem": redacao.origem,
            "provedor": self.supervisor.nome_llm if redacao.origem == "llm" else None,
            "latencia_s": round(redacao.latencia_s, 3),
            "observacao": redacao.observacao,
        }

    def _entrada_log(self, bloco: int, grupo: dict[str, Any], fatos: dict[str, Any], redacao: Redacao,
                     intencao: dict[str, Any]) -> dict[str, Any]:
        continuacao = intencao["modo"] == "continuacao"
        detectado_por = []
        if fatos.get("n_eventos"):
            detectado_por.append("especialista_jerk")
        if fatos.get("ml_atipica"):
            detectado_por.append("especialista_ml")
        return {
            "tipo": "log",
            "bloco": bloco,
            "t_ini": fatos["t_ini"],
            "t_fim": fatos["t_fim"],
            "t_pico": None,
            "hora_local": fatos["hora_ini"],
            "nivel": grupo["estado"][2],
            "causa": None,
            "evento": None,
            "texto": redacao.campos["resumo"],
            "diagnostico": None,
            "acao": None,
            "n_janelas": len(grupo["janelas"]),
            "continuacao": continuacao,
            "mesclado_desde": round(intencao["desde"], 2) if continuacao else None,
            "detectado_por": detectado_por,
            "fonte": fatos["fonte"],
            **self._metadados_redacao(redacao),
            "fatos": fatos,
        }

    def _entrada_problema(self, bloco: int, evento: EventoJerk, redacao: Redacao) -> dict[str, Any]:
        diagnostico, acao = redacao.campos["diagnostico"], redacao.campos["acao"]
        return {
            "tipo": "problema",
            "bloco": bloco,
            "t_ini": evento.t_ini,
            "t_fim": evento.t_fim,
            "t_pico": evento.t_pico,
            "hora_local": evento.hora,
            "nivel": evento.nivel,
            "causa": evento.causa,
            "evento": evento.tipo,
            "texto": f"{diagnostico} Ação: {acao}",
            "diagnostico": diagnostico,
            "acao": acao,
            "n_janelas": None,
            "continuacao": False,
            "mesclado_desde": None,
            "detectado_por": ["especialista_jerk"],
            "fonte": evento.fonte,
            **self._metadados_redacao(redacao),
            "fatos": evento.fatos(),
            "ajuste": self.planejador.ajuste(evento),
        }
