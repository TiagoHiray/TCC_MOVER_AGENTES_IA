"""Sessão da corrida autônoma dentro do servidor e difusão das mensagens para os WebSockets.

Uma sessão = uma corrida do caminhão autônomo (uma ou mais voltas). Quem dirige é o processo do
piloto (mover.autonomo.dirigir): ele manda ao servidor a telemetria medida em lotes (0,5 s) e as
decisões da camada agêntica assim que ficam prontas. O servidor:
- guarda a telemetria e publica a série reduzida (5 Hz) para os gráficos, que se montam ao vivo;
- publica cada decisão (alteração) no log e o bloco seguinte: o plano que o caminhão mandou x o
  plano alterado pelos agentes;
- roda a camada de logs (Etapa 2) sobre a telemetria medida, bloco a bloco, numa thread: o que
  de fato aconteceu (trechos e eventos de jerk) vira log em linguagem natural;
- mantém as métricas ao vivo (combustível, custo, CO2) e a comparação com a linha de base (o plano
  do caminhão sem agentes, previsto pelo modelo do caminhão, na mesma distância percorrida).

Mensagens do WebSocket (campo "tipo"):
- historico:  ao conectar; sessão, entradas, série já percorrida, previsão, último estado e métricas;
- sessao:     evento "iniciada" ou "encerrada" (com o resumo no fim);
- entrada:    uma entrada nova do log: intervencao, log ou problema;
- telemetria: pontos novos da série (5 Hz) e o estado atual do caminhão com as métricas;
- previsao:   a previsão da decisão mais recente (curvas e elementos do plano à frente).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import queue
import threading
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from mover.agentes.grafo import CamadaAgentica
from mover.agentes.supervisor import Supervisor
from mover.agentes.telemetria_viva import para_agentes

log = logging.getLogger("mover.servidor")

# série dos gráficos e do mapa (5 Hz)
COLUNAS_SERIE = ("t", "speed_kmh", "acc_long", "v_desejada_kmh", "v_limite_kmh", "x", "y", "rumo_graus", "d", "combustivel_ml")
# colunas do estado atual do caminhão
COLUNAS_ESTADO = ("t", "x", "y", "rumo_graus", "speed_kmh", "acc_long", "v_desejada_kmh", "v_limite_kmh", "v_alvo_kmh",
                  "d", "s", "volta", "bloco", "modo_tick", "limitado_por", "throttle", "brake", "steer",
                  "combustivel_ml", "custo_r", "co2_g", "fonte", "modo", "hora_local")
PASSO_SERIE_S = 0.2


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
    modo: str = "agentes"                  # agentes | bruto (o caminhão sem alterações, para comparar)
    voltas: int = 1
    simulador: str = "substituto"          # carla | substituto
    provedor: str | None = None            # LLM dos logs e do chat (padrão: .env ou YAML)
    modelo: str | None = None
    llm_decisoes: str | None = None        # nome do LLM das decisões, informado pelo piloto
    fator_tempo: float = 1.0
    tempo_real: bool = True
    duracao_bloco_s: float = 10.0
    comprimento_volta_m: float = 0.0
    duracao_prevista_s: float = 0.0
    linha_de_base: dict[str, Any] | None = None
    combustivel: str = "diesel"
    preco_r_l: float = 6.0
    co2_kg_l: float = 2.68
    logs_medidos: bool = True              # roda a camada de logs sobre a telemetria medida


class SessaoEncerrada(Exception):
    pass


@dataclass
class Metricas:
    tempo_s: float = 0.0
    distancia_m: float = 0.0
    combustivel_ml: float = 0.0
    custo_r: float = 0.0
    co2_g: float = 0.0
    economia_prevista_ml: float = 0.0       # soma da economia prevista pelas decisões (plano alterado x plano do caminhão)
    decisoes: int = 0
    decisoes_llm: int = 0
    intervencoes: int = 0                   # decisões com opção diferente de "nenhuma"
    problemas_medidos: int = 0
    base_combustivel_ml: float | None = None  # linha de base (plano sem agentes, previsto) na mesma distância
    base_eventos: int | None = None
    economia_medida_ml: float | None = None
    economia_medida_pct: float | None = None
    latencias_llm_s: list[float] = field(default_factory=list)

    def para_json(self) -> dict[str, Any]:
        dados = asdict(self)
        lat = dados.pop("latencias_llm_s")
        dados["latencia_llm_media_s"] = round(float(np.mean(lat)), 2) if lat else None
        return para_json({k: (round(v, 3) if isinstance(v, float) else v) for k, v in dados.items()})


class SessaoAutonoma:
    """Uma corrida do caminhão autônomo: telemetria, decisões, logs medidos e métricas."""

    def __init__(self, cfg_agentes: dict[str, Any], opcoes: OpcoesSessao, difusor: Difusor, pasta_sessoes: Path,
                 supervisor: Supervisor | None = None):
        self.cfg = cfg_agentes
        self.opcoes = opcoes
        self.difusor = difusor
        self.duracao_bloco = float(opcoes.duracao_bloco_s)
        self.linhas: list[dict[str, Any]] = []
        self.serie: dict[str, list[Any]] = {c: [] for c in COLUNAS_SERIE}
        self.entradas: list[dict[str, Any]] = []
        self.previsao: dict[str, Any] | None = None
        self.ultimo_estado: dict[str, Any] | None = None
        self.metricas = Metricas()
        self.encerrada = False
        self.resumo_piloto: dict[str, Any] | None = None
        self._proximo_id = 1
        self._t_serie = -math.inf
        self._a_suave = 0.0
        self._trava = threading.RLock()
        self._base = opcoes.linha_de_base
        self.supervisor = supervisor
        self.camada: CamadaAgentica | None = None
        if opcoes.logs_medidos:
            if self.supervisor is None:
                from mover.agentes.rodar_agentes import criar_supervisor

                self.supervisor = criar_supervisor(cfg_agentes, opcoes.provedor, opcoes.modelo)
            self.camada = CamadaAgentica(cfg_agentes, None, self.supervisor, None, t0=0.0)
        self.margem_s = float(cfg_agentes.get("blocos", {}).get("margem_eventos_s", 2.0))
        self.taxa_hz = float(cfg_agentes.get("entrada", {}).get("taxa_hz", 20))
        self._blocos_pedidos = 0          # blocos já mandados para a camada de logs
        self._fila_logs: queue.Queue = queue.Queue()
        self._thread_logs: threading.Thread | None = None
        if self.camada is not None:
            self._thread_logs = threading.Thread(target=self._laco_logs, name="logs-medidos", daemon=True)
            self._thread_logs.start()
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
    @property
    def llm_logs(self) -> str:
        llm = getattr(self.supervisor, "llm", None)
        return getattr(llm, "nome", None) or "texto-modelo"

    def info(self) -> dict[str, Any]:
        op = self.opcoes
        return para_json({
            "id": self.id,
            "modo": op.modo,
            "voltas": op.voltas,
            "simulador": op.simulador,
            "duracao_bloco_s": self.duracao_bloco,
            "comprimento_volta_m": op.comprimento_volta_m,
            "distancia_total_m": op.comprimento_volta_m * op.voltas,
            "duracao_prevista_s": op.duracao_prevista_s,
            "fator_tempo": op.fator_tempo,
            "tempo_real": op.tempo_real,
            "llm": op.llm_decisoes or self.llm_logs,
            "llm_decisoes": op.llm_decisoes,
            "llm_logs": self.llm_logs,
            "provedor": op.provedor,
            "modelo": op.modelo,
            "combustivel": op.combustivel,
            "preco_r_l": op.preco_r_l,
            "co2_kg_l": op.co2_kg_l,
            "tem_linha_de_base": self._base is not None,
            "n_entradas": len(self.entradas),
            "encerrada": self.encerrada,
            "arquivo": str(self.arquivo),
        })

    def previsao_atual(self) -> dict[str, Any] | None:
        with self._trava:
            return self.previsao

    def historico(self) -> dict[str, Any]:
        """Tudo o que uma página que conecta no meio da corrida precisa para se montar."""
        with self._trava:
            return para_json({"sessao": self.info(), "entradas": list(self.entradas),
                              "serie": {c: list(v) for c, v in self.serie.items()}, "previsao": self.previsao,
                              "estado": self.ultimo_estado, "metricas": self.metricas.para_json(),
                              "linha_de_base": self._base})

    # ------------------------------------------------------------------ telemetria
    def receber_telemetria(self, linhas: list[dict[str, Any]]) -> None:
        if self.encerrada:
            raise SessaoEncerrada("a sessão já foi encerrada")
        if not linhas:
            return
        pontos: dict[str, list[Any]] = {c: [] for c in COLUNAS_SERIE}
        with self._trava:
            for linha in linhas:
                linha = dict(linha)
                t = float(linha.get("t", linha.get("sim_time", 0.0)))
                linha["t"] = t
                # aceleração mostrada suavizada (~0,3 s): a do CARLA vem com o ruído da física
                dt = 1.0 / self.taxa_hz
                self._a_suave += (float(linha.get("acc_long") or 0.0) - self._a_suave) * min(1.0, dt / 0.3)
                linha["acc_long_suave"] = self._a_suave
                self.linhas.append(linha)
                if t >= self._t_serie + PASSO_SERIE_S - 1e-6:
                    self._t_serie = t
                    for c in COLUNAS_SERIE:
                        valor = self._a_suave if c == "acc_long" else linha.get(c)
                        valor = round(float(valor), 3) if isinstance(valor, (int, float)) else valor
                        self.serie[c].append(valor)
                        pontos[c].append(valor)
            ultima = self.linhas[-1]
            self._atualizar_metricas(ultima)
            estado = {c: ultima.get(c) for c in COLUNAS_ESTADO}
            estado["acc_long"] = self._a_suave
            total = self.opcoes.comprimento_volta_m * self.opcoes.voltas
            estado["progresso"] = round(min(1.0, max(0.0, float(ultima.get("d", 0.0)) / total)), 4) if total > 0 else None
            self.ultimo_estado = para_json(estado)
            mensagem = {"tipo": "telemetria", "pontos": pontos, "estado": self.ultimo_estado,
                        "metricas": self.metricas.para_json()}
        self.difusor.publicar(mensagem)
        self._pedir_logs(float(ultima["t"]))

    def _atualizar_metricas(self, linha: dict[str, Any]) -> None:
        m = self.metricas
        m.tempo_s = float(linha.get("t", 0.0))
        m.distancia_m = float(linha.get("d", 0.0))
        m.combustivel_ml = float(linha.get("combustivel_ml", 0.0) or 0.0)
        m.custo_r = float(linha.get("custo_r", 0.0) or 0.0)
        m.co2_g = float(linha.get("co2_g", 0.0) or 0.0)
        if self._base and self._base.get("d"):
            d = np.asarray(self._base["d"], float)
            m.base_combustivel_ml = float(np.interp(m.distancia_m, d, np.asarray(self._base["combustivel_ml"], float)))
            m.base_eventos = int(np.interp(m.distancia_m, d, np.asarray(self._base["eventos"], float)))
            m.economia_medida_ml = m.base_combustivel_ml - m.combustivel_ml
            m.economia_medida_pct = 100.0 * m.economia_medida_ml / m.base_combustivel_ml if m.base_combustivel_ml > 1 else None

    # ------------------------------------------------------------------ decisões
    def receber_decisao(self, entrada: dict[str, Any]) -> dict[str, Any]:
        if self.encerrada:
            raise SessaoEncerrada("a sessão já foi encerrada")
        with self._trava:
            entrada = self._publicar_entrada(entrada)
            m = self.metricas
            m.decisoes += 1
            if entrada.get("texto_origem") == "llm":
                m.decisoes_llm += 1
            if entrada.get("latencia_llm_s"):
                m.latencias_llm_s.append(float(entrada["latencia_llm_s"]))
            if entrada.get("opcao") not in (None, "nenhuma"):
                m.intervencoes += 1
            m.economia_prevista_ml += float((entrada.get("fatos") or {}).get("economia_prevista_ml", 0.0) or 0.0)
            self.previsao = para_json({
                "id": entrada["id"], "bloco": entrada.get("bloco"), "t_ini": entrada.get("t_ini"),
                "t_fim": entrada.get("t_fim"), "t_pedido": entrada.get("t_pedido"), "opcao": entrada.get("opcao"),
                "acao": entrada.get("acao"), "texto_origem": entrada.get("texto_origem"), "nivel": entrada.get("nivel"),
                "curvas": entrada.get("curvas") or {}, "instancias": entrada.get("instancias") or [],
                "brutas": entrada.get("brutas") or [], "fatos": entrada.get("fatos") or {},
                "fonte": entrada.get("fonte"), "base": entrada.get("base"),
            })
            previsao = self.previsao
        self.difusor.publicar({"tipo": "previsao", "sessao": self.id, **previsao})
        return entrada

    def _publicar_entrada(self, entrada: dict[str, Any]) -> dict[str, Any]:
        with self._trava:
            entrada = para_json({**entrada, "id": self._proximo_id})
            self._proximo_id += 1
            self.entradas.append(entrada)
            self._arquivo.write(json.dumps(entrada, ensure_ascii=False) + "\n")
            self._arquivo.flush()
            if entrada.get("tipo") == "problema":
                self.metricas.problemas_medidos += 1
        self.difusor.publicar({"tipo": "entrada", "entrada": entrada})
        return entrada

    # ------------------------------------------------------------------ logs medidos (Etapa 2 ao vivo)
    def _pedir_logs(self, t_atual: float, fim: bool = False) -> None:
        """Manda para a camada de logs os blocos cuja telemetria (com a margem) já chegou."""
        if self.camada is None:
            return
        while True:
            k = self._blocos_pedidos
            t_fim = (k + 1) * self.duracao_bloco
            if not fim and t_atual < t_fim + self.margem_s:
                break
            if fim and k * self.duracao_bloco >= t_atual - 1e-6:
                break
            self._blocos_pedidos += 1
            self._fila_logs.put(k)

    def _trecho(self, k: int) -> Any:
        t_ini, t_fim = k * self.duracao_bloco, (k + 1) * self.duracao_bloco
        with self._trava:
            linhas = [l for l in self.linhas if t_ini - self.margem_s - 1.0 <= l["t"] < t_fim + self.margem_s]
        dados = para_agentes(linhas, self.taxa_hz)
        if len(dados):
            dados["sim_time"] = [l["t"] for l in linhas]
        return dados

    def _laco_logs(self) -> None:
        while True:
            k = self._fila_logs.get()
            if k is None:
                return
            try:
                dados = self._trecho(k)
                if len(dados) < 2:
                    continue
                for entrada in self.camada.processar_trecho(k, dados):
                    self._publicar_entrada(entrada)
            except Exception:
                log.exception("Erro na camada de logs (bloco %d); a corrida continua.", k)

    # ------------------------------------------------------------------ fim
    def resumo(self) -> dict[str, Any]:
        problemas = [e for e in self.entradas if e.get("tipo") == "problema"]
        decisoes = [e for e in self.entradas if e.get("tipo") == "intervencao"]
        return para_json({
            "entradas": len(self.entradas),
            "logs": sum(e.get("tipo") == "log" for e in self.entradas),
            "problemas": len(problemas),
            "problemas_por_nivel": dict(Counter(e.get("nivel") for e in problemas)),
            "decisoes": len(decisoes),
            "opcoes": dict(Counter(e.get("opcao") for e in decisoes)),
            "decisoes_por_origem": dict(Counter(e.get("texto_origem") for e in decisoes)),
            "texto_origem": dict(Counter(e.get("texto_origem") for e in self.entradas)),
            "metricas": self.metricas.para_json(),
            "piloto": self.resumo_piloto,
        })

    def encerrar(self, resumo_piloto: dict[str, Any] | None = None, prazo_logs_s: float = 60.0) -> dict[str, Any]:
        with self._trava:
            if self.encerrada:
                return self.resumo()
            self.encerrada = True
            self.resumo_piloto = resumo_piloto
            t_final = float(self.linhas[-1]["t"]) if self.linhas else 0.0
        if self._thread_logs is not None:  # o último bloco (parcial) também vira log
            self._pedir_logs(t_final, fim=True)
            self._fila_logs.put(None)
            self._thread_logs.join(timeout=prazo_logs_s)
        with self._trava:
            self._arquivo.close()
            resumo = self.resumo()
            with self.arquivo.with_name(self.arquivo.stem + "_resumo.json").open("w", encoding="utf-8") as f:
                json.dump(resumo, f, ensure_ascii=False, indent=2)
            self.previsao = None
        self.difusor.publicar({"tipo": "sessao", "evento": "encerrada", "sessao": self.info(), "resumo": resumo})
        return resumo


__all__ = ["COLUNAS_SERIE", "Difusor", "OpcoesSessao", "SessaoAutonoma", "SessaoEncerrada", "para_json"]
