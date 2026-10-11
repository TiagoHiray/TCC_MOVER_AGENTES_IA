"""Laço da corrida autônoma: governador -> piloto automático (CARLA ou substituto) -> telemetria.

A cada tick (20 Hz):
1. o governador calcula a velocidade desejada a partir do plano de missão e das alterações já
   combinadas com a camada agêntica;
2. o simulador aplica a velocidade (Traffic Manager do CARLA ou o substituto) e devolve a telemetria;
3. a linha ganha o consumo acumulado (modelo VT-CPFM) e entra no lote mandado ao servidor (0,5 s);
4. ao entrar num bloco de 10 s, o caminhão monta o plano do bloco seguinte a partir do estado atual
   da corrida (mover.autonomo.planejador) e o manda aos agentes, antes de executá-lo.

Cada decisão chega em duas etapas: primeiro a da regra (rápida, aplicada na hora) e depois a do
Supervisor (LLM, com prazo), que pode trocar a alteração enquanto o caminhão ainda não chegou ao
elemento. Ao vivo, os agentes rodam numa thread e a corrida não espera por eles. No benchmark, a
simulação espera cada decisão, e o resultado não depende da velocidade do computador.
"""

from __future__ import annotations

import concurrent.futures as futures
import logging
import math
import queue
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np

from mover.agentes.decisao import Decisao, Decisor, janela_do_bloco, janela_inicial
from mover.agentes.telemetria_viva import MedidorConsumo
from mover.autonomo.consumo import Limiares, ParametrosConsumo, metricas
from mover.autonomo.governador import NOMES_OPCAO, Governador, Intervencao
from mover.autonomo.missao import PlanoMissao
from mover.autonomo.planejador import PlanejadorCaminhao, PlanoFuturo
from mover.autonomo.veiculo import ParametrosCaminhao

log = logging.getLogger("mover.autonomo")


class Simulador(Protocol):
    nome: str
    dt: float

    def iniciar(self) -> None: ...
    def reiniciar(self) -> None: ...
    def passo(self, v_desejada: float) -> dict[str, Any]: ...
    def encerrar(self) -> None: ...


class Relogio:
    """Faz a corrida andar no ritmo do relógio (vezes fator_tempo) e se reancora depois de esperas.

    Se o simulador atrasar (CARLA lento), a corrida segue do ponto em que está, sem acelerar para
    "recuperar" o tempo perdido.
    """

    def __init__(self, fator_tempo: float = 1.0, sem_espera: bool = False, tolerancia_s: float = 0.25):
        self.fator = max(float(fator_tempo), 1e-6)
        self.sem_espera = sem_espera
        self.tolerancia_s = tolerancia_s
        self.atrasos = 0
        self._t_sim0 = 0.0
        self._t_real0 = time.monotonic()

    def ancorar(self, t_sim: float) -> None:
        self._t_sim0, self._t_real0 = float(t_sim), time.monotonic()

    def esperar_ate(self, t_sim: float) -> None:
        if self.sem_espera:
            return
        restante = self._t_real0 + (float(t_sim) - self._t_sim0) / self.fator - time.monotonic()
        if restante > 0:
            time.sleep(restante)
        elif restante < -self.tolerancia_s:
            self.atrasos += 1
            self.ancorar(t_sim)


@dataclass
class OpcoesConducao:
    voltas: int = 1
    usar_llm: bool = True
    assincrona: bool = True          # decisões numa thread (ao vivo); False: a simulação espera (benchmark)
    fator_tempo: float = 1.0
    sem_espera: bool = False         # não segue o relógio (o mais rápido possível)
    lote_s: float = 0.5              # telemetria mandada ao servidor a cada 0,5 s de simulação
    duracao_bloco_s: float = 10.0
    tempo_max_s: float = 0.0         # 0: 5 min por volta (proteção contra caminhão preso)
    parado_max_s: float = 30.0       # parado esse tempo pedindo para andar: encerra (caminhão preso)


@dataclass
class ResultadoConducao:
    modo: str
    simulador: str
    voltas: int
    linhas: list[dict[str, Any]] = field(default_factory=list)
    decisoes: list[dict[str, Any]] = field(default_factory=list)
    motivo_fim: str = ""             # chegada | tempo_max | preso | interrompido
    duracao_real_s: float = 0.0
    atrasos_relogio: int = 0
    aplicacoes: dict[str, int] = field(default_factory=lambda: {"regra": 0, "trocas_llm": 0, "atrasadas": 0})

    @property
    def completa(self) -> bool:
        return self.motivo_fim == "chegada"

    def coluna(self, nome: str) -> np.ndarray:
        return np.array([float(linha.get(nome, math.nan)) for linha in self.linhas])


def metricas_da_corrida(resultado: ResultadoConducao, comprimento_m: float, caminhao: ParametrosCaminhao,
                        consumo: ParametrosConsumo, limiares: Limiares) -> dict[str, Any]:
    """Métricas da corrida inteira e de cada volta (mesmo modelo de consumo para os dois modos)."""
    if len(resultado.linhas) < 2:
        return {"total": {}, "voltas": []}
    t, v, a = resultado.coluna("sim_time"), resultado.coluna("speed_mps"), resultado.coluna("acc_long")
    d = resultado.coluna("d")
    rampa = np.nan_to_num(resultado.coluna("rampa"))
    total = metricas(t, v, a, caminhao, consumo, limiares, rampa=rampa)
    voltas = []
    indice = np.floor(np.maximum(d, 0.0) / comprimento_m + 1e-9).astype(int)
    indice = np.minimum(indice, max(0, resultado.voltas - 1))  # a parada final fica na última volta
    for k in range(resultado.voltas):
        sel = indice == k
        if sel.sum() < 2:
            continue
        m = metricas(t[sel], v[sel], a[sel], caminhao, consumo, limiares, rampa=rampa[sel])
        voltas.append({"volta": k + 1, **m})
    return {"total": total, "voltas": voltas}


class Conducao:
    """Uma corrida de N voltas num modo (bruto: o piloto sozinho; agentes: com a camada agêntica)."""

    def __init__(self, plano: PlanoMissao, simulador: Simulador, governador: Governador, decisor: Decisor | None = None,
                 opcoes: OpcoesConducao | None = None, caminhao: ParametrosCaminhao | None = None,
                 consumo: ParametrosConsumo | None = None,
                 ao_lote: Callable[[list[dict[str, Any]]], None] | None = None,
                 ao_decisao: Callable[[dict[str, Any]], None] | None = None,
                 imprimir: Callable[[str], None] | None = None, planejador: PlanejadorCaminhao | None = None):
        self.plano = plano
        self.sim = simulador
        self.gov = governador
        self.decisor = decisor if governador.modo == "agentes" else None
        # o lado do caminhão: monta o plano dos próximos 10 s a partir da corrida atual
        self.planejador = planejador or (self.decisor.planejador() if self.decisor is not None else None)
        self.op = opcoes or OpcoesConducao()
        self.caminhao = caminhao or ParametrosCaminhao()
        self.consumo = consumo or ParametrosConsumo()
        self.ao_lote = ao_lote
        self.ao_decisao = ao_decisao
        self.imprimir = imprimir or (lambda texto: None)
        self.estado = {"t": 0.0, "d": 0.0, "v": 0.0, "a": 0.0}
        self._fila: queue.Queue = queue.Queue()
        self._executor: futures.ThreadPoolExecutor | None = None
        self._futuro: futures.Future | None = None
        self._pendente: tuple[int, tuple[float, float]] | None = None
        self._dono: dict[str, int] = {}          # elemento -> decisão (bloco) que o definiu
        self._resultado: ResultadoConducao | None = None

    # ------------------------------------------------------------------ decisões
    def _planejar(self, bloco: int, janela: tuple[float, float]) -> PlanoFuturo | None:
        """O caminhão monta o plano do bloco seguinte a partir do estado atual da corrida.

        No CARLA, as manobras à frente vêm do próprio Traffic Manager (get_all_actions).
        """
        acoes = None
        obter = getattr(self.sim, "acoes_do_tm", None)
        if callable(obter):
            try:
                acoes = obter()
            except Exception as erro:  # o plano segue com as manobras tiradas da rota
                log.debug("Sem as ações do Traffic Manager: %s", erro)
        return self.planejador.planejar(self.gov, dict(self.estado), bloco, janela, acoes)

    def _pedir(self, bloco: int, janela: tuple[float, float]) -> None:
        if self.decisor is None:
            return
        if not self.op.assincrona:
            self._decidir_agora(bloco, janela)
            return
        if self._futuro is None or self._futuro.done():
            self._lancar(bloco, janela)
        else:  # os agentes ainda trabalham no pedido anterior: este espera (só o mais novo)
            self._pendente = (bloco, janela)

    def _lancar(self, bloco: int, janela: tuple[float, float]) -> None:
        plano = self._planejar(bloco, janela)  # o caminhão monta o plano agora e o manda aos agentes
        if plano is None:
            return
        if self._executor is None:
            self._executor = futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="agentes")
        self._futuro = self._executor.submit(self._trabalho, plano, self.gov.clonar())

    def _trabalho(self, plano: PlanoFuturo, governador: Governador) -> None:
        """Na thread dos agentes: regra primeiro (vai para o caminhão na hora), depois o Supervisor."""
        decisao = self.decisor.avaliar(plano, governador)
        if decisao is None:
            return
        self._fila.put(("regra", decisao.bloco, decisao.intervencoes_para(decisao.recomendada, self.decisor.opcoes, "regra")))
        if self.op.usar_llm:
            self.decisor.supervisionar(decisao)
        self._fila.put(("final", decisao))

    def _decidir_agora(self, bloco: int, janela: tuple[float, float]) -> None:
        plano = self._planejar(bloco, janela)
        decisao = self.decisor.avaliar(plano, self.gov)
        if decisao is None:
            return
        self._aplicar(("regra", decisao.bloco, decisao.intervencoes_para(decisao.recomendada, self.decisor.opcoes, "regra")))
        if self.op.usar_llm:
            self.decisor.supervisionar(decisao)
        self._aplicar(("final", decisao))

    def _recolher(self) -> None:
        while True:
            try:
                item = self._fila.get_nowait()
            except queue.Empty:
                break
            self._aplicar(item)
        if self._futuro is not None and self._futuro.done():
            erro = self._futuro.exception()
            if erro is not None:
                log.error("Erro na camada agêntica (a corrida segue sem esta decisão): %s: %s", type(erro).__name__, erro)
            self._futuro = None
            if self._pendente is not None:
                bloco, janela = self._pendente
                self._pendente = None
                self._lancar(bloco, janela)

    def _aplicar(self, item: tuple) -> None:
        res = self._resultado
        if item[0] == "regra":
            _, bloco, intervencoes = item
            for chave, interv in intervencoes.items():
                if self.gov.decidida(chave):
                    continue  # já decidido por outra decisão (pedidos que se cruzaram)
                if self.gov.pode_intervir(chave):
                    self.gov.definir(chave, interv)
                    self._dono[chave] = bloco
                    res.aplicacoes["regra"] += 1
                else:
                    res.aplicacoes["atrasadas"] += 1
            return
        decisao: Decisao = item[1]
        aplicada = []
        if decisao.escolhida != decisao.recomendada:  # o Supervisor trocou a alteração da regra
            trocas = decisao.intervencoes_para(decisao.escolhida, self.decisor.opcoes, decisao.origem)
            for chave, interv in trocas.items():
                if self._dono.get(chave) == decisao.bloco and self.gov.pode_intervir(chave):
                    self.gov.definir(chave, interv)
                    res.aplicacoes["trocas_llm"] += 1
        for inst in decisao.instancias:
            interv = self.gov.intervencoes.get(inst["chave"])
            aplicada.append({"chave": inst["chave"], "opcao": interv.opcao if interv else None,
                             "a_tempo": self._dono.get(inst["chave"]) == decisao.bloco})
        entrada = decisao.para_entrada()
        entrada["aplicacao"] = aplicada
        entrada["t_aplicada"] = round(self.estado["t"], 2)
        res.decisoes.append(entrada)
        origem = "LLM" if decisao.origem == "llm" else "regra"
        self.imprimir(f"[{self.estado['t']:6.1f} s] decisão do bloco {decisao.bloco:02d}: "
                      f"{NOMES_OPCAO.get(decisao.escolhida, decisao.escolhida)} ({origem}, {decisao.latencia_s:.1f} s) - "
                      f"{decisao.texto}")
        if self.ao_decisao is not None:
            self.ao_decisao(entrada)

    # ------------------------------------------------------------------ corrida
    def rodar(self) -> ResultadoConducao:
        op, gov, sim = self.op, self.gov, self.sim
        res = ResultadoConducao(modo=gov.modo, simulador=getattr(sim, "nome", "?"), voltas=op.voltas)
        self._resultado = res
        dt = float(sim.dt)
        dur = float(op.duracao_bloco_s)
        tempo_max = op.tempo_max_s or 300.0 * op.voltas
        medidor = MedidorConsumo(self.caminhao, self.consumo)
        relogio = Relogio(op.fator_tempo, op.sem_espera)
        inicio = time.monotonic()
        lote: list[dict[str, Any]] = []
        proximo_lote = op.lote_s
        bloco = 0
        parado_desde: float | None = None
        try:
            if self.decisor is not None:  # antes da largada: os dois primeiros blocos (a simulação espera)
                self._decidir_agora(0, janela_inicial(0.0, dur))
            relogio.ancorar(0.0)
            while True:
                self._recolher()
                v_des = gov.velocidade_desejada(self.estado["d"], self.estado["v"], self.estado["a"], dt)
                linha = sim.passo(v_des)
                t, d, v, a = float(linha["sim_time"]), float(linha["d"]), float(linha["speed_mps"]), float(linha["acc_long"])
                self.estado = {"t": t, "d": d, "v": v, "a": a}
                litros = medidor.passo(v, a, dt, float(linha.get("rampa", 0.0) or 0.0))
                ultimo = gov.ultimo
                linha.update({
                    "t": round(t, 3),
                    "speed_kmh": v * 3.6,
                    "v_desejada_kmh": v_des * 3.6,
                    "v_limite_kmh": 0.0 if ultimo.get("parado") else self.plano.limite_kmh(d),
                    "v_alvo_kmh": ultimo.get("v_alvo_kmh"),
                    "modo_tick": ultimo.get("modo_tick"),
                    "limitado_por": ultimo.get("limitado_por"),
                    "modo": gov.modo,
                    "bloco": int(t // dur),
                    "volta": min(int(d // self.plano.comprimento), op.voltas - 1),
                    "combustivel_ml": 1000.0 * litros,
                    "custo_r": litros * self.consumo.preco_r_l,
                    "co2_g": 1000.0 * litros * self.consumo.co2_kg_l,
                })
                res.linhas.append(linha)
                lote.append(linha)
                if t >= proximo_lote - 1e-9:
                    self._mandar(lote)
                    lote = []
                    proximo_lote += op.lote_s
                k = int(t // dur)
                while bloco < k:  # entrou num bloco novo: pede a decisão do bloco seguinte
                    bloco += 1
                    self._pedir(bloco + 1, janela_do_bloco(0.0, bloco, dur))
                    if not op.assincrona:
                        relogio.ancorar(t)  # a espera pelos agentes não conta como atraso
                if gov.terminou and v <= 0.05:
                    res.motivo_fim = "chegada"
                    break
                if v_des > 1.0 and v < 0.1:
                    parado_desde = t if parado_desde is None else parado_desde
                    if t - parado_desde > op.parado_max_s:
                        res.motivo_fim = "preso"
                        log.error("O caminhão ficou %.0f s parado pedindo %.0f km/h; corrida encerrada.",
                                  t - parado_desde, v_des * 3.6)
                        break
                else:
                    parado_desde = None
                if t > tempo_max:
                    res.motivo_fim = "tempo_max"
                    log.error("A corrida passou de %.0f s sem chegar; encerrada.", tempo_max)
                    break
                relogio.esperar_ate(t)
        except KeyboardInterrupt:
            res.motivo_fim = "interrompido"
            self.imprimir("Corrida interrompida (Ctrl+C).")
        finally:
            if lote:
                self._mandar(lote)
            self._encerrar_agentes()
        res.duracao_real_s = round(time.monotonic() - inicio, 3)
        res.atrasos_relogio = relogio.atrasos
        return res

    def _mandar(self, lote: list[dict[str, Any]]) -> None:
        if self.ao_lote is not None:
            try:
                self.ao_lote(lote)
            except Exception as erro:  # o painel é só para ver: a corrida continua
                log.warning("Falha ao mandar a telemetria: %s", erro)

    def _encerrar_agentes(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        try:  # decisões que ficaram prontas no fim: vão para o log, mas não mudam mais nada
            while True:
                item = self._fila.get_nowait()
                if item[0] == "final":
                    self._aplicar(item)
        except queue.Empty:
            pass


__all__ = ["Conducao", "OpcoesConducao", "Relogio", "ResultadoConducao", "Simulador", "metricas_da_corrida"]
