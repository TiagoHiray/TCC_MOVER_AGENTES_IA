"""Decisão da camada agêntica sobre o plano que o caminhão manda para os próximos 10 s.

O caminhão autônomo (X) decide antes de agir: a cada bloco, ele manda à camada agêntica o plano do
bloco seguinte, montado a partir do estado atual da corrida (mover.autonomo.planejador). Então:
1. Especialista de jerk: aponta as decisões bruscas do plano (frenagens e arrancadas bruscas, com a
   zona que causa cada uma);
2. Especialista de ML (mover.agentes.suavizador_ml): propõe quanto suavizar cada uma delas
   (desaceleração antecipada, aceleração da retomada e teto), uma alteração leve do plano;
3. Especialista de consumo: simula, com o modelo do caminhão, o plano sem alteração, a proposta do
   ML e as opções fixas de suavização, e mede combustível, tempo, desacelerações e eventos de jerk;
4. Regra: sem decisão brusca no plano, não altera nada. Com decisão brusca, fica a proposta do ML
   se ela passar na conferência (sem mais eventos bruscos que as opções fixas e sem alongar demais
   o trecho); senão, a opção fixa de menor custo;
5. Supervisor (LLM local): aprova ou troca a alteração e escreve o log em linguagem natural, com
   prazo. Se atrasar, falhar ou citar um número sem respaldo, vale a recomendação da regra.

A alteração escolhida vai para o governador, que muda de verdade a velocidade desejada mandada ao
piloto automático: o caminhão executa o plano alterado (Y). O LLM nunca mexe no pedal: só escolhe
entre alterações já simuladas.
"""

from __future__ import annotations

import concurrent.futures as futures
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from mover.agentes.llm import MARCADOR_RASCUNHO, ClienteLLM
from mover.agentes.supervisor import motivo_rejeicao, numeros_por_unidade
from mover.agentes.textos import num
from mover.autonomo.governador import NOMES_OPCAO, OPCOES_PADRAO, Governador, Intervencao
from mover.autonomo.missao import Instancia, PlanoMissao
from mover.autonomo.planejador import PlanejadorCaminhao, PlanoFuturo, ate_distancia
from mover.autonomo.previsao import ModeloInterno, Trajetoria

log = logging.getLogger("mover.agentes")

NOMES_TIPO = {"lombada": "lombada", "faixa": "faixa de pedestre", "curva": "curva", "parada": "parada",
              "chegada": "chegada", "partida": "partida", "portaria": "portaria", "zona": "zona"}

SISTEMA_DECISAO = (
    "Você é o Supervisor da camada agêntica do MOVER, o gêmeo digital de um caminhão autônomo que dá a "
    "volta no campus do IMT. O caminhão decide antes de agir: a cada bloco, ele manda o plano dos próximos "
    "10 s (a velocidade e a aceleração que vai aplicar). Esse plano segue o plano de missão em degrau e tem "
    "frenagens e arrancadas bruscas perto das zonas. Um modelo de ML propôs quanto suavizar cada uma, e o "
    "modelo do caminhão simulou essa proposta e as opções fixas. Você aprova a alteração recomendada ou "
    "escolhe outra da lista e escreve o log para o operador.\n"
    "Regras:\n"
    "- Escolha exatamente uma opção da lista (use o nome entre aspas).\n"
    "- Prefira opções sem eventos bruscos e de menor custo; a recomendação da regra é um bom ponto de partida. "
    "Opções com mais eventos bruscos que a da regra são recusadas.\n"
    "- O rascunho em JSON traz a opção da regra e um texto-modelo: melhore o texto ou troque a opção se tiver motivo.\n"
    "- Texto em português do Brasil, até {max_palavras} palavras: o que o plano do caminhão faria nesse trecho e o "
    "efeito previsto da alteração escolhida, com números da lista.\n"
    "- Não invente números. Velocidades sem casas decimais; acelerações e jerk com uma casa.\n"
    '- Responda somente com JSON no formato {{"opcao": "...", "texto": "..."}}.'
)


class RespostaDecisao(BaseModel):
    opcao: str = Field(description="nome de uma das opções listadas")
    texto: str = Field(description="log da decisão para o operador, em português")


@dataclass
class OpcaoAvaliada:
    opcao: str
    combustivel_ml: float
    tempo_s: float
    eventos_atencao: int
    eventos_criticos: int
    desacel_max_mps2: float
    acel_max_mps2: float
    jerk_max_mps3: float
    custo_r: float
    economia_ml: float = 0.0
    economia_pct: float = 0.0
    tempo_extra_s: float = 0.0
    tempo_extra_pct: float = 0.0

    @property
    def eventos(self) -> int:
        return self.eventos_atencao + self.eventos_criticos


@dataclass
class Decisao:
    bloco: int
    t_pedido: float
    janela: tuple[float, float]
    estado: dict[str, float]
    instancias: list[dict[str, Any]]
    brutas: list[dict[str, Any]]            # as decisões bruscas do plano do caminhão
    opcoes: list[OpcaoAvaliada]
    recomendada: str
    escolhida: str
    origem: str                             # llm | regra
    texto: str
    motivo: str | None = None               # por que a regra valeu (LLM fora, atrasado ou recusado)
    latencia_s: float = 0.0                 # total: especialistas + regra + LLM
    latencia_llm_s: float = 0.0
    provedor: str | None = None
    propostas_ml: dict[str, dict[str, Any]] = field(default_factory=dict)   # elemento -> suavização proposta pelo ML
    motivo_ml: str | None = None            # por que a proposta do ML não valeu
    modelo_ml: str | None = None
    plano: dict[str, Any] = field(default_factory=dict)                   # o plano que o caminhão mandou (resumo)
    curvas: dict[str, Any] = field(default_factory=dict)
    trajetorias: dict[str, Trajetoria] = field(default_factory=dict, repr=False, compare=False)

    @property
    def opcao_escolhida(self) -> OpcaoAvaliada:
        return next(o for o in self.opcoes if o.opcao == self.escolhida)

    @property
    def opcao_bruta(self) -> OpcaoAvaliada:
        return next(o for o in self.opcoes if o.opcao == "nenhuma")

    @property
    def opcao_recomendada(self) -> OpcaoAvaliada:
        return next(o for o in self.opcoes if o.opcao == self.recomendada)

    def atualizar_curvas(self) -> None:
        """Curvas (5 Hz) do plano do caminhão e do plano alterado, para o gráfico da página."""
        if self.trajetorias:
            self.curvas = {"bruto": self.trajetorias["nenhuma"].para_json(),
                           "escolhido": self.trajetorias[self.escolhida].para_json(),
                           "fonte_bruto": "plano"}

    def intervencoes_para(self, opcao: str, opcoes: dict[str, Any] | None = None, origem: str | None = None) -> dict[str, Intervencao]:
        """A alteração de cada elemento da decisão para a opção dada (a do ML é uma por elemento)."""
        origem = origem or self.origem
        if opcao == "ml":
            saida = {}
            for inst in self.instancias:
                p = self.propostas_ml.get(inst["chave"])
                saida[inst["chave"]] = (Intervencao("ml", float(p["desacel_mps2"]), float(p["acel_mps2"]), p.get("teto_kmh"), origem)
                                        if p else Intervencao.da_opcao("nenhuma", origem=origem))
            return saida
        interv = Intervencao.da_opcao(opcao, opcoes, origem=origem)
        return {inst["chave"]: interv for inst in self.instancias}

    def para_entrada(self) -> dict[str, Any]:
        """Entrada do log (tipo "intervencao") com os números da decisão."""
        escolhida, bruta = self.opcao_escolhida, self.opcao_bruta
        nivel = "info"
        if any(b.get("nivel") == "critico" for b in self.brutas):
            nivel = "critico"
        elif self.brutas:
            nivel = "atencao"
        return {
            "tipo": "intervencao",
            "bloco": self.bloco,
            "t_ini": round(self.janela[0], 2),
            "t_fim": round(self.janela[1], 2),
            "t_pico": round(self.brutas[0]["t_s"], 2) if self.brutas else None,
            "t_pedido": round(self.t_pedido, 2),
            "nivel": nivel,
            "causa": None,
            "evento": self.brutas[0]["tipo"] if self.brutas else None,
            "texto": self.texto,
            "diagnostico": None,
            "acao": NOMES_OPCAO.get(self.escolhida, self.escolhida),
            "opcao": self.escolhida,
            "recomendada": self.recomendada,
            "texto_origem": self.origem,
            "autor": "supervisor",
            "provedor": self.provedor if self.origem == "llm" else None,
            "latencia_s": round(self.latencia_s, 3),
            "latencia_llm_s": round(self.latencia_llm_s, 3),
            "observacao": self.motivo,
            "fonte": "plano",
            "plano": self.plano,
            "ml": {"modelo": self.modelo_ml, "propostas": self.propostas_ml, "motivo": self.motivo_ml},
            "instancias": self.instancias,
            "brutas": self.brutas,
            "opcoes": [asdict(o) for o in self.opcoes],
            "curvas": self.curvas,
            "fatos": {
                "combustivel_bruto_ml": round(bruta.combustivel_ml, 1),
                "combustivel_escolhido_ml": round(escolhida.combustivel_ml, 1),
                "economia_prevista_ml": round(escolhida.economia_ml, 1),
                "economia_prevista_pct": round(escolhida.economia_pct, 1),
                "tempo_extra_s": round(escolhida.tempo_extra_s, 1),
                "eventos_brutos": bruta.eventos,
                "eventos_escolhidos": escolhida.eventos,
                "desacel_bruta_mps2": round(bruta.desacel_max_mps2, 1),
                "desacel_escolhida_mps2": round(escolhida.desacel_max_mps2, 1),
            },
        }


class Decisor:
    def __init__(self, plano: PlanoMissao, modelo: ModeloInterno, cfg: dict[str, Any] | None = None,
                 llm: ClienteLLM | None = None, opcoes: dict[str, Any] | None = None, n_voltas: int | None = None,
                 suavizador: Any = None):
        cfg = cfg or {}
        self.plano = plano
        self.modelo = modelo
        self.llm = llm
        self.suavizador = suavizador
        self.opcoes = {k: tuple(v) for k, v in (opcoes or OPCOES_PADRAO).items()}
        self.n_voltas = n_voltas
        self.prazo_s = float(cfg.get("prazo_llm_s", 6.0))
        self.valor_tempo_r_s = float(cfg.get("valor_tempo_r_h", 30.0)) / 3600.0
        self.max_tempo_extra_pct = float(cfg.get("max_tempo_extra_pct", 60.0))
        self.max_palavras = int(cfg.get("max_palavras", 45))
        self.duracao_bloco = float(cfg.get("horizonte_s", 10.0))
        self.margem_m = float(cfg.get("margem_elementos_m", 20.0))
        self._executor = futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm-decisao")
        self.chamadas_llm = 0

    @property
    def nome_llm(self) -> str | None:
        return getattr(self.llm, "nome", None)

    @property
    def nome_ml(self) -> str | None:
        return getattr(self.suavizador, "nome", None) if self.suavizador is not None else None

    def planejador(self) -> PlanejadorCaminhao:
        """O planejador do caminhão com os mesmos parâmetros (quem monta o plano é o lado do caminhão)."""
        return PlanejadorCaminhao(self.plano, self.modelo, self.n_voltas, self.margem_m)

    def encerrar(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ decisão
    def decidir(self, plano: PlanoFuturo | None, governador: Governador, usar_llm: bool = True) -> Decisao | None:
        """Decide as alterações do plano que o caminhão mandou. None se não houver o que decidir."""
        decisao = self.avaliar(plano, governador)
        if decisao is not None and usar_llm:
            self.supervisionar(decisao)
        return decisao

    def avaliar(self, plano: PlanoFuturo | None, governador: Governador) -> Decisao | None:
        """Etapa rápida (~0,3 s): especialistas e regra. A recomendação já pode ser aplicada.

        O governador recebido não é alterado (as simulações usam clones).
        """
        if plano is None or not plano.pendentes:
            return None
        inicio = time.monotonic()
        t0, d0, v0, a0 = (float(plano.estado[k]) for k in ("t", "d", "v", "a"))
        pendentes = plano.pendentes
        trajetorias: dict[str, Trajetoria] = {"nenhuma": plano.trajetoria}
        avaliadas = [self._avaliar("nenhuma", plano.trajetoria)]
        for opcao in self.opcoes:
            interv = Intervencao.da_opcao(opcao, self.opcoes)
            trajetorias[opcao] = self._simular(governador, plano, {i.chave: interv for i in pendentes})
            avaliadas.append(self._avaliar(opcao, trajetorias[opcao]))
        propostas = self._propostas_ml(plano)
        if propostas:
            alteracoes = {i.chave: (Intervencao("ml", p["desacel_mps2"], p["acel_mps2"], p["teto_kmh"], "regra")
                                    if (p := propostas.get(i.chave)) else Intervencao.da_opcao("nenhuma"))
                          for i in pendentes}
            trajetorias["ml"] = self._simular(governador, plano, alteracoes)
            avaliadas.append(self._avaliar("ml", trajetorias["ml"]))
        bruta = avaliadas[0]
        for o in avaliadas:
            o.economia_ml = bruta.combustivel_ml - o.combustivel_ml
            o.economia_pct = 100.0 * o.economia_ml / bruta.combustivel_ml if bruta.combustivel_ml > 0 else 0.0
            o.tempo_extra_s = o.tempo_s - bruta.tempo_s
            o.tempo_extra_pct = 100.0 * o.tempo_extra_s / bruta.tempo_s if bruta.tempo_s > 0 else 0.0
        recomendada, motivo_ml = self._recomendar(avaliadas, plano.decisoes_bruscas)
        instancias = [self._descrever(i, d0, v0, plano.projecao) for i in pendentes]
        decisao = Decisao(bloco=plano.bloco, t_pedido=t0, janela=plano.janela, estado={"t": t0, "d": d0, "v": v0, "a": a0},
                          instancias=instancias, brutas=list(plano.decisoes_bruscas), opcoes=avaliadas,
                          recomendada=recomendada, escolhida=recomendada, origem="regra", texto="", provedor=self.nome_llm,
                          propostas_ml=propostas, motivo_ml=motivo_ml, modelo_ml=self.nome_ml, plano=plano.resumo())
        decisao.texto = self.texto_modelo(decisao)
        decisao.trajetorias = trajetorias
        decisao.atualizar_curvas()
        decisao.latencia_s = time.monotonic() - inicio
        return decisao

    def supervisionar(self, decisao: Decisao) -> Decisao:
        """Etapa do Supervisor (LLM, com prazo): pode trocar a opção e reescrever o log.

        Sem LLM, ou sem decisão brusca no plano, a decisão da regra fica como está.
        """
        if self.llm is None or not decisao.brutas:
            return decisao
        inicio = time.monotonic()
        self._perguntar_llm(decisao)
        if decisao.origem == "llm":
            decisao.atualizar_curvas()
        decisao.latencia_llm_s = time.monotonic() - inicio
        decisao.latencia_s += decisao.latencia_llm_s
        return decisao

    # ------------------------------------------------------------------ especialistas
    def _simular(self, governador: Governador, plano: PlanoFuturo, alteracoes: dict[str, Intervencao]) -> Trajetoria:
        gov = governador.clonar()
        for chave, interv in alteracoes.items():
            gov.definir(chave, interv)
        e = plano.estado
        return ate_distancia(self.modelo, gov, e["d"], e["v"], e["a"], e["t"], plano.d_fim)

    def _avaliar(self, opcao: str, traj: Trajetoria) -> OpcaoAvaliada:
        m = self.modelo.metricas(traj)
        tempo = m["tempo_s"]
        ml = 1000.0 * m["combustivel_l"]
        custo = m["custo_r"] + self.valor_tempo_r_s * tempo
        return OpcaoAvaliada(opcao=opcao, combustivel_ml=ml, tempo_s=tempo, eventos_atencao=m["eventos_atencao"],
                             eventos_criticos=m["eventos_criticos"], desacel_max_mps2=-min(0.0, m["acel_min_mps2"]),
                             acel_max_mps2=max(0.0, m["acel_max_mps2"]), jerk_max_mps3=m["jerk_max_abs_mps3"], custo_r=custo)

    def _propostas_ml(self, plano: PlanoFuturo) -> dict[str, dict[str, Any]]:
        """Especialista de ML: para cada elemento com decisão brusca no plano, quanto suavizar."""
        if self.suavizador is None or not plano.decisoes_bruscas:
            return {}
        from mover.agentes.suavizador_ml import caracteristicas

        por_elemento: dict[str, list[dict[str, Any]]] = {}
        for b in plano.decisoes_bruscas:
            por_elemento.setdefault(b.get("elemento") or "", []).append(b)
        d0, v0 = float(plano.estado["d"]), float(plano.estado["v"])
        cruzeiro = float(self.plano.cruzeiro_kmh)
        adiante = self.plano.instancias(max(0.0, d0 - 1.0), plano.d_fim + 250.0, self.n_voltas)
        propostas = {}
        for inst in plano.pendentes:
            brutas = por_elemento.get(inst.chave)
            if not brutas:
                continue
            seguinte = next((i for i in adiante if i.d_ini > inst.d_fim + 1e-6 and i.tipo != "partida"), None)
            partida = inst.tipo == "partida"
            x = caracteristicas(
                v_atual_kmh=v0 * 3.6, v_alvo_kmh=cruzeiro if partida else inst.v_kmh, cruzeiro_kmh=cruzeiro,
                distancia_m=inst.d_ini - d0, comprimento_m=inst.d_fim - inst.d_ini, duracao_s=inst.duracao_s,
                parada=inst.e_parada, gap_seguinte_m=(seguinte.d_ini - inst.d_fim) if seguinte else None,
                v_alvo_seguinte_kmh=seguinte.v_kmh if seguinte else None,
                desacel_plano_mps2=min((b["acel_mps2"] for b in brutas if b["tipo"] == "frenagem_brusca"), default=0.0),
                acel_plano_mps2=max((b["acel_mps2"] for b in brutas if b["tipo"] == "arrancada_brusca"), default=0.0))
            try:
                desacel, acel, teto = self.suavizador.propor(x)
            except Exception as erro:  # modelo quebrado: os agentes seguem com as opções fixas
                log.warning("Especialista de ML sem proposta (%s: %s).", type(erro).__name__, erro)
                return {}
            propostas[inst.chave] = {"nome": inst.nome, "desacel_mps2": float(desacel), "acel_mps2": float(acel),
                                     "teto_kmh": float(teto) if teto else None}
        return propostas

    def _recomendar(self, avaliadas: list[OpcaoAvaliada], brutas: list[dict[str, Any]]) -> tuple[str, str | None]:
        """Sem decisão brusca no plano, nenhuma alteração. Com, a proposta do ML se passar na conferência;
        senão, a opção fixa de menor custo entre as de menos trancos."""
        if not brutas:
            return "nenhuma", None
        fixas = [o for o in avaliadas if o.opcao not in ("nenhuma", "ml")]
        ml = next((o for o in avaliadas if o.opcao == "ml"), None)
        if not fixas:
            return ("ml", None) if ml is not None else ("nenhuma", None)
        ativas = [o for o in fixas if o.tempo_extra_pct <= self.max_tempo_extra_pct] or fixas
        menor = min(o.eventos for o in ativas)
        melhor_fixa = min([o for o in ativas if o.eventos == menor], key=lambda o: o.custo_r).opcao
        if ml is None:
            return melhor_fixa, None
        nome = NOMES_OPCAO.get(melhor_fixa, melhor_fixa)
        if ml.eventos > menor:
            return melhor_fixa, (f"a proposta do ML teria {ml.eventos} evento(s) brusco(s) no modelo do caminhão; "
                                 f"valeu a opção {nome}")
        if ml.tempo_extra_pct > self.max_tempo_extra_pct:
            return melhor_fixa, f"a proposta do ML alongaria demais o trecho; valeu a opção {nome}"
        return "ml", None

    def _descrever(self, inst: Instancia, d0: float, v0: float, projecao: Trajetoria) -> dict[str, Any]:
        distancia = inst.d_ini - d0
        if len(projecao.d) and projecao.d[-1] >= inst.d_ini:
            tempo_ate = float(np.interp(inst.d_ini, projecao.d, projecao.t)) - float(projecao.t[0]) + self.modelo.dt
        else:
            tempo_ate = distancia / max(v0, 1.0)
        return {"chave": inst.chave, "tipo": inst.tipo, "nome": inst.nome, "d_ini": round(inst.d_ini, 1),
                "d_fim": round(inst.d_fim, 1), "v_max_kmh": inst.v_kmh, "duracao_s": inst.duracao_s,
                "distancia_m": round(max(0.0, distancia), 0), "tempo_ate_s": round(max(0.0, tempo_ate), 1)}

    # ------------------------------------------------------------------ supervisor (LLM)
    def fatos(self, decisao: Decisao) -> dict[str, Any]:
        """Os números que o texto pode citar, com a unidade no sufixo da chave."""
        return {
            "t_atual_s": decisao.t_pedido,
            "v_atual_kmh": decisao.estado["v"] * 3.6,
            "elementos": [{"distancia_m": i["distancia_m"], "tempo_ate_s": i["tempo_ate_s"], "v_max_kmh": i["v_max_kmh"],
                           "duracao_s": i["duracao_s"]} for i in decisao.instancias],
            "brutas": [{"em_s": b["em_s"], "acel_mps2": b["acel_mps2"], "jerk_mps3": b["jerk_mps3"], "vel_kmh": b["vel_kmh"]}
                       for b in decisao.brutas],
            "manobras": [{"distancia_m": m.get("distancia_m") or 0, "em_s": m.get("em_s") or 0,
                          "angulo_graus": m.get("angulo_graus") or 0} for m in (decisao.plano.get("manobras") or [])],
            "opcoes": [{"combustivel_ml": o.combustivel_ml, "tempo_s": o.tempo_s, "eventos": o.eventos,
                        "desacel_mps2": o.desacel_max_mps2, "acel_mps2": o.acel_max_mps2, "economia_ml": o.economia_ml,
                        "economia_pct": o.economia_pct, "tempo_extra_s": o.tempo_extra_s, "custo_r": o.custo_r}
                       for o in decisao.opcoes],
            "opcoes_parametros": [{"desacel_mps2": v[0], "acel_mps2": v[1], "teto_kmh": v[2] or 0} for v in self.opcoes.values()],
            "ml": [{"desacel_mps2": p["desacel_mps2"], "acel_mps2": p["acel_mps2"], "teto_kmh": p.get("teto_kmh") or 0}
                   for p in decisao.propostas_ml.values()],
            "contagens": [len(decisao.instancias), len(decisao.brutas), 1, 2, 10],
        }

    @staticmethod
    def _texto_proposta(p: dict[str, Any]) -> str:
        teto = f", até {num(p['teto_kmh'], 0)} km/h" if p.get("teto_kmh") else ""
        return f"frear com até {num(p['desacel_mps2'])} m/s² e retomar com até {num(p['acel_mps2'])} m/s²{teto}"

    def prompt(self, decisao: Decisao) -> str:
        linhas = [f"Instante: {num(decisao.t_pedido)} s (bloco {decisao.bloco}). Velocidade atual: "
                  f"{num(decisao.estado['v'] * 3.6, 0)} km/h.",
                  f"Plano que o caminhão mandou para o próximo bloco ({num(decisao.janela[0], 0)}–{num(decisao.janela[1], 0)} s), "
                  "ainda não executado. Elementos do plano de missão:"]
        for i in decisao.instancias:
            nome = i["nome"] if i["tipo"] not in ("parada", "chegada") else f"{i['nome']} (parada)"
            limite = (f"parar por {num(i['duracao_s'], 0)} s" if i["tipo"] in ("parada",) else
                      "parar no fim da corrida" if i["tipo"] == "chegada" else
                      "sair parado" if i["tipo"] == "partida" else f"limite {num(i['v_max_kmh'], 0)} km/h")
            linhas.append(f"- {nome}: em {num(i['distancia_m'], 0)} m (~{num(i['tempo_ate_s'])} s), {limite}")
        manobras = decisao.plano.get("manobras") or []
        if manobras:
            linhas.append("Manobras do trajeto: " + "; ".join(
                f"{m['tipo']} em {num(m.get('distancia_m') or 0, 0)} m" + (f" ({num(m['angulo_graus'], 0)}°)" if m.get("angulo_graus") else "")
                for m in manobras[:4]))
        linhas.append("Decisões bruscas do plano:" if decisao.brutas else "Decisões bruscas do plano: nenhuma.")
        for b in decisao.brutas:
            tipo = "frenagem brusca" if b["tipo"] == "frenagem_brusca" else "arrancada brusca"
            alvo = f" ({b['nome_elemento']})" if b.get("nome_elemento") else ""
            linhas.append(f"- {tipo}{alvo} em {num(b['em_s'])} s: {num(b['acel_mps2'])} m/s², jerk {num(b['jerk_mps3'])} m/s³")
        if decisao.propostas_ml:
            linhas.append("Proposta do modelo de ML: " + "; ".join(f"{p.get('nome') or chave}: {self._texto_proposta(p)}"
                                                                    for chave, p in decisao.propostas_ml.items()))
        linhas.append("Alterações simuladas pelo modelo do caminhão (mesmo trecho):")
        for o in decisao.opcoes:
            parametros = ""
            if o.opcao in self.opcoes:
                dec, acel, teto = (list(self.opcoes[o.opcao]) + [None])[:3]
                parametros = f" [frear com até {num(dec)} m/s², retomar com até {num(acel)} m/s²" + \
                             (f", teto {num(teto, 0)} km/h" if teto else "") + "]"
            elif o.opcao == "ml":
                parametros = " [proposta do ML]"
            elif o.opcao == "nenhuma":
                parametros = " [plano sem alteração]"
            linhas.append(f"- \"{o.opcao}\"{parametros}: combustível {num(o.combustivel_ml, 0)} mL "
                          f"({num(o.economia_ml, 0)} mL a menos), tempo {num(o.tempo_s)} s "
                          f"({'+' if o.tempo_extra_s >= 0 else ''}{num(o.tempo_extra_s)} s), {o.eventos} eventos bruscos, "
                          f"desaceleração máxima {num(o.desacel_max_mps2)} m/s²")
        if decisao.recomendada == "ml":
            linhas.append("Recomendação da regra: \"ml\" (a proposta do ML passou na conferência do modelo do caminhão).")
        else:
            motivo = f"; {decisao.motivo_ml}" if decisao.motivo_ml else ""
            linhas.append(f"Recomendação da regra: \"{decisao.recomendada}\" (menor custo de combustível e tempo sem eventos "
                          f"bruscos{motivo}).")
        rascunho = {"opcao": decisao.recomendada, "texto": self.texto_modelo(decisao)}
        linhas.append(f"{MARCADOR_RASCUNHO} {json.dumps(rascunho, ensure_ascii=False)}")
        return "\n".join(linhas)

    def _perguntar_llm(self, decisao: Decisao) -> None:
        sistema = SISTEMA_DECISAO.format(max_palavras=self.max_palavras)
        usuario = self.prompt(decisao)
        self.chamadas_llm += 1
        try:
            futuro = self._executor.submit(self.llm.gerar, sistema, usuario, RespostaDecisao)
            resposta = futuro.result(timeout=self.prazo_s)
        except futures.TimeoutError:
            decisao.motivo = f"o LLM não respondeu em {num(self.prazo_s)} s; valeu a regra"
            return
        except Exception as erro:  # rede, modelo ausente, JSON fora do esquema...
            decisao.motivo = f"erro do LLM ({type(erro).__name__}); valeu a regra"
            log.warning("Decisão sem LLM: %s: %s", type(erro).__name__, erro)
            return
        opcao = str(resposta.opcao).strip().strip('"').lower().replace("ô", "o")
        texto = " ".join(str(resposta.texto).split())
        validas = {o.opcao for o in decisao.opcoes}
        if opcao not in validas:
            decisao.motivo = f"o LLM escolheu uma opção que não existe ({opcao[:30]}); valeu a regra"
            return
        # guarda de segurança: o LLM escolhe só entre opções tão seguras quanto a da regra
        candidata = next(o for o in decisao.opcoes if o.opcao == opcao)
        regra = decisao.opcao_recomendada
        if candidata.eventos > regra.eventos:
            decisao.motivo = (f"o LLM escolheu \"{opcao}\", com {candidata.eventos} evento(s) brusco(s) contra "
                              f"{regra.eventos} da regra; valeu a regra")
            return
        if candidata.tempo_extra_pct > self.max_tempo_extra_pct >= regra.tempo_extra_pct:
            decisao.motivo = f"o LLM escolheu \"{opcao}\", que alonga demais o trecho; valeu a regra"
            return
        permitidos = numeros_por_unidade(self.fatos(decisao))
        motivo = motivo_rejeicao(texto, permitidos, self.max_palavras)
        if motivo:
            decisao.motivo = f"texto do LLM recusado ({motivo}); valeu a regra"
            return
        decisao.escolhida, decisao.origem, decisao.texto = opcao, "llm", texto

    # ------------------------------------------------------------------ texto-modelo
    def texto_modelo(self, decisao: Decisao) -> str:
        escolhida = decisao.opcao_escolhida
        if not decisao.brutas:
            nomes = ", ".join(i["nome"] for i in decisao.instancias[:3])
            return f"Próximo bloco ({nomes}): o plano do caminhão não tem decisões bruscas; sem alteração."
        b = decisao.brutas[0]
        tipo = "frenagem brusca" if b["tipo"] == "frenagem_brusca" else "arrancada brusca"
        alvo = f" para a {b['nome_elemento']}" if b.get("nome_elemento") and b["tipo"] == "frenagem_brusca" else (
            f" depois da {b['nome_elemento']}" if b.get("nome_elemento") else "")
        extra = f" e mais {len(decisao.brutas) - 1} evento(s)" if len(decisao.brutas) > 1 else ""
        inicio = (f"O caminhão planeja {tipo}{alvo} em {num(b['em_s'])} s ({num(b['acel_mps2'])} m/s², jerk "
                  f"{num(b['jerk_mps3'])} m/s³){extra}.")
        if escolhida.opcao == "nenhuma":
            return f"{inicio} Mantido sem alteração."
        efeito = (f"{escolhida.eventos} evento(s) bruscos e {num(escolhida.economia_ml, 0)} mL a menos "
                  f"({'+' if escolhida.tempo_extra_s >= 0 else ''}{num(escolhida.tempo_extra_s)} s)")
        if escolhida.opcao == "ml" and decisao.propostas_ml:
            p = decisao.propostas_ml.get(b.get("elemento") or "") or next(iter(decisao.propostas_ml.values()))
            return f"{inicio} O ML suavizou: {self._texto_proposta(p)}; {efeito}."
        return (f"{inicio} Alteração {NOMES_OPCAO.get(escolhida.opcao, escolhida.opcao)}: desaceleração máxima de "
                f"{num(escolhida.desacel_max_mps2)} m/s², {efeito}.")


def janela_do_bloco(t0: float, bloco: int, duracao: float) -> tuple[float, float]:
    """Janela analisada quando o caminhão entra no bloco `bloco`: o bloco seguinte."""
    return t0 + (bloco + 1) * duracao, t0 + (bloco + 2) * duracao


def janela_inicial(t0: float, duracao: float) -> tuple[float, float]:
    """Antes da largada: os dois primeiros blocos."""
    return t0, t0 + 2 * duracao


def resumo_opcoes(decisao: Decisao) -> str:
    return "; ".join(f"{o.opcao}: {o.combustivel_ml:.0f} mL, {o.tempo_s:.1f} s, {o.eventos} ev" for o in decisao.opcoes)


__all__ = ["Decisao", "Decisor", "OpcaoAvaliada", "RespostaDecisao", "janela_do_bloco", "janela_inicial", "resumo_opcoes"]
