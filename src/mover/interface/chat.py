"""Chat sobre o log (painel 4): perguntas do operador respondidas a partir das entradas da camada agêntica.

Contexto enviado ao LLM (o mesmo provedor plugável dos agentes: .env > config/agentes.yaml):
- as entradas publicadas até agora: intervenções (decisões dos agentes para o trecho seguinte, com
  a economia prevista), logs dos trechos percorridos e problemas medidos (eventos de jerk);
- as métricas ao vivo: combustível, custo, CO2 e a comparação com o caminhão sem agentes.
Para caber no contexto de modelos pequenos (gemma2:2b), entram todos os problemas, as intervenções
recentes, as entradas do trecho citado na pergunta e as mais recentes, até `chat.max_entradas_contexto`.

A resposta deve ser curta e citar as entradas no formato (#id, tempo). Antes de ir para a tela, uma
guarda local confere se cada número citado aparece no contexto (fatos, tempos, ids, métricas) ou na
pergunta, e se os ids citados existem. Se a guarda recusar, ou se o LLM falhar, vale a resposta
determinística, montada por palavras-chave sobre o log. A guarda não é um revisor: ela só barra
números e citações sem respaldo, não confere o sentido da frase. O provedor "falso" usa sempre a
resposta determinística.
"""

from __future__ import annotations

import copy
import logging
import math
import re
import threading
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, Field

from mover.agentes.fatos import NOMES_NIVEL, ORDEM_NIVEL
from mover.agentes.llm import ClienteLLM, criar_llm
from mover.agentes.supervisor import (NUMEROS_LIVRES, QUALQUER_UNIDADE, Permitidos, descrever_numero, juntar_permitidos,
                                      numero_com_respaldo, numeros_com_unidade, numeros_no_texto, numeros_por_unidade)
from mover.agentes.textos import NOMES_CAUSA, maiuscula, num

log = logging.getLogger("mover.interface")

PADRAO_CHAT: dict[str, Any] = {
    "max_palavras": 70,
    "max_entradas_contexto": 16,
    "rodadas_historico": 3,
    "num_ctx": 4096,
    "timeout_s": 60,
    "validar_numeros": True,
}

SISTEMA_CHAT = (
    "Você é o assistente do operador no projeto MOVER, o gêmeo digital de um caminhão autônomo que dá a volta "
    "no campus do IMT. A camada agêntica prevê as decisões bruscas do piloto automático e intervém (suaviza "
    "frenagens e arrancadas); ela também registra o que aconteceu. Você responde perguntas sobre esse log.\n"
    "Regras:\n"
    "- Português do Brasil, no máximo {max_palavras} palavras.\n"
    "- Use só as entradas do log e as métricas abaixo. Se a resposta não estiver nelas, diga que o log não tem essa informação.\n"
    "- Cite as entradas usadas no formato (#id, tempo), por exemplo (#12, 51,4 s).\n"
    "- Intervenções são decisões para o trecho seguinte: o caminhão manda o plano dele (o que vai fazer nos próximos "
    "10 s, ainda não executado), e o efeito da alteração é uma previsão.\n"
    "- Não invente números, causas nem peças. Velocidades sem casas decimais; acelerações e jerk com uma casa.\n"
    '- Responda somente com JSON no formato {{"resposta": "...", "citacoes": [ids das entradas citadas]}}.'
)


class RespostaChat(BaseModel):
    resposta: str = Field(description="Resposta curta em português, citando as entradas como (#id, tempo)")
    citacoes: list[int] = Field(default_factory=list, description="ids das entradas citadas")


# ---------------------------------------------------------------------------------------------
# Contexto
# ---------------------------------------------------------------------------------------------
@dataclass
class ContextoChat:
    """O que o chat sabe num instante: o log publicado e as métricas da corrida."""

    publicadas: list[dict[str, Any]]            # todas as entradas publicadas
    selecionadas: list[dict[str, Any]]          # as que vão para o prompt do LLM
    metricas: dict[str, Any] = field(default_factory=dict)
    t_atual: float | None = None
    bloco_atual: int | None = None
    duracao_bloco: float = 10.0
    t0: float = 0.0
    encerrada: bool = False
    modo: str = "agentes"
    combustivel: str = "diesel"
    fonte_base: str = "previsto"                # o plano sem agentes, previsto pelo modelo do caminhão

    @property
    def todas(self) -> list[dict[str, Any]]:
        return self.publicadas

    @property
    def intervencoes(self) -> list[dict[str, Any]]:
        return [e for e in self.publicadas if e["tipo"] == "intervencao"]

    def por_id(self) -> dict[int, tuple[dict[str, Any], bool]]:
        """id -> (entrada, é previsão). Intervenções são previsões do trecho seguinte."""
        return {e["id"]: (e, e["tipo"] == "intervencao") for e in self.publicadas}


def sem_acentos(texto: str) -> str:
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii").lower()


def _tem(texto: str, chaves: tuple[str, ...]) -> bool:
    return any(chave in texto for chave in chaves)


_INTERVALO = re.compile(r"entre\s+(\d+(?:[.,]\d+)?)\s*(?:s|seg\w*)?\s+e\s+(\d+(?:[.,]\d+)?)\s*(?:s\b|seg)")
_INSTANTE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:s\b|seg)")
_BLOCO = re.compile(r"bloco\s*(\d+)")


def filtro_de_tempo(pergunta: str, t0: float = 0.0, duracao_bloco: float = 10.0) -> tuple[float, float] | None:
    """Trecho citado na pergunta: 'entre 40 e 60 s', 'aos 51 s' (± 2,5 s) ou 'bloco 5'."""
    texto = sem_acentos(pergunta)
    numero = lambda s: float(s.replace(",", "."))  # noqa: E731
    if m := _INTERVALO.search(texto):
        a, b = sorted((numero(m.group(1)), numero(m.group(2))))
        return a, b
    if m := _BLOCO.search(texto):
        k = int(m.group(1))
        return t0 + k * duracao_bloco, t0 + (k + 1) * duracao_bloco
    if m := _INSTANTE.search(texto):
        t = numero(m.group(1))
        return t - 2.5, t + 2.5
    return None


def no_trecho(entrada: dict[str, Any], trecho: tuple[float, float]) -> bool:
    a, b = trecho
    if entrada["tipo"] == "problema":
        return a - 1.0 <= float(entrada["t_pico"]) <= b + 1.0
    return float(entrada["t_ini"]) < b and float(entrada["t_fim"]) > a


def tempo_da_entrada(entrada: dict[str, Any]) -> str:
    if entrada["tipo"] == "problema":
        return f"{num(entrada['t_pico'])} s"
    if entrada["tipo"] == "intervencao":
        return f"{num(entrada.get('t_pedido') or entrada['t_ini'])} s"
    return f"{num(entrada['t_ini'])}–{num(entrada['t_fim'])} s"


def citacao(entrada: dict[str, Any]) -> str:
    return f"(#{entrada['id']}, {tempo_da_entrada(entrada)})"


def nome_problema(entrada: dict[str, Any], conhecimento: dict[str, Any]) -> str:
    if entrada.get("causa") == "irregularidade_via":
        return "provável lombada"
    return conhecimento.get(entrada.get("evento") or "", {}).get("nome", str(entrada.get("evento", "evento")).replace("_", " "))


def _origem(entrada: dict[str, Any]) -> str:
    return "LLM" if entrada.get("texto_origem") == "llm" else "regra"


def linha_do_contexto(entrada: dict[str, Any], conhecimento: dict[str, Any]) -> str:
    """Uma entrada do log no prompt do chat: cabeçalho, texto publicado e os números."""
    f = entrada.get("fatos") or {}
    nivel = NOMES_NIVEL.get(entrada["nivel"], entrada["nivel"])
    if entrada["tipo"] == "intervencao":
        cabecalho = (f"#{entrada['id']} [decidida aos {num(entrada.get('t_pedido') or entrada['t_ini'])} s para "
                     f"{num(entrada['t_ini'], 0)}–{num(entrada['t_fim'], 0)} s, intervenção {entrada.get('acao')} "
                     f"({_origem(entrada)})]")
        fatos = (f"plano do caminhão: {num(f.get('combustivel_bruto_ml', 0), 0)} mL, {f.get('eventos_brutos', 0)} eventos bruscos, "
                 f"desaceleração até {num(f.get('desacel_bruta_mps2', 0))} m/s²; com a intervenção: "
                 f"{num(f.get('combustivel_escolhido_ml', 0), 0)} mL, {f.get('eventos_escolhidos', 0)} eventos, até "
                 f"{num(f.get('desacel_escolhida_mps2', 0))} m/s²; economia prevista {num(f.get('economia_prevista_ml', 0), 0)} mL "
                 f"({num(f.get('economia_prevista_pct', 0), 0)} %), tempo extra {num(f.get('tempo_extra_s', 0))} s")
    elif entrada["tipo"] == "problema":
        cabecalho = f"#{entrada['id']} [bloco {entrada['bloco']:02d}, pico aos {num(entrada['t_pico'])} s, problema {nivel}]"
        fatos = (f"{nome_problema(entrada, conhecimento)}; jerk máx {num(f.get('jerk_max_abs_mps3', 0))} m/s³; "
                 f"velocidade {num(f.get('vel_kmh', 0), 0)} km/h; causa: {NOMES_CAUSA.get(f.get('causa', ''), f.get('causa', ''))}")
    else:
        cabecalho = (f"#{entrada['id']} [bloco {entrada['bloco']:02d}, {num(entrada['t_ini'])}–{num(entrada['t_fim'])} s, "
                     f"log {nivel}]")
        giro = abs(int(f.get("mudanca_rumo_graus", 0)))
        rumo = "trajeto reto" if f.get("lado", "reto") == "reto" else f"virou {giro}° à {f['lado']}"
        fatos = (f"velocidade média {num(f.get('vel_media_kmh', 0), 0)} km/h (mín {num(f.get('vel_min_kmh', 0), 0)}, "
                 f"máx {num(f.get('vel_max_kmh', 0), 0)}); aceleração longitudinal {num(f.get('acel_long_min_mps2', 0))} a "
                 f"{num(f.get('acel_long_max_mps2', 0))} m/s²; jerk {num(f.get('jerk_long_min_mps3', 0))} a "
                 f"{num(f.get('jerk_long_max_mps3', 0))} m/s³; {rumo}")
    return f"{cabecalho} {entrada['texto']} Fatos: {fatos}."


def fonte_da_linha_de_base(linha: dict[str, Any] | None) -> str:
    """De onde vem o consumo sem agentes (hoje, sempre a previsão do plano sem alterações)."""
    return str((linha or {}).get("fonte_curta") or "previsto")


def quem_compara(fonte_base: str = "previsto") -> str:
    """Quem dá o consumo sem agentes: o plano do caminhão sem alterações, previsto pelo modelo."""
    return "o plano sem agentes previa"


def linha_das_metricas(m: dict[str, Any], combustivel: str = "diesel", fonte_base: str = "previsto") -> str:
    if not m:
        return "Métricas: ainda sem telemetria."
    partes = [f"Métricas até agora: {num(m.get('tempo_s', 0))} s, {num(m.get('distancia_m', 0), 0)} m, "
              f"{num(m.get('combustivel_ml', 0), 0)} mL de {combustivel} (R$ {num(m.get('custo_r', 0), 2)}, "
              f"{num(m.get('co2_g', 0) / 1000.0, 2)} kg de CO2)"]
    if m.get("base_combustivel_ml") is not None:
        partes.append(f"sem agentes, {quem_compara(fonte_base)} {num(m['base_combustivel_ml'], 0)} mL na mesma distância "
                      f"(economia de {num(m.get('economia_medida_ml') or 0, 0)} mL, {num(m.get('economia_medida_pct') or 0, 0)} %)")
    partes.append(f"{m.get('decisoes', 0)} decisões ({m.get('decisoes_llm', 0)} do LLM), economia prevista somada "
                  f"{num(m.get('economia_prevista_ml', 0), 0)} mL; {m.get('problemas_medidos', 0)} problemas medidos")
    return "; ".join(partes) + "."


# ---------------------------------------------------------------------------------------------
# Resposta determinística (provedor falso, LLM fora do ar ou recusado pela guarda)
# ---------------------------------------------------------------------------------------------
CHAVES_PREVISAO = ("previs", "proxim", "adiante", "a frente", "pela frente", "vem ai", "vai acontecer", "futuro", "seguinte")
CHAVES_ECONOMIA = ("economi", "combust", "gasolin", "diesel", "custo", "gast", "consum", "litro", "reais", "co2", "carbono",
                   "dinheiro", "100 voltas")
CHAVES_INTERVENCAO = ("interv", "suaviz", "decis", "decid", "agente", "opcao", "llm", "mudou", "corrig")
CHAVES_PROBLEMA = ("problema", "critic", "atenc", "alerta", "evento", "lombada", "irregular", "fren", "frei", "arranc",
                   "brusc", "jerk", "tranco", "perig", "risco")
CHAVES_VELOCIDADE = ("velocidade", "veloc", "km/h", "rapid", "devagar", "lent")
CHAVES_CURVA = ("curva", "virou", "vira", "rumo", "esquerda", "direita")


def _filtrar_problemas(problemas: list[dict[str, Any]], texto: str) -> tuple[list[dict[str, Any]], str]:
    """Filtra por nível e tipo citados na pergunta; devolve também a descrição do filtro."""
    filtros: list[tuple[str, Callable[[dict[str, Any]], bool]]] = []
    if "critic" in texto:
        filtros.append(("crítico", lambda e: e["nivel"] == "critico"))
    elif "atenc" in texto:
        filtros.append(("de atenção", lambda e: e["nivel"] == "atencao"))
    conducao = lambda eventos: lambda e: e.get("evento") in eventos and e.get("causa") != "irregularidade_via"  # noqa: E731
    if _tem(texto, ("lombada", "irregular", "buraco")):
        filtros.append(("de lombada", lambda e: e.get("causa") == "irregularidade_via"))
    elif _tem(texto, ("fren", "frei", "freou")):
        filtros.append(("de frenagem", conducao(("frenagem_brusca", "soltura_freio"))))
    elif _tem(texto, ("arranc", "acelerou")):
        filtros.append(("de arrancada", conducao(("arrancada_brusca", "corte_aceleracao"))))
    for _, filtro in filtros:
        problemas = [e for e in problemas if filtro(e)]
    return problemas, " ".join(d for d, _ in filtros)


_NIVEL_NA_CONTAGEM = {"critico": ("crítico", "críticos"), "atencao": ("de atenção", "de atenção")}


def _plural_problema(descricao: str) -> str:
    """'problema crítico de frenagem' -> 'problemas críticos de frenagem'."""
    return descricao.replace("problema", "problemas", 1).replace("crítico", "críticos", 1)


def _niveis_no_cabecalho(problemas: list[dict[str, Any]]) -> str:
    """' (3 críticos, 6 de atenção)' ou ', todos de atenção'; vazio se houver um só (o item já diz o nível)."""
    if len(problemas) < 2:
        return ""
    niveis = Counter(e["nivel"] for e in problemas)
    nome = lambda nivel, n: _NIVEL_NA_CONTAGEM.get(nivel, (nivel, nivel))[n != 1]  # noqa: E731
    if len(niveis) == 1:
        nivel = next(iter(niveis))
        return f", todos {nome(nivel, 2)}"
    ordem = sorted(niveis, key=lambda nivel: -ORDEM_NIVEL.get(nivel, 0))
    return " (" + ", ".join(f"{niveis[nivel]} {nome(nivel, niveis[nivel])}" for nivel in ordem) + ")"


def _resumo_intervencao(e: dict[str, Any]) -> str:
    f = e.get("fatos") or {}
    bruta = (e.get("brutas") or [{}])[0]
    alvo = f" para a {bruta['nome_elemento']}" if bruta.get("nome_elemento") else ""
    tipo = "frenagem brusca" if bruta.get("tipo") == "frenagem_brusca" else "arrancada brusca"
    if e.get("opcao") == "nenhuma" or not bruta:
        return f"sem intervenção no trecho de {num(e['t_ini'], 0)}–{num(e['t_fim'], 0)} s {citacao(e)}"
    return (f"intervenção {e.get('acao')} ({_origem(e)}) no lugar da {tipo}{alvo} ({num(bruta.get('acel_mps2', 0))} m/s²), "
            f"economia prevista de {num(f.get('economia_prevista_ml', 0), 0)} mL {citacao(e)}")


def resposta_deterministica(pergunta: str, ctx: ContextoChat | None, conhecimento: dict[str, Any]) -> tuple[str, list[int]]:
    """Resposta curta por palavras-chave sobre o log, com as citações (#id, tempo)."""
    if ctx is None:
        return "Ainda não há corrida em andamento. Rode o piloto (python -m mover.autonomo.dirigir) para o log começar.", []
    texto = sem_acentos(pergunta)
    trecho = filtro_de_tempo(pergunta, ctx.t0, ctx.duracao_bloco)
    no_pedido = (lambda e: no_trecho(e, trecho)) if trecho else (lambda e: True)  # noqa: E731
    sufixo = f" entre {num(trecho[0])} e {num(trecho[1])} s" if trecho else ""
    intervencoes = [e for e in ctx.intervencoes if no_pedido(e)]

    if _tem(texto, CHAVES_PREVISAO):
        if ctx.encerrada:
            return "A corrida terminou: não há trecho adiante para prever.", []
        if not ctx.intervencoes:
            return "Os agentes ainda não decidiram o próximo trecho.", []
        e = ctx.intervencoes[-1]
        return f"Próximo trecho ({num(e['t_ini'], 0)}–{num(e['t_fim'], 0)} s): {e['texto']} {citacao(e)}", [e["id"]]

    if _tem(texto, CHAVES_ECONOMIA):
        m = ctx.metricas or {}
        if not m:
            return "Ainda não há telemetria para medir o consumo.", []
        partes = [f"Até agora o caminhão gastou {num(m.get('combustivel_ml', 0), 0)} mL de {ctx.combustivel} "
                  f"(R$ {num(m.get('custo_r', 0), 2)}) em {num(m.get('distancia_m', 0), 0)} m."]
        if m.get("base_combustivel_ml") is not None and ctx.modo == "agentes":
            partes.append(f"Sem os agentes, {quem_compara(ctx.fonte_base)} {num(m['base_combustivel_ml'], 0)} mL na mesma distância: "
                          f"economia de {num(m.get('economia_medida_ml') or 0, 0)} mL ({num(m.get('economia_medida_pct') or 0, 0)} %).")
        if m.get("decisoes"):
            partes.append(f"As {m['decisoes']} decisões previram {num(m.get('economia_prevista_ml', 0), 0)} mL de economia.")
        partes.append("O comparativo medido em várias voltas sai do benchmark.")
        citadas = ctx.intervencoes[-1:]
        if citadas:
            partes.append(f"Última decisão {citacao(citadas[0])}.")
        return " ".join(partes), [e["id"] for e in citadas]

    if _tem(texto, CHAVES_INTERVENCAO) and not _tem(texto, ("problema", "critic", "atenc")):
        if not intervencoes:
            return f"Nenhuma intervenção dos agentes{sufixo} até agora.", []
        ativas = [e for e in intervencoes if e.get("opcao") not in (None, "nenhuma")]
        por_opcao = Counter(e.get("acao") for e in ativas)
        cabeca = (f"{len(intervencoes)} decisão(ões){sufixo}, {len(ativas)} com intervenção"
                  + (f" ({', '.join(f'{n} {op}' for op, n in por_opcao.most_common())})" if por_opcao else ""))
        ultimas = (ativas or intervencoes)[-2:]
        return f"{cabeca}. Últimas: " + "; ".join(_resumo_intervencao(e) for e in ultimas) + ".", [e["id"] for e in ultimas]

    if not ctx.publicadas:
        return "O log ainda está vazio: o caminhão acabou de partir.", []

    if _tem(texto, CHAVES_PROBLEMA):
        todos = [e for e in ctx.publicadas if e["tipo"] == "problema" and no_pedido(e)]
        escolhidos, filtro = _filtrar_problemas(todos, texto)
        descricao = f"problema{' ' + filtro if filtro else ''}"
        if not escolhidos:
            evitados = sum(int((e.get("fatos") or {}).get("eventos_brutos", 0) or 0) for e in intervencoes
                           if e.get("opcao") not in (None, "nenhuma"))
            extra = (f" Os agentes alteraram o plano antes de {evitados} evento(s) brusco(s) que o caminhão tinha planejado."
                     if evitados and not filtro else "")
            return f"Nenhum {descricao} medido{sufixo}.{extra}", [e["id"] for e in intervencoes[-1:]] if extra else []
        escolhidos.sort(key=lambda e: (-ORDEM_NIVEL.get(e["nivel"], 0), float(e["t_pico"])))
        filtrou_nivel = "critic" in texto or "atenc" in texto
        cabeca = (f"{len(escolhidos)} {descricao if len(escolhidos) == 1 else _plural_problema(descricao)} medido(s){sufixo}"
                  f"{'' if filtrou_nivel else _niveis_no_cabecalho(escolhidos)}")
        itens = []
        for e in escolhidos[:3]:
            f = e.get("fatos") or {}
            itens.append(f"{nome_problema(e, conhecimento)} ({NOMES_NIVEL[e['nivel']]}), "
                         f"jerk de {num(f.get('jerk_max_abs_mps3', 0))} m/s³ a {num(f.get('vel_kmh', 0), 0)} km/h {citacao(e)}")
        resto = f"; e mais {len(escolhidos) - 3}" if len(escolhidos) > 3 else ""
        return f"{cabeca}: " + "; ".join(itens) + resto + ".", [e["id"] for e in escolhidos[:3]]

    logs = [e for e in ctx.publicadas if e["tipo"] == "log" and no_pedido(e) and e.get("fatos")]
    if _tem(texto, CHAVES_VELOCIDADE) and logs:
        maior = max(logs, key=lambda e: e["fatos"]["vel_max_kmh"])
        menor = min(logs, key=lambda e: e["fatos"]["vel_min_kmh"])
        resposta = (f"Velocidade máxima de {num(maior['fatos']['vel_max_kmh'], 0)} km/h {citacao(maior)} e mínima de "
                    f"{num(menor['fatos']['vel_min_kmh'], 0)} km/h {citacao(menor)}{sufixo}.")
        return resposta, list(dict.fromkeys([maior["id"], menor["id"]]))

    if _tem(texto, CHAVES_CURVA) and logs:
        curvas = [e for e in logs if e["fatos"].get("lado", "reto") != "reto"]
        if not curvas:
            return f"Nenhuma curva registrada no log{sufixo}.", []
        itens = [f"{abs(int(e['fatos']['mudanca_rumo_graus']))}° à {e['fatos']['lado']} {citacao(e)}" for e in curvas[-3:]]
        return f"{len(curvas)} trecho(s) com curva{sufixo}; os últimos: " + "; ".join(itens) + ".", [e["id"] for e in curvas[-3:]]

    if trecho:
        do_trecho = [e for e in ctx.publicadas if no_trecho(e, trecho)]
        if not do_trecho:
            return f"O log não tem entradas{sufixo}.", []
        itens = [f"{maiuscula(e['texto'])} {citacao(e)}" for e in do_trecho[:3]]
        return f"No log{sufixo}: " + " ".join(itens), [e["id"] for e in do_trecho[:3]]

    problemas = [e for e in ctx.publicadas if e["tipo"] == "problema"]
    niveis = Counter(e["nivel"] for e in problemas)
    ultimo = ctx.publicadas[-1]
    resposta = (f"Até agora: {len(ctx.publicadas)} entradas, {len(ctx.intervencoes)} decisões dos agentes e "
                f"{len(problemas)} problema(s) medido(s) ({niveis.get('critico', 0)} crítico(s), "
                f"{niveis.get('atencao', 0)} de atenção). Última entrada: {ultimo['texto']} {citacao(ultimo)}")
    return resposta, [ultimo["id"]]


# ---------------------------------------------------------------------------------------------
# Serviço
# ---------------------------------------------------------------------------------------------
class ServicoChat:
    """Monta o contexto, chama o LLM, aplica a guarda e cai na resposta determinística quando preciso."""

    def __init__(self, cfg_agentes: dict[str, Any], fabrica_llm: Callable[[Any], ClienteLLM | None] | None = None):
        self.cfg_agentes = cfg_agentes
        self.cfg = {**PADRAO_CHAT, **(cfg_agentes.get("chat") or {})}
        self.conhecimento = cfg_agentes.get("conhecimento", {})
        jerk = cfg_agentes.get("limiares", {}).get("jerk", {})
        # limiares que a resposta pode citar; a chave diz a unidade (ver supervisor.unidade_da_chave)
        self._numeros_fixos = {"atencao_mps3": jerk.get("atencao_mps3"), "critico_mps3": jerk.get("critico_mps3"),
                               "contagens": list(NUMEROS_LIVRES), "voltas_comparacao": [100]}
        self._fabrica = fabrica_llm or self._llm_padrao
        self._llms: dict[tuple[Any, Any], ClienteLLM | None] = {}
        self._erros: dict[tuple[Any, Any], str] = {}
        self._trava = threading.Lock()

    # ------------------------------------------------------------------ LLM
    def _llm_padrao(self, opcoes: Any) -> ClienteLLM:
        cfg_llm = copy.deepcopy(self.cfg_agentes.get("llm", {}))
        cfg_llm.setdefault("ollama", {})["num_ctx"] = int(self.cfg["num_ctx"])  # o log não cabe em 2048 tokens
        cfg_llm["timeout_s"] = float(self.cfg["timeout_s"])
        return criar_llm(cfg_llm, getattr(opcoes, "provedor", None), getattr(opcoes, "modelo", None))

    def llm_para(self, opcoes: Any) -> tuple[ClienteLLM | None, str | None]:
        """Cliente do LLM do chat (um por provedor/modelo), ou None e o motivo."""
        chave = (getattr(opcoes, "provedor", None), getattr(opcoes, "modelo", None))
        with self._trava:
            if chave not in self._llms:
                try:
                    self._llms[chave] = self._fabrica(opcoes)
                except Exception as erro:  # pacote ausente, chave de API faltando...
                    log.warning("Chat sem LLM (%s: %s); usando respostas por busca no log.", type(erro).__name__, erro)
                    self._llms[chave], self._erros[chave] = None, f"LLM indisponível ({type(erro).__name__}: {erro})"[:200]
            return self._llms[chave], self._erros.get(chave)

    # ------------------------------------------------------------------ contexto
    def contexto(self, sessao: Any, pergunta: str = "") -> ContextoChat | None:
        if sessao is None:
            return None
        publicadas = list(sessao.entradas)
        estado = sessao.ultimo_estado or {}
        duracao = float(getattr(sessao, "duracao_bloco", 10.0))
        trecho = filtro_de_tempo(pergunta, 0.0, duracao)
        escolhidas = {e["id"]: e for e in publicadas if e["tipo"] == "problema"}
        escolhidas.update({e["id"]: e for e in [e for e in publicadas if e["tipo"] == "intervencao"][-4:]})
        if trecho:
            escolhidas.update({e["id"]: e for e in publicadas if no_trecho(e, trecho)})
        limite = max(int(self.cfg["max_entradas_contexto"]), len(escolhidas))
        for entrada in reversed(publicadas):  # completa com as mais recentes
            if len(escolhidas) >= limite:
                break
            escolhidas.setdefault(entrada["id"], entrada)
        t_atual = estado.get("t")
        opcoes = getattr(sessao, "opcoes", None)
        return ContextoChat(
            publicadas=publicadas,
            selecionadas=sorted(escolhidas.values(), key=lambda e: e["id"]),
            metricas=sessao.metricas.para_json() if hasattr(sessao, "metricas") else {},
            t_atual=t_atual,
            bloco_atual=int(t_atual // duracao) if t_atual is not None else None,
            duracao_bloco=duracao,
            encerrada=bool(sessao.encerrada),
            modo=getattr(opcoes, "modo", "agentes"),
            combustivel=getattr(opcoes, "combustivel", "diesel"),
            fonte_base=fonte_da_linha_de_base(getattr(opcoes, "linha_de_base", None)),
        )

    def prompt(self, ctx: ContextoChat, pergunta: str, historico: list[dict[str, str]]) -> str:
        linhas = []
        modo = "com a camada agêntica intervindo" if ctx.modo == "agentes" else "sem agentes (o caminhão sozinho, para comparar)"
        if ctx.encerrada:
            linhas.append(f"A corrida ({modo}) terminou; o log abaixo está completo.")
        elif ctx.t_atual is not None:
            linhas.append(f"Corrida {modo}. Instante atual: {num(ctx.t_atual)} s (bloco {ctx.bloco_atual:02d}).")
        linhas.append(linha_das_metricas(ctx.metricas, ctx.combustivel, ctx.fonte_base))
        omitidas = len(ctx.publicadas) - len(ctx.selecionadas)
        linhas.append(f"Log ({len(ctx.publicadas)} entradas" + (f"; {omitidas} omitidas" if omitidas else "") + "):")
        linhas += [linha_do_contexto(e, self.conhecimento) for e in ctx.selecionadas] or ["(vazio)"]
        rodadas = historico[-2 * int(self.cfg["rodadas_historico"]):]
        if rodadas:
            linhas.append("Conversa anterior:")
            linhas += [f"{'Operador' if t['papel'] == 'operador' else 'Assistente'}: {t['texto']}" for t in rodadas]
        linhas.append(f"Pergunta do operador: {pergunta}")
        return "\n".join(linhas)

    # ------------------------------------------------------------------ guarda
    def numeros_permitidos(self, ctx: ContextoChat, pergunta: str, historico: list[dict[str, str]]) -> Permitidos:
        """Números que a resposta pode citar, agrupados por unidade (ver supervisor.unidade_da_chave)."""
        grupos: list[Permitidos] = []
        for e in ctx.todas:
            grupos.append(numeros_por_unidade({"id": e["id"], "bloco": e["bloco"], "t_ini": e.get("t_ini"),
                                               "t_fim": e.get("t_fim"), "t_pico": e.get("t_pico"),
                                               "t_pedido": e.get("t_pedido"), "t_mesclado_desde": e.get("mesclado_desde"),
                                               "fatos": e.get("fatos"), "brutas": e.get("brutas"),
                                               "instancias": e.get("instancias")}))
            for valor, unidade in numeros_com_unidade(e.get("texto") or ""):  # o texto publicado, com a unidade escrita
                grupos.append({unidade: [abs(valor)]})
        m = ctx.metricas or {}
        metricas = {"tempo_s": m.get("tempo_s"), "distancia_m": m.get("distancia_m"), "combustivel_ml": m.get("combustivel_ml"),
                    "custo_r": m.get("custo_r"), "co2_kg": (m.get("co2_g") or 0.0) / 1000.0,
                    "base_combustivel_ml": m.get("base_combustivel_ml"), "economia_medida_ml": m.get("economia_medida_ml"),
                    "economia_medida_pct": m.get("economia_medida_pct"), "economia_prevista_ml": m.get("economia_prevista_ml")}
        grupos.append(numeros_por_unidade({k: v for k, v in metricas.items() if v is not None and math.isfinite(float(v))}))
        problemas = [e for e in ctx.todas if e["tipo"] == "problema"]
        contagens = [ctx.bloco_atual, len(ctx.publicadas), len(problemas), len(ctx.intervencoes),
                     list(Counter(e["nivel"] for e in problemas).values()), m.get("decisoes"), m.get("decisoes_llm"),
                     m.get("intervencoes"), m.get("problemas_medidos"), m.get("base_eventos")]
        grupos.append(numeros_por_unidade({"t_atual": ctx.t_atual, "contagens": [c for c in contagens if c is not None]},
                                          self._numeros_fixos))
        # o que o operador escreveu pode voltar na resposta com qualquer unidade ("passou de 40?" -> "40 km/h")
        do_operador = [pergunta] + [t["texto"] for t in historico if t["papel"] == "operador"]
        grupos.append({QUALQUER_UNIDADE: [abs(v) for texto in do_operador for v in numeros_no_texto(texto)]})
        return juntar_permitidos(*grupos)

    def motivo_rejeicao(self, resposta: str, citacoes: list[int], ctx: ContextoChat, pergunta: str,
                        historico: list[dict[str, str]]) -> str | None:
        if not resposta.strip():
            return "resposta vazia"
        palavras = len(resposta.split())
        if palavras > 2 * int(self.cfg["max_palavras"]):
            return f"resposta longa demais ({palavras} palavras)"
        ids = ctx.por_id()
        citados = set(citacoes) | {int(m) for m in re.findall(r"#(\d+)", resposta)}
        inexistentes = sorted(i for i in citados if i not in ids)
        if inexistentes:
            return f"cita entrada inexistente: #{inexistentes[0]}"
        if self.cfg["validar_numeros"]:
            permitidos = self.numeros_permitidos(ctx, pergunta, historico)
            for numero, unidade in numeros_com_unidade(resposta):
                if not numero_com_respaldo(numero, unidade, permitidos):
                    return f"número sem respaldo no log: {descrever_numero(numero, unidade)}"
        return None

    # ------------------------------------------------------------------ resposta
    def responder(self, sessao: Any, pergunta: str, historico: list[dict[str, str]] | None = None) -> dict[str, Any]:
        inicio = time.monotonic()
        historico = [t for t in (historico or []) if t.get("texto")]
        ctx = self.contexto(sessao, pergunta)
        texto, ids = resposta_deterministica(pergunta, ctx, self.conhecimento)
        origem, provedor, observacao = "modelo", None, None
        llm, erro_llm = self.llm_para(getattr(sessao, "opcoes", None))
        if ctx is None or not ctx.todas:
            observacao = "sem entradas no log ainda"
        elif llm is None:
            observacao = erro_llm or "sem LLM configurado"
        elif getattr(llm, "nome", "") == "falso":
            observacao = "provedor falso: resposta por busca no log"
        else:
            try:
                saida = llm.gerar(SISTEMA_CHAT.format(max_palavras=self.cfg["max_palavras"]),
                                  self.prompt(ctx, pergunta, historico), RespostaChat)
                resposta_llm = " ".join(str(saida.resposta).split())
                motivo = self.motivo_rejeicao(resposta_llm, list(saida.citacoes), ctx, pergunta, historico)
                if motivo:
                    observacao = f"resposta do LLM recusada pela guarda ({motivo})"
                else:
                    texto, origem, provedor = resposta_llm, "llm", llm.nome
                    no_texto = [int(m) for m in re.findall(r"#(\d+)", resposta_llm)]
                    ids = list(dict.fromkeys(no_texto + [int(i) for i in saida.citacoes]))
            except Exception as erro:  # rede, modelo ausente, JSON fora do esquema...
                observacao = f"erro do LLM: {type(erro).__name__}: {erro}"[:200]
        latencia = round(time.monotonic() - inicio, 3)
        log.info("Chat (%s, %.2f s): %s", origem, latencia, pergunta[:80])
        return {"resposta": texto, "citacoes": self._citacoes(ids, ctx), "origem": origem, "provedor": provedor,
                "observacao": observacao, "latencia_s": latencia,
                "contexto": {"entradas": len(ctx.publicadas) if ctx else 0,
                             "no_prompt": len(ctx.selecionadas) if ctx else 0,
                             "t_atual": ctx.t_atual if ctx else None}}

    @staticmethod
    def _citacoes(ids: list[int], ctx: ContextoChat | None) -> list[dict[str, Any]]:
        if ctx is None:
            return []
        por_id = ctx.por_id()
        saida = []
        for i in ids:
            if i in por_id:
                e, previsto = por_id[i]
                saida.append({"id": i, "tipo": e["tipo"], "nivel": e["nivel"], "bloco": e["bloco"],
                              "tempo": tempo_da_entrada(e), "previsao": previsto})
        return saida
