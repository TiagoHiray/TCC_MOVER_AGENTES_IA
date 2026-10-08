"""Chat sobre o log (painel 4): perguntas do operador respondidas a partir das entradas da camada agêntica.

Contexto enviado ao LLM (o mesmo provedor plugável dos agentes: .env > config/agentes.yaml):
- as entradas publicadas até o bloco atual, com os números (fatos) de cada janela;
- as entradas do próximo bloco, se a análise dele já terminou, marcadas como PREVISÃO (o caminhão
  ainda não passou por ali).
Para caber no contexto de modelos pequenos (gemma2:2b), entram todos os problemas, as entradas do
trecho citado na pergunta e as mais recentes, até `chat.max_entradas_contexto`.

A resposta deve ser curta e citar as entradas no formato (#id, tempo). Antes de ir para a tela, uma
guarda local confere se cada número citado aparece no contexto (fatos, tempos, ids) ou na pergunta,
e se os ids citados existem. Se a guarda recusar, ou se o LLM falhar, vale a resposta determinística,
montada por palavras-chave sobre o log. A guarda não é um revisor: ela só barra números e citações
sem respaldo, não confere o sentido da frase. O provedor "falso" usa sempre a resposta determinística.
"""

from __future__ import annotations

import copy
import logging
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
    "Você é o assistente do operador no projeto MOVER, o gêmeo digital de um caminhão. Você responde "
    "perguntas sobre o log de condução escrito pela camada agêntica.\n"
    "Regras:\n"
    "- Português do Brasil, no máximo {max_palavras} palavras.\n"
    "- Use só as entradas do log abaixo. Se a resposta não estiver nelas, diga que o log não tem essa informação.\n"
    "- Cite as entradas usadas no formato (#id, tempo), por exemplo (#12, 51,4 s).\n"
    "- Entradas marcadas PREVISÃO são do próximo bloco, que o caminhão ainda não percorreu: apresente-as como previsão.\n"
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
    """O que o chat sabe num instante: o log publicado e, se pronta, a previsão do próximo bloco."""

    publicadas: list[dict[str, Any]]            # todas as entradas publicadas
    selecionadas: list[dict[str, Any]]          # as que vão para o prompt do LLM
    previsao: list[dict[str, Any]] = field(default_factory=list)
    bloco_previsao: dict[str, Any] | None = None
    t_atual: float | None = None
    bloco_atual: int | None = None
    n_blocos: int = 0
    duracao_bloco: float = 10.0
    t0: float = 0.0
    encerrada: bool = False

    @property
    def todas(self) -> list[dict[str, Any]]:
        return self.publicadas + self.previsao

    def por_id(self) -> dict[int, tuple[dict[str, Any], bool]]:
        """id -> (entrada, é previsão)."""
        return {**{e["id"]: (e, False) for e in self.publicadas}, **{e["id"]: (e, True) for e in self.previsao}}


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
    return f"{num(entrada['t_ini'])}–{num(entrada['t_fim'])} s"


def citacao(entrada: dict[str, Any]) -> str:
    return f"(#{entrada['id']}, {tempo_da_entrada(entrada)})"


def nome_problema(entrada: dict[str, Any], conhecimento: dict[str, Any]) -> str:
    if entrada.get("causa") == "irregularidade_via":
        return "provável lombada"
    return conhecimento.get(entrada.get("evento") or "", {}).get("nome", str(entrada.get("evento", "evento")).replace("_", " "))


def linha_do_contexto(entrada: dict[str, Any], previsao: bool, conhecimento: dict[str, Any]) -> str:
    """Uma entrada do log no prompt do chat: cabeçalho, texto publicado e os números da janela."""
    f = entrada.get("fatos") or {}
    marca = "PREVISÃO " if previsao else ""
    sintetico = " [evento sintético]" if entrada.get("fonte") not in (None, "real") else ""
    nivel = NOMES_NIVEL.get(entrada["nivel"], entrada["nivel"])
    if entrada["tipo"] == "problema":
        cabecalho = f"{marca}#{entrada['id']} [bloco {entrada['bloco']:02d}, pico aos {num(entrada['t_pico'])} s, problema {nivel}]"
        fatos = (f"{nome_problema(entrada, conhecimento)}; jerk máx {num(f.get('jerk_max_abs_mps3', 0))} m/s³; "
                 f"velocidade {num(f.get('vel_kmh', 0), 0)} km/h; impacto vertical {num(f.get('acel_vert_max_abs_mps2', 0))} m/s²; "
                 f"causa: {NOMES_CAUSA.get(f.get('causa', ''), f.get('causa', ''))}")
    else:
        cabecalho = (f"{marca}#{entrada['id']} [bloco {entrada['bloco']:02d}, {num(entrada['t_ini'])}–{num(entrada['t_fim'])} s, "
                     f"log {nivel}]")
        giro = abs(int(f.get("mudanca_rumo_graus", 0)))
        rumo = "trajeto reto" if f.get("lado", "reto") == "reto" else f"virou {giro}° à {f['lado']}"
        fatos = (f"velocidade média {num(f.get('vel_media_kmh', 0), 0)} km/h (mín {num(f.get('vel_min_kmh', 0), 0)}, "
                 f"máx {num(f.get('vel_max_kmh', 0), 0)}); aceleração longitudinal {num(f.get('acel_long_min_mps2', 0))} a "
                 f"{num(f.get('acel_long_max_mps2', 0))} m/s²; jerk {num(f.get('jerk_long_min_mps3', 0))} a "
                 f"{num(f.get('jerk_long_max_mps3', 0))} m/s³; {rumo}")
    return f"{cabecalho}{sintetico} {entrada['texto']} Fatos: {fatos}."


# ---------------------------------------------------------------------------------------------
# Resposta determinística (provedor falso, LLM fora do ar ou recusado pela guarda)
# ---------------------------------------------------------------------------------------------
CHAVES_PREVISAO = ("previs", "proxim", "adiante", "a frente", "pela frente", "vem ai", "vai acontecer", "futuro", "seguinte")
CHAVES_PROBLEMA = ("problema", "critic", "atenc", "alerta", "evento", "lombada", "irregular", "fren", "frei", "arranc",
                   "brusc", "jerk", "tranco", "perig", "risco")
CHAVES_VELOCIDADE = ("velocidade", "veloc", "km/h", "rapid", "devagar", "lent")
CHAVES_CURVA = ("curva", "virou", "vira", "rumo", "esquerda", "direita")


def _filtrar_problemas(problemas: list[tuple[dict[str, Any], bool]], texto: str) -> tuple[list[tuple[dict[str, Any], bool]], str]:
    """Filtra por nível e tipo citados na pergunta; devolve também a descrição do filtro."""
    filtros: list[tuple[str, Callable[[dict[str, Any]], bool]]] = []
    if "critic" in texto:
        filtros.append(("crítico", lambda e: e["nivel"] == "critico"))
    elif "atenc" in texto:
        filtros.append(("de atenção", lambda e: e["nivel"] == "atencao"))
    # frenagem e arrancada são do motorista: as prováveis lombadas (causa na via) ficam de fora, mesmo que o
    # pico de jerk tenha a forma de uma soltura de freio
    conducao = lambda eventos: lambda e: e.get("evento") in eventos and e.get("causa") != "irregularidade_via"  # noqa: E731
    if _tem(texto, ("lombada", "irregular", "buraco")):
        filtros.append(("de lombada", lambda e: e.get("causa") == "irregularidade_via"))
    elif _tem(texto, ("fren", "frei", "freou")):
        filtros.append(("de frenagem", conducao(("frenagem_brusca", "soltura_freio"))))
    elif _tem(texto, ("arranc", "acelerou")):
        filtros.append(("de arrancada", conducao(("arrancada_brusca", "corte_aceleracao"))))
    for _, filtro in filtros:
        problemas = [(e, p) for e, p in problemas if filtro(e)]
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


def resposta_deterministica(pergunta: str, ctx: ContextoChat | None, conhecimento: dict[str, Any]) -> tuple[str, list[int]]:
    """Resposta curta por palavras-chave sobre o log, com as citações (#id, tempo)."""
    if ctx is None:
        return "Ainda não há volta em andamento. Inicie o replay para o log começar.", []
    texto = sem_acentos(pergunta)
    trecho = filtro_de_tempo(pergunta, ctx.t0, ctx.duracao_bloco)
    no_pedido = (lambda e: no_trecho(e, trecho)) if trecho else (lambda e: True)  # noqa: E731
    sufixo = f" entre {num(trecho[0])} e {num(trecho[1])} s" if trecho else ""

    if _tem(texto, CHAVES_PREVISAO):
        if ctx.encerrada:
            return "A volta terminou: não há próximo bloco para prever.", []
        if not ctx.previsao:
            return "O próximo bloco ainda está em análise; a previsão aparece assim que os agentes terminarem.", []
        bp = ctx.bloco_previsao
        partes = [f"Previsão para o bloco {bp['bloco']:02d} ({num(bp['t_ini'], 0)}–{num(bp['t_fim'], 0)} s), ainda não percorrido:"]
        problemas = [e for e in ctx.previsao if e["tipo"] == "problema"]
        logs = [e for e in ctx.previsao if e["tipo"] == "log"]
        if problemas:
            partes += [f"{nome_problema(e, conhecimento)} ({NOMES_NIVEL[e['nivel']]}) {citacao(e)}." for e in problemas[:3]]
        else:
            partes.append("nenhum problema previsto.")
        if logs:
            partes.append(f"{logs[0]['texto']} {citacao(logs[0])}")
        citadas = problemas[:3] + logs[:1]
        return " ".join(partes), [e["id"] for e in citadas]

    if not ctx.publicadas:
        return "O log ainda está vazio: o caminhão não entrou no primeiro bloco.", []

    if _tem(texto, CHAVES_PROBLEMA):
        todos = [(e, False) for e in ctx.publicadas if e["tipo"] == "problema" and no_pedido(e)]
        todos += [(e, True) for e in ctx.previsao if e["tipo"] == "problema" and no_pedido(e)]
        escolhidos, filtro = _filtrar_problemas(todos, texto)
        descricao = f"problema{' ' + filtro if filtro else ''}"
        if not escolhidos:
            return f"Nenhum {descricao} no log até agora{sufixo}.", []
        # só cabem 3 na resposta: críticos primeiro, depois o que já aconteceu antes da previsão, e por fim o tempo
        escolhidos.sort(key=lambda par: (-ORDEM_NIVEL.get(par[0]["nivel"], 0), par[1], float(par[0]["t_pico"])))
        publicados = [e for e, previsto in escolhidos if not previsto]
        n_prev = len(escolhidos) - len(publicados)
        if publicados:
            filtrou_nivel = "critic" in texto or "atenc" in texto
            cabeca = (f"{len(publicados)} {descricao if len(publicados) == 1 else _plural_problema(descricao)} no log{sufixo}"
                      f"{'' if filtrou_nivel else _niveis_no_cabecalho(publicados)}")
            if n_prev:
                cabeca += f", mais {n_prev} na previsão do próximo bloco"
        else:
            cabeca = f"Nenhum {descricao} no log até agora{sufixo}; {n_prev} na previsão do próximo bloco"
        itens = []
        for e, previsto in escolhidos[:3]:
            f = e.get("fatos") or {}
            marcas = [NOMES_NIVEL[e["nivel"]]] + (["previsão"] if previsto else []) + (
                ["sintético"] if e.get("fonte") not in (None, "real") else [])
            itens.append(f"{nome_problema(e, conhecimento)} ({', '.join(marcas)}), "
                         f"jerk de {num(f.get('jerk_max_abs_mps3', 0))} m/s³ a {num(f.get('vel_kmh', 0), 0)} km/h {citacao(e)}")
        resto = f"; e mais {len(escolhidos) - 3}" if len(escolhidos) > 3 else ""
        return f"{cabeca}: " + "; ".join(itens) + resto + ".", [e["id"] for e, _ in escolhidos[:3]]

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
            return f"O log não tem entradas{sufixo} (até agora vai até {num(ctx.publicadas[-1].get('t_fim') or 0, 0)} s).", []
        itens = [f"{maiuscula(e['texto'])} {citacao(e)}" for e in do_trecho[:3]]
        return f"No log{sufixo}: " + " ".join(itens), [e["id"] for e in do_trecho[:3]]

    problemas = [e for e in ctx.publicadas if e["tipo"] == "problema"]
    niveis = Counter(e["nivel"] for e in problemas)
    ultimo = next((e for e in reversed(ctx.publicadas) if e["tipo"] == "log"), ctx.publicadas[-1])
    blocos = len({e["bloco"] for e in ctx.publicadas})
    resposta = (f"Até agora: {len(ctx.publicadas)} entradas em {blocos} bloco(s), {len(problemas)} problema(s) "
                f"({niveis.get('critico', 0)} crítico(s), {niveis.get('atencao', 0)} de atenção). "
                f"Último trecho: {ultimo['texto']} {citacao(ultimo)}")
    citadas = [ultimo["id"]]
    previstos = [e for e in ctx.previsao if e["tipo"] == "problema"]
    if previstos:
        resposta += f" Previsão do próximo bloco: {len(previstos)} problema(s), o primeiro {citacao(previstos[0])}."
        citadas.append(previstos[0]["id"])
    return resposta, citadas


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
        vertical = cfg_agentes.get("limiares", {}).get("lombada", {})
        # limiares que a resposta pode citar; a chave diz a unidade (ver supervisor.unidade_da_chave)
        self._numeros_fixos = {"atencao_mps3": jerk.get("atencao_mps3"), "critico_mps3": jerk.get("critico_mps3"),
                               "acel_vertical_mps2": vertical.get("acel_vertical_mps2"), "contagens": list(NUMEROS_LIVRES)}
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
        previsao = sessao.previsao_atual()
        camada = sessao.camada
        estado = sessao.ultimo_estado or {}
        trecho = filtro_de_tempo(pergunta, camada.t0, camada.duracao_bloco)
        escolhidas = {e["id"]: e for e in publicadas if e["tipo"] == "problema"}
        if trecho:
            escolhidas.update({e["id"]: e for e in publicadas if no_trecho(e, trecho)})
        limite = max(int(self.cfg["max_entradas_contexto"]), len(escolhidas))
        for entrada in reversed(publicadas):  # completa com as mais recentes
            if len(escolhidas) >= limite:
                break
            escolhidas.setdefault(entrada["id"], entrada)
        return ContextoChat(
            publicadas=publicadas,
            selecionadas=sorted(escolhidas.values(), key=lambda e: e["id"]),
            previsao=list(previsao["entradas"]) if previsao else [],
            bloco_previsao=previsao,
            t_atual=estado.get("sim_time"),
            bloco_atual=sessao.proximo_bloco - 1 if sessao.proximo_bloco > 0 else None,
            n_blocos=camada.n_blocos,
            duracao_bloco=camada.duracao_bloco,
            t0=camada.t0,
            encerrada=sessao.encerrada,
        )

    def prompt(self, ctx: ContextoChat, pergunta: str, historico: list[dict[str, str]]) -> str:
        linhas = []
        if ctx.encerrada:
            linhas.append("A volta terminou; o log abaixo está completo.")
        elif ctx.t_atual is not None:
            bloco = f" (bloco {ctx.bloco_atual:02d} de {ctx.n_blocos})" if ctx.bloco_atual is not None else ""
            linhas.append(f"Instante atual da volta: {num(ctx.t_atual)} s{bloco}.")
        omitidas = len(ctx.publicadas) - len(ctx.selecionadas)
        linhas.append(f"Log publicado ({len(ctx.publicadas)} entradas" + (f"; {omitidas} antigas omitidas" if omitidas else "") + "):")
        linhas += [linha_do_contexto(e, False, self.conhecimento) for e in ctx.selecionadas] or ["(vazio)"]
        if ctx.previsao:
            bp = ctx.bloco_previsao
            linhas.append(f"PREVISÃO do bloco {bp['bloco']:02d} ({num(bp['t_ini'])}–{num(bp['t_fim'])} s), já analisado e "
                          "ainda não percorrido:")
            linhas += [linha_do_contexto(e, True, self.conhecimento) for e in ctx.previsao]
        elif not ctx.encerrada:
            linhas.append("O próximo bloco ainda está em análise (sem previsão por enquanto).")
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
                                               "t_mesclado_desde": e.get("mesclado_desde"), "fatos": e.get("fatos")}))
            for valor, unidade in numeros_com_unidade(e.get("texto") or ""):  # o texto publicado, com a unidade escrita
                grupos.append({unidade: [abs(valor)]})
        problemas = [e for e in ctx.todas if e["tipo"] == "problema"]
        contagens = [ctx.bloco_atual, ctx.n_blocos, len(ctx.publicadas), len(ctx.previsao), len(problemas),
                     list(Counter(e["nivel"] for e in problemas).values()), len({e["bloco"] for e in ctx.publicadas})]
        grupos.append(numeros_por_unidade({"t_atual": ctx.t_atual, "contagens": contagens}, self._numeros_fixos))
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
                             "previsao_bloco": ctx.bloco_previsao["bloco"] if ctx and ctx.bloco_previsao else None,
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
