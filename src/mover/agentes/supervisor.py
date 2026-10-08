"""Supervisor (LLM): escreve as entradas de log e o diagnóstico/ação dos problemas.

Segue o padrão Especialista + Supervisor da POC (src/poc-agents/app.py): os especialistas
detectam e o Supervisor interpreta e escreve. O Supervisor recebe apenas fatos medidos, a
base de conhecimento do YAML e um rascunho determinístico (textos.py), que ele reescreve
em linguagem natural.

Guarda de números (supervisor.validar_numeros no YAML): se o texto do LLM citar um número
sem respaldo nos fatos (com tolerância de arredondamento), a entrada usa o texto-modelo e
o motivo fica em `observacao`. Número com unidade (km/h, m/s², m/s³, s, %, °, m) só vale se
bater com um fato da mesma unidade. Não é um segundo agente revisor: é uma verificação local,
barata e determinística, e pode ser desligada.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from mover.agentes import textos
from mover.agentes.fatos import NOMES_MANOBRA, NOMES_NIVEL
from mover.agentes.llm import MARCADOR_RASCUNHO, ClienteLLM

log = logging.getLogger("mover.agentes")


class RespostaLog(BaseModel):
    resumo: str = Field(description="Entrada de log em português, 1 ou 2 frases objetivas")


class RespostaProblema(BaseModel):
    diagnostico: str = Field(description="Diagnóstico técnico curto do evento")
    acao: str = Field(description="Ação recomendada ao motorista")


@dataclass
class Redacao:
    """Texto final de uma entrada e de onde ele veio."""

    campos: dict[str, str]
    origem: str                      # "llm" ou "modelo"
    latencia_s: float = 0.0
    observacao: str | None = None    # por que o texto-modelo foi usado


SISTEMA_LOG = (
    "Você é o Supervisor da camada agêntica do projeto MOVER, o gêmeo digital de um caminhão. "
    "Você escreve o log de condução para o operador a partir de fatos medidos pela telemetria.\n"
    "Regras:\n"
    "- Português do Brasil, 1 ou 2 frases, no máximo {max_palavras} palavras.\n"
    "- Use apenas os fatos fornecidos. Não invente causas, peças, defeitos nem números.\n"
    "- Velocidades em km/h sem casas decimais; acelerações e jerk com uma casa decimal.\n"
    "- Não cite horários nem instantes em segundos.\n"
    '- Responda somente com JSON no formato {{"resumo": "..."}}.'
)

SISTEMA_PROBLEMA = (
    "Você é o Supervisor da camada agêntica do projeto MOVER, o gêmeo digital de um caminhão. "
    "O Especialista de jerk detectou uma variação brusca da aceleração. Escreva o diagnóstico "
    "e a ação recomendada ao motorista.\n"
    "Regras:\n"
    "- Português do Brasil; diagnóstico com até {max_palavras} palavras e ação com até {max_palavras} palavras.\n"
    "- Use apenas os fatos do evento e a base de conhecimento. Não invente peças, defeitos nem números.\n"
    '- Responda somente com JSON no formato {{"diagnostico": "...", "acao": "..."}}.'
)

TAREFA_LOG = ("Tarefa: reescreva o rascunho como uma entrada de log natural e objetiva para o operador "
              "do caminhão. Mantenha os fatos importantes e não acrescente números.")
TAREFA_PROBLEMA = ("Tarefa: reescreva o rascunho com linguagem natural e direta para o motorista, sem perder "
                   "o tipo do evento, a gravidade e a ação recomendada.")

# ---------------------------------------------------------------------------------------------
# Guarda de números
# ---------------------------------------------------------------------------------------------
# A guarda olha também a unidade escrita logo depois de cada número: "97 km/h" só passa se algum
# fato em km/h valer ~97, e não porque outro fato vale 100 (%). A unidade de cada fato vem do
# sufixo da chave (vel_media_kmh -> km/h, jerk_max_abs_mps3 -> m/s³, t_pico -> s). Número escrito
# sem unidade pode bater com qualquer fato, como antes.
_HORARIO = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
_NUMERO = re.compile(r"(?<!\w)[-−+]?\d+(?:[.,]\d+)?")
NUMEROS_LIVRES = (1.0, 2.0)  # contagens pequenas ("1 evento", "2 lombadas")
QUALQUER_UNIDADE = "*"  # grupo de números aceitos com qualquer unidade (lista simples, números da pergunta)

_UNIDADES_NO_TEXTO: tuple[tuple[str, re.Pattern[str]], ...] = (  # a ordem importa: m/s³ antes de m/s e de m
    ("km/h", re.compile(r"\s*km\s*/\s*h", re.IGNORECASE)),
    ("m/s³", re.compile(r"\s*m\s*/\s*s(?:³|\^?3)", re.IGNORECASE)),
    ("m/s²", re.compile(r"\s*m\s*/\s*s(?:²|\^?2)", re.IGNORECASE)),
    ("m/s", re.compile(r"\s*m\s*/\s*s\b", re.IGNORECASE)),
    ("%", re.compile(r"\s*(?:%|por\s*cento\b)", re.IGNORECASE)),
    ("°", re.compile(r"\s*(?:°|º|graus?\b)", re.IGNORECASE)),
    ("s", re.compile(r"\s*(?:s|seg|segundos?)\b", re.IGNORECASE)),
    ("m", re.compile(r"\s*(?:m|metros?)\b", re.IGNORECASE)),
)
_SUFIXOS_UNIDADE = (("_kmh", "km/h"), ("_mps3", "m/s³"), ("_mps2", "m/s²"), ("_pct", "%"), ("_graus", "°"),
                    ("_s", "s"), ("_m", "m"))

Permitidos = dict[str | None, list[float]]  # unidade (None = fato sem unidade conhecida) -> valores absolutos


def unidade_da_chave(chave: Any) -> str | None:
    """Unidade de um fato pelo nome da chave: vel_media_kmh -> km/h; t_pico -> s; id -> None."""
    if not isinstance(chave, str):
        return None
    if chave.startswith("t_"):  # instantes da volta (t_ini, t_fim, t_pico...)
        return "s"
    return next((unidade for sufixo, unidade in _SUFIXOS_UNIDADE if chave.endswith(sufixo)), None)


def numeros_com_unidade(texto: str) -> list[tuple[float, str | None]]:
    """Números citados no texto, com a unidade escrita logo depois (ou None), ignorando horários HH:MM[:SS]."""
    sem_horarios = _HORARIO.sub(" ", texto)
    saida: list[tuple[float, str | None]] = []
    for m in _NUMERO.finditer(sem_horarios):
        depois = sem_horarios[m.end():m.end() + 16]
        unidade = next((nome for nome, padrao in _UNIDADES_NO_TEXTO if padrao.match(depois)), None)
        saida.append((float(m.group().replace("−", "-").replace(",", ".")), unidade))
    return saida


def numeros_no_texto(texto: str) -> list[float]:
    """Números citados no texto (vírgula ou ponto decimal), ignorando horários HH:MM[:SS]."""
    return [valor for valor, _ in numeros_com_unidade(texto)]


def coletar_numeros(*fontes: Any) -> list[float]:
    """Valores absolutos de todos os números (int/float) dentro de dicionários e listas."""
    return [valor for valores in numeros_por_unidade(*fontes).values() for valor in valores]


def numeros_por_unidade(*fontes: Any) -> Permitidos:
    """Valores absolutos dos números de dicionários e listas, agrupados pela unidade da chave."""
    grupos: Permitidos = {}

    def visitar(valor: Any, unidade: str | None) -> None:
        if isinstance(valor, bool) or valor is None:
            return
        if isinstance(valor, (int, float)):
            if math.isfinite(valor):
                grupos.setdefault(unidade, []).append(abs(float(valor)))
        elif isinstance(valor, dict):
            for chave, item in valor.items():
                visitar(item, unidade_da_chave(chave) or unidade)
        elif isinstance(valor, (list, tuple)):
            for item in valor:
                visitar(item, unidade)

    for fonte in fontes:
        visitar(fonte, None)
    return grupos


def juntar_permitidos(*grupos: Permitidos) -> Permitidos:
    saida: Permitidos = {}
    for grupo in grupos:
        for unidade, valores in grupo.items():
            saida.setdefault(unidade, []).extend(valores)
    return saida


def numero_suportado(numero: float, permitidos: list[float], tol_abs: float = 0.6, tol_rel: float = 0.08) -> bool:
    """Verdadeiro se |numero| bate com algum valor permitido, com folga de arredondamento."""
    alvo = abs(numero)
    return any(abs(alvo - v) <= max(tol_abs, tol_rel * v) for v in permitidos)


def numero_com_respaldo(numero: float, unidade: str | None, permitidos: Permitidos | list[float]) -> bool:
    """Com unidade, o número tem de bater com um fato da mesma unidade; sem unidade, com qualquer fato.

    Uma lista simples de números (sem unidades) vale para qualquer unidade.
    """
    grupos = permitidos if isinstance(permitidos, dict) else {QUALQUER_UNIDADE: list(permitidos)}
    if unidade is None:
        candidatos = [valor for valores in grupos.values() for valor in valores]
    else:
        candidatos = grupos.get(unidade, []) + grupos.get(QUALQUER_UNIDADE, [])
    return numero_suportado(numero, candidatos)


def descrever_numero(numero: float, unidade: str | None) -> str:
    texto = textos.num(numero, 0 if numero == int(numero) else 1)
    if unidade is None:
        return texto
    return f"{texto}{'' if unidade == '°' else ' '}{unidade}"


def motivo_rejeicao(texto: str, permitidos: Permitidos | list[float], max_palavras: int,
                    checar_numeros: bool = True) -> str | None:
    """Motivo para recusar o texto do LLM, ou None se ele pode ser usado."""
    if not texto.strip():
        return "texto vazio"
    if not checar_numeros:
        return None
    palavras = len(texto.split())
    if palavras > 2 * max_palavras:
        return f"texto longo demais ({palavras} palavras)"
    for numero, unidade in numeros_com_unidade(texto):
        if not numero_com_respaldo(numero, unidade, permitidos):
            return f"número sem respaldo nos fatos: {descrever_numero(numero, unidade)}"
    return None


# ---------------------------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------------------------
def nome_evento(evento: dict[str, Any], conhecimento: dict[str, Any]) -> str:
    if evento["causa"] == "irregularidade_via":
        return "provável lombada"
    return conhecimento[evento["tipo"]]["nome"]


class Supervisor:
    """Transforma fatos e eventos em texto, com o LLM ou com o texto-modelo."""

    def __init__(self, llm: ClienteLLM | None, cfg: dict[str, Any]):
        cs = cfg["supervisor"]
        self.llm = llm
        self.nome_llm = llm.nome if llm is not None else None
        self.validar_numeros = bool(cs.get("validar_numeros", True))
        self.max_palavras_log = int(cs.get("max_palavras_log", 45))
        self.max_palavras_campo = int(cs.get("max_palavras_campo", 20))
        self.orcamento_s = float(cs.get("orcamento_bloco_s", 8.0))
        self.falhas_para_desligar = int(cs.get("falhas_para_desligar", 2))
        self.cfg_estado = cfg["estado"]
        self.conhecimento = cfg["conhecimento"]
        self.limiar_atencao = float(cfg["limiares"]["jerk"]["atencao_mps3"])
        self.limiar_critico = float(cfg["limiares"]["jerk"]["critico_mps3"])
        self.limiar_vertical = float(cfg["limiares"]["lombada"]["acel_vertical_mps2"])
        # números que o texto pode citar além dos fatos; a chave diz a unidade (ver unidade_da_chave)
        self._numeros_fixos = {"atencao_mps3": self.limiar_atencao, "critico_mps3": self.limiar_critico,
                               "acel_vertical_mps2": self.limiar_vertical, "contagens": list(NUMEROS_LIVRES)}
        self.llm_ativo = llm is not None
        self._falhas_seguidas = 0
        self._inicio_bloco = time.monotonic()
        self.estatisticas: Counter[str] = Counter()

    def iniciar_bloco(self) -> None:
        """Zera o relógio do orçamento de tempo de LLM do bloco."""
        self._inicio_bloco = time.monotonic()

    # ------------------------------------------------------------------ log
    def escrever_log(self, fatos: dict[str, Any], eventos: list[dict[str, Any]]) -> Redacao:
        rascunho = {"resumo": textos.texto_log(fatos, self.cfg_estado)}
        return self._redigir(
            SISTEMA_LOG.format(max_palavras=self.max_palavras_log),
            self._prompt_log(fatos, eventos, rascunho),
            RespostaLog,
            rascunho,
            numeros_por_unidade(fatos, eventos, self._numeros_fixos),
            {"resumo": self.max_palavras_log},
        )

    def _prompt_log(self, f: dict[str, Any], eventos: list[dict[str, Any]], rascunho: dict[str, str]) -> str:
        n = textos.num
        giro = abs(f["mudanca_rumo_graus"])
        if f["lado"] == "reto":
            rumo = f"- trajeto praticamente reto (mudança de rumo de {giro}°)"
        else:
            rumo = f"- mudança de rumo de {giro}° para a {f['lado']}"
        manobra = NOMES_MANOBRA.get(f["manobra_predominante"], f["manobra_predominante"])
        linhas = [
            f"Trecho com {n(f['duracao_s'], 0)} s de percurso.",
            "Fatos medidos:",
            f"- velocidade média {n(f['vel_media_kmh'])} km/h (mínima {n(f['vel_min_kmh'])}, máxima "
            f"{n(f['vel_max_kmh'])}); variação no trecho {'+' if f['variacao_vel_kmh'] > 0 else ''}"
            f"{n(f['variacao_vel_kmh'])} km/h",
            f"- aceleração longitudinal entre {n(f['acel_long_min_mps2'])} e {n(f['acel_long_max_mps2'])} m/s²; "
            f"aceleração lateral de até {n(f['acel_lat_max_abs_mps2'])} m/s²",
            f"- jerk longitudinal entre {n(f['jerk_long_min_mps3'])} e {n(f['jerk_long_max_mps3'])} m/s³ "
            f"(atenção a partir de {n(self.limiar_atencao)}, crítico a partir de {n(self.limiar_critico)})",
            rumo,
            f"- manobra predominante: {manobra} ({f['manobra_predominante_pct']} % do tempo)",
            f"- rampa média {n(f['rampa_media_pct'])} %; variação de altitude {n(f['variacao_alt_m'])} m; "
            f"distância percorrida {f['distancia_m']} m",
        ]
        if "ml_quadros_atipicos_pct" in f:
            situacao = "trecho atípico" if f["ml_atipica"] else "dentro do padrão da volta"
            linhas.append(f"- especialista de ML: {f['ml_quadros_atipicos_pct']} % dos quadros atípicos ({situacao})")
        if eventos:
            linhas.append("Eventos do especialista de jerk:")
            for ev in eventos:
                linhas.append(
                    f"- {nome_evento(ev, self.conhecimento)} (nível {NOMES_NIVEL[ev['nivel']]}): jerk de até "
                    f"{n(ev['jerk_max_abs_mps3'])} m/s³ a {n(ev['vel_kmh'], 0)} km/h; causa provável: "
                    f"{textos.NOMES_CAUSA[ev['causa']]}"
                )
        else:
            linhas.append("Eventos do especialista de jerk: nenhum.")
        if f["fonte"] != "real":
            linhas.append("Observação: o trecho contém eventos sintéticos de teste.")
        linhas += [f"{MARCADOR_RASCUNHO} {json.dumps(rascunho, ensure_ascii=False)}", TAREFA_LOG]
        return "\n".join(linhas)

    # ------------------------------------------------------------------ problema
    def escrever_problema(self, evento: dict[str, Any]) -> Redacao:
        rascunho = textos.texto_problema(evento, self.conhecimento)
        return self._redigir(
            SISTEMA_PROBLEMA.format(max_palavras=self.max_palavras_campo),
            self._prompt_problema(evento, rascunho),
            RespostaProblema,
            rascunho,
            numeros_por_unidade(evento, self._numeros_fixos),
            {"diagnostico": self.max_palavras_campo, "acao": self.max_palavras_campo},
        )

    def _prompt_problema(self, ev: dict[str, Any], rascunho: dict[str, str]) -> str:
        n = textos.num
        chave = "irregularidade_via" if ev["causa"] == "irregularidade_via" else ev["tipo"]
        base = self.conhecimento[chave]
        linhas = [
            f"Evento: {nome_evento(ev, self.conhecimento)}, nível {NOMES_NIVEL[ev['nivel']]}.",
            "Fatos do evento:",
            f"- jerk no pico {n(ev['jerk_pico_mps3'])} m/s³; maior |jerk| do evento {n(ev['jerk_max_abs_mps3'])} m/s³ "
            f"(atenção a partir de {n(self.limiar_atencao)}, crítico a partir de {n(self.limiar_critico)})",
            f"- velocidade {n(ev['vel_kmh'], 0)} km/h; aceleração longitudinal entre {n(ev['acel_long_min_mps2'])} "
            f"e {n(ev['acel_long_max_mps2'])} m/s²",
            f"- impacto vertical máximo {n(ev['acel_vert_max_abs_mps2'])} m/s² (acima de {n(self.limiar_vertical)} "
            "indica irregularidade na via)",
            f"- causa provável: {textos.NOMES_CAUSA[ev['causa']]}",
            f"Base de conhecimento: consequência: {base['consequencia']}; ação indicada: {base['acao']}.",
        ]
        if ev["fonte"] != "real":
            linhas.append("Observação: evento sintético de teste.")
        linhas += [f"{MARCADOR_RASCUNHO} {json.dumps(rascunho, ensure_ascii=False)}", TAREFA_PROBLEMA]
        return "\n".join(linhas)

    # ------------------------------------------------------------------ chamada ao LLM
    def _redigir(self, sistema: str, usuario: str, esquema: type[BaseModel], rascunho: dict[str, str],
                 permitidos: Permitidos, limites: dict[str, int]) -> Redacao:
        if self.llm is None:
            return Redacao(dict(rascunho), "modelo", observacao="sem LLM configurado")
        if not self.llm_ativo:
            return Redacao(dict(rascunho), "modelo", observacao="LLM desligado após falhas seguidas")
        if time.monotonic() - self._inicio_bloco > self.orcamento_s:
            self.estatisticas["orcamento_esgotado"] += 1
            return Redacao(dict(rascunho), "modelo", observacao="orçamento de tempo do bloco esgotado")

        inicio = time.monotonic()
        self.estatisticas["chamadas"] += 1
        log.debug("Prompt do Supervisor:\n[sistema]\n%s\n[usuário]\n%s", sistema, usuario)
        try:
            resposta = self.llm.gerar(sistema, usuario, esquema)
        except Exception as erro:  # rede, modelo ausente, JSON fora do esquema...
            latencia = time.monotonic() - inicio
            self._falhas_seguidas += 1
            self.estatisticas["erros"] += 1
            if self._falhas_seguidas >= self.falhas_para_desligar:
                self.llm_ativo = False
                log.warning("LLM %s desligado após %d falhas seguidas (último erro: %s). Seguindo com o texto-modelo.",
                            self.nome_llm, self._falhas_seguidas, erro)
            return Redacao(dict(rascunho), "modelo", latencia, f"erro do LLM: {type(erro).__name__}: {erro}"[:200])

        latencia = time.monotonic() - inicio
        self._falhas_seguidas = 0
        campos = {nome: " ".join(str(valor).split()) for nome, valor in resposta.model_dump().items()}
        log.debug("Resposta do LLM (%.2f s): %s", latencia, campos)
        for nome, texto in campos.items():
            motivo = motivo_rejeicao(texto, permitidos, limites[nome], self.validar_numeros)
            if motivo:
                self.estatisticas["recusadas"] += 1
                return Redacao(dict(rascunho), "modelo", latencia, motivo)
        self.estatisticas["aceitas"] += 1
        return Redacao(campos, "llm", latencia)
