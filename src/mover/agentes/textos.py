"""Textos-modelo determinísticos (rascunho do Supervisor e plano B quando o LLM falha).

Os números saem no padrão brasileiro (vírgula decimal). Esses textos são o rascunho que o
Supervisor reescreve com o LLM e também a saída usada quando o LLM está desligado, falha,
estoura o orçamento de tempo ou não passa na guarda de números.
"""

from __future__ import annotations

from typing import Any

from mover.agentes.fatos import NOMES_MANOBRA, NOMES_NIVEL

NOMES_CAUSA = {
    "conducao_brusca": "condução brusca",
    "irregularidade_via": "irregularidade na via (provável lombada)",
}


def num(valor: float, casas: int = 1) -> str:
    """Formata no padrão brasileiro: num(18.44) -> '18,4'; num(-0.04) -> '0,0'."""
    texto = f"{float(valor):.{casas}f}"
    if texto.lstrip("-").strip("0.") == "":  # evita "-0,0"
        texto = texto.lstrip("-")
    return texto.replace(".", ",")


def maiuscula(texto: str) -> str:
    return texto[:1].upper() + texto[1:]


def _plural(n: int, singular: str, plural: str) -> str:
    return f"{n} {singular if n == 1 else plural}"


def texto_log(fatos: dict[str, Any], cfg_estado: dict[str, Any]) -> str:
    """Resumo seco de um trecho (uma janela ou janelas mescladas)."""
    manobra = NOMES_MANOBRA.get(fatos["manobra_predominante"], fatos["manobra_predominante"])
    v_min, v_max = num(fatos["vel_min_kmh"], 0), num(fatos["vel_max_kmh"], 0)
    faixa = f" (entre {v_min} e {v_max} km/h)" if v_min != v_max else ""  # evita "entre 25 e 25 km/h"
    partes = [f"{maiuscula(manobra)} a {num(fatos['vel_media_kmh'], 0)} km/h em média{faixa}"]
    dv = fatos["variacao_vel_kmh"]
    if abs(dv) >= cfg_estado["variacao_vel_relevante_kmh"]:
        partes.append(f"{'ganhou' if dv > 0 else 'perdeu'} {num(abs(dv), 0)} km/h no trecho")
    giro = fatos["mudanca_rumo_graus"]
    if abs(giro) >= cfg_estado["curva_minima_graus"]:
        partes.append(f"virou {num(abs(giro), 0)}° à {fatos['lado']}")
    rampa = fatos["rampa_media_pct"]
    if abs(rampa) >= cfg_estado["rampa_relevante_pct"]:
        partes.append(f"{'subida' if rampa > 0 else 'descida'} média de {num(abs(rampa), 0)} %")
    n_eventos = int(fatos.get("n_eventos", 0))
    if n_eventos:
        nivel = NOMES_NIVEL[fatos.get("nivel", "atencao")]
        partes.append(f"{_plural(n_eventos, 'evento', 'eventos')} de jerk, nível {nivel}")
    if fatos.get("ml_atipica"):
        partes.append(f"trecho atípico para o modelo de ML ({fatos['ml_quadros_atipicos_pct']} % dos quadros)")
    return "; ".join(partes) + "."


def texto_continuacao(fatos: dict[str, Any], duracao_estado_s: float) -> str:
    """Linha curta para um trecho que repete o estado da entrada anterior (sem LLM)."""
    manobra = NOMES_MANOBRA.get(fatos["manobra_predominante"], fatos["manobra_predominante"])
    return (
        f"Segue em {manobra} a {num(fatos['vel_media_kmh'], 0)} km/h em média, sem novidades "
        f"(mesmo estado há {num(duracao_estado_s, 0)} s)."
    )


def texto_problema(evento: dict[str, Any], conhecimento: dict[str, Any]) -> dict[str, str]:
    """Diagnóstico e ação para um evento de jerk, a partir da base de conhecimento.

    O texto cita o maior |jerk| do evento, que é o valor que define o nível.
    """
    chave = "irregularidade_via" if evento["causa"] == "irregularidade_via" else evento["tipo"]
    base = conhecimento[chave]
    jerk = num(evento["jerk_max_abs_mps3"], 1)
    vel = num(evento["vel_kmh"], 0)
    if evento["causa"] == "irregularidade_via":
        diagnostico = (
            f"Provável lombada a {vel} km/h: jerk de até {jerk} m/s³ e impacto vertical de "
            f"{num(evento['acel_vert_max_abs_mps2'], 1)} m/s²; {base['consequencia']}."
        )
    else:
        diagnostico = f"{maiuscula(base['nome'])} a {vel} km/h, com jerk de até {jerk} m/s³; {base['consequencia']}."
    return {"diagnostico": diagnostico, "acao": maiuscula(base["acao"]) + "."}
