"""Ajustes de condução que a camada agêntica devolve ao caminhão (malha fechada com o CARLA).

Cada problema de jerk previsto vira uma intenção executável: uma zona, no relógio do plano do
caminhão autônomo (X), onde o caminhão com agentes (Y) suaviza a velocidade e, numa
irregularidade da via, também a limita. Os números saem destas regras e do YAML (seção
`ajustes`), nunca do LLM; quem executa a zona é o plano de velocidade da simulação
(mover/simulacao/plano_velocidade.py).

A zona começa `antecedencia_s` antes do evento, para dar tempo de frear antes com suavidade, e
termina `depois_s` depois dele, para a retomada também ser suave.
"""

from __future__ import annotations

from typing import Any

from mover.agentes.especialistas import EventoJerk
from mover.agentes.textos import num

PADRAO_AJUSTES: dict[str, Any] = {
    "ativo": True,
    "antecedencia_s": 6.0,
    "depois_s": 6.0,
    "janela_s": 4.0,
    "rampa_s": 1.0,
    "vel_lombada_kmh": 12.0,
    "margem_lombada_s": 0.5,
}


class PlanejadorAjustes:
    """Transforma um evento de jerk no ajuste que o caminhão deve executar."""

    def __init__(self, cfg: dict[str, Any] | None = None):
        c = {**PADRAO_AJUSTES, **(cfg or {})}
        self.ativo = bool(c["ativo"])
        self.antecedencia_s = float(c["antecedencia_s"])
        self.depois_s = float(c["depois_s"])
        self.janela_s = float(c["janela_s"])
        self.rampa_s = float(c["rampa_s"])
        self.vel_lombada_kmh = float(c["vel_lombada_kmh"])
        self.margem_lombada_s = float(c["margem_lombada_s"])

    def ajuste(self, evento: EventoJerk) -> dict[str, Any] | None:
        if not self.ativo:
            return None
        t_ini = round(evento.t_ini - self.antecedencia_s, 2)
        t_fim = round(evento.t_fim + self.depois_s, 2)
        janela = num(self.janela_s, 0)
        if evento.causa == "irregularidade_via":
            nucleo_ini = round(evento.t_ini - self.margem_lombada_s, 2)
            nucleo_fim = round(evento.t_fim + self.margem_lombada_s, 2)
            vel_max = self.vel_lombada_kmh
            descricao = (f"Limitar a {num(vel_max, 0)} km/h entre {num(nucleo_ini)} e {num(nucleo_fim)} s e suavizar "
                         f"a velocidade de {num(t_ini)} a {num(t_fim)} s (janela de {janela} s).")
            tipo = "limitar_velocidade"
        else:
            nucleo_ini, nucleo_fim, vel_max = evento.t_ini, evento.t_fim, None
            descricao = (f"Suavizar a velocidade de {num(t_ini)} a {num(t_fim)} s (janela de {janela} s), "
                         "sem passar da velocidade planejada.")
            tipo = "suavizar"
        return {
            "tipo": tipo,
            "t_ini": t_ini,
            "t_fim": t_fim,
            "nucleo_ini": nucleo_ini,
            "nucleo_fim": nucleo_fim,
            "vel_max_kmh": vel_max,
            "janela_s": self.janela_s,
            "rampa_s": self.rampa_s,
            "descricao": descricao,
        }
