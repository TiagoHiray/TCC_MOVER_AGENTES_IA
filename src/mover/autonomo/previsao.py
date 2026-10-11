"""Previsão dos próximos segundos: o modelo interno do gêmeo digital.

Simula o governador (com as intervenções já decididas e as candidatas), o controle do piloto e a
dinâmica do caminhão a partir do estado atual. A camada agêntica usa isso para:
- ver as decisões bruscas que o piloto vai tomar no próximo bloco (plano bruto);
- comparar as opções de intervenção (jerk, desaceleração, consumo e tempo) antes de escolher.

Com o CARLA, o caminhão de verdade é o do simulador; a previsão continua usando este modelo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from mover.autonomo.consumo import Limiares, ParametrosConsumo, metricas
from mover.autonomo.governador import Governador
from mover.autonomo.veiculo import DinamicaCaminhao, ParametrosCaminhao, ParametrosPiloto, PilotoTM


@dataclass
class Trajetoria:
    t: np.ndarray
    d: np.ndarray
    v: np.ndarray
    a: np.ndarray
    v_desejada: np.ndarray
    parado: np.ndarray

    def recorte(self, t_ini: float, t_fim: float) -> "Trajetoria":
        sel = (self.t >= t_ini - 1e-9) & (self.t < t_fim - 1e-9)
        return Trajetoria(self.t[sel], self.d[sel], self.v[sel], self.a[sel], self.v_desejada[sel], self.parado[sel])

    def para_json(self, passo: int = 4, casas: int = 2) -> dict[str, list[float]]:
        """Série reduzida para a interface (a 20 Hz, passo 4 = 5 Hz)."""
        idx = np.arange(0, len(self.t), max(1, passo))
        return {"t": np.round(self.t[idx], casas).tolist(), "v_kmh": np.round(self.v[idx] * 3.6, 1).tolist(),
                "a": np.round(self.a[idx], casas).tolist(), "d": np.round(self.d[idx], 1).tolist()}


class ModeloInterno:
    def __init__(self, caminhao: ParametrosCaminhao, piloto: ParametrosPiloto, dt: float = 0.05,
                 consumo: ParametrosConsumo | None = None, limiares: Limiares | None = None,
                 rampa: Callable[[float], float] | None = None):
        self.caminhao = caminhao
        self.piloto = piloto
        self.dt = dt
        self.consumo = consumo or ParametrosConsumo()
        self.limiares = limiares or Limiares()
        self.rampa = rampa or (lambda d: 0.0)

    def simular(self, governador: Governador, d0: float, v0: float, a0: float, t0: float, duracao_s: float) -> Trajetoria:
        """Roda `duracao_s` a partir do estado dado. O governador recebido é alterado: passe um clone."""
        n = max(1, int(round(duracao_s / self.dt)))
        dinamica = DinamicaCaminhao(self.caminhao, v0=v0, a0=a0, rampa=self.rampa(d0))
        piloto = PilotoTM(self.piloto, self.caminhao)
        t = np.empty(n)
        d = np.empty(n)
        v = np.empty(n)
        a = np.empty(n)
        v_des = np.empty(n)
        parado = np.zeros(n, dtype=bool)
        di, vi, ai = float(d0), float(v0), float(a0)
        for k in range(n):
            alvo = governador.velocidade_desejada(di, vi, ai, self.dt)
            rampa = self.rampa(di)
            acelerador, freio = piloto.comando(alvo, vi, self.dt, rampa)
            saida = dinamica.passo(acelerador, freio, self.dt, rampa)
            di += 0.5 * (vi + saida["v"]) * self.dt
            vi, ai = saida["v"], saida["a"]
            t[k], d[k], v[k], a[k], v_des[k] = t0 + (k + 1) * self.dt, di, vi, ai, alvo
            parado[k] = bool(governador.ultimo.get("parado"))
            if governador.terminou and vi <= 0.01:
                return Trajetoria(t[:k + 1], d[:k + 1], v[:k + 1], a[:k + 1], v_des[:k + 1], parado[:k + 1])
        return Trajetoria(t, d, v, a, v_des, parado)

    def metricas(self, trajetoria: Trajetoria, contexto: Trajetoria | None = None) -> dict[str, Any]:
        """Métricas do trecho; `contexto` (o trecho completo) evita o efeito de borda do filtro."""
        if contexto is not None and len(trajetoria.t) and len(contexto.t) > len(trajetoria.t):
            completo = metricas(contexto.t, contexto.v, contexto.a, self.caminhao, self.consumo, self.limiares)
            sel = (contexto.t >= trajetoria.t[0] - 1e-9) & (contexto.t <= trajetoria.t[-1] + 1e-9)
            from mover.autonomo.consumo import suavizar  # import local: evita ciclo na documentação
            a_suave = suavizar(contexto.a, self.dt, self.limiares.fc_hz)[sel]
            resultado = metricas(contexto.t[sel], contexto.v[sel], a_suave, self.caminhao, self.consumo, self.limiares,
                                 ja_suave=True)
            resultado["contexto_combustivel_l"] = completo["combustivel_l"]
            return resultado
        return metricas(trajetoria.t, trajetoria.v, trajetoria.a, self.caminhao, self.consumo, self.limiares)
