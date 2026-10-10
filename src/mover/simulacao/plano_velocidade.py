"""Plano de velocidade do caminhão com agentes (Y): o plano do autônomo (X) com os ajustes dos agentes.

X é a volta autônoma gravada no CARLA (poses e velocidade a 20 Hz). Y percorre exatamente o mesmo
caminho, reparametrizado no tempo: o relógio do plano X (tau) avança v_Y / v_X a cada segundo de Y.
Fora das zonas pedidas pelos agentes, v_Y = v_X e Y repete X. Dentro de uma zona:

- a velocidade de X (com o teto da lombada, se houver) passa por um mínimo móvel de `janela_s` e
  por duas médias móveis cujo suporte cabe na mesma janela. O resultado nunca passa de X no mesmo
  ponto do caminho, começa a frear antes (o plano é conhecido com antecedência) e tem aceleração
  contínua: no máximo ~2 dv / janela, com jerk de no máximo ~4 dv / janela² (dv = variação de
  velocidade de X na zona);
- nas bordas, X e o perfil suavizado se misturam com uma rampa de cosseno de `rampa_s`; a zona se
  alarga até os dois coincidirem na rampa, para não cortar uma retomada do próprio X no meio.

Um ajuste que chega com o caminhão já dentro da zona entra a partir do instante atual, com uma
rampa de `rampa_replanejamento_s`: a velocidade nunca salta.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d

V_REGULARIZACAO_MPS = 0.3  # evita 0/0 na razão v_Y / v_X com o caminhão parado


def rampa_cosseno(x: np.ndarray) -> np.ndarray:
    """0 para x <= 0, 1 para x >= 1 e meia onda de cosseno entre os dois."""
    return 0.5 * (1.0 - np.cos(math.pi * np.clip(x, 0.0, 1.0)))


def suavizar_abaixo(v: np.ndarray, n_meia: int) -> np.ndarray:
    """Perfil suave que nunca passa de v: mínimo móvel de 2n+1 amostras e duas médias móveis dentro dele.

    Cada amostra que entra na média do instante t é um mínimo sobre uma janela que contém t, então a
    média também fica abaixo de v(t).
    """
    v = np.asarray(v, float)
    if n_meia < 1 or v.size < 3:
        return v.copy()
    minimo = minimum_filter1d(v, size=2 * n_meia + 1, mode="nearest")
    m = n_meia + 1 if n_meia % 2 == 0 else n_meia  # ímpar, com o suporte das duas médias <= n_meia
    suave = uniform_filter1d(uniform_filter1d(minimo, m, mode="nearest"), m, mode="nearest")
    return np.minimum(suave, v)


def pesos_das_zonas(t: np.ndarray, zonas: Iterable[tuple[float, float, float]]) -> np.ndarray:
    """1 dentro das zonas (início, fim, rampa), 0 fora e rampa de cosseno nas bordas (sobrepostas: o maior)."""
    w = np.zeros_like(t, dtype=float)
    for ini, fim, r in zonas:
        if fim > ini:
            w = np.maximum(w, np.minimum(rampa_cosseno((t - ini) / r), rampa_cosseno((fim - t) / r)))
    return w


def zonas_efetivas(t: np.ndarray, v_x: np.ndarray, suave: np.ndarray, ajustes: Iterable[dict[str, Any]],
                   hz: float, tolerancia_mps: float = 0.05) -> list[tuple[float, float, float]]:
    """(início, fim, rampa) de cada zona, alargada até X e o perfil suavizado coincidirem nas duas rampas.

    Assim a mistura nas bordas não corta no meio uma retomada (ou frenagem) do próprio X.
    """
    ok = (v_x - suave) <= tolerancia_mps
    zonas = []
    for a in ajustes:
        ini, fim = float(a["t_ini"]), float(a["t_fim"])
        if fim <= ini:
            continue
        r = max(min(float(a.get("rampa_s", 1.0)), (fim - ini) / 2), 1.0 / hz)
        n_r = max(1, int(round(r * hz)))
        # estavel[j]: as n_r + 1 amostras que terminam em j têm X e o perfil suavizado praticamente iguais
        estavel = np.convolve(ok.astype(int), np.ones(n_r + 1, dtype=int))[: t.size] >= n_r + 1
        i_fim = int(np.searchsorted(t, fim))
        candidatos = np.flatnonzero(estavel[i_fim:])
        j_fim = min(i_fim + int(candidatos[0]), t.size - 1) if candidatos.size else t.size - 1
        comeca = np.zeros_like(estavel)
        comeca[: t.size - n_r] = estavel[n_r:]
        candidatos = np.flatnonzero(comeca[: int(np.searchsorted(t, ini)) + 1])
        j_ini = int(candidatos[-1]) if candidatos.size else 0
        zonas.append((float(t[j_ini]), float(t[j_fim]), r))
    return zonas


def perfil_ajustado(t: np.ndarray, v_x: np.ndarray, ajustes: Iterable[dict[str, Any]], hz: float) -> np.ndarray:
    """Velocidade de Y (m/s) no relógio de X com todos os `ajustes`; nunca passa de v_x."""
    ajustes = list(ajustes)
    v_x = np.clip(np.asarray(v_x, float), 0.0, None)
    if not ajustes:
        return v_x.copy()
    limite = v_x.copy()
    for a in ajustes:
        if a.get("vel_max_kmh"):
            nucleo = (t >= float(a["nucleo_ini"])) & (t <= float(a["nucleo_fim"]))
            limite[nucleo] = np.minimum(limite[nucleo], float(a["vel_max_kmh"]) / 3.6)
    janela = max(float(a.get("janela_s", 4.0)) for a in ajustes)
    suave = suavizar_abaixo(limite, int(round(janela / 2 * hz)))
    return v_x - pesos_das_zonas(t, zonas_efetivas(t, v_x, suave, ajustes, hz)) * (v_x - suave)


class PlanoVelocidade:
    """Velocidade alvo de Y ao longo do relógio de X, atualizada a cada ajuste que chega dos agentes."""

    def __init__(self, t: np.ndarray, v_x: np.ndarray, rampa_replanejamento_s: float = 1.5):
        self.t = np.asarray(t, float)
        self.v_x = np.clip(np.asarray(v_x, float), 0.0, None)
        self.hz = 1.0 / float(np.median(np.diff(self.t)))
        self.rampa_s = max(float(rampa_replanejamento_s), 1e-6)
        self.alvo = self.v_x.copy()
        self.ajustes: dict[Any, dict[str, Any]] = {}
        self.chegadas: list[dict[str, Any]] = []

    def atualizar(self, ajustes: Iterable[tuple[Any, dict[str, Any] | None]], tau: float) -> list[tuple[Any, dict[str, Any]]]:
        """Incorpora os ajustes (id, ajuste) ainda não vistos a partir do instante `tau` do plano; devolve os novos."""
        novos = [(i, a) for i, a in ajustes if a and i not in self.ajustes]
        if not novos:
            return []
        for i, a in novos:
            self.ajustes[i] = a
            self.chegadas.append({"id": i, "tau_chegada_s": round(float(tau), 3), "t_ini": a["t_ini"],
                                  "t_fim": a["t_fim"], "tipo": a["tipo"], "atrasado": bool(tau > a["t_ini"])})
        bruto = perfil_ajustado(self.t, self.v_x, self.ajustes.values(), self.hz)
        futuro = self.t >= tau
        beta = rampa_cosseno((self.t[futuro] - tau) / self.rampa_s)
        self.alvo[futuro] += beta * (bruto[futuro] - self.alvo[futuro])
        return novos

    def velocidade_x(self, tau: float) -> float:
        return float(np.interp(tau, self.t, self.v_x))

    def velocidade(self, tau: float) -> float:
        return float(np.interp(tau, self.t, self.alvo))

    def taxa(self, tau: float) -> float:
        """Quanto o relógio de X avança por segundo de Y: (v_alvo + c) / (v_X + c), em (0, 1]."""
        c = V_REGULARIZACAO_MPS
        return min(1.0, (self.velocidade(tau) + c) / (self.velocidade_x(tau) + c))
