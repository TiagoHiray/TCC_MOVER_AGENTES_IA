"""Consumo de combustível, custo, CO2 e métricas de condução de um trecho ou de uma volta.

O CARLA não simula combustível. O consumo vem do modelo VT-CPFM (Rakha et al.): linha de Willans
mais um termo quadrático, a partir da potência nas rodas, com a massa do caminhão:

    P_rodas = (m * a + arrasto(v) + rolamento + m * g * rampa) * v
    P_motor = max(0, P_rodas) / eficiência
    combustível (L/s) = marcha_lenta + a1 * P_motor * (1 + P_motor / P2),  a1 = BSFC / densidade

Frenando ou em desaceleração, o motor gasta só a marcha lenta. A energia tirada pelos freios é
a que o motor já tinha pago antes: é ela que a camada agêntica tenta não desperdiçar.

Há duas formas de chegar à potência:
- consumo medido (MedidorConsumo e consumo_medido): o painel ao vivo, a corrida e o benchmark usam o
  mesmo medidor. A potência de cada tick sai da variação da energia cinética da velocidade
  suavizada (média de dois ticks e média móvel exponencial causal de 0,16 s, corte de ~1 Hz) mais
  as resistências. Assim cada arrancada e cada frenagem entram com a energia inteira, o tremor de
  10 Hz da velocidade do CARLA não vira consumo e nada é somado antes de uma freada brusca (o
  filtro é causal e não oscila);
- previsões do modelo do caminhão (metricas(), usada pelo Decisor e pelo treino do modelo de ML): a
  aceleração simulada, suavizada em 1,5 Hz.

Os eventos bruscos (jerk) usam sempre a aceleração suavizada em 1,5 Hz, a mesma convenção da camada
de logs.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from mover.autonomo.veiculo import ParametrosCaminhao
from mover.tratamento.sinais import passa_baixa


@dataclass
class ParametrosConsumo:
    combustivel: str = "diesel"
    densidade_kg_l: float = 0.835
    bsfc_g_kwh: float = 225.0          # consumo específico do motor diesel (g por kWh)
    marcha_lenta_l_h: float = 2.0
    eficiencia_transmissao: float = 0.88
    potencia_quadratica_kw: float = 1000.0  # termo a2 do VT-CPFM: +10 % de consumo marginal a 100 kW
    preco_r_l: float = 6.00
    co2_kg_l: float = 2.68             # diesel fóssil

    @classmethod
    def de_dict(cls, dados: dict[str, Any] | None) -> "ParametrosConsumo":
        nomes = {f.name for f in fields(cls)}
        convertidos = {}
        for k, v in (dados or {}).items():
            if k in nomes:
                convertidos[k] = v if k == "combustivel" else float(v)
        return cls(**convertidos)


@dataclass
class Limiares:
    atencao_mps3: float = 2.5
    critico_mps3: float = 5.0
    agrupar_s: float = 1.0
    fc_hz: float = 1.5                 # a aceleração é suavizada em 1,5 Hz antes do jerk


def potencia_rodas_kw(v: np.ndarray, a: np.ndarray, rampa: np.ndarray | float, c: ParametrosCaminhao) -> np.ndarray:
    v = np.asarray(v, float)
    a = np.asarray(a, float)
    rampa = np.broadcast_to(np.asarray(rampa, float), v.shape)
    aero = 0.5 * c.densidade_ar * c.cd * c.area_frontal_m2 * v * v
    rolamento = np.where(v > 1e-3, c.massa_kg * c.g * c.cr, 0.0)
    forca = c.massa_kg * a + aero + rolamento + c.massa_kg * c.g * rampa
    return forca * v / 1000.0


def taxa_combustivel_l_s(p_rodas_kw: np.ndarray, p: ParametrosConsumo) -> np.ndarray:
    """VT-CPFM: marcha lenta + a1*P + a2*P^2 com P >= 0 (potência do motor); só marcha lenta com P < 0."""
    p_motor = np.maximum(np.asarray(p_rodas_kw, float), 0.0) / p.eficiencia_transmissao
    a1 = p.bsfc_g_kwh / (p.densidade_kg_l * 1000.0) / 3600.0  # L por kW·s
    extra = a1 * p_motor * (1.0 + p_motor / p.potencia_quadratica_kw)
    return p.marcha_lenta_l_h / 3600.0 + extra


TAU_CONSUMO_S = 0.16  # suavização da velocidade no consumo medido: média móvel exponencial com corte de ~1 Hz


class MedidorConsumo:
    """Consumo medido, acumulado tick a tick: o mesmo no painel ao vivo, na corrida e no benchmark.

    A potência nas rodas de cada tick sai da variação da energia cinética da velocidade suavizada
    mais o arrasto, o rolamento e a rampa. A velocidade passa por uma média de dois ticks, que zera o
    tremor de 10 Hz (tick a tick) da velocidade do CARLA, e por uma média móvel exponencial causal
    de `tau_s` (corte de ~1 Hz). Assim:
    - cada arrancada e cada frenagem entram com a energia inteira (m·Δ(v²)/2), sem a defasagem de
      multiplicar uma aceleração filtrada pela velocidade do instante;
    - o tremor não vira consumo (com a potência baixa, a parte positiva dele seria somada);
    - nada é somado antes de uma freada brusca: um filtro de fase zero (o Butterworth de 1,5 Hz usado
      antes) oscila antes do degrau e somava consumo que não existiu.
    """

    def __init__(self, caminhao: ParametrosCaminhao, consumo: ParametrosConsumo, tau_s: float = TAU_CONSUMO_S):
        self.caminhao = caminhao
        self.consumo = consumo
        self.tau_s = float(tau_s)
        self.v_suave: float | None = None
        self._v_tick_anterior: float | None = None
        self.litros = 0.0
        self.energia_rodas_kwh = 0.0
        self.energia_freio_kwh = 0.0

    def incrementos(self, v: float, dt: float, rampa: float = 0.0) -> tuple[float, float, float]:
        """Avança um tick e devolve (litros, kWh nas rodas, kWh nos freios) desse tick, sem acumular."""
        v = max(0.0, float(v))
        media = v if self._v_tick_anterior is None else 0.5 * (v + self._v_tick_anterior)
        self._v_tick_anterior = v
        anterior = media if self.v_suave is None else self.v_suave
        atual = anterior + (media - anterior) * min(1.0, dt / self.tau_s)
        self.v_suave = atual
        c = self.caminhao
        cinetica_w = 0.5 * c.massa_kg * (atual * atual - anterior * anterior) / dt
        p_kw = (cinetica_w + c.resistencia_n(atual, float(rampa)) * atual) / 1000.0
        litros = float(taxa_combustivel_l_s(p_kw, self.consumo)) * dt
        return litros, max(p_kw, 0.0) * dt / 3600.0, max(-p_kw, 0.0) * dt / 3600.0

    def passo(self, v: float, dt: float, rampa: float = 0.0) -> float:
        """Acumula um tick (v em m/s, dt em s, rampa = tangente da inclinação) e devolve os litros acumulados."""
        litros, rodas, freio = self.incrementos(v, dt, rampa)
        self.litros += litros
        self.energia_rodas_kwh += rodas
        self.energia_freio_kwh += freio
        return self.litros

    @property
    def custo_r(self) -> float:
        return self.litros * self.consumo.preco_r_l


def consumo_medido(v: np.ndarray, dt: float, c: ParametrosCaminhao, p: ParametrosConsumo,
                   rampa: np.ndarray | float = 0.0, tau_s: float = TAU_CONSUMO_S) -> dict[str, np.ndarray]:
    """O MedidorConsumo sobre uma série inteira: litros, kWh nas rodas e kWh nos freios de cada tick.

    A soma acumulada de "litros" é a coluna combustivel_ml que a corrida grava ao vivo (em L).
    """
    v = np.asarray(v, float)
    rampa = np.broadcast_to(np.asarray(rampa, float), v.shape)
    medidor = MedidorConsumo(c, p, tau_s)
    saida = np.array([medidor.incrementos(float(vi), dt, float(ri)) for vi, ri in zip(v, rampa)], float).reshape(-1, 3)
    return {"litros": saida[:, 0], "rodas_kwh": saida[:, 1], "freio_kwh": saida[:, 2]}


def suavizar(a: np.ndarray, dt: float, fc_hz: float) -> np.ndarray:
    return passa_baixa(np.asarray(a, float), 1.0 / dt, fc_hz, ordem=4)


def jerk(a_suave: np.ndarray, dt: float) -> np.ndarray:
    return np.gradient(np.asarray(a_suave, float), dt) if len(a_suave) > 1 else np.zeros(len(a_suave))


def eventos_jerk(t: np.ndarray, j: np.ndarray, limiares: Limiares) -> list[dict[str, Any]]:
    """Grupos de |jerk| acima do limiar de atenção (picos a menos de `agrupar_s` formam um evento)."""
    t = np.asarray(t, float)
    j = np.asarray(j, float)
    acima = np.flatnonzero(np.abs(j) >= limiares.atencao_mps3)
    if acima.size == 0:
        return []
    cortes = np.flatnonzero(np.diff(t[acima]) >= limiares.agrupar_s) + 1
    eventos = []
    for grupo in np.split(acima, cortes):
        k = int(grupo[np.argmax(np.abs(j[grupo]))])
        maximo = float(abs(j[k]))
        eventos.append({"t_ini": float(t[grupo[0]]), "t_fim": float(t[grupo[-1]]), "t_pico": float(t[k]),
                        "jerk_mps3": float(j[k]), "nivel": "critico" if maximo >= limiares.critico_mps3 else "atencao"})
    return eventos


def metricas(t: np.ndarray, v: np.ndarray, a: np.ndarray, c: ParametrosCaminhao, p: ParametrosConsumo,
             limiares: Limiares | None = None, rampa: np.ndarray | float = 0.0, ja_suave: bool = False) -> dict[str, Any]:
    """Métricas de um trecho amostrado a passo constante (t, v em m/s, a em m/s²).

    O consumo e as energias daqui usam a aceleração suavizada em 1,5 Hz: servem às previsões do modelo
    do caminhão. Na telemetria medida (corrida e benchmark), metricas_da_corrida troca esses campos
    pelos do MedidorConsumo, o mesmo do painel.
    """
    limiares = limiares or Limiares()
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    if len(t) < 2:
        return {"tempo_s": 0.0, "distancia_m": 0.0, "combustivel_l": 0.0, "custo_r": 0.0, "co2_kg": 0.0,
                "eventos_atencao": 0, "eventos_criticos": 0, "jerk_max_abs_mps3": 0.0, "acel_min_mps2": 0.0,
                "acel_max_mps2": 0.0, "energia_rodas_kwh": 0.0, "energia_freio_kwh": 0.0, "vel_media_kmh": 0.0}
    dt = float(np.median(np.diff(t)))
    a_suave = np.asarray(a, float) if ja_suave else suavizar(a, dt, limiares.fc_hz)
    j = jerk(a_suave, dt)
    p_rodas = potencia_rodas_kw(v, a_suave, rampa, c)
    combustivel = float(np.sum(taxa_combustivel_l_s(p_rodas, p)) * dt)
    eventos = eventos_jerk(t, j, limiares)
    distancia = float(np.sum(v) * dt)
    tempo = float(len(t) * dt)
    return {
        "tempo_s": round(tempo, 2),
        "distancia_m": round(distancia, 1),
        "vel_media_kmh": round(3.6 * distancia / tempo, 2) if tempo > 0 else 0.0,
        "combustivel_l": combustivel,
        "custo_r": combustivel * p.preco_r_l,
        "co2_kg": combustivel * p.co2_kg_l,
        "energia_rodas_kwh": float(np.sum(np.maximum(p_rodas, 0.0)) * dt / 3600.0),
        "energia_freio_kwh": float(np.sum(np.maximum(-p_rodas, 0.0)) * dt / 3600.0),
        "acel_min_mps2": round(float(a_suave.min()), 2),
        "acel_max_mps2": round(float(a_suave.max()), 2),
        "jerk_max_abs_mps3": round(float(np.abs(j).max()), 2),
        "eventos_atencao": sum(e["nivel"] == "atencao" for e in eventos),
        "eventos_criticos": sum(e["nivel"] == "critico" for e in eventos),
    }
