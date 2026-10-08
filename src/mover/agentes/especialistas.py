"""Especialistas da camada agêntica.

- EspecialistaJerk: regras sobre o jerk longitudinal (o "problema" da v1), com dois níveis
  (atenção e crítico), causa provável (lombada ou condução brusca) e nome do evento.
- EspecialistaML: o IsolationForest da POC, retreinado com a telemetria tratada e avaliado
  por janela (fração de quadros atípicos).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mover.agentes.fatos import fonte_do_trecho, hora_curta

log = logging.getLogger("mover.agentes")


@dataclass(frozen=True)
class EventoJerk:
    """Variação brusca da aceleração longitudinal detectada pelo EspecialistaJerk."""

    t_ini: float                   # início do grupo de picos (s)
    t_fim: float                   # fim do grupo de picos (s)
    t_pico: float                  # instante do 1º pico, que define o sinal do evento (s)
    hora: str                      # hora local do 1º pico (HH:MM:SS)
    jerk_pico_mps3: float          # jerk no 1º pico (com sinal)
    jerk_max_abs_mps3: float       # maior |jerk| do grupo; define o nível
    nivel: str                     # atencao | critico
    sinal: str                     # positivo | negativo
    tipo: str                      # frenagem_brusca | corte_aceleracao | arrancada_brusca | soltura_freio
    causa: str                     # conducao_brusca | irregularidade_via
    vel_kmh: float                 # velocidade no 1º pico
    acel_long_min_mps2: float      # no grupo ± margem da lombada
    acel_long_max_mps2: float
    acel_vert_max_abs_mps2: float
    fonte: str                     # real | sintetico | misto

    @property
    def chave_conhecimento(self) -> str:
        """Entrada da base de conhecimento do YAML usada no diagnóstico e na ação."""
        return "irregularidade_via" if self.causa == "irregularidade_via" else self.tipo

    def fatos(self) -> dict[str, Any]:
        return asdict(self)


class EspecialistaJerk:
    """Encontra e classifica picos de |jerk| acima do limiar de atenção."""

    def __init__(self, limiares: dict[str, Any]):
        cj, cl, cr = limiares["jerk"], limiares["lombada"], limiares["rotulo"]
        self.coluna = cj.get("coluna", "jerk_long")
        self.atencao = float(cj["atencao_mps3"])
        self.critico = float(cj["critico_mps3"])
        self.agrupar_s = float(cj["agrupar_picos_s"])
        self.coluna_vertical = cl.get("coluna", "acc_vert")
        self.limiar_vertical = float(cl["acel_vertical_mps2"])
        self.margem_vertical = float(cl["margem_s"])
        self.acel_forte = float(cr["acel_forte_mps2"])
        self.janela_apos = float(cr["janela_apos_s"])

    def detectar(self, dados: pd.DataFrame, t_ini: float, t_fim: float) -> list[EventoJerk]:
        """Eventos cujo 1º pico cai em [t_ini, t_fim).

        `dados` pode ter linhas antes e depois do intervalo: elas só servem para delimitar
        grupos de picos que cruzam a fronteira do bloco.
        """
        t = dados["sim_time"].to_numpy(dtype=float)
        jerk = dados[self.coluna].to_numpy(dtype=float)
        acima = np.flatnonzero(np.abs(jerk) >= self.atencao)
        if acima.size == 0:
            return []
        cortes = np.flatnonzero(np.diff(t[acima]) >= self.agrupar_s) + 1
        eventos = []
        for grupo in np.split(acima, cortes):
            evento = self._classificar(dados, t, jerk, grupo)
            if t_ini - 1e-9 <= evento.t_pico < t_fim - 1e-9:
                eventos.append(evento)
        return eventos

    def _classificar(self, dados: pd.DataFrame, t: np.ndarray, jerk: np.ndarray, grupo: np.ndarray) -> EventoJerk:
        # 1º pico: maior |jerk| da primeira sequência contínua de amostras acima do limiar.
        # Numa frenagem, a entrada no freio (jerk negativo) vem antes da soltura (positivo).
        primeira = np.split(grupo, np.flatnonzero(np.diff(grupo) > 1) + 1)[0]
        k = int(primeira[np.argmax(np.abs(jerk[primeira]))])
        max_abs = float(np.max(np.abs(jerk[grupo])))
        nivel = "critico" if max_abs >= self.critico else "atencao"
        sinal = "positivo" if jerk[k] > 0 else "negativo"

        a_long = dados["acc_long"].to_numpy(dtype=float)
        apos = (t > t[k]) & (t <= t[k] + self.janela_apos)
        if sinal == "negativo":
            forte = apos.any() and a_long[apos].min() <= -self.acel_forte
            tipo = "frenagem_brusca" if forte else "corte_aceleracao"
        else:
            forte = apos.any() and a_long[apos].max() >= self.acel_forte
            tipo = "arrancada_brusca" if forte else "soltura_freio"

        vizinhanca = (t >= t[grupo[0]] - self.margem_vertical) & (t <= t[grupo[-1]] + self.margem_vertical)
        vertical = float(np.max(np.abs(dados[self.coluna_vertical].to_numpy(dtype=float)[vizinhanca])))
        causa = "irregularidade_via" if vertical >= self.limiar_vertical else "conducao_brusca"

        return EventoJerk(
            t_ini=round(float(t[grupo[0]]), 2),
            t_fim=round(float(t[grupo[-1]]), 2),
            t_pico=round(float(t[k]), 2),
            hora=hora_curta(dados["hora_local"].iloc[k]),
            jerk_pico_mps3=round(float(jerk[k]), 2),
            jerk_max_abs_mps3=round(max_abs, 2),
            nivel=nivel,
            sinal=sinal,
            tipo=tipo,
            causa=causa,
            vel_kmh=round(float(dados["speed_kmh"].iloc[k]), 1),
            acel_long_min_mps2=round(float(a_long[vizinhanca].min()), 2),
            acel_long_max_mps2=round(float(a_long[vizinhanca].max()), 2),
            acel_vert_max_abs_mps2=round(vertical, 2),
            fonte=fonte_do_trecho(dados["fonte"][vizinhanca]),
        )


class EspecialistaML:
    """IsolationForest (como na POC) avaliado por janela: fração de quadros atípicos."""

    def __init__(self, pacote: dict[str, Any], fracao_atipica: float):
        self.modelo = pacote["modelo"]
        self.features = list(pacote["features"])
        self.fracao_atipica = float(fracao_atipica)

    @classmethod
    def carregar(cls, caminho_modelo: Path, fracao_atipica: float) -> "EspecialistaML":
        import joblib  # import tardio: só quem usa o especialista de ML precisa do joblib

        pacote = joblib.load(caminho_modelo)
        log.info("Especialista de ML carregado: %s (features: %s)", caminho_modelo, ", ".join(pacote["features"]))
        return cls(pacote, fracao_atipica)

    def avaliar(self, trecho: pd.DataFrame) -> dict[str, Any]:
        x = trecho[self.features]
        rotulos = self.modelo.predict(x)              # -1 = anômalo, como na POC
        pontuacao = self.modelo.decision_function(x)  # < 0 = anômalo
        fracao = float(np.mean(rotulos == -1))
        return {
            "ml_quadros_atipicos_pct": int(round(100 * fracao)),
            "ml_pontuacao_min": round(float(pontuacao.min()), 3),
            "ml_atipica": bool(fracao >= self.fracao_atipica),
        }
