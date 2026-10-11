"""Monta as peças da corrida autônoma a partir do config/autonomo.yaml e do plano de missão.

Usado pela corrida ao vivo (dirigir.py), pelo benchmark e pelos testes, para que todos usem os
mesmos parâmetros de caminhão, piloto, governador, consumo e decisão.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mover.agentes.decisao import Decisor
from mover.agentes.llm import ClienteLLM
from mover.agentes.telemetria_viva import MedidorConsumo
from mover.autonomo.consumo import Limiares, ParametrosConsumo, eventos_jerk, jerk, suavizar
from mover.autonomo.governador import OPCOES_PADRAO, ConfigGovernador, Governador, Intervencao
from mover.autonomo.missao import PlanoMissao
from mover.autonomo.planejador import PlanejadorCaminhao
from mover.autonomo.previsao import ModeloInterno, Trajetoria
from mover.autonomo.simulador import SimuladorSubstituto
from mover.autonomo.veiculo import ParametrosCaminhao, ParametrosPiloto
from mover.config import caminho, carregar_yaml

PASSO_S = 0.05  # 20 Hz, o mesmo fixed_delta_seconds do CARLA


@dataclass
class Montagem:
    cfg: dict[str, Any]
    plano: PlanoMissao
    caminhao: ParametrosCaminhao
    piloto: ParametrosPiloto
    consumo: ParametrosConsumo
    limiares: Limiares
    cfg_governador: ConfigGovernador
    opcoes: dict[str, tuple]
    dt: float = PASSO_S

    @property
    def cfg_decisao(self) -> dict[str, Any]:
        return dict(self.cfg.get("decisao") or {})

    @property
    def duracao_bloco_s(self) -> float:
        return float(self.cfg_decisao.get("horizonte_s", 10.0))

    def governador(self, modo: str, voltas: int) -> Governador:
        return Governador(self.plano, self.cfg_governador, modo=modo, n_voltas=voltas, opcoes=self.opcoes)

    def modelo(self) -> ModeloInterno:
        return ModeloInterno(self.caminhao, self.piloto, self.dt, self.consumo, self.limiares)

    def decisor(self, voltas: int, llm: ClienteLLM | None = None, suavizador: Any = None) -> Decisor:
        """O Decisor da camada agêntica; com `suavizador` (modelo de ML), ele propõe quanto suavizar."""
        return Decisor(self.plano, self.modelo(), self.cfg_decisao, llm=llm, opcoes=self.opcoes, n_voltas=voltas,
                       suavizador=suavizador)

    def planejador(self, voltas: int) -> PlanejadorCaminhao:
        """O planejador do caminhão: o plano dos próximos 10 s, a partir do estado atual da corrida."""
        return PlanejadorCaminhao(self.plano, self.modelo(), voltas, float(self.cfg_decisao.get("margem_elementos_m", 20.0)))

    def suavizador(self, treinar_se_faltar: bool = True) -> Any:
        """O modelo de ML de suavização (config/autonomo.yaml, seção suavizador_ml); treina um se faltar."""
        from mover.agentes.suavizador_ml import carregar_suavizador

        return carregar_suavizador(self.cfg, treinar_se_faltar)

    def simulador_substituto(self) -> SimuladorSubstituto:
        return SimuladorSubstituto(self.plano.rota, self.caminhao, self.piloto, self.dt)

    # ------------------------------------------------------------------ previsões da corrida inteira
    def simular_corrida(self, modo: str, voltas: int, opcao: str = "suave") -> Trajetoria:
        """A corrida inteira no modelo interno: bruto, ou com `opcao` em todos os elementos do plano."""
        gov = self.governador(modo, voltas)
        if modo == "agentes":
            interv = Intervencao.da_opcao(opcao, self.opcoes)
            for inst in self.plano.instancias(0.0, voltas * self.plano.comprimento + 1.0, voltas):
                gov.definir(inst.chave, interv)
        return self.modelo().simular(gov, 0.0, 0.0, 0.0, 0.0, 400.0 * voltas)

    def linha_de_base(self, voltas: int, passo_m: float = 5.0) -> dict[str, Any]:
        """O plano do caminhão sem agentes, previsto pelo modelo: consumo e eventos acumulados pela distância.

        Numa corrida só não dá para medir as duas versões; a página compara a corrida com essa previsão.
        A comparação medida (N voltas sem e N com os agentes) é a do benchmark.
        """
        traj = self.simular_corrida("bruto", voltas)
        medidor = MedidorConsumo(self.caminhao, self.consumo)
        litros = np.array([medidor.passo(float(v), float(a), self.dt) for v, a in zip(traj.v, traj.a)])
        a_suave = suavizar(traj.a, self.dt, self.limiares.fc_hz)
        eventos = eventos_jerk(traj.t, jerk(a_suave, self.dt), self.limiares)
        d_eventos = np.sort([float(np.interp(e["t_pico"], traj.t, traj.d)) for e in eventos])
        grade = np.arange(0.0, float(traj.d[-1]) + passo_m, passo_m)
        d_mono = np.maximum.accumulate(traj.d)
        primeiro = np.searchsorted(d_mono, grade, side="left").clip(0, len(traj.t) - 1)
        return {
            "fonte": "plano do caminhão sem agentes, previsto pelo modelo do caminhão",
            "fonte_curta": "previsto",
            "passo_m": passo_m,
            "d": np.round(grade, 1).tolist(),
            "t": np.round(traj.t[primeiro], 2).tolist(),
            "combustivel_ml": np.round(1000.0 * litros[primeiro], 1).tolist(),
            "eventos": np.searchsorted(d_eventos, grade, side="right").tolist(),
            "total": {"tempo_s": round(float(traj.t[-1]), 1), "combustivel_ml": round(1000.0 * float(litros[-1]), 1),
                      "eventos": len(eventos)},
        }


def montar(cfg: dict[str, Any] | None = None, arquivo_missao: str | Path | None = None,
           plano: PlanoMissao | None = None) -> Montagem:
    cfg = cfg if cfg is not None else carregar_yaml("config/autonomo.yaml")
    if plano is None:
        arquivo = arquivo_missao or (cfg.get("missao") or {}).get("arquivo", "config/missao.yaml")
        destino = caminho(arquivo)
        if not destino.exists():
            raise SystemExit(f"Plano de missão não encontrado: {destino}. Gere com: python -m mover.autonomo.gerar_missao")
        plano = PlanoMissao.carregar(destino)
    lim = cfg.get("limiares") or {}
    opcoes = {k: tuple(v) for k, v in (cfg.get("opcoes") or OPCOES_PADRAO).items()}
    return Montagem(
        cfg=cfg,
        plano=plano,
        caminhao=ParametrosCaminhao.de_dict(cfg.get("veiculo")),
        piloto=ParametrosPiloto.de_dict(cfg.get("piloto")),
        consumo=ParametrosConsumo.de_dict(cfg.get("consumo")),
        limiares=Limiares(float(lim.get("atencao_mps3", 2.5)), float(lim.get("critico_mps3", 5.0)),
                          float(lim.get("agrupar_s", 1.0)), float(lim.get("fc_hz", 1.5))),
        cfg_governador=ConfigGovernador.de_dict(cfg.get("governador")),
        opcoes=opcoes,
    )


__all__ = ["Montagem", "PASSO_S", "montar"]
