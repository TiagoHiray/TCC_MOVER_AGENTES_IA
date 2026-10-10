"""Liga o estimador ao vivo à ingestão: uma instância por gravação, CSV contínuo e rotas HTTP.

    GET /gemeo/estado[?token=...]  -> último estado de cada gravação + contadores do filtro

Ativado pela seção `gemeo` do config/simulacao.yaml (ativo: true).
"""

from __future__ import annotations

import csv
import logging
import threading
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from mover.config import caminho, carregar_yaml
from mover.gemeo.calibracao_fixa import CalibracaoFixa
from mover.gemeo.estimador import EstadoVeiculo, EstimadorOnline
from mover.ingestao.armazem import ArmazemIngestao, nome_seguro

log = logging.getLogger("mover.gemeo")

COLUNAS = [f.name for f in fields(EstadoVeiculo)]


class ServicoGemeo:
    def __init__(self, cal: CalibracaoFixa, cfg_fusao: dict[str, Any], pasta: Path):
        self.cal = cal
        self.cfg_fusao = cfg_fusao
        self.pasta = pasta
        self._trava = threading.Lock()
        self._sessoes: dict[str, dict[str, Any]] = {}

    def _sessao(self, sid: str) -> dict[str, Any]:
        s = self._sessoes.get(sid)
        if s is None:
            pasta = self.pasta / f"{datetime.now():%Y%m%d-%H%M%S}_{nome_seguro(sid, 'anon')[:8]}"
            pasta.mkdir(parents=True, exist_ok=True)
            s = self._sessoes[sid] = {"estimador": EstimadorOnline(self.cal, self.cfg_fusao),
                                      "csv": pasta / "estado_ao_vivo.csv", "n": 0, "ultimo": None}
            with open(s["csv"], "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(COLUNAS)
            log.info("Gêmeo: nova gravação %s -> %s", sid, s["csv"])
        return s

    def receber(self, msg: dict[str, Any]) -> list[EstadoVeiculo]:
        payload = msg.get("payload")
        if not isinstance(payload, list):
            return []
        with self._trava:
            s = self._sessao(str(msg.get("sessionId") or "anon"))
            est: EstimadorOnline = s["estimador"]
            for leitura in payload:
                if isinstance(leitura, dict) and isinstance(leitura.get("time"), (int, float)) \
                        and isinstance(leitura.get("values"), dict):
                    est.adicionar(str(leitura.get("name", "")), int(leitura["time"]), leitura["values"])
            novos = est.processar()
            if novos:
                with open(s["csv"], "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    for e in novos:
                        d = e.para_dict()
                        w.writerow([round(d[c], 4) if isinstance(d[c], float) else d[c] for c in COLUNAS])
                s["n"] += len(novos)
                s["ultimo"] = novos[-1].para_dict()
            return novos

    def status(self) -> dict[str, Any]:
        with self._trava:
            return {"calibracao": self.cal.origem, "com_mapa": self.cal.mapa is not None,
                    "gravacoes": {sid: {"csv": str(s["csv"]), "estados": s["n"], "ultimo": s["ultimo"],
                                        "inicializado": s["estimador"].t0 is not None,
                                        "filtro": dict(s["estimador"].contagem)}
                                  for sid, s in self._sessoes.items()}}


def instalar_gemeo(app: FastAPI, cfg_simulacao: dict[str, Any], armazem: ArmazemIngestao,
                   forcar: bool = False) -> ServicoGemeo | None:
    cfg = cfg_simulacao.get("gemeo", {})
    if not (forcar or cfg.get("ativo", False)):
        return None
    relatorio = caminho(cfg.get("calibracao", "data/tratado/relatorio_tratamento.json"))
    if not relatorio.exists():
        log.warning("Gêmeo desligado: calibração %s não encontrada (rode o tratamento da volta de calibração).",
                    relatorio)
        return None
    cal = CalibracaoFixa.carregar(relatorio, caminho(cfg.get("alinhamento", "data/simulacao/alinhamento.json")))
    cfg_fusao = carregar_yaml(cfg.get("config_tratamento", "config/tratamento.yaml"))["fusao"]
    servico = ServicoGemeo(cal, cfg_fusao, caminho(cfg.get("pasta", "data/gemeo")))
    armazem.ouvintes.append(servico.receber)
    app.state.gemeo = servico
    autorizar = app.state.autorizar_ingestao
    log.info("Gêmeo ativo: calibração de %s%s", cal.origem, " + alinhamento ao mapa" if cal.mapa else "")

    @app.get("/gemeo/estado")
    async def estado(token: str | None = None) -> dict[str, Any]:
        autorizar(token)
        return servico.status()

    return servico
