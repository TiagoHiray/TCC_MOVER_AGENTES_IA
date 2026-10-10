"""Testes do gêmeo ao vivo: estimador causal contra o tratamento offline, serviço HTTP e repasse."""

from __future__ import annotations

import csv
import json
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pytest
from auxiliares import telemetria
from fastapi.testclient import TestClient

from mover.agentes.llm import FalsoLLM
from mover.agentes.supervisor import Supervisor
from mover.config import caminho, carregar_yaml
from mover.gemeo.calibracao_fixa import CalibracaoFixa
from mover.gemeo.estimador import EstimadorOnline
from mover.ingestao.replay_csv import arquivos_da_gravacao, lotes, reenviar
from mover.ingestao.repasse import Repassador
from mover.servidor.app import criar_app

RELATORIO = caminho("data/tratado/relatorio_tratamento.json")
ALINHAMENTO = caminho("data/simulacao/alinhamento.json")
TELEMETRIA = caminho("data/tratado/telemetria_tratada.csv")
GRAVACAO = caminho("data/csv_maua")
SENSORES = ["accelerometer", "gyroscope", "location"]

precisa_dados = pytest.mark.skipif(not (RELATORIO.exists() and TELEMETRIA.exists() and GRAVACAO.exists()),
                                   reason="volta gravada e tratamento não disponíveis")


@pytest.fixture(scope="module")
def cfg() -> dict[str, Any]:
    return carregar_yaml("config/agentes.yaml")


@precisa_dados
def test_estimador_ao_vivo_acompanha_o_tratamento_offline():
    cal = CalibracaoFixa.carregar(RELATORIO, ALINHAMENTO)
    est = EstimadorOnline(cal, carregar_yaml("config/tratamento.yaml")["fusao"])
    estados = []
    for _, _, payload in lotes(arquivos_da_gravacao(GRAVACAO, SENSORES), 1.0):  # lotes de 1 s, como o app
        for r in payload:
            est.adicionar(r["name"], r["time"], r["values"])
        estados += est.processar()

    df = pd.DataFrame([e.para_dict() for e in estados])
    ref = pd.read_csv(TELEMETRIA)
    dentro = (df.t_unix >= ref.t_unix.iloc[0]) & (df.t_unix <= ref.t_unix.iloc[-1])
    erro = np.hypot(df.x - np.interp(df.t_unix, ref.t_unix, ref.x), df.y - np.interp(df.t_unix, ref.t_unix, ref.y))[dentro]
    erro_v = (df.v - np.interp(df.t_unix, ref.t_unix, ref.speed_mps)).abs()[dentro]

    assert len(df) > 0.95 * len(ref)
    assert np.allclose(np.diff(df.t_unix), 0.05, atol=1e-3)
    assert erro.median() < 2.0 and erro.quantile(0.95) < 7.0
    assert erro_v.median() < 0.3
    assert est.contagem["retrocessos"] > 100 and est.contagem["fixes_antigos"] == 0
    assert df.x_carla.notna().all()


def test_servico_publica_estado_pela_ingestao(cfg, tmp_path):
    if not RELATORIO.exists():
        pytest.skip("calibração não disponível")
    cfg_sim = {"servidor": {"pasta_sessoes": str(tmp_path / "sessoes")},
               "ingestao": {"pasta": str(tmp_path / "ingestao"), "token": None},
               "gemeo": {"ativo": True, "pasta": str(tmp_path / "gemeo"), "calibracao": str(RELATORIO),
                         "alinhamento": str(ALINHAMENTO)}}
    app = criar_app(cfg, cfg_sim, telemetria=telemetria(10.0),
                    fabrica_supervisor=lambda opcoes: Supervisor(FalsoLLM(), cfg), sem_ml=True)
    with TestClient(app) as cliente:
        resumo = reenviar(arquivos_da_gravacao(GRAVACAO, SENSORES),
                          lambda m: cliente.post("/ingestao/sensorlogger", json=m).status_code == 200,
                          periodo_s=1.0, duracao_s=30, sem_espera=True)
        status = cliente.get("/gemeo/estado").json()

    gravacao = status["gravacoes"][resumo["sessao"]]
    assert gravacao["inicializado"] and gravacao["estados"] > 400
    assert gravacao["ultimo"]["x_carla"] is not None
    with open(gravacao["csv"], newline="") as f:
        linhas = list(csv.DictReader(f))
    assert len(linhas) == gravacao["estados"] and "yaw_carla" in linhas[0]


def test_gemeo_desligado_sem_secao_no_yaml(cfg, tmp_path):
    app = criar_app(cfg, {"servidor": {"pasta_sessoes": str(tmp_path)}, "ingestao": {"pasta": str(tmp_path)}},
                    telemetria=telemetria(10.0), fabrica_supervisor=lambda o: Supervisor(FalsoLLM(), cfg), sem_ml=True)
    with TestClient(app) as cliente:
        assert cliente.get("/gemeo/estado").status_code == 404


def test_repassador_envia_em_ordem_e_conta_falhas():
    recebidas: list[int] = []

    def destino(req: httpx.Request) -> httpx.Response:
        mid = json.loads(req.content)["messageId"]
        if mid == 2:
            return httpx.Response(500)
        recebidas.append(mid)
        return httpx.Response(200)

    rep = Repassador("http://vm:8000/ingestao/sensorlogger", cliente=httpx.Client(transport=httpx.MockTransport(destino)))
    for mid in range(5):
        rep.receber({"messageId": mid, "sessionId": "s", "payload": []})
    rep.fechar()
    assert recebidas == [0, 1, 3, 4]
    st = rep.status()
    assert (st["enviadas"], st["falhas"], st["ultimo_erro"]) == (4, 1, "HTTP 500")


def test_url_de_repasse_invalida():
    with pytest.raises(ValueError):
        Repassador("ftp://vm/x", cliente=object())
