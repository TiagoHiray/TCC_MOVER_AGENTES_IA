"""Testes do servidor do painel (FastAPI + WebSocket) para a corrida autônoma, sem LLM e sem CARLA.

Rodar a partir da raiz do repositório:
    python -m pytest tests -q

O piloto é imitado: as linhas de telemetria e as decisões vêm de uma corrida curta no simulador
substituto (plano do estádio de tests/auxiliares.py). A camada de logs usa o provedor falso. As
mensagens do WebSocket são lidas com prazo, para um erro virar falha e não travar o pytest.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import anyio
import pytest
from auxiliares import plano_estadio
from fastapi.testclient import TestClient

from mover.agentes.llm import FalsoLLM
from mover.agentes.supervisor import Supervisor
from mover.autonomo.conducao import Conducao, OpcoesConducao
from mover.autonomo.montagem import montar
from mover.config import carregar_yaml
from mover.servidor.app import criar_app


@pytest.fixture(scope="module")
def cfg() -> dict[str, Any]:
    return carregar_yaml("config/agentes.yaml")


@pytest.fixture(scope="module")
def corrida() -> dict[str, Any]:
    """Uma volta com os agentes no substituto: as linhas e as decisões que o piloto mandaria."""
    m = montar(carregar_yaml("config/autonomo.yaml"), plano=plano_estadio())
    decisor = m.decisor(1, FalsoLLM())
    res = Conducao(m.plano, m.simulador_substituto(), m.governador("agentes", 1), decisor,
                   OpcoesConducao(voltas=1, assincrona=False, sem_espera=True), m.caminhao, m.consumo).rodar()
    decisor.encerrar()
    return {"m": m, "linhas": res.linhas, "decisoes": res.decisoes, "base": m.linha_de_base(1)}


def montar_app(cfg: dict[str, Any], pasta: Path, fabrica_llm_chat: Callable[[Any], Any] | None = None,
               cfg_autonomo: dict[str, Any] | None = None):
    cfg_sim = {"servidor": {"pasta_sessoes": str(pasta / "sessoes")}, "mapa": {"arquivo": str(pasta / "nao_existe.xodr")}}
    return criar_app(cfg, cfg_sim, fabrica_supervisor=lambda opcoes: Supervisor(FalsoLLM(), cfg),
                     fabrica_llm_chat=fabrica_llm_chat or (lambda opcoes: FalsoLLM()),
                     cfg_autonomo=cfg_autonomo or {"benchmark": {"pasta": str(pasta / "benchmark")},
                                                   "missao": {"arquivo": str(pasta / "nao_existe.yaml")}})


def opcoes_sessao(corrida: dict[str, Any], **extra: Any) -> dict[str, Any]:
    m = corrida["m"]
    return {"modo": "agentes", "voltas": 1, "simulador": "substituto", "llm_decisoes": "falso",
            "comprimento_volta_m": m.plano.comprimento, "duracao_prevista_s": 60.0, "linha_de_base": corrida["base"],
            "tempo_real": False, **extra}


def receber(ws, prazo_s: float = 5.0) -> dict[str, Any]:
    """Próxima mensagem do WebSocket de teste (o receive do TestClient não tem prazo)."""
    if not hasattr(ws, "_send_rx"):  # outra versão do Starlette: sem prazo
        return ws.receive_json()

    async def proxima():
        with anyio.fail_after(prazo_s):
            return await ws._send_rx.receive()

    mensagem = ws.portal.call(proxima)
    ws._raise_on_close(mensagem)
    return json.loads(mensagem["text"])


def receber_ate(ws, condicao: Callable[[dict[str, Any]], bool], prazo_s: float = 5.0) -> dict[str, Any]:
    limite = time.monotonic() + prazo_s
    while True:
        mensagem = receber(ws, max(0.1, limite - time.monotonic()))
        if condicao(mensagem):
            return mensagem


def mandar_corrida(http, corrida: dict[str, Any], ate_t: float | None = None) -> None:
    """Telemetria em lotes de 0,5 s e cada decisão no instante em que ficou pronta, como o piloto faz."""
    linhas = [l for l in corrida["linhas"] if ate_t is None or l["t"] <= ate_t]
    decisoes = sorted(corrida["decisoes"], key=lambda d: d["t_aplicada"])
    k = 0
    for i in range(0, len(linhas), 10):
        lote = linhas[i:i + 10]
        while k < len(decisoes) and decisoes[k]["t_aplicada"] <= lote[-1]["t"]:
            assert http.post("/sessao/decisoes", json=decisoes[k]).status_code == 200
            k += 1
        assert http.post("/sessao/telemetria", json={"linhas": lote}).status_code == 204


def esperar(condicao: Callable[[], bool], prazo_s: float = 10.0) -> None:
    limite = time.monotonic() + prazo_s
    while not condicao():
        assert time.monotonic() < limite, "a condição não ficou verdadeira no prazo"
        time.sleep(0.02)


def test_saude_e_rotas_sem_sessao(cfg, tmp_path):
    with TestClient(montar_app(cfg, tmp_path)) as http:
        assert http.get("/saude").json() == {"ok": True, "sessao": None, "clientes_ws": 0}
        for metodo, rota in (("get", "/sessao"), ("post", "/sessao/telemetria"), ("delete", "/sessao")):
            corpo = {"json": {"linhas": []}} if metodo == "post" else {}
            assert getattr(http, metodo)(rota, **corpo).status_code == 404
        assert http.post("/sessao", json={"modo": "turbo"}).status_code == 422
        assert http.get("/benchmark/ultimo").status_code == 404


def test_corrida_publica_serie_decisoes_logs_medidos_e_metricas(cfg, tmp_path, corrida):
    with TestClient(montar_app(cfg, tmp_path)) as http:
        with http.websocket_connect("/ws") as ws:
            assert receber(ws) == {"tipo": "historico", "sessao": None, "entradas": [], "serie": None, "previsao": None,
                                   "estado": None, "metricas": None, "linha_de_base": None}
            info = http.post("/sessao", json=opcoes_sessao(corrida)).json()
            assert info["modo"] == "agentes" and info["llm_decisoes"] == "falso" and info["tem_linha_de_base"]
            iniciada = receber_ate(ws, lambda m: m["tipo"] == "sessao")
            assert iniciada["evento"] == "iniciada" and iniciada["linha_de_base"]["total"]["eventos"] > 0

            mandar_corrida(http, corrida)
            telemetria = receber_ate(ws, lambda m: m["tipo"] == "telemetria")
            assert len(telemetria["pontos"]["t"]) == 3  # 10 linhas a 20 Hz -> pontos a 5 Hz
            assert {"t", "x", "y", "speed_kmh", "v_limite_kmh", "progresso"} <= set(telemetria["estado"])
            previsao = receber_ate(ws, lambda m: m["tipo"] == "previsao")
            assert previsao["sessao"] == info["id"] and previsao["curvas"]["bruto"]["t"]

        # a camada de logs roda numa thread: espera os logs dos blocos medidos
        sessao = http.app.state.estado["sessao"]
        esperar(lambda: any(e["tipo"] == "log" for e in sessao.entradas))
        entradas = http.get("/sessao/log").json()["entradas"]
        assert [e["id"] for e in entradas] == list(range(1, len(entradas) + 1))  # ids únicos, na ordem de publicação
        decisoes = [e for e in entradas if e["tipo"] == "intervencao"]
        assert len(decisoes) == len(corrida["decisoes"])
        logs = [e for e in entradas if e["tipo"] == "log"]
        assert logs and all(e["fonte"] == "substituto" for e in logs)

        m = http.get("/saude").json()["sessao"]
        assert m["n_entradas"] == len(sessao.entradas)
        metricas = sessao.metricas.para_json()
        assert metricas["combustivel_ml"] == pytest.approx(corrida["linhas"][-1]["combustivel_ml"], rel=1e-3)
        assert metricas["base_combustivel_ml"] > metricas["combustivel_ml"] and metricas["economia_medida_pct"] > 5
        assert metricas["decisoes"] == len(decisoes) and metricas["economia_prevista_ml"] > 0

        fim = http.request("DELETE", "/sessao", json={"resumo": {"motivo_fim": "chegada"}}).json()
        resumo = fim["resumo"]
        assert fim["sessao"]["encerrada"] and resumo["decisoes"] == len(decisoes) and resumo["piloto"] == {"motivo_fim": "chegada"}
        assert http.post("/sessao/telemetria", json={"linhas": corrida["linhas"][:2]}).status_code == 409
        arquivo = Path(fim["sessao"]["arquivo"])
        assert len(arquivo.read_text(encoding="utf-8").splitlines()) == len(sessao.entradas)
        assert json.loads(arquivo.with_name(arquivo.stem + "_resumo.json").read_text(encoding="utf-8"))["decisoes"] == len(decisoes)


def test_pagina_aberta_no_meio_da_corrida_recebe_o_historico(cfg, tmp_path, corrida):
    with TestClient(montar_app(cfg, tmp_path)) as http:
        http.post("/sessao", json=opcoes_sessao(corrida))
        mandar_corrida(http, corrida, ate_t=12.0)
        with http.websocket_connect("/ws") as ws:
            historico = receber(ws)
        assert historico["sessao"]["id"] == http.get("/sessao").json()["id"]
        serie = historico["serie"]
        assert serie["t"] and max(serie["t"]) <= 12.0 and len(serie["t"]) == len(serie["speed_kmh"]) == len(serie["rumo_graus"])
        assert historico["estado"]["t"] == pytest.approx(max(l["t"] for l in corrida["linhas"] if l["t"] <= 12.0))
        assert historico["previsao"] is not None and historico["metricas"]["combustivel_ml"] > 0


def test_ultimo_benchmark_vem_do_resumo_mais_recente(cfg, tmp_path):
    pasta = tmp_path / "benchmark"
    for nome, economia in (("20261009_100000", 1.0), ("20261010_100000", 2.5)):
        (pasta / nome).mkdir(parents=True)
        (pasta / nome / "resumo.json").write_text(json.dumps({"extrapolacao": {"economia_l": economia}}), encoding="utf-8")
        time.sleep(0.01)
    with TestClient(montar_app(cfg, tmp_path)) as http:
        dados = http.get("/benchmark/ultimo").json()
        assert dados["extrapolacao"]["economia_l"] == 2.5 and dados["pasta"].endswith("20261010_100000")
