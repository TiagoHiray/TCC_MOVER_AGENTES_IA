"""Testes do servidor da camada agêntica (FastAPI + WebSocket), sem LLM e sem CARLA.

Rodar a partir da raiz do repositório:
    python -m pytest tests -q

O app é montado com telemetria sintética, o provedor falso e sem o especialista de ML. As
mensagens do WebSocket são lidas com prazo, para um erro virar falha e não travar o pytest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anyio
import pytest
from auxiliares import FRENAGEM, telemetria
from fastapi.testclient import TestClient

from mover.agentes import eventos_sinteticos
from mover.agentes.llm import FalsoLLM
from mover.agentes.supervisor import Supervisor
from mover.config import carregar_yaml
from mover.servidor.app import criar_app


@pytest.fixture(scope="module")
def cfg() -> dict[str, Any]:
    return carregar_yaml("config/agentes.yaml")


def montar_app(cfg: dict[str, Any], pasta: Path, dados=None):
    """App com telemetria sintética (25 s = 3 blocos, frenagem brusca aos 5 s) e provedor falso."""
    if dados is None:
        dados = eventos_sinteticos.injetar(telemetria(25.0), [FRENAGEM], {})
    return criar_app(cfg, {"servidor": {"pasta_sessoes": str(pasta)}}, telemetria=dados,
                     fabrica_supervisor=lambda opcoes: Supervisor(FalsoLLM(), cfg), sem_ml=True)


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


def receber_sem_previsao(ws, prazo_s: float = 5.0) -> dict[str, Any]:
    """Próxima mensagem que não seja "previsao" (a análise do bloco seguinte termina em paralelo)."""
    while True:
        mensagem = receber(ws, prazo_s)
        if mensagem["tipo"] != "previsao":
            return mensagem


def test_volta_completa_pelo_servidor(cfg, tmp_path):
    app = montar_app(cfg, tmp_path)
    with TestClient(app) as cliente:
        assert cliente.get("/saude").json()["sessao"] is None
        assert cliente.post("/sessao/blocos/0").status_code == 404  # ainda sem sessão

        info = cliente.post("/sessao", json={}).json()
        assert (info["n_blocos"], info["duracao_bloco_s"], info["t0"]) == (3, 10.0, 0.0)
        assert (info["llm"], info["especialista_ml"], info["proximo_bloco"]) == ("falso", False, 0)

        with cliente.websocket_connect("/ws") as ws:
            historico = receber(ws)
            assert historico["tipo"] == "historico"
            assert historico["sessao"]["id"] == info["id"] and historico["entradas"] == []

            fora = cliente.post("/sessao/blocos/1")
            assert fora.status_code == 409 and fora.json()["detail"]["proximo_bloco"] == 0

            bloco0 = cliente.post("/sessao/blocos/0").json()
            assert bloco0["proximo_bloco"] == 1
            problemas = [e for e in bloco0["entradas"] if e["tipo"] == "problema"]
            assert len(problemas) == 1 and problemas[0]["evento"] == "frenagem_brusca"
            assert problemas[0]["nivel"] == "critico"

            # o WebSocket recebe as mesmas entradas, na ordem, e depois o aviso do bloco
            for esperada in bloco0["entradas"]:
                mensagem = receber_sem_previsao(ws)
                assert mensagem["tipo"] == "entrada" and mensagem["entrada"]["id"] == esperada["id"]
            aviso = receber_sem_previsao(ws)
            assert (aviso["tipo"], aviso["bloco"], aviso["n_problemas"]) == ("bloco", 0, 1)
            assert (aviso["t_ini"], aviso["t_fim"]) == (0.0, 10.0)

            # pose do caminhão -> estado com a linha mais próxima da telemetria
            r = cliente.post("/sessao/estado", json={"sim_time": 5.02, "x": 10.0, "y": -2.0, "yaw": 90.0})
            assert r.status_code == 204
            estado = receber_sem_previsao(ws)
            assert estado["tipo"] == "estado"
            assert (estado["sim_time"], estado["bloco"], estado["quadro"]) == (5.0, 0, 100)
            assert (estado["x"], estado["y"], estado["yaw"]) == (10.0, -2.0, 90.0)
            assert estado["speed_kmh"] == 25.0 and estado["fonte"] in ("sintetico", "misto")

            for k in (1, 2):
                assert cliente.post(f"/sessao/blocos/{k}").status_code == 200
            assert cliente.post("/sessao/blocos/3").status_code == 404  # a volta só tem 3 blocos

            log = cliente.get("/sessao/log", params={"desde": 1}).json()
            assert log["total"] >= 4 and len(log["entradas"]) == log["total"] - 1

            fim = cliente.delete("/sessao").json()
            assert fim["sessao"]["encerrada"] and fim["resumo"]["blocos_publicados"] == 3
            assert fim["resumo"]["problemas"] == 1
            ultima = receber(ws)
            while ultima["tipo"] != "sessao":  # descarta as entradas dos blocos 1 e 2
                ultima = receber(ws)
            assert ultima["evento"] == "encerrada"
            assert cliente.post("/sessao/blocos/0").status_code == 409  # sessão encerrada

    linhas = Path(info["arquivo"]).read_text(encoding="utf-8").splitlines()
    assert len(linhas) == log["total"]
    assert json.loads(linhas[0])["id"] == 1
    assert Path(info["arquivo"]).with_name(f"sessao_{info['id']}_resumo.json").exists()


def test_nova_sessao_encerra_a_anterior_e_o_fim_do_app_encerra_a_ultima(cfg, tmp_path):
    app = montar_app(cfg, tmp_path, telemetria(12.0))
    with TestClient(app) as cliente:
        primeira = cliente.post("/sessao", json={}).json()
        segunda = cliente.post("/sessao", json={"injetar_eventos": False}).json()
        assert primeira["id"] != segunda["id"]  # mesmo segundo: o id ganha um sufixo
        assert cliente.get("/sessao").json()["id"] == segunda["id"]
        assert Path(primeira["arquivo"]).with_name(f"sessao_{primeira['id']}_resumo.json").exists()
    assert app.state.estado["sessao"].encerrada  # o lifespan encerra a sessão aberta
    assert Path(segunda["arquivo"]).with_name(f"sessao_{segunda['id']}_resumo.json").exists()


def test_injetar_eventos_vale_para_a_telemetria_do_servidor(cfg, tmp_path):
    app = montar_app(cfg, tmp_path, telemetria(130.0))
    with TestClient(app) as cliente:
        cliente.post("/sessao", json={"injetar_eventos": True})
        entradas = []
        for k in range(cliente.get("/sessao").json()["n_blocos"]):
            entradas += cliente.post(f"/sessao/blocos/{k}").json()["entradas"]
    eventos = {e["evento"] for e in entradas if e["tipo"] == "problema"}
    assert {"frenagem_brusca", "arrancada_brusca"} <= eventos  # os dois eventos do config/agentes.yaml
