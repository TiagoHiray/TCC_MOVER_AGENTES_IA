"""Testes da ingestão ao vivo do Sensor Logger (HTTP Push) e do replay de CSV, sem rede e sem CARLA."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
from auxiliares import telemetria
from fastapi.testclient import TestClient

from mover.agentes.llm import FalsoLLM
from mover.agentes.supervisor import Supervisor
from mover.config import carregar_yaml
from mover.ingestao.replay_csv import arquivos_da_gravacao, lotes, reenviar
from mover.servidor.app import criar_app

T0 = 1_791_048_995_000_000_000
URL = "/ingestao/sensorlogger"


@pytest.fixture(scope="module")
def cfg() -> dict[str, Any]:
    return carregar_yaml("config/agentes.yaml")


def montar_app(cfg: dict[str, Any], pasta: Path, token: str | None = None):
    cfg_sim = {"servidor": {"pasta_sessoes": str(pasta / "sessoes")},
               "ingestao": {"pasta": str(pasta / "ingestao"), "max_corpo_bytes": 10_000, "token": token}}
    return criar_app(cfg, cfg_sim, telemetria=telemetria(10.0),
                     fabrica_supervisor=lambda opcoes: Supervisor(FalsoLLM(), cfg), sem_ml=True)


def gravacao(pasta: Path, segundos: float = 2.0) -> Path:
    """Export mínimo do Sensor Logger: acelerômetro a 100 Hz, GPS a 1 Hz e Metadata."""
    pasta.mkdir(parents=True, exist_ok=True)
    with open(pasta / "Accelerometer.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "seconds_elapsed", "z", "y", "x"])
        for i in range(int(segundos * 100)):
            w.writerow([T0 + i * 10_000_000, i / 100, 0.1, 0.2, 0.3])
    with open(pasta / "Location.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "seconds_elapsed", "latitude", "longitude", "speed"])
        for i in range(int(segundos)):
            w.writerow([T0 + i * 1_000_000_000 + 5_000_000, i, -23.6479, -46.5732, 5.0])
    (pasta / "Metadata.csv").write_text("version,platform\n2,ios\n")
    return pasta


def mensagem(mid: int, sessao: str = "abc123") -> dict[str, Any]:
    return {"messageId": mid, "sessionId": sessao, "deviceId": "d", "payload": [
        {"name": "accelerometer", "time": T0 + mid, "values": {"x": 1.0, "y": 2.0, "z": 3.0}},
        {"name": "location", "time": T0 + mid, "values": {"latitude": -23.6, "longitude": -46.5, "speed": 4.0}},
    ]}


def test_lotes_agrupam_por_periodo_e_ignoram_metadata(tmp_path):
    arquivos = arquivos_da_gravacao(gravacao(tmp_path / "rec"))
    assert [p.stem for p in arquivos] == ["Accelerometer", "Location"]

    resultado = list(lotes(arquivos, periodo_s=0.2))
    assert len(resultado) == 10
    leituras = [r for _, _, payload in resultado for r in payload]
    assert len(leituras) == 202
    assert [r["time"] for r in leituras] == sorted(r["time"] for r in leituras)
    assert {r["name"] for r in leituras} == {"accelerometer", "location"}
    assert leituras[0]["values"] == {"z": 0.1, "y": 0.2, "x": 0.3}
    for t0, fim, payload in resultado:
        assert all(fim - 200_000_000 <= r["time"] < fim for r in payload)

    assert sum(len(p) for _, _, p in lotes(arquivos, 0.2, duracao_s=0.5)) == 52


def test_push_grava_um_csv_por_sensor(cfg, tmp_path):
    with TestClient(montar_app(cfg, tmp_path)) as cliente:
        for mid in (0, 1, 1):
            r = cliente.post(URL, json=mensagem(mid))
            assert r.status_code == 200 and r.json() == {"ok": True, "leituras": 2}
        status = cliente.get("/ingestao/status").json()

    gravacao_ = status["gravacoes"]["abc123"]
    assert (gravacao_["mensagens"], gravacao_["fora_de_ordem"]) == (3, 1)
    assert gravacao_["sensores"]["location"]["ultima"]["latitude"] == -23.6
    pasta = Path(gravacao_["pasta"])
    with open(pasta / "accelerometer.csv", newline="") as f:
        linhas = list(csv.DictReader(f))
    assert len(linhas) == 3 and list(linhas[0]) == ["time", "recv_ns", "x", "y", "z"]
    assert len((pasta / "raw.jsonl").read_text(encoding="utf-8").splitlines()) == 3


def test_push_rejeita_token_e_corpo_invalidos(cfg, tmp_path):
    with TestClient(montar_app(cfg, tmp_path, token="segredo")) as cliente:
        assert cliente.post(URL, json=mensagem(0)).status_code == 401
        assert cliente.post(URL + "?token=errado", json=mensagem(0)).status_code == 401
        assert cliente.get("/ingestao/status").status_code == 401
        assert cliente.post(URL + "?token=segredo", content=b"{nao json").status_code == 400
        assert cliente.post(URL + "?token=segredo", content=b"[1, 2]").status_code == 400
        assert cliente.post(URL + "?token=segredo", json={"payload": "x"}).status_code == 400
        assert cliente.post(URL + "?token=segredo", content=json.dumps({"p": "x" * 20_000})).status_code == 413
        assert cliente.post(URL + "?token=segredo", json=mensagem(0)).status_code == 200


def test_replay_ponta_a_ponta(cfg, tmp_path):
    arquivos = arquivos_da_gravacao(gravacao(tmp_path / "rec"))
    with TestClient(montar_app(cfg, tmp_path)) as cliente:
        resumo = reenviar(arquivos, lambda msg: cliente.post(URL, json=msg).status_code == 200, sem_espera=True)
        status = cliente.get("/ingestao/status").json()

    assert (resumo["mensagens"], resumo["falhas"], resumo["leituras"]) == (10, 0, 202)
    sensores = status["gravacoes"][resumo["sessao"]]["sensores"]
    assert sensores["accelerometer"]["amostras"] == 200 and sensores["location"]["amostras"] == 2
    assert sensores["accelerometer"]["taxa_hz"] == pytest.approx(100, rel=0.01)
    assert sensores["accelerometer"]["ultima"]["time"] > T0  # rebase para o relógio atual
