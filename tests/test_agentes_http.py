"""Testes da conversa do caminhão com a camada agêntica por HTTP (sem CARLA e sem LLM de verdade).

Rodar a partir da raiz do repositório:
    python -m pytest tests -q

O caminhão manda o plano dos próximos 10 s e o estado do governador em JSON (POST /agentes/plano)
e busca as alterações (GET /agentes/decisoes). O servidor roda no TestClient do FastAPI; a corrida,
no simulador substituto com o plano do estádio de tests/auxiliares.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from auxiliares import plano_estadio
from fastapi.testclient import TestClient

from mover.agentes.decisao import janela_inicial
from mover.agentes.llm import FalsoLLM
from mover.autonomo.canal_agentes import CanalHttp, ErroAgentes
from mover.autonomo.conducao import Conducao, OpcoesConducao, metricas_da_corrida
from mover.autonomo.governador import Intervencao
from mover.autonomo.montagem import Montagem, montar
from mover.autonomo.planejador import PlanoFuturo
from mover.config import carregar_yaml
from mover.servidor.app import criar_app


@pytest.fixture(scope="module")
def m() -> Montagem:
    return montar(carregar_yaml("config/autonomo.yaml"), plano=plano_estadio())


def app_dos_agentes(pasta: Path, montagem: Montagem | None, missao: str = "nao_existe.yaml"):
    cfg_sim = {"servidor": {"pasta_sessoes": str(pasta / "sessoes")}, "mapa": {"arquivo": str(pasta / "nao_existe.xodr")}}
    cfg_autonomo = {"benchmark": {"pasta": str(pasta / "benchmark")}, "missao": {"arquivo": str(pasta / missao)}}
    return criar_app(carregar_yaml("config/agentes.yaml"), cfg_sim, fabrica_llm_chat=lambda opcoes: FalsoLLM(),
                     cfg_autonomo=cfg_autonomo, montagem=montagem)


def abrir(canal: CanalHttp, m: Montagem, **extra: Any) -> dict[str, Any]:
    return canal.abrir({"voltas": 1, "usar_llm": True, "usar_ml": False, "provedor": "falso",
                        "comprimento_volta_m": m.plano.comprimento, **extra})


def test_plano_e_governador_vao_e_voltam_em_json_sem_mudar_a_decisao(m):
    sim = m.simulador_substituto()
    sim.iniciar()
    gov = m.governador("agentes", 1)
    gov.definir("lombada_1@0", Intervencao.da_opcao("suave", m.opcoes))
    estado = {"t": 0.0, "d": 0.0, "v": 0.0, "a": 0.0}
    for _ in range(300):  # 15 s de corrida: o governador acumula referência, elementos feitos e disparados
        linha = sim.passo(gov.velocidade_desejada(estado["d"], estado["v"], estado["a"], m.dt))
        estado = {"t": linha["sim_time"], "d": linha["d"], "v": linha["speed_mps"], "a": linha["acc_long"]}
    em_json = json.loads(json.dumps(gov.exportar_estado()))
    copia = m.governador("agentes", 1).restaurar_estado(em_json)
    original = gov.clonar()
    for k in range(200):  # mesmas entradas: mesma velocidade desejada, tick a tick
        d, v = estado["d"] + 0.4 * k, max(0.0, estado["v"] - 0.01 * k)
        assert copia.velocidade_desejada(d, v, 0.0, m.dt) == original.velocidade_desejada(d, v, 0.0, m.dt)
    assert copia.intervencoes == gov.intervencoes and copia.ja_passou("partida@0") == gov.ja_passou("partida@0")

    plano = m.planejador(1).planejar(gov, estado, 2, (20.0, 30.0))
    assert plano is not None and plano.pendentes
    recebido = PlanoFuturo.de_json(json.loads(json.dumps(plano.para_json())))
    decisor = m.decisor(1)
    a = decisor.decidir(plano, gov, usar_llm=False)
    b = decisor.decidir(recebido, m.governador("agentes", 1).restaurar_estado(em_json), usar_llm=False)
    decisor.encerrar()
    assert b.recomendada == a.recomendada and [o.combustivel_ml for o in b.opcoes] == [o.combustivel_ml for o in a.opcoes]
    assert [i.chave for i in recebido.pendentes] == [i.chave for i in plano.pendentes]


def test_corrida_com_os_agentes_por_http_decide_igual_a_corrida_local(m, tmp_path):
    op = OpcoesConducao(voltas=1, assincrona=False, sem_espera=True, duracao_bloco_s=10.0)
    decisor = m.decisor(1, FalsoLLM())
    local = Conducao(m.plano, m.simulador_substituto(), m.governador("agentes", 1), decisor, op, m.caminhao, m.consumo).rodar()
    decisor.encerrar()

    with TestClient(app_dos_agentes(tmp_path, m)) as http:
        canal = CanalHttp(http)
        info = abrir(canal, m)
        assert info["llm"] == "falso" and info["ml"] is None and info["comprimento_volta_m"] == pytest.approx(m.plano.comprimento, abs=0.1)
        res = Conducao(m.plano, m.simulador_substituto(), m.governador("agentes", 1), canal, op, m.caminhao, m.consumo,
                       planejador=m.planejador(1)).rodar()
        saude = http.get("/saude").json()["agentes"]
        canal.encerrar()
        assert http.get("/saude").json()["agentes"] is None

    assert res.completa and [d["opcao"] for d in res.decisoes] == [d["opcao"] for d in local.decisoes]
    assert res.linhas[-1]["d"] == pytest.approx(local.linhas[-1]["d"], abs=1e-6)
    assert saude["planos_recebidos"] == len(res.decisoes) == canal.requisicoes - 2  # + abrir e encerrar a sessão
    met = metricas_da_corrida(res, m.plano.comprimento, m.caminhao, m.consumo, m.limiares)["total"]
    assert met["eventos_atencao"] + met["eventos_criticos"] == 0


def test_corrida_ao_vivo_busca_as_decisoes_por_http_sem_esperar_os_agentes(m, tmp_path):
    op = OpcoesConducao(voltas=1, assincrona=True, fator_tempo=10.0, duracao_bloco_s=10.0)
    with TestClient(app_dos_agentes(tmp_path, m)) as http:
        canal = CanalHttp(http, intervalo_s=0.02)
        abrir(canal, m)
        res = Conducao(m.plano, m.simulador_substituto(), m.governador("agentes", 1), canal, op, m.caminhao, m.consumo,
                       planejador=m.planejador(1)).rodar()
        canal.encerrar()
    assert res.completa and len(res.decisoes) >= 2 and not canal.ocupado
    assert all(e["t_aplicada"] >= e["t_pedido"] for e in res.decisoes)
    met = metricas_da_corrida(res, m.plano.comprimento, m.caminhao, m.consumo, m.limiares)["total"]
    assert met["eventos_criticos"] <= 1  # a regra chega na hora; o LLM só confirma ou troca


def test_rotas_dos_agentes_recusam_pedidos_fora_de_ordem_ou_invalidos(m, tmp_path):
    with TestClient(app_dos_agentes(tmp_path, m)) as http:
        assert http.post("/agentes/plano", json={"plano": {}, "governador": {}}).status_code == 404
        assert http.get("/agentes/decisoes").status_code == 404
        canal = CanalHttp(http)
        with pytest.raises(ErroAgentes, match="plano de missão diferente"):
            abrir(canal, m, comprimento_volta_m=m.plano.comprimento + 50.0)
        abrir(canal, m)
        assert http.post("/agentes/plano", json={"plano": {"bloco": 1}, "governador": {}}).status_code == 422
        assert http.get("/agentes/decisoes", params={"desde": 0}).json() == {"mensagens": [], "total": 0, "ocupado": False}
        assert http.delete("/agentes/sessao").status_code == 200
        assert http.get("/agentes/decisoes").status_code == 404
    with TestClient(app_dos_agentes(tmp_path, None)) as http:  # servidor sem o plano de missão
        resposta = http.post("/agentes/sessao", json={"voltas": 1, "usar_llm": False, "usar_ml": False})
        assert resposta.status_code == 503 and "gerar_missao" in json.dumps(resposta.json(), ensure_ascii=False)


def test_plano_da_largada_tem_os_dois_primeiros_blocos_e_cabe_em_json(m):
    plano = m.planejador(1).planejar(m.governador("agentes", 1), {"t": 0.0, "d": 0.0, "v": 0.0, "a": 0.0}, 0,
                                     janela_inicial(0.0, 10.0))
    corpo = json.dumps(plano.para_json())
    assert len(corpo) < 200_000 and json.loads(corpo)["janela"] == [0.0, 20.0]
