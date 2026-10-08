"""Testes da camada agêntica (pytest), com telemetria sintética de 20 Hz.

Rodar a partir da raiz do repositório:
    python -m pytest tests -q

Os testes não precisam de LLM instalado: usam o provedor falso ou clientes de teste que
simulam respostas e erros. O último teste usa a telemetria tratada da volta real, se ela
existir em data/tratado/.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest
from auxiliares import FRENAGEM, marcar, telemetria

from mover.agentes import eventos_sinteticos
from mover.agentes.especialistas import EspecialistaJerk
from mover.agentes.executor import ExecutorBlocos
from mover.agentes.grafo import CamadaAgentica
from mover.agentes.llm import FalsoLLM, interpretar_json
from mover.agentes.supervisor import (RespostaLog, RespostaProblema, Supervisor, motivo_rejeicao, numeros_com_unidade,
                                      numeros_no_texto, numeros_por_unidade, unidade_da_chave)
from mover.config import caminho, carregar_yaml


# ---------------------------------------------------------------------------------------------
# Auxiliares
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cfg() -> dict[str, Any]:
    return carregar_yaml("config/agentes.yaml")


def rodar(cfg: dict[str, Any], dados: pd.DataFrame, llm: Any = None) -> tuple[CamadaAgentica, list[dict[str, Any]]]:
    """Processa todos os blocos em ordem (sem o especialista de ML) e devolve as entradas."""
    camada = CamadaAgentica(cfg, dados, Supervisor(llm or FalsoLLM(), cfg), especialista_ml=None)
    entradas = [e for k in range(camada.n_blocos) for e in camada.processar_bloco(k)]
    return camada, entradas


class LLMFixo:
    """Cliente de teste que sempre devolve o mesmo resumo de log."""

    nome = "fixo"

    def __init__(self, resumo: str):
        self.resumo = resumo

    def gerar(self, sistema: str, usuario: str, esquema: type) -> Any:
        return esquema(resumo=self.resumo)


class LLMQuebrado:
    """Cliente de teste que sempre falha, como um servidor do Ollama fora do ar."""

    nome = "quebrado"

    def __init__(self):
        self.chamadas = 0

    def gerar(self, sistema: str, usuario: str, esquema: type) -> Any:
        self.chamadas += 1
        raise ConnectionError("servidor do LLM fora do ar")


# ---------------------------------------------------------------------------------------------
# Guarda de números e leitura da resposta do LLM
# ---------------------------------------------------------------------------------------------
def test_numeros_no_texto_ignora_horarios():
    texto = "Às 14:36:35 a velocidade caiu de 18,4 para 9 km/h, com jerk de -2.5 m/s³."
    assert numeros_no_texto(texto) == [18.4, 9.0, -2.5]


def test_guarda_recusa_numero_inventado_e_aceita_arredondamento():
    permitidos = [18.4, 25.0, 2.5, 5.0]
    assert motivo_rejeicao("Seguiu a 18 km/h, sem eventos.", permitidos, 45) is None
    assert motivo_rejeicao("Seguiu a 47 km/h, sem eventos.", permitidos, 45) == "número sem respaldo nos fatos: 47 km/h"
    assert motivo_rejeicao("Seguiu a 47 km/h.", permitidos, 45, checar_numeros=False) is None
    assert motivo_rejeicao("palavra " * 100, permitidos, 45).startswith("texto longo demais")
    assert motivo_rejeicao("   ", permitidos, 45) == "texto vazio"


def test_supervisor_usa_texto_modelo_quando_o_llm_inventa_numero(cfg):
    _, entradas = rodar(cfg, telemetria(10.0), LLMFixo("Seguiu a 47 km/h, sem novidades."))
    assert len(entradas) == 1
    entrada = entradas[0]
    assert entrada["texto_origem"] == "modelo"
    assert entrada["observacao"] == "número sem respaldo nos fatos: 47 km/h"
    assert entrada["texto"].startswith("Velocidade constante a 25 km/h em média")

    _, entradas = rodar(cfg, telemetria(10.0), LLMFixo("Seguiu em velocidade constante, perto de 25 km/h."))
    assert (entradas[0]["texto_origem"], entradas[0]["provedor"]) == ("llm", "fixo")
    assert entradas[0]["texto"] == "Seguiu em velocidade constante, perto de 25 km/h."


def test_guarda_confere_a_unidade_do_numero(cfg):
    texto = "A 25 km/h, jerk de 9,3 m/s³, impacto de 4.1 m/s2, 51,4 s, 100 % do tempo, virou 90° e andou 35 metros."
    assert numeros_com_unidade(texto) == [(25.0, "km/h"), (9.3, "m/s³"), (4.1, "m/s²"), (51.4, "s"), (100.0, "%"),
                                          (90.0, "°"), (35.0, "m")]
    assert numeros_com_unidade("bloco 3, 2 eventos, 10 sintéticos, (#4, 5,0–10,0 s)") == [
        (3.0, None), (2.0, None), (10.0, None), (4.0, None), (5.0, None), (10.0, "s")]
    assert unidade_da_chave("vel_media_kmh") == "km/h" and unidade_da_chave("t_pico") == "s"
    assert unidade_da_chave("manobra_predominante_pct") == "%" and unidade_da_chave("id") is None

    fatos = {"vel_media_kmh": 25.0, "manobra_predominante_pct": 100, "duracao_s": 10.0, "contagens": [1.0, 2.0]}
    permitidos = numeros_por_unidade(fatos)
    assert motivo_rejeicao("Manteve o padrão em 100 % do tempo, a 25 km/h.", permitidos, 45) is None
    assert motivo_rejeicao("Chegou a 97 km/h.", permitidos, 45) == "número sem respaldo nos fatos: 97 km/h"
    assert motivo_rejeicao("Foram 100 metros.", permitidos, 45) == "número sem respaldo nos fatos: 100 m"
    assert motivo_rejeicao("Foram 100 manobras.", permitidos, 45) is None  # sem unidade: vale qualquer fato

    # no Supervisor de verdade: o fato de 100 % (manobra predominante) não respalda uma velocidade
    _, entradas = rodar(cfg, telemetria(10.0), LLMFixo("Seguiu a 100 km/h, sem novidades."))
    assert (entradas[0]["texto_origem"], entradas[0]["observacao"]) == ("modelo", "número sem respaldo nos fatos: 100 km/h")
    _, entradas = rodar(cfg, telemetria(10.0), LLMFixo("Velocidade constante em 100 % do trecho, perto de 25 km/h."))
    assert entradas[0]["texto_origem"] == "llm"


def test_interpretar_json_tolera_texto_em_volta():
    assert interpretar_json('{"resumo": "Tudo certo."}', RespostaLog).resumo == "Tudo certo."
    assert interpretar_json('Claro! {"resumo": "Freou forte."} Até mais.', RespostaLog).resumo == "Freou forte."
    assert interpretar_json('"Texto solto do modelo"', RespostaLog).resumo == "Texto solto do modelo"
    resposta = interpretar_json('```json\n{"diagnostico": "a", "acao": "b"}\n```', RespostaProblema)
    assert (resposta.diagnostico, resposta.acao) == ("a", "b")
    with pytest.raises(ValueError):
        interpretar_json("sem JSON nenhum", RespostaProblema)


# ---------------------------------------------------------------------------------------------
# Especialista de jerk e eventos sintéticos
# ---------------------------------------------------------------------------------------------
def test_injecao_nao_altera_os_dados_reais():
    real = telemetria(20.0)
    original = real.copy(deep=True)
    com_eventos = eventos_sinteticos.injetar(real, [FRENAGEM], {})
    pd.testing.assert_frame_equal(real, original)
    assert (com_eventos["fonte"] == "sintetico").any()
    assert np.array_equal(com_eventos["speed_kmh"], real["speed_kmh"])  # a velocidade não muda
    assert com_eventos["acc_long"].min() < -3.0


def test_frenagem_sintetica_vira_frenagem_brusca_critica(cfg):
    dados = eventos_sinteticos.injetar(telemetria(20.0), [FRENAGEM], {})
    eventos = EspecialistaJerk(cfg["limiares"]).detectar(dados, 0.0, 20.0)
    assert len(eventos) == 1  # entrada e soltura do freio formam um só evento
    ev = eventos[0]
    assert (ev.tipo, ev.nivel, ev.causa, ev.sinal) == ("frenagem_brusca", "critico", "conducao_brusca", "negativo")
    assert 4.8 <= ev.t_pico <= 5.6
    assert ev.jerk_max_abs_mps3 >= 5.0
    assert ev.fonte in ("sintetico", "misto")


def test_pico_vertical_vira_irregularidade_na_via(cfg):
    dados = telemetria(10.0, vel_kmh=10.0)
    marcar(dados, "jerk_long", 5.0, 5.1, 3.0)    # soltura do freio na lombada (1º pico positivo)
    marcar(dados, "jerk_long", 5.4, 5.5, -3.0)
    marcar(dados, "acc_vert", 5.1, 5.3, 3.0)     # impacto vertical da lombada
    _, entradas = rodar(cfg, dados)
    problemas = [e for e in entradas if e["tipo"] == "problema"]
    assert len(problemas) == 1
    p = problemas[0]
    assert (p["evento"], p["causa"], p["nivel"], p["t_pico"]) == ("soltura_freio", "irregularidade_via", "atencao", 5.0)
    assert p["texto"].startswith("Provável lombada a 10 km/h: jerk de até 3,0 m/s³")
    assert "Ação: Chegar mais devagar" in p["texto"]
    assert p["texto_origem"] == "llm"  # o texto do provedor falso passa na guarda de números


# ---------------------------------------------------------------------------------------------
# Janelas, mesclagem e blocos
# ---------------------------------------------------------------------------------------------
def test_cruzeiro_longo_mescla_janelas_e_continua_entre_blocos(cfg):
    _, entradas = rodar(cfg, telemetria(40.0))
    assert [e["tipo"] for e in entradas] == ["log"] * 4
    assert [e["bloco"] for e in entradas] == [0, 1, 2, 3]
    assert [e["n_janelas"] for e in entradas] == [2, 2, 2, 2]
    assert [e["continuacao"] for e in entradas] == [False, True, False, True]  # mescla até 20 s
    assert [e["texto_origem"] for e in entradas] == ["llm", "modelo", "llm", "modelo"]
    assert "mesmo estado há 20 s" in entradas[1]["texto"]
    assert entradas[3]["mesclado_desde"] == 20.0
    json.dumps(entradas, ensure_ascii=False)  # o JSONL não pode ter tipos do numpy


def test_sobra_curta_no_fim_vai_para_a_janela_anterior(cfg):
    _, entradas = rodar(cfg, telemetria(17.0))
    ultimo = entradas[-1]
    assert (ultimo["bloco"], ultimo["n_janelas"], ultimo["t_ini"], ultimo["t_fim"]) == (1, 1, 10.0, 17.0)


def test_evento_na_fronteira_entre_blocos_sai_uma_vez_so(cfg):
    dados = telemetria(20.0)
    marcar(dados, "jerk_long", 9.7, 9.8, -3.0)    # 1º pico no bloco 0
    marcar(dados, "jerk_long", 10.3, 10.4, 6.0)   # maior pico do mesmo grupo, já no bloco 1
    _, entradas = rodar(cfg, dados)
    problemas = [e for e in entradas if e["tipo"] == "problema"]
    assert len(problemas) == 1
    p = problemas[0]
    assert (p["bloco"], p["t_pico"], p["nivel"], p["evento"]) == (0, 9.7, "critico", "corte_aceleracao")
    assert p["fatos"]["jerk_max_abs_mps3"] == 6.0
    assert sum(e["tipo"] == "log" and e["bloco"] == 1 for e in entradas) >= 1  # todo bloco tem log


def test_llm_com_erro_e_desligado_apos_duas_falhas(cfg):
    llm = LLMQuebrado()
    camada, entradas = rodar(cfg, telemetria(60.0), llm)
    assert llm.chamadas == 2
    assert not camada.supervisor.llm_ativo
    assert all(e["texto_origem"] == "modelo" for e in entradas)
    novas = [e["observacao"].split(":")[0] for e in entradas if not e["continuacao"]]
    assert novas == ["erro do LLM", "erro do LLM", "LLM desligado após falhas seguidas"]


def test_executor_publica_os_blocos_em_ordem_com_ids_sequenciais(cfg):
    camada = CamadaAgentica(cfg, telemetria(40.0), Supervisor(FalsoLLM(), cfg))
    publicados = []
    resultados = ExecutorBlocos(camada, ao_publicar=publicados.append).rodar()
    assert [r.bloco for r in resultados] == [0, 1, 2, 3]
    assert publicados == resultados
    ids = [e["id"] for r in resultados for e in r.entradas]
    assert ids == list(range(1, len(ids) + 1))


# ---------------------------------------------------------------------------------------------
# Regressão com a volta real
# ---------------------------------------------------------------------------------------------
VOLTA_REAL = caminho("data/tratado/telemetria_tratada.csv")


@pytest.mark.skipif(not VOLTA_REAL.exists(), reason="telemetria tratada da volta real não encontrada")
def test_volta_real_tem_sete_lombadas_e_nenhuma_conducao_brusca(cfg):
    _, entradas = rodar(cfg, pd.read_csv(VOLTA_REAL))
    problemas = [e for e in entradas if e["tipo"] == "problema"]
    assert len(problemas) == 7
    assert {e["causa"] for e in problemas} == {"irregularidade_via"}
    assert sum(e["nivel"] == "critico" for e in problemas) == 2
    blocos_com_log = {e["bloco"] for e in entradas if e["tipo"] == "log"}
    assert blocos_com_log == set(range(16))
