"""Testes do consumo medido: o mesmo medidor no painel ao vivo, na corrida e no benchmark.

Rodar a partir da raiz do repositório:
    python -m pytest tests/test_consumo.py -q

Conferem que o medidor (MedidorConsumo / consumo_medido):
- não soma consumo antes de uma freada brusca (o Butterworth de fase zero usado antes somava);
- conta a energia de cada frenagem inteira, seja a freada seca ou suave;
- não transforma o tremor de ~10 Hz da velocidade do CARLA em consumo;
- dá no benchmark o mesmo consumo que a corrida gravou ao vivo;
- e que o --recalcular refaz os números de um benchmark salvo a partir das telemetrias.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml
from auxiliares import plano_estadio

from mover.agentes.llm import FalsoLLM
from mover.autonomo.conducao import Conducao, OpcoesConducao, metricas_da_corrida
from mover.autonomo.consumo import (MedidorConsumo, ParametrosConsumo, consumo_medido, potencia_rodas_kw, suavizar,
                                    taxa_combustivel_l_s)
from mover.autonomo.montagem import montar
from mover.autonomo.veiculo import ParametrosCaminhao
from mover.config import RAIZ, carregar_yaml

DT = 0.05


def _freada(duracao_freada_s: float, v0: float = 8.0, v1: float = 2.0, total_s: float = 20.0, t0: float = 10.0):
    """Velocidade constante v0, freada linear até v1 em `duracao_freada_s` a partir de t0, e v1 até o fim."""
    t = np.arange(0.0, total_s, DT)
    v = np.where(t < t0, v0, np.maximum(v1, v0 - (v0 - v1) * (t - t0) / duracao_freada_s))
    a = np.gradient(v, DT)
    return t, v, a


def test_medidor_nao_soma_consumo_antes_de_uma_freada_brusca():
    c, p = ParametrosCaminhao(), ParametrosConsumo()
    t, v, a = _freada(0.2)
    litros = consumo_medido(v, DT, c, p)["litros"]
    cruzeiro = float(taxa_combustivel_l_s(potencia_rodas_kw(np.array([8.0]), np.array([0.0]), 0.0, c), p)[0]) * DT
    antes = t < 10.0
    # o medidor é causal: até a freada, cada tick gasta exatamente o de cruzeiro
    assert litros[antes] == pytest.approx(np.full(antes.sum(), cruzeiro), rel=1e-9)
    # o Butterworth de fase zero oscila antes do degrau e soma consumo que não existiu
    a_bw = suavizar(a, DT, 1.5)
    p_bw = potencia_rodas_kw(v, a_bw, 0.0, c)
    litros_bw = taxa_combustivel_l_s(p_bw, p) * DT
    janela = (t > 9.0) & (t < 10.0)
    assert litros_bw[janela].sum() > 1.05 * litros[janela].sum()


def test_energia_de_freio_e_a_energia_cinetica_perdida_seja_a_freada_seca_ou_suave():
    c = ParametrosCaminhao(cd=0.0, cr=0.0)  # sem arrasto e sem rolamento: só a energia cinética
    p = ParametrosConsumo()
    esperado_kwh = 0.5 * c.massa_kg * (8.0 ** 2 - 2.0 ** 2) / 3.6e6
    for duracao in (0.2, 1.0, 4.0):
        _, v, _ = _freada(duracao)
        freio = consumo_medido(v, DT, c, p)["freio_kwh"].sum()
        assert freio == pytest.approx(esperado_kwh, rel=0.01), duracao


def test_tremor_de_10hz_na_velocidade_nao_vira_consumo():
    c, p = ParametrosCaminhao(), ParametrosConsumo()
    n = 2000
    liso = np.full(n, 6.0)
    tremido = liso + 0.02 * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)  # ±0,02 m/s a 10 Hz, como no CARLA a 0,85
    base = consumo_medido(liso, DT, c, p)["litros"].sum()
    assert consumo_medido(tremido, DT, c, p)["litros"].sum() == pytest.approx(base, rel=0.03)
    # multiplicar a aceleração bruta (diferença tick a tick, como a do CARLA) pela velocidade somaria o tremor
    a = np.diff(tremido, prepend=tremido[0]) / DT
    bruto = (taxa_combustivel_l_s(potencia_rodas_kw(tremido, a, 0.0, c), p) * DT).sum()
    assert bruto > 1.2 * base


def test_medidor_ao_vivo_e_consumo_medido_sao_o_mesmo_calculo():
    c, p = ParametrosCaminhao(), ParametrosConsumo()
    _, v, _ = _freada(0.5)
    rampa = 0.02 * np.sin(np.arange(len(v)) * DT / 3.0)
    medidor = MedidorConsumo(c, p)
    acumulado = [medidor.passo(float(vi), DT, float(ri)) for vi, ri in zip(v, rampa)]
    serie = consumo_medido(v, DT, c, p, rampa)
    assert np.cumsum(serie["litros"]) == pytest.approx(np.array(acumulado), rel=1e-12)
    assert serie["freio_kwh"].sum() == pytest.approx(medidor.energia_freio_kwh, rel=1e-12)
    assert serie["rodas_kwh"].sum() == pytest.approx(medidor.energia_rodas_kwh, rel=1e-12)


def test_benchmark_usa_o_consumo_que_a_corrida_gravou_ao_vivo():
    m = montar(carregar_yaml("config/autonomo.yaml"), plano=plano_estadio())
    decisor = m.decisor(1, FalsoLLM())
    op = OpcoesConducao(voltas=1, usar_llm=True, assincrona=False, sem_espera=True, duracao_bloco_s=10.0)
    for modo, agentes in (("bruto", None), ("agentes", decisor)):
        res = Conducao(m.plano, m.simulador_substituto(), m.governador(modo, 1), agentes, op, m.caminhao,
                       m.consumo).rodar()
        assert res.completa
        met = metricas_da_corrida(res, m.plano.comprimento, m.caminhao, m.consumo, m.limiares)
        assert 1000.0 * met["total"]["combustivel_l"] == pytest.approx(res.linhas[-1]["combustivel_ml"], rel=1e-9)
        assert sum(v["combustivel_l"] for v in met["voltas"]) == pytest.approx(met["total"]["combustivel_l"], rel=1e-9)
        assert met["total"]["custo_r"] == pytest.approx(met["total"]["combustivel_l"] * m.consumo.preco_r_l)
    decisor.encerrar()


def test_recalcular_benchmark_salvo_refaz_os_numeros_das_telemetrias(tmp_path, monkeypatch):
    import mover.autonomo.benchmark as modulo

    monkeypatch.setattr(modulo, "montar", lambda cfg, missao=None: montar(cfg, plano=plano_estadio()))
    cfg = carregar_yaml("config/autonomo.yaml")
    cfg["suavizador_ml"] = {**(cfg.get("suavizador_ml") or {}), "modelo": str(tmp_path / "suavizador.joblib"),
                            "situacoes": 40, "n_arvores": 10, "processos": 1}
    arquivo = tmp_path / "autonomo.yaml"
    arquivo.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    pasta = tmp_path / "bench"
    assert modulo.main(["--config", str(arquivo), "--voltas", "1", "--sem-carla", "--provedor", "falso", "--silencioso",
                        "--saida", str(pasta)]) == 0
    original = pd.read_csv(pasta / "resultados.csv")
    resumo_original = json.loads((pasta / "resumo.json").read_text(encoding="utf-8"))
    telemetria = (pasta / "telemetria_com_agentes.csv").read_bytes()
    # o benchmark grava o consumo do mesmo medidor do painel: a coluna ao vivo bate com o resultados.csv
    tel = pd.read_csv(pasta / "telemetria_sem_agentes.csv")
    sem = original[(original["tipo"] == "volta") & (original["bateria"] == "sem_agentes")]
    assert sem["combustivel_ml"].sum() == pytest.approx(tel["combustivel_ml"].iloc[-1], abs=0.02)

    (pasta / "resultados.csv").write_text("estragado\n", encoding="utf-8")
    assert modulo.main(["--config", str(arquivo), "--recalcular", str(pasta)]) == 0
    refeito = pd.read_csv(pasta / "resultados.csv")
    pd.testing.assert_frame_equal(refeito, original)
    resumo = json.loads((pasta / "resumo.json").read_text(encoding="utf-8"))
    assert resumo["recalculado_em"] and resumo["data"] == resumo_original["data"]
    assert resumo["decisoes"] == resumo_original["decisoes"] and resumo["extrapolacao"] == resumo_original["extrapolacao"]
    assert "energia cinética" in resumo["modelo_consumo"]
    # dentro do repositório o resumo guarda caminhos relativos (nada da máquina de quem rodou)
    assert modulo._no_resumo(RAIZ / "data" / "benchmark" / "x" / "resultados.csv") == "data/benchmark/x/resultados.csv"
    assert (pasta / "telemetria_com_agentes.csv").read_bytes() == telemetria  # as telemetrias não mudam
