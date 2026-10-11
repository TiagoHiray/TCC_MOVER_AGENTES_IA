"""Testes do plano futuro do caminhão (sempre da corrida atual) e do especialista de ML de suavização.

O caminhão autônomo (X) decide antes de agir: a cada bloco, a partir do estado medido agora na
corrida, ele monta o plano dos próximos 10 s e o manda aos agentes, que o alteram levemente (o modelo
de ML propõe quanto suavizar, o modelo do caminhão confere) antes de ele ser executado (Y). Aqui: o
plano e as decisões bruscas dele, as manobras (pela rota e pelo Traffic Manager), o modelo de ML, o
Decisor com o ML, a corrida com o ML, a corrida ao vivo (dirigir, que treina o ML na primeira vez) e
a análise offline da corrida. Sem CARLA e sem Ollama; a pista é o estádio de tests/auxiliares.py.

Rodar a partir da raiz do repositório:
    python -m pytest tests/test_planejador.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml
from auxiliares import plano_estadio

from mover.agentes.decisao import janela_inicial
from mover.agentes.rodar_agentes import carregar_telemetria
from mover.autonomo.conducao import Conducao, OpcoesConducao, metricas_da_corrida
from mover.autonomo.governador import Intervencao
from mover.autonomo.montagem import Montagem, montar
from mover.autonomo.planejador import PlanoFuturo, manobras_da_rota, manobras_do_tm
from mover.config import carregar_yaml

PARADO = {"t": 0.0, "d": 0.0, "v": 0.0, "a": 0.0}


@pytest.fixture(scope="module")
def m() -> Montagem:
    return montar(carregar_yaml("config/autonomo.yaml"), plano=plano_estadio())


class SuavizadorFixo:
    """Modelo de ML de teste: propõe sempre a mesma suavização e guarda as entradas que recebeu."""

    nome = "fixo (teste)"

    def __init__(self, desacel: float, acel: float, teto: float | None = None) -> None:
        self.proposta = (desacel, acel, teto)
        self.entradas: list[dict[str, float]] = []

    def propor(self, x: dict[str, float]):
        self.entradas.append(dict(x))
        return self.proposta


# ---------------------------------------------------------------------------------------------
# O plano do caminhão
# ---------------------------------------------------------------------------------------------
def test_plano_sai_do_estado_atual_e_aponta_as_decisoes_bruscas_com_a_zona_que_causa(m):
    gov = m.governador("agentes", 1)
    plano = m.planejador(1).planejar(gov, PARADO, 0, janela_inicial(0.0, 10.0))
    assert isinstance(plano, PlanoFuturo) and plano.estado == PARADO
    assert [i.chave for i in plano.pendentes] == ["partida@0", "lombada_1@0", "curva_1@0"]
    traj = plano.trajetoria
    assert traj.d[0] < 0.1 and traj.d[-1] >= plano.d_fim - 0.5 and (traj.v_desejada > 0).any()
    brutas = plano.decisoes_bruscas
    assert brutas and all(b["fonte"] == "plano" and b["em_s"] >= 0 for b in brutas)
    assert {b["tipo"] for b in brutas} == {"frenagem_brusca", "arrancada_brusca"}
    freia_na_lombada = next(b for b in brutas if b["elemento"] == "lombada_1@0" and b["tipo"] == "frenagem_brusca")
    assert freia_na_lombada["acel_mps2"] < -3.5 and freia_na_lombada["d_m"] < 30.0  # freia tarde, antes da lombada
    assert not gov.decidida("lombada_1@0")  # planejar não muda o governador
    resumo = plano.resumo()
    assert resumo["t_ini"] == 0.0 and len(resumo["t"]) == len(resumo["v_kmh"]) == len(resumo["a_mps2"]) > 10
    json.dumps(resumo)  # vai para o log e para a página


def test_plano_muda_com_o_estado_medido_e_com_o_que_ja_foi_decidido(m):
    planejador = m.planejador(1)
    gov = m.governador("agentes", 1)
    agora = {"t": 20.0, "d": 120.0, "v": 8.0, "a": 0.0}
    plano = planejador.planejar(gov, agora, 2, (30.0, 40.0))
    assert [i.chave for i in plano.pendentes] == ["lombada_2@0", "portao_1@0"]
    assert plano.trajetoria.d[0] == pytest.approx(120.0 + 8.0 * m.dt, abs=0.05)  # parte de onde o caminhão está
    assert {b["elemento"] for b in plano.decisoes_bruscas} == {"lombada_2@0", "portao_1@0"}
    # o que já foi alterado sai do plano seguinte (e entra na projeção)
    gov.definir("lombada_2@0", Intervencao.da_opcao("suave", m.opcoes))
    depois = planejador.planejar(gov, agora, 2, (30.0, 40.0))
    assert [i.chave for i in depois.pendentes] == ["portao_1@0"]
    assert {b["elemento"] for b in depois.decisoes_bruscas} == {"portao_1@0"}


def test_manobras_pela_rota_e_pelo_traffic_manager(m):
    rota = m.plano.rota
    gov = m.governador("agentes", 1)
    plano = m.planejador(1).planejar(gov, PARADO, 0, janela_inicial(0.0, 10.0))
    # sem o CARLA: pela curvatura da rota (o estádio roda no sentido anti-horário: curvas à esquerda)
    assert plano.fonte_manobras == "rota"
    curva = plano.manobras[0]
    assert curva["tipo"] == "esquerda" and curva["d_m"] == pytest.approx(61.0, abs=3.0)
    assert curva["angulo_graus"] == pytest.approx(180.0, abs=6.0) and curva["em_s"] > 0
    volta = manobras_da_rota(m.plano, 0.0, m.plano.comprimento, plano.trajetoria, 0.0)
    assert [c["tipo"] for c in volta] == ["esquerda", "esquerda"]
    # no CARLA: as ações que o Traffic Manager já decidiu (get_all_actions), localizadas na rota
    acoes = [("LaneFollow", rota.ponto(5.0).x, rota.ponto(5.0).y), ("Left", rota.ponto(60.0).x, rota.ponto(60.0).y),
             ("Left", rota.ponto(64.0).x, rota.ponto(64.0).y), ("Straight", rota.ponto(130.0).x, rota.ponto(130.0).y)]
    manobras = manobras_do_tm(m.plano, acoes, 0.0, 200.0, plano.trajetoria, 0.0)
    assert [(c["tipo"], c["fonte"]) for c in manobras] == [("esquerda", "traffic_manager"), ("reto", "traffic_manager")]
    assert manobras[0]["d_m"] == pytest.approx(60.0, abs=1.0)  # a ação repetida em waypoints seguidos conta uma vez
    com_tm = m.planejador(1).planejar(gov, PARADO, 0, janela_inicial(0.0, 10.0), acoes)
    assert com_tm.fonte_manobras == "traffic_manager" and com_tm.manobras[0]["tipo"] == "esquerda"


# ---------------------------------------------------------------------------------------------
# Especialista de ML de suavização
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def modelo_pequeno(tmp_path_factory) -> tuple[Path, dict]:
    pytest.importorskip("sklearn")
    from mover.agentes.suavizador_ml import treinar

    arquivo = tmp_path_factory.mktemp("ml") / "suavizador.joblib"
    meta = treinar(carregar_yaml("config/autonomo.yaml"), situacoes=40, saida=arquivo, n_arvores=10, processos=1)
    return arquivo, meta


def test_modelo_de_ml_treina_com_situacoes_simuladas_e_propoe_uma_suavizacao_leve(modelo_pequeno):
    from mover.agentes.suavizador_ml import ALVOS, FEATURES, SuavizadorML, caracteristicas, carregar_suavizador

    arquivo, meta = modelo_pequeno
    assert meta["amostras"] >= 10 and set(meta["r2_teste"]) == set(ALVOS) and meta["features"] == list(FEATURES)
    assert 0.0 <= meta["validacao"]["sem_eventos_pct"] <= 100.0
    assert json.loads(arquivo.with_suffix(".json").read_text(encoding="utf-8"))["amostras"] == meta["amostras"]
    modelo = SuavizadorML.carregar(arquivo)
    x = caracteristicas(v_atual_kmh=29.0, v_alvo_kmh=10.0, cruzeiro_kmh=30.0, distancia_m=40.0, comprimento_m=12.0,
                        duracao_s=0.0, parada=False, gap_seguinte_m=None, v_alvo_seguinte_kmh=None,
                        desacel_plano_mps2=-5.7, acel_plano_mps2=1.6)
    assert x["gap_seguinte_m"] == 200.0 and x["desacel_plano_mps2"] == 5.7  # sem zona seguinte: gap máximo
    desacel, acel, teto = modelo.propor(x)
    assert 0.4 <= desacel <= 1.8 and 0.3 <= acel <= 1.2 and (teto is None or 12.0 <= teto < 30.0)
    cfg = carregar_yaml("config/autonomo.yaml")
    assert carregar_suavizador(dict(cfg, suavizador_ml={"modelo": str(arquivo)}), False).metadados["amostras"] == meta["amostras"]
    assert carregar_suavizador(dict(cfg, suavizador_ml={"ativo": False})) is None
    assert carregar_suavizador(dict(cfg, suavizador_ml={"modelo": str(arquivo.parent / "nao.joblib")}), False) is None


def test_decisor_fica_com_a_proposta_do_ml_quando_ela_passa_na_conferencia(m):
    gov = m.governador("agentes", 1)
    plano = m.planejador(1).planejar(gov, PARADO, 0, janela_inicial(0.0, 10.0))
    ml = SuavizadorFixo(0.7, 0.5)
    dec = m.decisor(1, llm=None, suavizador=ml).decidir(plano, gov)
    assert dec.recomendada == dec.escolhida == "ml" and dec.motivo_ml is None
    assert dec.opcao_escolhida.eventos == 0 < dec.opcao_bruta.eventos
    assert dec.opcao_escolhida.combustivel_ml < dec.opcao_bruta.combustivel_ml
    assert set(dec.propostas_ml) == {b["elemento"] for b in dec.brutas} == {"partida@0", "lombada_1@0", "curva_1@0"}
    # as entradas do ML vêm do plano: a velocidade agora, a zona e a frenagem que o próprio plano traz
    lombada = next(x for x in ml.entradas if x["v_alvo_kmh"] == 10.0)
    assert lombada["distancia_m"] == 30.0 and lombada["desacel_plano_mps2"] > 3.5 and lombada["v_atual_kmh"] == 0.0
    interv = dec.intervencoes_para("ml")
    assert interv["lombada_1@0"] == Intervencao("ml", 0.7, 0.5, None, "regra")
    entrada = dec.para_entrada()
    assert entrada["acao"] == "ajuste do ML" and entrada["ml"]["modelo"] == "fixo (teste)"
    assert entrada["ml"]["propostas"]["lombada_1@0"]["desacel_mps2"] == 0.7
    assert "O caminhão planeja" in dec.texto and "O ML suavizou" in dec.texto


@pytest.mark.parametrize("proposta, motivo", [((3.5, 1.2), "teria 2 evento(s) brusco(s)"),
                                              ((0.2, 0.1), "alongaria demais o trecho")])
def test_proposta_do_ml_reprovada_pelo_modelo_do_caminhao_cai_na_opcao_fixa(m, proposta, motivo):
    gov = m.governador("agentes", 1)
    plano = m.planejador(1).planejar(gov, PARADO, 0, janela_inicial(0.0, 10.0))
    dec = m.decisor(1, llm=None, suavizador=SuavizadorFixo(*proposta)).decidir(plano, gov)
    assert dec.escolhida not in ("ml", "nenhuma") and dec.opcao_escolhida.eventos == 0
    assert motivo in dec.motivo_ml and "valeu a opção" in dec.motivo_ml
    assert "ml" in {o.opcao for o in dec.opcoes}  # a proposta foi simulada e fica no log


class LLMProibido:
    nome = "proibido"

    def gerar(self, sistema, usuario, esquema):
        raise AssertionError("o Supervisor não deveria ser chamado")


def test_sem_decisao_brusca_no_plano_os_agentes_nao_alteram_nada(m):
    from mover.agentes.decisao import Decisor
    from mover.autonomo.governador import Governador
    from mover.autonomo.missao import PlanoMissao, Zona
    from mover.autonomo.planejador import PlanejadorCaminhao

    # só uma curva aberta (limite de 35 km/h, acima do cruzeiro de 30 km/h): o plano não freia nem arranca
    leve = PlanoMissao(m.plano.rota, 30.0, [Zona("curva_leve", "curva", 100.0, 140.0, 35.0, "curva aberta")], [])
    gov = Governador(leve, m.cfg_governador, modo="agentes", n_voltas=None, opcoes=m.opcoes)
    plano = PlanejadorCaminhao(leve, m.modelo()).planejar(gov, {"t": 10.0, "d": 40.0, "v": 30.0 / 3.6, "a": 0.0}, 2,
                                                          (20.0, 30.0))
    assert [i.chave for i in plano.pendentes] == ["curva_leve@0"] and plano.decisoes_bruscas == []
    ml = SuavizadorFixo(0.7, 0.5)
    decisor = Decisor(leve, m.modelo(), m.cfg_decisao, llm=LLMProibido(), opcoes=m.opcoes, suavizador=ml)
    dec = decisor.decidir(plano, gov)
    decisor.encerrar()
    assert dec.escolhida == dec.recomendada == "nenhuma" and dec.origem == "regra"
    assert dec.propostas_ml == {} and ml.entradas == []  # o ML nem é consultado
    # e sem nada pendente à frente (tudo já decidido), o caminhão nem manda plano
    tudo_decidido = m.governador("agentes", 1)
    for chave in ("lombada_2@0", "portao_1@0", "chegada@1"):
        tudo_decidido.definir(chave, Intervencao.da_opcao("suave", m.opcoes))
    agora = {"t": 20.0, "d": 120.0, "v": 8.0, "a": 0.0}
    assert m.planejador(1).planejar(tudo_decidido, agora, 2, (30.0, 40.0)) is None
    assert m.decisor(1).decidir(None, tudo_decidido) is None


# ---------------------------------------------------------------------------------------------
# Corrida com o ML (substituto)
# ---------------------------------------------------------------------------------------------
def test_corrida_com_o_ml_altera_o_plano_de_cada_bloco_feito_na_hora(m):
    decisor = m.decisor(1, None, SuavizadorFixo(0.7, 0.5))
    op = OpcoesConducao(voltas=1, usar_llm=False, assincrona=False, sem_espera=True, fator_tempo=40.0, duracao_bloco_s=10.0)
    res = Conducao(m.plano, m.simulador_substituto(), m.governador("agentes", 1), decisor, op, m.caminhao, m.consumo).rodar()
    decisor.encerrar()
    assert res.completa and len(res.decisoes) >= 2 and res.aplicacoes["atrasadas"] == 0
    assert all(d["fonte"] == "plano" and d["opcao"] == "ml" for d in res.decisoes)
    assert all(a["opcao"] == "ml" and a["a_tempo"] for d in res.decisoes for a in d["aplicacao"])
    met = metricas_da_corrida(res, m.plano.comprimento, m.caminhao, m.consumo, m.limiares)["total"]
    assert met["eventos_atencao"] + met["eventos_criticos"] == 0
    # cada plano parte do estado medido naquela hora, nesta mesma corrida
    tel = pd.DataFrame(res.linhas)
    for d in res.decisoes[1:]:
        linha = tel.iloc[(tel["t"] - d["plano"]["t_ini"]).abs().argmin()]
        assert linha["t"] == pytest.approx(d["plano"]["t_ini"], abs=0.01)
        assert linha["d"] == pytest.approx(d["plano"]["d_ini"], abs=0.06)
        assert d["plano"]["v_kmh"][0] == pytest.approx(linha["speed_kmh"], abs=1.0)


# ---------------------------------------------------------------------------------------------
# Corrida ao vivo (dirigir) e análise offline da corrida
# ---------------------------------------------------------------------------------------------
def _config_temporaria(pasta: Path) -> Path:
    cfg = carregar_yaml("config/autonomo.yaml")
    cfg["corridas"] = {"pasta": str(pasta / "corridas")}
    cfg["suavizador_ml"] = {**(cfg.get("suavizador_ml") or {}), "modelo": str(pasta / "suavizador.joblib"),
                            "situacoes": 40, "n_arvores": 10, "processos": 1}
    arquivo = pasta / "autonomo.yaml"
    arquivo.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return arquivo


def _dirigir(pasta: Path, *extra: str) -> int:
    import mover.autonomo.dirigir as dirigir

    pasta.mkdir(parents=True, exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(dirigir, "montar", lambda cfg, missao=None: montar(cfg, plano=plano_estadio()))
        return dirigir.main(["--config", str(_config_temporaria(pasta)), "--sem-carla", "--sem-painel", "--sem-llm",
                             "--sem-espera", *extra])


@pytest.fixture(scope="module")
def corrida_ao_vivo(tmp_path_factory) -> Path:
    """Uma volta com agentes pelo dirigir, sem modelo de ML gravado (ele treina um pequeno na hora)."""
    pytest.importorskip("sklearn")
    pasta = tmp_path_factory.mktemp("dirigir")
    assert _dirigir(pasta) == 0
    return pasta


def test_dirigir_treina_o_ml_na_primeira_vez_e_grava_o_plano_de_cada_bloco(m, corrida_ao_vivo):
    assert (corrida_ao_vivo / "suavizador.joblib").is_file()
    corrida = next((corrida_ao_vivo / "corridas").glob("*_agentes"))
    resumo = json.loads((corrida / "resumo.json").read_text(encoding="utf-8"))
    assert resumo["suavizador_ml"].startswith("floresta aleatória") and resumo["simulador"] == "substituto"
    assert resumo["fonte_linha_de_base"].startswith("plano do caminhão sem agentes")
    assert json.loads((corrida / "linha_de_base.json").read_text(encoding="utf-8"))["fonte_curta"] == "previsto"
    decisoes = [json.loads(linha) for linha in (corrida / "decisoes.jsonl").read_text(encoding="utf-8").splitlines() if linha]
    assert decisoes and all(d["fonte"] == "plano" and d["plano"]["t"] and d["ml"]["modelo"] for d in decisoes)
    assert all(d["opcao"] != "nenhuma" for d in decisoes if d["brutas"])
    tel = pd.read_csv(corrida / "telemetria.csv")
    assert tel["d"].iloc[-1] == pytest.approx(m.plano.comprimento, abs=15.0)


def test_dirigir_sem_ml_usa_so_as_opcoes_fixas(tmp_path):
    assert _dirigir(tmp_path, "--sem-ml") == 0
    assert not (tmp_path / "suavizador.joblib").exists()
    corrida = next((tmp_path / "corridas").glob("*_agentes"))
    assert json.loads((corrida / "resumo.json").read_text(encoding="utf-8"))["suavizador_ml"] is None
    decisoes = [json.loads(linha) for linha in (corrida / "decisoes.jsonl").read_text(encoding="utf-8").splitlines() if linha]
    assert decisoes and all(d["opcao"] != "ml" and not d["ml"]["propostas"] for d in decisoes)


def test_analise_offline_le_a_corrida_mais_recente_no_formato_da_camada_de_logs(corrida_ao_vivo, tmp_path):
    cfg = carregar_yaml("config/agentes.yaml")
    assert cfg["entrada"]["telemetria"] == "corrida"
    cfg["entrada"]["pasta_corridas"] = str(corrida_ao_vivo / "corridas")
    corrida = next((corrida_ao_vivo / "corridas").glob("*_agentes"))
    dados, origem = carregar_telemetria(cfg)
    assert origem == {"tipo": "corrida", "corrida": corrida.name, "simulador": "substituto", "modo": "agentes", "voltas": 1}
    for coluna in ("sim_time", "speed_mps", "acc_long", "jerk_long", "manobra", "fonte"):
        assert coluna in dados.columns, coluna
    assert set(dados["fonte"]) == {"substituto"}
    assert carregar_telemetria(cfg, corrida=str(corrida))[1]["corrida"] == corrida.name
    with pytest.raises(SystemExit, match="legado"):
        carregar_telemetria(cfg, injetar_eventos=True)
    cfg["entrada"]["pasta_corridas"] = str(tmp_path)  # nenhuma corrida ainda: pede uma volta antes
    with pytest.raises(SystemExit, match="mover.autonomo.dirigir"):
        carregar_telemetria(cfg)


def test_especialista_de_ml_dos_logs_treina_com_a_corrida(corrida_ao_vivo, tmp_path):
    from mover.agentes.treinar_especialista_ml import ler_metadados, treinar

    cfg = carregar_yaml("config/agentes.yaml")
    cfg["entrada"]["pasta_corridas"] = str(corrida_ao_vivo / "corridas")
    corrida = next((corrida_ao_vivo / "corridas").glob("*_agentes"))
    modelo = tmp_path / "especialista_ml.joblib"
    meta = treinar(cfg, saida=modelo)
    assert meta["corrida"] == corrida.name and meta["amostras"] > 100
    assert ler_metadados(modelo)["corrida"] == corrida.name
