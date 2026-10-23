"""Testes da corrida autônoma (sem CARLA e sem LLM de verdade).

Rodar a partir da raiz do repositório:
    python -m pytest tests -q

Cobre a rota, o plano de missão, o governador (o piloto sozinho tem trancos; com intervenções,
não), o modelo de consumo, o Decisor (regra, LLM válido, inválido, atrasado e fora do conjunto
seguro), o laço de condução no simulador substituto (síncrono e assíncrono) e o benchmark. O plano
de teste é uma volta de ~246 m no estádio de tests/auxiliares.py.
"""

from __future__ import annotations

import importlib.util
import json
import math
import time
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from auxiliares import plano_estadio

from mover.agentes.decisao import Decisor, janela_inicial
from mover.agentes.llm import FalsoLLM
from mover.autonomo import benchmark
from mover.autonomo.conducao import Conducao, OpcoesConducao, Relogio, metricas_da_corrida
from mover.autonomo.consumo import Limiares, ParametrosConsumo, metricas, potencia_rodas_kw, taxa_combustivel_l_s
from mover.autonomo.governador import ConfigGovernador, Governador, Intervencao
from mover.autonomo.missao import PlanoMissao
from mover.autonomo.montagem import Montagem, montar
from mover.autonomo.previsao import ModeloInterno
from mover.autonomo.rota import Rota, de_carla, para_carla
from mover.autonomo.rota_mapa import Faixa, grafo_de_faixas, itinerario_de_faixas, rota_do_itinerario
from mover.autonomo.veiculo import ParametrosCaminhao, ParametrosPiloto
from mover.config import caminho, carregar_yaml


@pytest.fixture(scope="module")
def m() -> Montagem:
    return montar(carregar_yaml("config/autonomo.yaml"), plano=plano_estadio())


class LLMRoteiro:
    """LLM de teste: devolve as respostas do roteiro, na ordem (ou levanta a exceção da vez)."""

    nome = "roteiro"

    def __init__(self, roteiro: list[Any] | None = None, atraso_s: float = 0.0) -> None:
        self.roteiro = list(roteiro or [])
        self.atraso_s = atraso_s
        self.prompts: list[str] = []

    def gerar(self, sistema: str, usuario: str, esquema):
        self.prompts.append(usuario)
        if self.atraso_s:
            time.sleep(self.atraso_s)
        resposta = self.roteiro.pop(0)
        if isinstance(resposta, Exception):
            raise resposta
        return esquema(**resposta)


def eventos(traj, dt: float = 0.05) -> int:
    met = metricas(traj.t, traj.v, traj.a, ParametrosCaminhao(), ParametrosConsumo(), Limiares())
    return met["eventos_atencao"] + met["eventos_criticos"]


# ---------------------------------------------------------------------------------------------
# Rota e plano de missão
# ---------------------------------------------------------------------------------------------
def test_rota_projeta_e_conta_a_distancia_na_passagem_pela_largada():
    plano = plano_estadio()
    rota = plano.rota
    p = rota.ponto(100.0)
    s, lateral = rota.projetar(p.x - 0.5 * math.sin(p.rumo), p.y + 0.5 * math.cos(p.rumo))
    assert s == pytest.approx(100.0, abs=0.3) and lateral == pytest.approx(0.5, abs=0.05)  # 0,5 m à esquerda
    L = rota.comprimento
    d = rota.avancar_distancia(L - 1.0, 0.5)  # passou pela largada: a distância continua crescendo
    assert d == pytest.approx(L + 0.5, abs=1e-6)
    x, y, yaw = para_carla(10.0, 20.0, math.radians(30.0))
    assert (x, y, yaw) == pytest.approx((10.0, -20.0, -30.0))
    assert de_carla(x, y, yaw) == pytest.approx((10.0, 20.0, math.radians(30.0)))


def test_plano_tem_partida_e_chegada_e_volta_igual_do_yaml(tmp_path):
    plano = plano_estadio()
    L = plano.comprimento
    inst = plano.instancias(0.0, 2 * L + 1.0, n_voltas=2)
    chaves = [i.chave for i in inst]
    assert chaves[0] == "partida@0" and chaves[-1] == "chegada@2"
    assert chaves.count("lombada_1@0") == chaves.count("lombada_1@1") == 1 and "lombada_1@2" not in chaves
    assert plano.limite_kmh(35.0) == 10.0 and plano.limite_kmh(L + 35.0) == 10.0 and plano.limite_kmh(20.0) == 30.0
    plano.salvar(tmp_path / "missao.yaml", "# teste")
    copia = PlanoMissao.carregar(tmp_path / "missao.yaml")
    assert copia.zonas == plano.zonas and copia.paradas == plano.paradas
    assert copia.comprimento == pytest.approx(L, rel=5e-3)  # o YAML guarda a rota a cada 5 m


def _reta(a: tuple[float, float], b: tuple[float, float], passo: float = 1.0) -> np.ndarray:
    n = max(2, int(round(math.dist(a, b) / passo)) + 1)
    return np.column_stack([np.linspace(a[0], b[0], n), np.linspace(a[1], b[1], n)])


def test_rota_do_mapa_acha_os_conectores_e_fecha_a_volta():
    """Quadrado de 4 vias (10-13) com conectores (90-93) nas quinas e uma via que segue reto (14)."""
    trechos = {10: ((5, 0), (45, 0)), 90: ((45, 0), (50, 5)), 11: ((50, 5), (50, 45)), 91: ((50, 45), (45, 50)),
               12: ((45, 50), (5, 50)), 92: ((5, 50), (0, 45)), 13: ((0, 45), (0, 5)), 93: ((0, 5), (5, 0)),
               14: ((45, 0), (90, 0))}
    faixas = {(v, 0): Faixa(v, 0, _reta(a, b)) for v, (a, b) in trechos.items()}
    grafo = grafo_de_faixas(faixas)
    assert set(grafo[(10, 0)]) == {(90, 0), (14, 0)}
    sequencia = itinerario_de_faixas(faixas, grafo, [10, 11, 12, 13])
    assert [v for v, _ in sequencia] == [10, 90, 11, 91, 12, 92, 13, 93]
    rota = rota_do_itinerario(faixas, sequencia, largada=(25.0, 0.0), suavizar_m=0.0)
    assert rota.comprimento == pytest.approx(4 * 40 + 4 * math.sqrt(50), abs=2.0)
    assert math.hypot(rota.x[0] - 25.0, rota.y[0]) < 1.0
    with pytest.raises(ValueError):
        itinerario_de_faixas(faixas, grafo, [10, 77])


def test_plano_do_mapa_real_tem_as_lombadas_e_faixas_do_proprio_mapa():
    cfg_sim = carregar_yaml("config/simulacao.yaml")
    if not caminho(cfg_sim["mapa"]["arquivo"]).exists():
        pytest.skip(f"sem {cfg_sim['mapa']['arquivo']}")
    from mover.autonomo.gerar_missao import gerar, objetos_do_mapa

    cfg_auto = carregar_yaml("config/autonomo.yaml")
    plano = gerar(cfg_sim, cfg_auto["gerar_missao"])
    assert 700.0 < plano.comprimento < 860.0
    lombadas = [z for z in plano.zonas if z.tipo == "lombada"]
    no_mapa = objetos_do_mapa(caminho(cfg_sim["mapa"]["arquivo"]), "speedBump")
    # campus_maua.xodr: 6 lombadas no mapa, 4 no anel R01-R10+R07-R02-R11 (as outras ficam na R03)
    assert len(no_mapa) == 6 and len(lombadas) == 4 and all("do mapa" in z.origem for z in lombadas)
    assert sum(z.tipo == "faixa" for z in plano.zonas) >= 4 and [p.id for p in plano.paradas] == ["portao_1"]
    assert plano.sentido == "anti-horário" and plano.mapa.endswith("campus_maua.xodr")
    # sem itinerário (mapas como o Town10HD): a volta automática acha o mesmo anel de ~784 m
    automatico = gerar(cfg_sim, {**cfg_auto["gerar_missao"], "rota": {"vias": "auto", "alvo_m": 784.0}})
    assert automatico.comprimento == pytest.approx(plano.comprimento, abs=3.0)
    assert "1, 5, 2, 6, achadas automaticamente" in automatico.observacoes[0]


def test_volta_automatica_numa_cidade_contorna_quarteiroes_sem_ir_e_voltar_pela_mesma_rua():
    from mover.autonomo.rota_mapa import largada_automatica, volta_automatica

    # grade de 4 x 4 cruzamentos a cada 100 m, ruas de mão dupla e conectores (junções) em cada cruzamento
    faixas, juncoes, via, chegadas, saidas = {}, set(), 1, {}, {}
    for i in range(4):
        for j in range(4):
            for dx, dy in ((1, 0), (0, 1)):
                if i + dx > 3 or j + dy > 3:
                    continue
                for sentido in (1, -1):
                    ux, uy = dx * sentido, dy * sentido
                    a, b = ((i, j), (i + dx, j + dy)) if sentido == 1 else ((i + dx, j + dy), (i, j))
                    lado = np.array([uy, -ux]) * 1.75  # mão à direita
                    p0 = np.array(a) * 100.0 + np.array([ux, uy]) * 10.0 + lado
                    p1 = np.array(b) * 100.0 - np.array([ux, uy]) * 10.0 + lado
                    faixas[(via, 0)] = Faixa(via, 0, _reta(tuple(p0), tuple(p1)))
                    saidas[(a, (ux, uy))], chegadas[(b, (ux, uy))] = (via, 0), (via, 0)
                    via += 1
    for (no, d_in), entra in chegadas.items():
        for (no2, d_out), sai in saidas.items():
            if no2 == no and d_out != (-d_in[0], -d_in[1]):
                faixas[(via, 0)] = Faixa(via, 0, _reta(tuple(faixas[entra].fim), tuple(faixas[sai].inicio), 0.5))
                juncoes.add(via)
                via += 1
    grafo = grafo_de_faixas(faixas)
    for alvo, quarteiroes in ((400.0, 1), (800.0, 2)):
        seq = volta_automatica(faixas, grafo, alvo, 300.0, 1500.0, juncoes)
        rota = rota_do_itinerario(faixas, seq, largada_automatica(faixas, seq, juncoes))
        ruas = [v for v, _ in seq if v not in juncoes]
        assert rota.comprimento == pytest.approx(alvo, rel=0.06) and len(ruas) == 4 * quarteiroes == len(set(ruas))
    with pytest.raises(ValueError):
        volta_automatica(faixas, grafo, 5000.0, 4000.0, 6000.0, juncoes)


def test_apresentacao_seco_cruza_as_situacoes_do_mapa_com_os_modos(capsys, tmp_path):
    from mover.autonomo import apresentacao

    assert apresentacao.main(["--seco", "--mapa", "campus", "--saida", str(tmp_path / "a")]) == 0
    plano = json.loads(capsys.readouterr().out)
    modos = {m["id"]: m for m in plano["modos"]}
    assert set(modos) == {m.id for m in apresentacao.MODOS}
    assert "--sem-agentes" in modos["so_carla"]["corrida"] and "completo" in modos["so_carla"]["benchmark"]
    assert {"--sem-ml", "--sem-llm"} <= set(modos["so_regra"]["benchmark"]) and "--sem-llm" in modos["so_ml"]["corrida"]
    assert "--sem-ml" in modos["so_llm"]["corrida"] and "--sem-llm" not in modos["so_llm"]["corrida"]
    assert modos["devagar_20"]["benchmark"][modos["devagar_20"]["benchmark"].index("--missao") + 1].endswith(
        "missao_cruzeiro20.yaml")
    assert all("--sem-navegador" in m["corrida"] for m in modos.values())
    assert [s["id"] for s in plano["situacoes"]] == ["curva", "pedestre", "veiculo", "lombada", "parada"]
    assert not (tmp_path / "a").exists()
    # Town10 no --seco: sem tocar no CARLA nem escrever nada; as situações saem depois de exportar o mapa
    assert apresentacao.main(["--seco", "--modos", "so_carla", "--saida", str(tmp_path / "t")]) == 0
    plano = json.loads(capsys.readouterr().out)
    assert [m["id"] for m in plano["modos"]] == ["so_carla", "completo"]  # a bateria sem agentes é a do completo
    assert plano["configs"]["config_simulacao"].endswith("simulacao.yaml") and not (tmp_path / "t").exists()


def _janelas_separadas(sits, L):
    marcas = np.zeros(int(L) + 1, int)
    for s in sits:
        largura = (s.s_fim - s.s_ini) % L
        marcas[(np.arange(int(s.s_ini) + 1, int(s.s_ini + largura))) % len(marcas)] += 1
    return marcas.max() <= 1


def test_situacoes_do_campus_ficam_separadas_e_criam_a_faixa_quando_o_mapa_nao_tem():
    from mover.autonomo.situacoes import escolher_situacoes, figurantes_do_cenario

    plano = PlanoMissao.carregar(caminho("config/missao.yaml"))
    L = plano.comprimento
    sits = {s.id: s for s in escolher_situacoes(plano)}
    assert list(sits) == ["curva", "pedestre", "veiculo", "lombada", "parada"] and _janelas_separadas(sits.values(), L)
    assert sits["curva"].s_ini < sits["curva"].s_fim < L - 40  # longe da chegada: a volta do vídeo não está parando
    pedestre = sits["pedestre"].figurantes["pedestres"][0]["s_m"]
    assert any(z.tipo == "faixa" and z.s_ini <= pedestre <= z.s_fim for z in plano.zonas)
    veiculo = sits["veiculo"].figurantes["veiculos"][0]
    assert (sits["veiculo"].s_fim - sits["veiculo"].s_ini) % L >= 120 and veiculo["percurso_m"] > 50
    cenario = figurantes_do_cenario(list(sits.values()))
    assert len(cenario["pedestres"]) == 1 and len(cenario["veiculos"]) == 1

    sem_faixa = PlanoMissao.carregar(caminho("config/missao.yaml"))
    sem_faixa.zonas = [z for z in sem_faixa.zonas if z.tipo != "faixa"]
    sits = {s.id: s for s in escolher_situacoes(sem_faixa)}
    criada = [z for z in sem_faixa.zonas if z.id == "faixa_cenario"]
    assert len(criada) == 1 and criada[0].s_ini <= sits["pedestre"].figurantes["pedestres"][0]["s_m"] <= criada[0].s_fim
    assert _janelas_separadas(sits.values(), L)


def test_metricas_e_recorte_de_cada_situacao_saem_da_telemetria(tmp_path):
    from mover.autonomo import apresentacao
    from mover.autonomo.situacoes import Situacao, metricas_situacao

    L, dt = 200.0, 0.05
    t = np.arange(0.0, 2 * L / 5.0 + 1.0, dt)
    tel = pd.DataFrame({"sim_time": t, "d": np.minimum(5.0 * t, 2 * L), "speed_mps": 5.0, "acc_long": 0.0,
                        "combustivel_ml": t, "t_parede": 1000.0 + 1.5 * t})
    tel["volta"] = np.minimum(tel["d"] // L, 1).astype(int)
    meio, largada = Situacao("meio", "meio", "?", 50.0, 90.0), Situacao("largada", "largada", "?", 180.0, 20.0)
    lim = Limiares()
    met = metricas_situacao(tel, meio, L, lim)
    assert met["voltas"] == 2 and met["tempo_s"] == pytest.approx(8.0, abs=0.1) and met["eventos"] == 0
    assert metricas_situacao(tel, largada, L, lim)["voltas"] == 1  # a segunda volta termina na chegada
    trechos = apresentacao.trechos_do_video(tel, [meio, largada], L, t_gravacao=1005.0)
    assert trechos["meio"] == pytest.approx((6.0, 20.0), abs=0.1)  # 4 s antes e depois, no relógio de parede
    assert trechos["largada"][0] == pytest.approx(1000 + 1.5 * 36 - 1005 - 4, abs=0.1)

    modos = [m for m in apresentacao.MODOS if m.id in ("so_carla", "completo")]
    pastas = {"completo": tmp_path / "modos" / "05_completo", "so_carla": tmp_path / "modos" / "01_so_carla"}
    (pastas["completo"] / "benchmark").mkdir(parents=True)
    tel.to_csv(pastas["completo"] / "benchmark" / "telemetria_sem_agentes.csv", index=False)
    (pastas["completo"] / "benchmark" / "resumo.json").write_text(json.dumps(
        {"sem_agentes": {"por_volta": {"tempo_s": 40.0, "combustivel_ml": 40.0, "eventos_total": 0}}}), encoding="utf-8")
    videos = {("meio", "so_carla"): str(tmp_path / "situacoes" / "meio" / "01_so_carla.mp4")}
    linhas = apresentacao.matriz(modos, [meio], L, pastas, lim, videos)
    so_carla = {l["situacao"]: l for l in linhas if l["modo"] == "so_carla"}
    assert so_carla["meio"]["tempo_s"] == pytest.approx(8.0, abs=0.1) and so_carla["volta"]["diesel_ml"] == 40.0
    assert all(l["tempo_s"] == "" for l in linhas if l["modo"] == "completo" and l["situacao"] == "meio")
    apresentacao.escrever_indice(tmp_path, linhas, [meio], modos, "teste", 3)
    texto = (tmp_path / "indice.md").read_text(encoding="utf-8")
    assert "## 1. meio" in texto and "[ver](situacoes/meio/01_so_carla.mp4)" in texto and "Só o CARLA" in texto


class _AtorFalso:
    def __init__(self, x, y):
        self.local = type("L", (), {"x": x, "y": y})()
        self.destruido, self.controle = False, None

    def get_location(self):
        return self.local

    def apply_control(self, controle):
        self.controle = controle

    def set_autopilot(self, *args):
        pass

    def destroy(self):
        self.destruido = True


def test_figurantes_disparam_uma_vez_por_volta_e_saem_depois_da_situacao():
    from types import SimpleNamespace

    from mover.autonomo.figurantes import Figurantes
    from mover.autonomo.rota import para_carla

    criados = []

    def spawn(bp, tf):
        criados.append((bp.id, _AtorFalso(tf.location.x, tf.location.y)))
        return criados[-1][1]

    bp = lambda nome: SimpleNamespace(id=nome, has_attribute=lambda _: False)  # noqa: E731
    mundo = SimpleNamespace(get_blueprint_library=lambda: SimpleNamespace(find=bp), try_spawn_actor=spawn)
    carla = SimpleNamespace(Location=lambda **k: SimpleNamespace(**k), Rotation=lambda **k: SimpleNamespace(**k),
                            Transform=lambda local, rot: SimpleNamespace(location=local, rotation=rot),
                            WalkerControl=SimpleNamespace, Vector3D=lambda *v: v)
    caminhos = []
    tm = SimpleNamespace(get_port=lambda: 8100, auto_lane_change=lambda *a: None, ignore_lights_percentage=lambda *a: None,
                         ignore_signs_percentage=lambda *a: None, set_desired_speed=lambda *a: None,
                         set_path=lambda ator, pontos, vazio: caminhos.append(pontos))
    rota = plano_estadio().rota
    L = rota.comprimento
    fig = Figurantes(carla, mundo, tm, rota, {"pedestres": [{"s_m": 100.0, "gatilho_m": 30.0}],
                                              "veiculos": [{"s_m": 150.0, "a_frente_m": 20.0, "percurso_m": 40.0}]},
                     lambda x, y: 0.0, lambda a, b: [a, b])
    assert fig.atualizar(60.0, 0.0, 0.0) is None and not criados
    assert fig.atualizar(71.0, 0.0, 0.0) is not None and [c[0] for c in criados] == ["walker.pedestrian.0001"]
    p = rota.ponto(100.0)
    xc, yc, _ = para_carla(p.x, p.y, p.rumo)
    pedestre = criados[0][1]
    assert math.hypot(pedestre.local.x - xc, pedestre.local.y - yc) == pytest.approx(6.0, abs=0.05)
    assert pedestre.controle.speed == 1.0
    fig.atualizar(80.0, 0.0, 0.0)
    assert len(criados) == 1  # uma vez por volta
    fig.atualizar(126.0, 0.0, 0.0)
    assert pedestre.destruido and not fig.ativos  # passou da faixa
    fig.atualizar(149.0, 0.0, 0.0)
    assert criados[-1][0] == "vehicle.tesla.model3" and caminhos == [[177.0, 249.0]]
    fig.atualizar(240.0, 0.0, 0.0)
    assert criados[-1][1].destruido
    fig.atualizar(L + 71.0, 0.0, 0.0)
    assert len(criados) == 3 and criados[-1][0] == "walker.pedestrian.0001"  # a volta seguinte
    fig.limpar()
    assert criados[-1][1].destruido and not fig.ativos


# ---------------------------------------------------------------------------------------------
# Governador e consumo
# ---------------------------------------------------------------------------------------------
def _volta(m: Montagem, modo: str, opcao: str | None = None):
    gov = m.governador(modo, 1)
    if opcao:
        interv = Intervencao.da_opcao(opcao, m.opcoes)
        for inst in m.plano.instancias(0.0, m.plano.comprimento + 1.0, 1):
            gov.definir(inst.chave, interv)
    return m.modelo().simular(gov, 0.0, 0.0, 0.0, 0.0, 200.0), gov


def test_piloto_sozinho_da_trancos_e_com_intervencao_suave_nao(m):
    bruto, gov_b = _volta(m, "bruto")
    suave, gov_s = _volta(m, "agentes", "suave")
    assert gov_b.terminou and gov_s.terminou and bruto.v[-1] < 0.05 and suave.v[-1] < 0.05
    assert eventos(bruto) >= 4 and eventos(suave) == 0
    assert bruto.a.min() < -3.5  # frenagem tardia perto das zonas
    # com a intervenção, a velocidade dentro das zonas não passa do limite (tolerância de 1 km/h)
    for z in m.plano.zonas:
        dentro = (suave.d >= z.s_ini) & (suave.d < z.s_fim)
        assert suave.v[dentro].max() * 3.6 <= z.v_max_kmh + 1.0, z.id
    # e para na portaria (2 s parado perto do ponto)
    parada = m.plano.paradas[0]
    perto = np.abs(suave.d - parada.s) < 8.0
    assert (suave.v[perto] < 0.05).sum() * 0.05 >= 1.9


def test_governador_nao_aceita_intervencao_depois_da_frenagem_bruta(m):
    gov = m.governador("agentes", 1)
    chave = "lombada_1@0"
    assert gov.pode_intervir(chave)
    modelo = m.modelo()
    modelo.simular(gov, 0.0, 0.0, 0.0, 0.0, 6.0)  # perto da lombada, sem decisão: o piloto freia em degrau
    assert chave in gov._disparadas and not gov.pode_intervir(chave)
    clone = gov.clonar()
    clone.definir("lombada_2@0", Intervencao.da_opcao("suave", m.opcoes))
    assert not gov.decidida("lombada_2@0")  # o clone não muda o original


def test_governador_nao_deixa_o_caminhao_rastejando_com_um_piloto_que_ignora_pedidos_pequenos(m):
    # o TM do CARLA quase não acelera um caminhão de 9 t pedindo +1 km/h: aqui, abaixo de 0,4 m/s de diferença, nada
    gov = m.governador("agentes", 1)
    gov.definir("partida@0", Intervencao.da_opcao("suave", m.opcoes))
    d, v, v_max, dt = 0.0, 0.0, 0.0, 0.05
    for _ in range(int(20.0 / dt)):
        v_des = gov.velocidade_desejada(d, v, 0.0, dt)
        v = v + 0.8 * dt if v_des - v > 0.4 else max(0.0, v - 0.2 * dt)
        d, v_max = d + v * dt, max(v_max, v)
    assert v_max > 2.0 and d > 20.0  # na trava antiga, o caminhão ficava abaixo de 0,5 m/s para sempre
    assert gov.cfg.soltar_freio_s == 1.5


def test_tm_do_carla_nunca_recebe_um_alvo_minusculo_quando_o_governador_quer_andar(m):
    # o PID do TM do CARLA, com erro normalizado pelo alvo, oscila com alvos de 1-3 km/h (corrida 20261011_121132)
    decisor = m.decisor(1, None)
    op = OpcoesConducao(voltas=1, usar_llm=False, assincrona=False, sem_espera=True, v_min_comando_mps=5.0 / 3.6)
    res = Conducao(m.plano, m.simulador_substituto(), m.governador("agentes", 1), decisor, op, m.caminhao, m.consumo).rodar()
    decisor.encerrar()
    assert res.completa
    andando = [l for l in res.linhas if l["modo_tick"] != "parado" and (l["v_alvo_kmh"] or 0.0) > 5.0
               and l["v_desejada_kmh"] > 0.05 * 3.6]  # pedir zero (frear) continua valendo
    assert andando and min(l["v_desejada_kmh"] for l in andando) >= 5.0 - 1e-9
    # sem degrau no pedido ao sair da arrancada (no CARLA, o degrau virava freio cheio na saída das lombadas)
    pedidos = [l["v_desejada_kmh"] for l in res.linhas if l["modo_tick"] != "parado"]
    assert max(a - b for a, b in zip(pedidos, pedidos[1:])) < 1.0
    assert m.v_min_comando("carla") == pytest.approx(5.0 / 3.6) and m.v_min_comando("substituto") == 0.0


def test_consumo_so_marcha_lenta_freando_e_metricas_contam_eventos():
    c, p = ParametrosCaminhao(), ParametrosConsumo()
    freando = taxa_combustivel_l_s(potencia_rodas_kw(np.array([8.0]), np.array([-3.0]), 0.0, c), p)[0]
    assert freando == pytest.approx(p.marcha_lenta_l_h / 3600.0)
    cruzeiro = taxa_combustivel_l_s(potencia_rodas_kw(np.array([8.0]), np.array([0.0]), 0.0, c), p)[0]
    assert cruzeiro > freando
    t = np.arange(0, 20, 0.05)
    a = np.where((t > 10) & (t < 11), -4.0, 0.0)  # degrau de frenagem: jerk alto
    v = np.maximum(0.0, 8.0 + np.cumsum(a) * 0.05)
    met = metricas(t, v, a, c, p, Limiares())
    assert met["eventos_criticos"] >= 1 and met["combustivel_l"] > 0 and met["custo_r"] == pytest.approx(met["combustivel_l"] * 6.0)


# ---------------------------------------------------------------------------------------------
# Decisor (sobre o plano que o caminhão monta a partir do estado atual da corrida)
# ---------------------------------------------------------------------------------------------
PARADO = {"t": 0.0, "d": 0.0, "v": 0.0, "a": 0.0}


def test_decisor_pela_regra_escolhe_opcao_sem_trancos_e_decide_a_chegada(m):
    gov = m.governador("agentes", 1)
    decisor = m.decisor(1, llm=None)  # sem o modelo de ML: só as opções fixas
    plano = m.planejador(1).planejar(gov, PARADO, 0, janela_inicial(0.0, 10.0))
    dec = decisor.decidir(plano, gov)
    assert dec is not None and dec.origem == "regra" and dec.brutas and dec.escolhida not in ("nenhuma", "ml")
    escolhida, bruta = dec.opcao_escolhida, dec.opcao_bruta
    assert escolhida.eventos == 0 < bruta.eventos and escolhida.combustivel_ml < bruta.combustivel_ml
    assert {i["chave"] for i in dec.instancias} >= {"partida@0", "lombada_1@0"}
    entrada = dec.para_entrada()
    assert entrada["tipo"] == "intervencao" and entrada["fatos"]["economia_prevista_ml"] > 0
    assert {"bruto", "escolhido"} <= set(entrada["curvas"]) and entrada["curvas"]["bruto"]["t"]
    assert entrada["fonte"] == "plano" and entrada["curvas"]["fonte_bruto"] == "plano"
    assert all(b["fonte"] == "plano" for b in entrada["brutas"]) and entrada["plano"]["d_ini"] == 0.0
    assert entrada["ml"]["propostas"] == {} and entrada["ml"]["modelo"] is None

    # perto do fim da volta, a chegada (o caminhão para ~1 m antes do ponto) também é decidida
    L = m.plano.comprimento
    gov_fim = m.governador("agentes", 1)
    for inst in m.plano.instancias(0.0, L - 30.0, 1):
        gov_fim._feitas.add(inst.chave)
    plano_fim = m.planejador(1).planejar(gov_fim, {"t": 50.0, "d": L - 60.0, "v": 8.0, "a": 0.0}, 5, (60.0, 70.0))
    dec_fim = decisor.decidir(plano_fim, gov_fim)
    assert dec_fim is not None and "chegada@1" in {i["chave"] for i in dec_fim.instancias}
    decisor.encerrar()


def _decisao_com_llm(m: Montagem, roteiro: list[Any], atraso_s: float = 0.0, prazo_s: float | None = None):
    cfg = dict(m.cfg_decisao)
    if prazo_s is not None:
        cfg["prazo_llm_s"] = prazo_s
    llm = LLMRoteiro(roteiro, atraso_s)
    decisor = Decisor(m.plano, m.modelo(), cfg, llm=llm, opcoes=m.opcoes, n_voltas=1)
    gov = m.governador("agentes", 1)
    dec = decisor.decidir(decisor.planejador().planejar(gov, PARADO, 0, janela_inicial(0.0, 10.0)), gov)
    decisor.encerrar()
    return dec, llm


def test_supervisor_llm_escolhe_dentro_das_opcoes_seguras_com_guarda_de_numeros(m):
    regra, _ = _decisao_com_llm(m, [])  # sem roteiro: só para saber a recomendação e os números
    segura = next(o for o in regra.opcoes if o.opcao != regra.recomendada and o.opcao != "nenhuma"
                  and o.eventos <= regra.opcao_recomendada.eventos and o.tempo_extra_pct <= 60)
    dec, llm = _decisao_com_llm(m, [{"opcao": segura.opcao, "texto": f"Intervenção {segura.opcao} com {round(segura.combustivel_ml)} mL."}])
    assert dec.origem == "llm" and dec.escolhida == segura.opcao and dec.recomendada == regra.recomendada
    assert "Rascunho (JSON):" in llm.prompts[0] and dec.latencia_llm_s > 0
    assert dec.curvas["escolhido"] == dec.trajetorias[segura.opcao].para_json()

    casos = [
        ({"opcao": "turbo", "texto": "x"}, "opção que não existe"),
        ({"opcao": "nenhuma", "texto": "Sem alteração."}, "evento(s) brusco(s)"),
        ({"opcao": regra.recomendada, "texto": "Economia de 987 mL."}, "texto do LLM recusado"),
        (RuntimeError("modelo fora do ar"), "erro do LLM"),
    ]
    for resposta, motivo in casos:
        dec, _ = _decisao_com_llm(m, [resposta])
        assert dec.origem == "regra" and dec.escolhida == regra.recomendada and motivo in dec.motivo, (resposta, dec.motivo)
    dec, _ = _decisao_com_llm(m, [{"opcao": regra.recomendada, "texto": "atrasado"}], atraso_s=0.5, prazo_s=0.1)
    assert dec.origem == "regra" and "não respondeu" in dec.motivo

    # resposta atrasada ainda ocupando o LLM: a decisão seguinte não entra na fila atrás dela (cascata no CARLA)
    cfg = dict(m.cfg_decisao, prazo_llm_s=0.1)
    llm = LLMRoteiro([{"opcao": regra.recomendada, "texto": "atrasado"}] * 3, atraso_s=1.5)
    decisor = Decisor(m.plano, m.modelo(), cfg, llm=llm, opcoes=m.opcoes, n_voltas=1)
    gov = m.governador("agentes", 1)
    plano = decisor.planejador().planejar(gov, PARADO, 0, janela_inicial(0.0, 10.0))
    motivos = [decisor.decidir(plano, gov).motivo]
    motivos.append(decisor.decidir(plano, gov).motivo)
    time.sleep(1.6)
    motivos.append(decisor.decidir(plano, gov).motivo)
    decisor.encerrar()
    assert "não respondeu" in motivos[0] and "ainda respondia" in motivos[1] and "não respondeu" in motivos[2]
    assert len(llm.prompts) == 2


# ---------------------------------------------------------------------------------------------
# Laço de condução (simulador substituto)
# ---------------------------------------------------------------------------------------------
def _corrida(m: Montagem, modo: str, llm=None, assincrona: bool = False, voltas: int = 1, fator_tempo: float = 40.0):
    lotes: list[list[dict[str, Any]]] = []
    decisoes: list[dict[str, Any]] = []
    decisor = m.decisor(voltas, llm) if modo == "agentes" else None
    op = OpcoesConducao(voltas=voltas, usar_llm=llm is not None, assincrona=assincrona, sem_espera=not assincrona,
                        fator_tempo=fator_tempo, duracao_bloco_s=10.0)
    res = Conducao(m.plano, m.simulador_substituto(), m.governador(modo, voltas), decisor, op, m.caminhao, m.consumo,
                   ao_lote=lotes.append, ao_decisao=decisoes.append).rodar()
    if decisor:
        decisor.encerrar()
    return res, lotes, decisoes


def test_corrida_com_agentes_gasta_menos_e_nao_tem_trancos(m):
    bruto, lotes_b, _ = _corrida(m, "bruto")
    agentes, lotes_a, decisoes = _corrida(m, "agentes", FalsoLLM())
    assert bruto.completa and agentes.completa
    assert sum(len(l) for l in lotes_a) == len(agentes.linhas) and all(len(l) <= 11 for l in lotes_a)
    met_b = metricas_da_corrida(bruto, m.plano.comprimento, m.caminhao, m.consumo, m.limiares)["total"]
    met_a = metricas_da_corrida(agentes, m.plano.comprimento, m.caminhao, m.consumo, m.limiares)["total"]
    assert met_a["combustivel_l"] < 0.9 * met_b["combustivel_l"]
    assert met_a["eventos_atencao"] + met_a["eventos_criticos"] == 0 < met_b["eventos_criticos"]
    assert decisoes == agentes.decisoes and len(decisoes) >= 2
    assert {d["texto_origem"] for d in decisoes} <= {"llm", "regra"} and agentes.aplicacoes["atrasadas"] == 0
    linha = agentes.linhas[-1]
    assert linha["combustivel_ml"] > 0 and linha["modo"] == "agentes" and linha["v_desejada_kmh"] == 0.0
    assert {"t", "speed_kmh", "v_limite_kmh", "custo_r", "co2_g", "bloco", "volta"} <= set(linha)
    # o diagnóstico por trecho soma o tempo da corrida e diz quem limitava em cada trecho
    from mover.autonomo.diagnostico import eventos_da_corrida, resumo_por_trecho

    tabela = resumo_por_trecho(pd.DataFrame(agentes.linhas))
    assert tabela["tempo_s"].sum() == pytest.approx(len(agentes.linhas) * 0.05)
    assert {"v_desejada_kmh", "v_alvo_kmh", "limitado_por", "modo_tick", "trocas_acel", "trocas_marcha"} <= set(tabela)
    eventos = eventos_da_corrida(pd.DataFrame(bruto.linhas), m.limiares, m.plano.comprimento)
    assert (eventos["nivel"] == "critico").sum() == met_b["eventos_criticos"]
    assert {"s_m", "jerk_mps3", "acelerador", "freio", "limitado_por"} <= set(eventos)


def test_corrida_ao_vivo_com_agentes_numa_thread_e_llm_lento(m):
    # 10x: cada 0,1 s de cálculo dos agentes vale 1 s da corrida, bem dentro do bloco de 10 s mesmo com a
    # máquina ocupada (a 1x, no CARLA, sobra ainda mais tempo)
    res, _, decisoes = _corrida(m, "agentes", FalsoLLM(atraso_s=0.05), assincrona=True, fator_tempo=10.0)
    assert res.completa and decisoes
    met = metricas_da_corrida(res, m.plano.comprimento, m.caminhao, m.consumo, m.limiares)["total"]
    assert met["eventos_criticos"] <= 1  # a regra é aplicada na hora; o LLM só confirma ou troca
    assert all(e["t_aplicada"] >= e["t_pedido"] for e in decisoes)


def test_relogio_segue_o_fator_de_tempo_e_nao_tenta_recuperar_atraso():
    relogio = Relogio(fator_tempo=20.0)
    relogio.ancorar(0.0)
    inicio = time.monotonic()
    for t in np.arange(0.0, 2.0, 0.05):  # 2 s de simulação a 20x = 0,1 s
        relogio.esperar_ate(t)
    assert 0.08 < time.monotonic() - inicio < 0.5
    relogio.ancorar(0.0)
    time.sleep(0.4)
    antes = time.monotonic()
    relogio.esperar_ate(0.05)  # atrasado: reancora em vez de correr
    assert relogio.atrasos == 1 and time.monotonic() - antes < 0.05


# ---------------------------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------------------------
def test_extrapolacao_usa_a_primeira_a_ultima_e_a_media_das_do_meio():
    voltas = [{"x": 10.0}, {"x": 4.0}, {"x": 6.0}, {"x": 12.0}]
    assert benchmark.extrapolar(voltas, "x", 100) == pytest.approx(10 + 12 + 5 * 98)
    assert benchmark.extrapolar(voltas[:2], "x", 100) == pytest.approx(7 * 100)


def test_benchmark_no_substituto_devolve_o_csv_e_o_resumo(tmp_path, monkeypatch):
    import mover.autonomo.benchmark as modulo

    monkeypatch.setattr(modulo, "montar", lambda cfg, missao=None: montar(cfg, plano=plano_estadio()))
    # um modelo de ML pequeno, treinado na pasta temporária (sem tocar em data/modelos/)
    cfg = carregar_yaml("config/autonomo.yaml")
    cfg["suavizador_ml"] = {**(cfg.get("suavizador_ml") or {}), "modelo": str(tmp_path / "suavizador.joblib"),
                            "situacoes": 40, "n_arvores": 10, "processos": 1}
    arquivo = tmp_path / "autonomo.yaml"
    arquivo.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    codigo = modulo.main(["--config", str(arquivo), "--voltas", "1", "--sem-carla", "--provedor", "falso", "--silencioso",
                          "--saida", str(tmp_path / "bench")])
    assert codigo == 0
    tabela = pd.read_csv(tmp_path / "bench" / "resultados.csv")
    assert list(tabela["tipo"]) == ["volta", "volta", "media_por_volta", "media_por_volta", "diferenca_por_volta",
                                    "diferenca_pct", "total_100_voltas", "total_100_voltas"]
    sem, com = tabela[tabela["tipo"] == "volta"].itertuples()
    assert com.combustivel_ml < sem.combustivel_ml and com.eventos_total < sem.eventos_total
    resumo = json.loads((tmp_path / "bench" / "resumo.json").read_text(encoding="utf-8"))
    e = resumo["extrapolacao"]
    assert e["voltas"] == 100 and e["economia_l"] > 0 and e["economia_r"] == pytest.approx(e["economia_l"] * 6.0, rel=0.02)
    assert resumo["simulador"] == "substituto" and resumo["decisoes"] >= 2
    if importlib.util.find_spec("sklearn") is not None:  # sem o scikit-learn, só as opções fixas
        assert resumo["suavizador_ml"].startswith("floresta aleatória") and (tmp_path / "suavizador.joblib").is_file()
    assert (tmp_path / "bench" / "telemetria_com_agentes.csv").exists()
