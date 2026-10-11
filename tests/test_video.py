"""Testes do vídeo (sem CARLA): HUD sobre um quadro e codificação H.264 com quadros sintéticos.

Rodar a partir da raiz do repositório:
    python -m pytest tests -q

A codificação só é testada se o imageio-ffmpeg estiver instalado (ele traz o ffmpeg).
"""

from __future__ import annotations

import json
import subprocess

import numpy as np
import pandas as pd
import pytest
from auxiliares import plano_estadio

from mover.video.gravar_video import CenaCarla, GravadorMP4, bgra_para_rgb, compor, resolver_corrida
from mover.video.hud import DadosCorrida, desenhar_hud


def corrida_sintetica(pasta, com_base: bool = True):
    t = np.arange(0.0, 20.0, 0.05)
    plano = plano_estadio()
    d = 5.0 * t
    pontos = [plano.rota.ponto(x) for x in d]
    tel = pd.DataFrame({"t": t, "sim_time": t, "d": d, "x": [p.x for p in pontos], "y": [p.y for p in pontos],
                        "rumo_graus": [np.degrees(p.rumo) for p in pontos], "speed_kmh": 18.0, "speed_mps": 5.0,
                        "acc_long": 0.0, "v_desejada_kmh": 18.0, "v_limite_kmh": 30.0, "volta": 0, "modo": "agentes",
                        "combustivel_ml": 2.0 * t, "custo_r": 0.012 * t})
    pasta.mkdir(parents=True, exist_ok=True)
    tel.to_csv(pasta / "telemetria.csv", index=False)
    decisao = {"t_pedido": 4.0, "t_aplicada": 4.2, "opcao": "economica", "texto_origem": "llm",
               "texto": "O piloto faria frenagem brusca para a lombada 1; intervenção econômica, 21 mL a menos."}
    (pasta / "decisoes.jsonl").write_text(json.dumps(decisao, ensure_ascii=False) + "\n", encoding="utf-8")
    (pasta / "resumo.json").write_text(json.dumps({"modo": "agentes", "voltas": 1, "simulador": "carla"}), encoding="utf-8")
    if com_base:
        (pasta / "linha_de_base.json").write_text(json.dumps({"d": [0.0, 100.0], "combustivel_ml": [0.0, 80.0]}), encoding="utf-8")
    return DadosCorrida.carregar(pasta, plano)


def test_dados_da_corrida_consultam_o_instante_a_decisao_e_a_linha_de_base(tmp_path):
    dados = corrida_sintetica(tmp_path / "c")
    assert dados.duracao_s == pytest.approx(19.95)
    assert dados.no_instante(10.0)["d"] == pytest.approx(50.0)
    assert dados.decisao_em(3.0) is None and dados.decisao_em(6.0)["opcao"] == "economica"
    assert dados.decisao_em(30.0) is None  # some da tela ~9 s depois de aplicada
    assert dados.base_ml(50.0) == pytest.approx(40.0) and len(dados.rastro_ate(1.0)) == 21


def test_hud_desenha_por_cima_do_quadro_sem_mudar_o_tamanho(tmp_path):
    from PIL import Image

    dados = corrida_sintetica(tmp_path / "c")
    fundo = Image.new("RGB", (1280, 720), (60, 80, 90))
    com_hud = desenhar_hud(fundo, dados, 6.0)
    assert com_hud.size == (1280, 720) and com_hud.mode == "RGB"
    antes, depois = np.asarray(fundo), np.asarray(com_hud)
    assert (antes != depois).any(axis=2).mean() > 0.05  # o HUD cobre uma parte do quadro
    assert (antes[360, 640] == depois[360, 640]).all()  # e deixa o centro livre para o caminhão


def test_bgra_vira_rgb_e_o_mp4_sai_em_h264(tmp_path):
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    bgra = np.zeros((36, 64, 4), np.uint8)
    bgra[..., 0] = 255  # azul em BGRA
    rgb = bgra_para_rgb(bgra.tobytes(), 64, 36)
    assert rgb.shape == (36, 64, 3) and (rgb[0, 0] == [0, 0, 255]).all()

    dados = corrida_sintetica(tmp_path / "c")
    destino = tmp_path / "video.mp4"
    gravador = GravadorMP4(destino, 320, 180, fps=30, crf=23, preset="ultrafast")
    for k in range(30):
        quadro = np.full((180, 320, 3), (k * 8) % 255, np.uint8)
        gravador.escrever(compor(quadro, dados, k / 30.0, hud=True))
    with pytest.raises(ValueError):
        gravador.escrever(np.zeros((10, 10, 3), np.uint8))
    gravador.fechar()
    assert destino.stat().st_size > 1000 and gravador.quadros == 30
    saida = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(destino)], capture_output=True,
                           text=True).stderr
    assert "h264" in saida and "yuv420p" in saida and "320x180" in saida and "30 fps" in saida


def test_sem_corrida_o_video_usa_a_ultima_gravada_ou_a_ultima(tmp_path):
    corridas = tmp_path / "corridas"
    for nome, gravada in (("20261010_090000_agentes", True), ("20261010_100000_bruto", False)):
        pasta = corridas / nome
        pasta.mkdir(parents=True)
        (pasta / "telemetria.csv").write_text("t\n0\n", encoding="utf-8")
        (pasta / "resumo.json").write_text(json.dumps({"gravacao_carla": "x.log" if gravada else None}), encoding="utf-8")
    (corridas / "20261010_110000_sem_telemetria").mkdir()
    # a mais recente sem gravação perde para a gravada; a pasta sem telemetria.csv é ignorada
    assert resolver_corrida(None, corridas).name == "20261010_090000_agentes"
    assert resolver_corrida("ultima", corridas).name == "20261010_090000_agentes"
    (corridas / "20261010_090000_agentes" / "resumo.json").write_text("{}", encoding="utf-8")
    assert resolver_corrida("última", corridas).name == "20261010_100000_bruto"  # nenhuma gravada: a mais recente
    explicita = corridas / "20261010_090000_agentes"
    assert resolver_corrida(str(explicita), corridas) == explicita
    with pytest.raises(SystemExit):
        resolver_corrida(None, tmp_path / "vazia")


def test_reproducao_seleciona_apenas_o_blueprint_configurado():
    blueprint = "vehicle.carlamotors.european_hgv"

    class Ator:
        def __init__(self, type_id, role_name):
            self.type_id, self.attributes, self.id = type_id, {"role_name": role_name}, 1

    class Atores(list):
        def filter(self, _):
            return self

    class Mundo:
        def __init__(self):
            self.ticks = 0

        def tick(self):
            self.ticks += 1

        def get_actors(self):
            return Atores([Ator("vehicle.carlamotors.firetruck", "mover_gemeo"), Ator(blueprint, "outro")])

    class Cliente:
        def set_replayer_ignore_spectator(self, _):
            pass

        def set_replayer_time_factor(self, _):
            pass

        def replay_file(self, *_):
            return ""

    cena = object.__new__(CenaCarla)
    cena.client, cena.world = Cliente(), Mundo()
    cena.cv, cena.reproduzindo = {"blueprint": blueprint, "role_name": "mover_gemeo"}, False

    cena.reproduzir("corrida.log")

    assert cena.ator.type_id == blueprint
    assert cena.world.ticks == 3
