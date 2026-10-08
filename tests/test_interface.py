"""Testes da interface (Etapa 4): página, câmera, cena 2D, previsão no WebSocket e chat.

Rodar a partir da raiz do repositório:
    python -m pytest tests -q

Sem navegador, sem CARLA e sem LLM de verdade: o chat é testado com o provedor falso (resposta
por busca no log) e com um "LLM de roteiro" que devolve respostas prontas, para exercitar a
guarda de números e de citações. O JS não roda aqui; um teste estático confere que os ids e os
nomes importados pelos módulos existem.
"""

from __future__ import annotations

import copy
import io
import re
import time
from pathlib import Path
from typing import Any, Callable

import pytest
from auxiliares import FRENAGEM, bgra_azul, pista_estadio_xodr, telemetria, volta_no_estadio
from fastapi.testclient import TestClient
from test_servidor import receber

from mover.agentes import eventos_sinteticos
from mover.agentes.llm import FalsoLLM
from mover.agentes.supervisor import Supervisor
from mover.agentes.textos import num
from mover.config import carregar_yaml
from mover.interface.cena import montar_cena
from mover.interface.chat import ContextoChat, resposta_deterministica
from mover.interface.rotas import PASTA_ESTATICA
from mover.servidor.app import criar_app
from mover.simulacao.alinhamento import alinhar
from mover.simulacao.camera_painel import codificar_jpeg
from mover.simulacao.opendrive import ler_xodr

FRENAGEM_15S = {**FRENAGEM, "tempo_s": 15.0}  # cai no bloco 1: vira problema da previsão


@pytest.fixture(scope="module")
def cfg() -> dict[str, Any]:
    return carregar_yaml("config/agentes.yaml")


def montar_app(cfg: dict[str, Any], pasta: Path, fabrica_llm_chat: Callable[[Any], Any] | None = None,
               cfg_simulacao: dict[str, Any] | None = None):
    """25 s de telemetria sintética (3 blocos) com frenagens bruscas aos 5 s e aos 15 s."""
    dados = eventos_sinteticos.injetar(telemetria(25.0), [FRENAGEM, FRENAGEM_15S], {})
    cfg_sim = {"servidor": {"pasta_sessoes": str(pasta)}, **(cfg_simulacao or {})}
    return criar_app(cfg, cfg_sim, telemetria=dados, fabrica_supervisor=lambda opcoes: Supervisor(FalsoLLM(), cfg),
                     sem_ml=True, fabrica_llm_chat=fabrica_llm_chat or (lambda opcoes: FalsoLLM()))


def receber_ate(ws, condicao: Callable[[dict[str, Any]], bool], prazo_s: float = 5.0) -> dict[str, Any]:
    """Descarta mensagens do WebSocket até a primeira que satisfaz a condição."""
    limite = time.monotonic() + prazo_s
    while True:
        mensagem = receber(ws, max(0.1, limite - time.monotonic()))
        if condicao(mensagem):
            return mensagem


def esperar_previsao(http, bloco: int, prazo_s: float = 5.0) -> None:
    """A análise do bloco seguinte roda numa thread: espera ela virar a previsão da sessão."""
    limite = time.monotonic() + prazo_s
    while http.get("/sessao").json()["previsao_bloco"] != bloco:
        assert time.monotonic() < limite, f"a previsão do bloco {bloco} não ficou pronta"
        time.sleep(0.02)


# ---------------------------------------------------------------------------------------------
# Página e arquivos estáticos
# ---------------------------------------------------------------------------------------------
def test_pagina_tem_os_4_paineis_e_os_estaticos_tem_o_tipo_certo(cfg, tmp_path):
    with TestClient(montar_app(cfg, tmp_path)) as http:
        pagina = http.get("/")
        assert pagina.status_code == 200 and pagina.headers["cache-control"] == "no-cache"
        for id_ in ("painel-simulacao", "painel-dashboard", "painel-log", "painel-chat", "canvas-perseguicao", "camera",
                    "canvas-mapa", "canvas-velocidade", "canvas-aceleracao", "lista-log", "ir-fim-log", "previsao",
                    "form-chat", "pergunta"):
            assert f'id="{id_}"' in pagina.text, id_
        for nome in ("painel.js", "desenho.js", "mapa.js", "graficos.js", "log.js", "chat.js"):
            r = http.get(f"/estatico/{nome}")
            assert r.status_code == 200 and r.headers["content-type"].startswith("text/javascript"), nome
            assert r.headers["cache-control"] == "no-cache"  # versão nova do JS sem limpar o cache
        assert http.get("/estatico/painel.css").headers["content-type"].startswith("text/css")


def _exportados(texto: str) -> set[str]:
    nomes = set(re.findall(r"export\s+(?:async\s+)?(?:function\*?|const|let|var|class)\s+([\w$]+)", texto))
    for lista in re.findall(r"export\s*\{([^}]*)\}", texto):
        nomes |= {parte.split(" as ")[-1].strip() for parte in lista.split(",") if parte.strip()}
    return nomes


def test_ids_e_nomes_importados_pelo_js_existem():
    """Sem rodar o JS: um id errado ou um import com nome errado deixaria a página em branco."""
    html = (PASTA_ESTATICA / "index.html").read_text(encoding="utf-8")
    assert 'href="/estatico/painel.css"' in html and 'src="/estatico/painel.js"' in html
    ids_html = set(re.findall(r'\bid="([\w-]+)"', html))
    usados = set(re.findall(r'\$\("([\w-]+)"\)', (PASTA_ESTATICA / "painel.js").read_text(encoding="utf-8")))
    assert usados and not usados - ids_html, f"ids usados pelo painel.js e ausentes do HTML: {usados - ids_html}"
    for js in sorted(PASTA_ESTATICA.glob("*.js")):
        texto = js.read_text(encoding="utf-8")
        for nomes, modulo in re.findall(r'import\s*\{([^}]*)\}\s*from\s*"\./([\w.-]+)"', texto, flags=re.S):
            alvo = PASTA_ESTATICA / modulo
            assert alvo.exists(), f"{js.name} importa {modulo}, que não existe"
            exportados = _exportados(alvo.read_text(encoding="utf-8"))
            for nome in (n.split(" as ")[0].strip() for n in nomes.split(",") if n.strip()):
                assert nome in exportados, f"{js.name} importa {nome} de {modulo}, que não o exporta"


# ---------------------------------------------------------------------------------------------
# Câmera do CARLA (POST /camera -> MJPEG)
# ---------------------------------------------------------------------------------------------
def test_codificar_jpeg_converte_bgra_em_rgb():
    from PIL import Image

    jpeg = codificar_jpeg(bgra_azul(32, 16), 32, 16, qualidade=90)
    imagem = Image.open(io.BytesIO(jpeg))
    assert jpeg.startswith(b"\xff\xd8") and imagem.size == (32, 16) and imagem.mode == "RGB"
    r, g, b = imagem.getpixel((16, 8))
    assert r < 30 and g < 30 and b > 220  # continua azul: os canais não foram trocados


def test_camera_recebe_quadros_e_transmite_mjpeg(cfg, tmp_path):
    jpeg = codificar_jpeg(bgra_azul(32, 16), 32, 16)
    tipo_jpeg = {"Content-Type": "image/jpeg"}
    with TestClient(montar_app(cfg, tmp_path)) as http:
        assert http.get("/camera/info").json() == {"quadros": 0, "idade_s": None, "ativa": False}
        assert http.get("/camera.jpg").status_code == 404
        assert http.post("/camera", content=jpeg, headers={"Content-Type": "image/png"}).status_code == 415
        assert http.post("/camera", content=b"isto nao e um jpeg", headers=tipo_jpeg).status_code == 400
        grande = b"\xff\xd8" + bytes(4 * 1024 * 1024)
        assert http.post("/camera", content=grande, headers=tipo_jpeg).status_code == 413

        assert http.post("/camera", content=jpeg, headers=tipo_jpeg).status_code == 204
        info = http.get("/camera/info").json()
        assert info["quadros"] == 1 and info["ativa"] and 0 <= info["idade_s"] < 2
        quadro = http.get("/camera.jpg")
        assert quadro.content == jpeg and quadro.headers["content-type"] == "image/jpeg"

        # o TestClient espera o corpo inteiro: max_quadros encerra a transmissão
        mjpeg = http.get("/camera.mjpg", params={"max_quadros": 1})
        assert mjpeg.headers["content-type"] == "multipart/x-mixed-replace; boundary=quadro"
        cabecalho = f"--quadro\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
        assert mjpeg.content == cabecalho + jpeg + b"\r\n"


# ---------------------------------------------------------------------------------------------
# Cena 2D (PUT/GET /cena)
# ---------------------------------------------------------------------------------------------
def test_cena_montada_pelo_replay_e_servida_a_pagina(cfg, tmp_path):
    mapa = ler_xodr(pista_estadio_xodr(), 0.5)
    alinhamento = alinhar(mapa, volta_no_estadio(12.0), {"icp": {"ativo": False}})
    cena = montar_cena(mapa, alinhamento)
    poses = alinhamento.poses
    assert cena["origem"] == "replay" and cena["vias"]
    assert all(len(v["pontos"]) >= 2 and v["largura_m"] == pytest.approx(3.2) for v in cena["vias"])
    tr = cena["trajeto"]
    assert len(tr["t"]) == len(tr["x"]) == len(tr["rumo_graus"]) == (len(poses) + 1) // 2  # um quadro a cada 2
    assert tr["rumo_graus"][3] == pytest.approx(-poses["yaw_carla"].iloc[6], abs=0.06)  # y do CARLA invertido
    assert tr["y"][3] == pytest.approx(poses["y_mapa"].iloc[6], abs=0.006)
    x0, y0, x1, y1 = cena["limites"]
    xs = [p[0] for v in cena["vias"] for p in v["pontos"]] + tr["x"]
    ys = [p[1] for v in cena["vias"] for p in v["pontos"]] + tr["y"]
    assert (x0, y0, x1, y1) == pytest.approx((min(xs), min(ys), max(xs), max(ys)), abs=0.006)

    # sem mapa no config, a página só tem cena depois que o replay a manda
    sem_mapa = {"mapa": {"arquivo": str(tmp_path / "nao_existe.xodr")}}
    with TestClient(montar_app(cfg, tmp_path, cfg_simulacao=sem_mapa)) as http:
        r = http.get("/cena")
        assert r.status_code == 404 and "cena indisponível" in r.json()["detail"]["mensagem"]
        assert http.put("/cena", json=cena).status_code == 204
        assert http.get("/cena").json() == cena
        ruim = copy.deepcopy(cena)
        ruim["trajeto"]["x"] = ruim["trajeto"]["x"][:-1]
        assert http.put("/cena", json=ruim).status_code == 422  # t, x, y e rumo com tamanhos diferentes


# ---------------------------------------------------------------------------------------------
# Previsão do próximo bloco no WebSocket
# ---------------------------------------------------------------------------------------------
def test_previsao_vai_pelo_websocket_e_entra_no_historico(cfg, tmp_path):
    with TestClient(montar_app(cfg, tmp_path)) as http:
        with http.websocket_connect("/ws") as ws:
            assert receber(ws)["tipo"] == "historico"
            info = http.post("/sessao", json={}).json()
            # a previsão do bloco 0 (analisado antes da partida) vai junto com o aviso "iniciada"
            iniciada = receber_ate(ws, lambda m: m["tipo"] == "sessao")
            assert iniciada["evento"] == "iniciada" and iniciada["sessao"]["id"] == info["id"]
            previsao0 = iniciada["previsao"]
            assert previsao0["bloco"] == 0 and previsao0["n_problemas"] == 1

            bloco0 = http.post("/sessao/blocos/0").json()
            # os ids são dados na análise: a entrada publicada é a mesma que estava na previsão
            assert [e["id"] for e in bloco0["entradas"]] == [e["id"] for e in previsao0["entradas"]]

            previsao1 = receber_ate(ws, lambda m: m["tipo"] == "previsao" and m["bloco"] == 1)
            assert previsao1["sessao"] == info["id"] and (previsao1["t_ini"], previsao1["t_fim"]) == (10.0, 20.0)
            assert min(e["id"] for e in previsao1["entradas"]) > max(e["id"] for e in bloco0["entradas"])
            problemas = [e for e in previsao1["entradas"] if e["tipo"] == "problema"]
            assert len(problemas) == 1 and problemas[0]["evento"] == "frenagem_brusca"
            assert 14.0 < problemas[0]["t_pico"] < 17.0
            assert len(previsao1["telemetria"]["t"]) == 50  # 10 s a 5 Hz, para o tracejado dos gráficos

        esperar_previsao(http, 1)
        with http.websocket_connect("/ws") as ws:  # página aberta no meio da volta
            historico = receber(ws)
            assert historico["previsao"]["bloco"] == 1
            assert [e["id"] for e in historico["entradas"]] == [e["id"] for e in bloco0["entradas"]]
            assert [b["bloco"] for b in historico["blocos"]] == [0]
            assert len(historico["telemetria"]["t"]) == 50 and max(historico["telemetria"]["t"]) < 10.0

        bloco1 = http.post("/sessao/blocos/1").json()
        assert [e["id"] for e in bloco1["entradas"]] == [e["id"] for e in previsao1["entradas"]]


# ---------------------------------------------------------------------------------------------
# Chat sobre o log
# ---------------------------------------------------------------------------------------------
def test_chat_por_busca_no_log_cita_o_problema_e_a_previsao(cfg, tmp_path):
    with TestClient(montar_app(cfg, tmp_path)) as http:
        sem_volta = http.post("/chat", json={"pergunta": "Teve algum problema?"}).json()
        assert sem_volta["citacoes"] == [] and sem_volta["observacao"] == "sem entradas no log ainda"
        assert http.post("/chat", json={"pergunta": ""}).status_code == 422

        http.post("/sessao", json={})
        bloco0 = http.post("/sessao/blocos/0").json()
        frenagem = next(e for e in bloco0["entradas"] if e["tipo"] == "problema")
        esperar_previsao(http, 1)

        r = http.post("/chat", json={"pergunta": "Teve algum problema?"}).json()
        assert r["origem"] == "modelo" and r["observacao"] == "provedor falso: resposta por busca no log"
        assert f"#{frenagem['id']}" in r["resposta"]
        citacao = next(c for c in r["citacoes"] if c["id"] == frenagem["id"])
        assert (citacao["tipo"], citacao["nivel"], citacao["previsao"]) == ("problema", "critico", False)
        assert r["contexto"]["previsao_bloco"] == 1 and r["contexto"]["entradas"] == len(bloco0["entradas"])

        r = http.post("/chat", json={"pergunta": "O que vem pela frente?",
                                     "historico": [{"papel": "operador", "texto": "Teve algum problema?"}]}).json()
        assert "bloco 01" in r["resposta"]
        publicadas = {e["id"] for e in bloco0["entradas"]}
        previstas = [c for c in r["citacoes"] if c["previsao"]]
        assert previstas and not {c["id"] for c in previstas} & publicadas
        assert any(c["tipo"] == "problema" and c["nivel"] == "critico" for c in previstas)  # a frenagem dos 15 s

        http.delete("/sessao")
        r = http.post("/chat", json={"pergunta": "O que vem pela frente?"}).json()
        assert r["resposta"].startswith("A volta terminou") and r["citacoes"] == []


def _problema(id_: int, t_pico: float, nivel: str, evento: str, causa: str, fonte: str = "real") -> dict[str, Any]:
    return {"id": id_, "tipo": "problema", "bloco": int(t_pico // 10), "t_ini": t_pico - 1.0, "t_fim": t_pico + 1.0,
            "t_pico": t_pico, "nivel": nivel, "evento": evento, "causa": causa, "fonte": fonte, "texto": "",
            "fatos": {"jerk_max_abs_mps3": 6.0 if nivel == "critico" else 3.0, "vel_kmh": 12.0}}


def test_resposta_por_busca_poe_os_criticos_primeiro_e_separa_lombada_de_frenagem(cfg):
    lombadas = [_problema(1, 10.0, "atencao", "soltura_freio", "irregularidade_via"),
                _problema(2, 20.0, "atencao", "soltura_freio", "irregularidade_via"),
                _problema(3, 30.0, "critico", "soltura_freio", "irregularidade_via")]
    frenagem = _problema(4, 45.0, "critico", "frenagem_brusca", "conducao_brusca", fonte="misto")
    prevista = _problema(5, 55.0, "critico", "arrancada_brusca", "conducao_brusca")
    publicadas = lombadas + [frenagem]
    ctx = ContextoChat(publicadas=publicadas, selecionadas=publicadas, previsao=[prevista],
                       bloco_previsao={"bloco": 5, "t_ini": 50.0, "t_fim": 60.0})
    responder = lambda pergunta: resposta_deterministica(pergunta, ctx, cfg["conhecimento"])  # noqa: E731

    # só cabem 3: os críticos vêm antes das lombadas de atenção, que são mais antigas
    texto, ids = responder("Quais problemas aconteceram?")
    assert texto.startswith("4 problemas no log (2 críticos, 2 de atenção), mais 1 na previsão do próximo bloco:")
    assert ids == [3, 4, 5] and texto.endswith("; e mais 2.")
    assert "frenagem brusca (crítico, sintético)" in texto and "(crítico, previsão)" in texto

    texto, ids = responder("Algum problema crítico?")
    assert texto.startswith("2 problemas críticos no log, mais 1 na previsão do próximo bloco:") and ids == [3, 4, 5]

    texto, ids = responder("Teve frenagem brusca?")  # a lombada tem forma de soltura de freio, mas a causa é a via
    assert texto.startswith("1 problema de frenagem no log:") and ids == [4]

    texto, ids = responder("Houve lombadas?")
    assert texto.startswith("3 problemas de lombada no log (1 crítico, 2 de atenção):") and ids == [3, 1, 2]

    texto, ids = responder("Algum problema entre 5 e 25 s?")
    assert texto.startswith(f"2 problemas no log entre {num(5.0)} e {num(25.0)} s, todos de atenção:") and ids == [1, 2]


class LLMRoteiro:
    """LLM de teste: devolve as respostas do roteiro, na ordem (ou levanta a exceção da vez)."""

    nome = "roteiro"

    def __init__(self) -> None:
        self.roteiro: list[dict[str, Any] | Exception] = []
        self.prompts: list[str] = []

    def gerar(self, sistema: str, usuario: str, esquema):
        self.prompts.append(usuario)
        resposta = self.roteiro.pop(0)
        if isinstance(resposta, Exception):
            raise resposta
        return esquema(**resposta)


def test_chat_com_llm_passa_pela_guarda_de_numeros_e_citacoes(cfg, tmp_path):
    llm = LLMRoteiro()
    with TestClient(montar_app(cfg, tmp_path, fabrica_llm_chat=lambda opcoes: llm)) as http:
        http.post("/sessao", json={})
        bloco0 = http.post("/sessao/blocos/0").json()
        esperar_previsao(http, 1)
        frenagem = next(e for e in bloco0["entradas"] if e["tipo"] == "problema")
        id_, pico = frenagem["id"], frenagem["t_pico"]
        pergunta = {"pergunta": "Teve algum problema?"}

        llm.roteiro = [
            {"resposta": f"Sim: frenagem brusca crítica aos {num(pico)} s (#{id_}).", "citacoes": [id_]},
            {"resposta": f"A velocidade chegou a 97 km/h (#{id_}).", "citacoes": [id_]},
            {"resposta": "Foi registrada uma frenagem brusca (#99).", "citacoes": [99]},
            RuntimeError("modelo fora do ar"),
        ]
        valida = http.post("/chat", json=pergunta).json()
        assert (valida["origem"], valida["provedor"], valida["observacao"]) == ("llm", "roteiro", None)
        assert valida["resposta"] == f"Sim: frenagem brusca crítica aos {num(pico)} s (#{id_})."
        assert [c["id"] for c in valida["citacoes"]] == [id_]
        assert "Pergunta do operador: Teve algum problema?" in llm.prompts[0]
        assert f"#{id_} [bloco 00, pico aos {num(pico)} s, problema crítico]" in llm.prompts[0]
        assert "PREVISÃO do bloco 01" in llm.prompts[0]

        # cada recusa cai na resposta montada direto do log, que continua citando o problema
        for motivo in ("número sem respaldo no log: 97", "cita entrada inexistente: #99", "erro do LLM: RuntimeError"):
            r = http.post("/chat", json=pergunta).json()
            assert r["origem"] == "modelo" and r["provedor"] is None
            assert motivo in r["observacao"], r["observacao"]
            assert f"#{id_}" in r["resposta"] and id_ in [c["id"] for c in r["citacoes"]]
        assert not llm.roteiro
