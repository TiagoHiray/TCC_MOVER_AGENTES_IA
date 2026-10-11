"""Testes da interface: página, câmera, cena 2D (plano de missão) e chat sobre o log.

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
from typing import Any

import pytest
from auxiliares import bgra_azul, pista_estadio_xodr, plano_estadio
from fastapi.testclient import TestClient
from test_servidor import corrida, mandar_corrida, montar_app, opcoes_sessao  # noqa: F401 (corrida é fixture)

from mover.agentes.textos import num
from mover.config import carregar_yaml
from mover.interface.cena import montar_cena
from mover.interface.chat import ContextoChat, resposta_deterministica
from mover.interface.rotas import PASTA_ESTATICA
from mover.simulacao.camera_painel import codificar_jpeg
from mover.simulacao.opendrive import ler_xodr


@pytest.fixture(scope="module")
def cfg() -> dict[str, Any]:
    return carregar_yaml("config/agentes.yaml")


# ---------------------------------------------------------------------------------------------
# Página e arquivos estáticos
# ---------------------------------------------------------------------------------------------
def test_pagina_tem_os_4_paineis_e_os_estaticos_tem_o_tipo_certo(cfg, tmp_path):
    with TestClient(montar_app(cfg, tmp_path)) as http:
        pagina = http.get("/")
        assert pagina.status_code == 200 and pagina.headers["cache-control"] == "no-cache"
        for id_ in ("painel-simulacao", "painel-dashboard", "painel-log", "painel-chat", "canvas-perseguicao", "camera",
                    "canvas-mapa", "canvas-velocidade", "canvas-aceleracao", "lista-log", "ir-fim-log", "previsao",
                    "form-chat", "pergunta", "m-combustivel", "m-economia", "m-benchmark", "aviso-proximo", "progresso"):
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
        mjpeg = http.get("/camera.mjpg", params={"max_quadros": 1})
        assert mjpeg.headers["content-type"] == "multipart/x-mixed-replace; boundary=quadro"
        cabecalho = f"--quadro\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
        assert mjpeg.content == cabecalho + jpeg + b"\r\n"


# ---------------------------------------------------------------------------------------------
# Cena 2D (PUT/GET /cena)
# ---------------------------------------------------------------------------------------------
def test_cena_tem_rota_zonas_e_paradas_do_plano(cfg, tmp_path):
    plano = plano_estadio()
    mapa = ler_xodr(pista_estadio_xodr(), 0.5)
    cena = montar_cena(plano, mapa)
    assert cena["origem"] == "piloto" and cena["vias"] and cena["sentido"] == "anti-horário"
    rota = cena["rota"]
    assert len(rota["s"]) == len(rota["x"]) == len(rota["y"]) and rota["comprimento_m"] == pytest.approx(plano.comprimento, abs=0.01)
    lombada = next(z for z in cena["zonas"] if z["id"] == "lombada_1")
    assert lombada["v_max_kmh"] == 10.0 and len(lombada["pontos"]) >= 12  # um ponto por metro
    p = plano.rota.ponto(lombada["s_ini"])
    assert lombada["pontos"][0] == pytest.approx([p.x, p.y], abs=0.01)
    assert cena["paradas"][0]["id"] == "portao_1" and cena["paradas"][0]["duracao_s"] == 2.0
    x0, y0, x1, y1 = cena["limites"]
    assert x0 <= min(rota["x"]) and x1 >= max(rota["x"]) and y0 <= min(rota["y"]) and y1 >= max(rota["y"])

    with TestClient(montar_app(cfg, tmp_path)) as http:
        r = http.get("/cena")  # sem plano no config: só a cena que o piloto mandar
        assert r.status_code == 404 and "cena indisponível" in r.json()["detail"]["mensagem"]
        assert http.put("/cena", json=cena).status_code == 204
        assert http.get("/cena").json()["zonas"] == cena["zonas"]
        ruim = copy.deepcopy(cena)
        ruim["rota"]["x"] = ruim["rota"]["x"][:-1]
        assert http.put("/cena", json=ruim).status_code == 422  # s, x e y com tamanhos diferentes


def test_cena_do_config_usa_o_plano_e_o_mapa(cfg, tmp_path):
    plano = plano_estadio()
    plano.mapa = str(tmp_path / "estadio.xodr")
    (tmp_path / "estadio.xodr").write_text(pista_estadio_xodr(), encoding="utf-8")
    plano.salvar(tmp_path / "missao.yaml")
    app = montar_app(cfg, tmp_path, cfg_autonomo={"missao": {"arquivo": str(tmp_path / "missao.yaml")}})
    with TestClient(app) as http:
        cena = http.get("/cena").json()
        assert cena["origem"] == "config" and cena["vias"] and len(cena["zonas"]) == 3


# ---------------------------------------------------------------------------------------------
# Chat sobre o log
# ---------------------------------------------------------------------------------------------
def test_chat_por_busca_responde_economia_intervencoes_e_proximo_trecho(cfg, tmp_path, corrida):  # noqa: F811
    with TestClient(montar_app(cfg, tmp_path)) as http:
        sem_corrida = http.post("/chat", json={"pergunta": "Teve algum problema?"}).json()
        assert sem_corrida["citacoes"] == [] and sem_corrida["observacao"] == "sem entradas no log ainda"
        assert http.post("/chat", json={"pergunta": ""}).status_code == 422

        http.post("/sessao", json=opcoes_sessao(corrida))
        mandar_corrida(http, corrida)
        sessao = http.app.state.estado["sessao"]
        decisoes = [e for e in sessao.entradas if e["tipo"] == "intervencao"]

        r = http.post("/chat", json={"pergunta": "Quanto diesel os agentes economizaram?"}).json()
        assert r["origem"] == "modelo" and r["observacao"] == "provedor falso: resposta por busca no log"
        m = sessao.metricas.para_json()
        assert f"{num(m['combustivel_ml'], 0)} mL de diesel" in r["resposta"] and "Sem os agentes" in r["resposta"]
        assert r["citacoes"] and r["citacoes"][0]["tipo"] == "intervencao"

        r = http.post("/chat", json={"pergunta": "Quais intervenções foram feitas?"}).json()
        assert r["resposta"].startswith(f"{len(decisoes)} decisão(ões)") and all(c["tipo"] == "intervencao" for c in r["citacoes"])

        r = http.post("/chat", json={"pergunta": "O que vem pela frente?"}).json()
        assert r["resposta"].startswith("Próximo trecho") and r["citacoes"][0]["id"] == decisoes[-1]["id"]
        assert r["citacoes"][0]["previsao"] is True

        r = http.post("/chat", json={"pergunta": "Teve frenagem brusca?"}).json()
        assert r["resposta"].startswith("Nenhum problema de frenagem medido")

        http.delete("/sessao")
        r = http.post("/chat", json={"pergunta": "O que vem pela frente?"}).json()
        assert r["resposta"].startswith("A corrida terminou") and r["citacoes"] == []


def _problema(id_: int, t_pico: float, nivel: str, evento: str, causa: str) -> dict[str, Any]:
    return {"id": id_, "tipo": "problema", "bloco": int(t_pico // 10), "t_ini": t_pico - 1.0, "t_fim": t_pico + 1.0,
            "t_pico": t_pico, "nivel": nivel, "evento": evento, "causa": causa, "fonte": "carla", "texto": "",
            "fatos": {"jerk_max_abs_mps3": 6.0 if nivel == "critico" else 3.0, "vel_kmh": 12.0}}


def test_resposta_por_busca_poe_os_criticos_primeiro_e_separa_lombada_de_frenagem(cfg):
    publicadas = [_problema(1, 10.0, "atencao", "soltura_freio", "irregularidade_via"),
                  _problema(2, 20.0, "atencao", "soltura_freio", "irregularidade_via"),
                  _problema(3, 30.0, "critico", "soltura_freio", "irregularidade_via"),
                  _problema(4, 45.0, "critico", "frenagem_brusca", "conducao_brusca")]
    ctx = ContextoChat(publicadas=publicadas, selecionadas=publicadas)
    responder = lambda pergunta: resposta_deterministica(pergunta, ctx, cfg["conhecimento"])  # noqa: E731
    texto, ids = responder("Quais problemas aconteceram?")
    assert texto.startswith("4 problemas medido(s) (2 críticos, 2 de atenção):") and ids == [3, 4, 1]
    assert texto.endswith("; e mais 1.")
    texto, ids = responder("Teve frenagem brusca?")  # a lombada tem forma de soltura de freio, mas a causa é a via
    assert texto.startswith("1 problema de frenagem medido(s):") and ids == [4]
    texto, ids = responder("Houve lombadas?")
    assert texto.startswith("3 problemas de lombada medido(s) (1 crítico, 2 de atenção):") and ids == [3, 1, 2]
    texto, ids = responder("Algum problema entre 5 e 25 s?")
    assert texto.startswith(f"2 problemas medido(s) entre {num(5.0)} e {num(25.0)} s, todos de atenção:") and ids == [1, 2]


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


def test_chat_com_llm_passa_pela_guarda_de_numeros_e_citacoes(cfg, tmp_path, corrida):  # noqa: F811
    llm = LLMRoteiro()
    with TestClient(montar_app(cfg, tmp_path, fabrica_llm_chat=lambda opcoes: llm)) as http:
        http.post("/sessao", json=opcoes_sessao(corrida))
        mandar_corrida(http, corrida)
        sessao = http.app.state.estado["sessao"]
        decisao = next(e for e in sessao.entradas if e["tipo"] == "intervencao" and e["opcao"] != "nenhuma")
        id_, economia = decisao["id"], decisao["fatos"]["economia_prevista_ml"]
        pergunta = {"pergunta": "Os agentes economizaram diesel?"}
        llm.roteiro = [
            {"resposta": f"Sim: a intervenção previu {num(economia, 0)} mL a menos (#{id_}).", "citacoes": [id_]},
            {"resposta": f"A economia foi de 987 mL (#{id_}).", "citacoes": [id_]},
            {"resposta": "Houve uma intervenção (#999).", "citacoes": [999]},
            RuntimeError("modelo fora do ar"),
        ]
        valida = http.post("/chat", json=pergunta).json()
        assert (valida["origem"], valida["provedor"], valida["observacao"]) == ("llm", "roteiro", None)
        assert [c["id"] for c in valida["citacoes"]] == [id_] and valida["citacoes"][0]["previsao"] is True
        assert "Pergunta do operador: Os agentes economizaram diesel?" in llm.prompts[0]
        assert f"#{id_} [decidida aos" in llm.prompts[0] and "Métricas até agora:" in llm.prompts[0]
        for motivo in ("número sem respaldo no log: 987", "cita entrada inexistente: #999", "erro do LLM: RuntimeError"):
            r = http.post("/chat", json=pergunta).json()
            assert r["origem"] == "modelo" and r["provedor"] is None
            assert motivo in r["observacao"], r["observacao"]
        assert not llm.roteiro


def test_espera_curta():
    """Os testes da interface não dependem de tempo; este só garante que o relógio anda."""
    inicio = time.monotonic()
    time.sleep(0.01)
    assert time.monotonic() > inicio and Path(PASTA_ESTATICA).is_dir()
