"""Testes dos arquivos da raiz e do scripts/rodar_gemeo.sh (sem CARLA e sem Ollama).

Rodar a partir da raiz do repositório:
    python -m pytest tests -q

O rodar_gemeo.sh é testado no modo seco (MOVER_SECO=1), que só mostra os comandos. No Windows os
testes que chamam o bash são pulados: o bash do PATH pode ser o do WSL, que não recebe as variáveis.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
SCRIPT = RAIZ / "scripts" / "rodar_gemeo.sh"
BASH = shutil.which("bash")
precisa_bash = pytest.mark.skipif(BASH is None or sys.platform == "win32", reason="sem bash (ou Windows)")


def rodar(*args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHON": sys.executable, "MOVER_SECO": "1"}
    return subprocess.run([BASH, str(SCRIPT), *args], cwd=RAIZ, env=env, capture_output=True, text=True,
                          encoding="utf-8", timeout=60)


def comandos_python(saida: str) -> list[str]:
    """O que vem depois de ' -m ' em cada comando que o modo seco mostraria a partir de src/."""
    return [linha.split(" -m ", 1)[1].strip() for linha in saida.splitlines() if linha.startswith("[MOVER] (cd src) ")]


def test_arquivos_da_raiz_tem_os_nomes_padrao():
    for nome in ("README.md", "requirements.txt", ".env.example", ".gitattributes"):
        assert (RAIZ / nome).is_file(), nome
    for antigo in ("README-mover.md", "requirements-mover.txt", ".env.mover.example"):
        assert not (RAIZ / antigo).exists(), antigo


def test_requirements_tem_o_cliente_do_carla_0_9_16_onde_o_pypi_tem_o_pacote():
    requirements = pytest.importorskip("packaging.requirements")
    linhas = [linha.split("#")[0].strip() for linha in (RAIZ / "requirements.txt").read_text(encoding="utf-8").splitlines()]
    carla = [requirements.Requirement(linha) for linha in linhas if linha.startswith("carla")]
    assert len(carla) == 1 and str(carla[0].specifier) == "==0.9.16"
    marcador = carla[0].marker
    assert marcador.evaluate({"python_version": "3.12", "sys_platform": "win32"})
    assert marcador.evaluate({"python_version": "3.10", "sys_platform": "linux"})
    assert not marcador.evaluate({"python_version": "3.13", "sys_platform": "win32"})  # sem pacote: pula
    assert not marcador.evaluate({"python_version": "3.12", "sys_platform": "darwin"})


def test_readme_tem_o_comando_de_cada_tarefa():
    texto = (RAIZ / "README.md").read_text(encoding="utf-8")
    for comando in ("python -m mover.autonomo.dirigir --sem-painel",            # coletar os dados de uma volta
                    "python -m mover.autonomo.dirigir",                         # visual
                    "python -m mover.autonomo.dirigir --gravar --sair-no-fim",  # gravar
                    "python -m mover.video.gravar_video",                       # vídeo
                    "python -m mover.autonomo.dirigir --sem-carla",             # painel sem CARLA
                    "python -m mover.autonomo.benchmark --voltas 3",
                    "python -m mover.agentes.rodar_agentes",                    # análise offline da corrida
                    "python -m mover.agentes.suavizador_ml",                    # treino do modelo de ML
                    "python -m mover.autonomo.gerar_missao",                    # plano pelo mapa
                    "./scripts/rodar_gemeo.sh visual", "./scripts/rodar_gemeo.sh coleta",
                    "./scripts/rodar_gemeo.sh gravar", "./scripts/rodar_gemeo.sh painel",
                    "pip install -r requirements.txt", "-quality-level=Epic", "http://127.0.0.1:8000/"):
        assert comando in texto, comando


def test_readme_poe_o_plano_da_corrida_atual_no_centro_e_o_sensor_logger_so_no_legado():
    texto = (RAIZ / "README.md").read_text(encoding="utf-8")
    legado = texto.find("## Legado")
    assert legado > 0, "o README precisa de uma seção Legado no fim"
    for termo in ("data/autonomo/corridas", "planejador.py", "suavizador_ml.py", "get_all_actions"):
        assert termo in texto[:legado], termo
    for antigo in ("coletar_base", "data/base", "--sem-base"):  # nada de voltas coletadas antes no laço
        assert antigo not in texto, antigo
    for termo in ("csv_maua", "Sensor Logger", "tratar_dados", "verificar_mapa"):
        antes = texto[:legado]
        assert termo not in antes or "legado" in antes[max(0, antes.find(termo) - 300):antes.find(termo) + 300].lower(), termo


def test_scripts_sh_comecam_pelo_bash_e_usam_lf():
    for script in (RAIZ / "scripts").glob("*.sh"):
        conteudo = script.read_bytes()
        assert conteudo.startswith(b"#!/usr/bin/env bash\n"), script.name
        assert b"\r\n" not in conteudo, script.name


@precisa_bash
def test_sintaxe_dos_scripts_sh():
    for script in (RAIZ / "scripts").glob("*.sh"):
        r = subprocess.run([BASH, "-n", str(script)], capture_output=True, text=True)
        assert r.returncode == 0, f"{script.name}: {r.stderr}"


@precisa_bash
@pytest.mark.parametrize("args, esperado", [
    ((), ["mover.autonomo.dirigir"]),                                   # sem modo = visual
    (("visual", "--voltas", "2"), ["mover.autonomo.dirigir --voltas 2"]),
    (("coleta",), ["mover.autonomo.dirigir --sem-painel"]),
    (("coleta", "--sem-carla", "--voltas", "2"), ["mover.autonomo.dirigir --sem-painel --sem-carla --voltas 2"]),
    (("coleta", "--sem-agentes"), ["mover.autonomo.dirigir --sem-painel --sem-agentes"]),
    (("gravar",), ["mover.autonomo.dirigir --gravar --sair-no-fim", "mover.video.gravar_video"]),
    (("video", "--camera", "frontal"), ["mover.video.gravar_video --camera frontal"]),
    (("painel",), ["mover.autonomo.dirigir --sem-carla"]),
    (("servidor",), ["mover.servidor.rodar_servidor"]),
    (("benchmark", "--voltas", "3"), ["mover.autonomo.benchmark --voltas 3"]),
    (("treinar", "--situacoes", "500"), ["mover.agentes.suavizador_ml --situacoes 500"]),
    (("analise", "--provedor", "falso"), ["mover.agentes.rodar_agentes --provedor falso"]),
])
def test_cada_modo_roda_o_comando_python_certo(args, esperado):
    r = rodar(*args)
    assert r.returncode == 0, r.stderr
    assert comandos_python(r.stdout) == esperado


@precisa_bash
def test_confere_o_carla_e_o_ollama_so_quando_precisa():
    visual = rodar("visual").stdout
    assert "confere o CARLA" in visual and "quality-level=Low" in visual and "confere o Ollama" in visual
    assert "quality-level=Epic" in rodar("gravar").stdout  # o vídeo precisa do CARLA em qualidade alta
    assert "confere o CARLA" not in rodar("coleta", "--sem-carla").stdout
    assert "confere o Ollama" in rodar("coleta").stdout  # a coleta é uma volta com os agentes
    assert "confere o Ollama" not in rodar("coleta", "--sem-agentes").stdout
    treinar = rodar("treinar").stdout  # o treino do ML só usa o modelo do caminhão
    assert "confere o CARLA" not in treinar and "confere o Ollama" not in treinar
    assert "confere o CARLA" not in rodar("video", "--teste-hud", "hud.png").stdout
    for args in (("visual", "--sem-llm"), ("visual", "--sem-agentes"), ("visual", "--provedor", "falso"),
                 ("visual", "--provedor=openai")):
        assert "confere o Ollama" not in rodar(*args).stdout, args
    assert "confere o Ollama" in rodar("visual", "--provedor", "ollama").stdout


@precisa_bash
def test_nenhum_modo_coleta_voltas_antes_da_corrida():
    """O laço usa só a corrida atual: nenhum modo coleta voltas antes nem lê dados de corridas anteriores."""
    texto = SCRIPT.read_text(encoding="utf-8")
    for antigo in ("coletar_base", "data/base", "--sem-base", "garantir_base"):
        assert antigo not in texto, antigo
    for args in (("visual",), ("gravar",), ("painel",), ("visual", "--sem-carla"), ("coleta",)):
        comandos = comandos_python(rodar(*args).stdout)
        assert comandos[0].startswith("mover.autonomo.dirigir"), args


@precisa_bash
def test_ajuda_lista_os_modos_e_modo_desconhecido_falha():
    ajuda = rodar("ajuda")
    assert ajuda.returncode == 0
    for modo in ("visual", "coleta", "gravar", "video", "painel", "servidor", "benchmark", "treinar", "analise", "ollama",
                 "carla", "testes"):
        assert f"rodar_gemeo.sh {modo}" in ajuda.stdout, modo
    errado = rodar("voar")
    assert errado.returncode != 0 and "Modo desconhecido" in errado.stderr
