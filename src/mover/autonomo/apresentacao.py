"""Vídeos e benchmarks da apresentação: cada situação (curva, pedestre, veículo, lombada, parada) em cada modo.

As situações saem do plano de missão (mover.autonomo.situacoes); o pedestre atravessando e o veículo lento
à frente são figurantes do CARLA disparados pela posição do caminhão. Os modos são os benchmarks: só o
CARLA (Traffic Manager), só a regra, só o ML, só o LLM, completo e, como controle, sem agentes a 20 km/h.

O script roda, em sequência:
1. um benchmark por modo (N voltas sem agentes x N com, o CARLA sem desenhar a cena, figurantes inclusos);
2. uma volta ao vivo por modo com o painel aberto, gravando a tela (câmera do CARLA embutida, gráficos, log
   das decisões e chat), e um recorte do vídeo para cada situação;
3. indice.md e indice.csv: para cada situação, a tabela dos modos (tempo, diesel, eventos bruscos, velocidade
   mínima, pior desaceleração, menor distância ao figurante) e os recortes; e a volta inteira.

Uso, no Windows da VM, com o CARLA aberto COM JANELA, o Ollama no ar e a área de trabalho visível (a
gravação é da tela: com a conexão remota minimizada, o vídeo sai preto), a partir de src/:
    python -m mover.autonomo.apresentacao                       # Town10HD_Opt, todos os modos
    python -m mover.autonomo.apresentacao --mapa campus         # no campus_maua.xodr
    python -m mover.autonomo.apresentacao --seco                # só mostra o que vai rodar
    python -m mover.autonomo.apresentacao --modos completo,so_carla --sem-benchmark
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # execução direta
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import yaml

from mover.autonomo.consumo import Limiares
from mover.autonomo.diagnostico import eventos_da_corrida
from mover.autonomo.missao import PlanoMissao
from mover.autonomo.situacoes import Situacao, escolher_situacoes, figurantes_do_cenario, metricas_situacao
from mover.config import RAIZ, caminho, carregar_yaml

log = logging.getLogger("mover.apresentacao")
SRC = RAIZ / "src"
TOWN10 = "Town10HD_Opt"
NAVEGADORES = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
)


@dataclass(frozen=True)
class Modo:
    id: str
    nome: str
    explica: str
    corrida: tuple[str, ...]              # opções do dirigir
    benchmark: tuple[str, ...] | None     # opções do benchmark (None: usa o de outro modo)
    bateria: str = "com"                  # com | sem: a bateria do benchmark que mede este modo
    missao: str = "padrao"                # padrao | cruzeiro20
    benchmark_de: str | None = None


MODOS: tuple[Modo, ...] = (
    Modo("so_carla", "Só o CARLA (Traffic Manager)",
         "Linha de base: o piloto automático do CARLA segue o limite de cada zona em degrau e freia em cima.",
         ("--sem-agentes",), None, bateria="sem", benchmark_de="completo"),
    Modo("so_regra", "Só a regra (opções fixas)",
         "A regra escolhe entre as 4 opções fixas, conferidas pelo modelo do caminhão. Sem ML e sem LLM.",
         ("--sem-ml", "--sem-llm"), ("--sem-ml", "--sem-llm")),
    Modo("so_ml", "Só o ML (floresta aleatória)",
         "O ML propõe a suavização zona a zona e a regra só confere. Sem LLM (também é o caso de o LLM cair).",
         ("--sem-llm",), ("--sem-llm",)),
    Modo("so_llm", "Só o LLM (sobre as opções fixas)",
         "O LLM escolhe entre as opções fixas já simuladas; a guarda recusa opção com mais eventos bruscos.",
         ("--sem-ml",), ("--sem-ml",)),
    Modo("completo", "Completo: ML + regra + LLM",
         "O ML propõe, o modelo do caminhão confere, a regra escolhe e o LLM supervisiona e explica no log.", (), ()),
    Modo("devagar_20", "Só o CARLA a 20 km/h (controle)",
         "Responde 'não bastava andar mais devagar?': a mesma volta sem agentes, com cruzeiro de 20 km/h.",
         ("--sem-agentes",), (), bateria="sem", missao="cruzeiro20"),
)


# ---------------------------------------------------------------------------------------------
# Mapa e plano de missão
# ---------------------------------------------------------------------------------------------
def _salvar_yaml(dados: dict[str, Any], destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(yaml.safe_dump(dados, allow_unicode=True, sort_keys=False), encoding="utf-8")


def preparar_configs(mapa: str, saida: Path, alvo_m: float) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    """Caminhos e conteúdo das configs da apresentação (escritas depois, em saida/config)."""
    pasta = saida / "config"
    caminhos = {"config": str(pasta / "autonomo.yaml"), "config_simulacao": str(pasta / "simulacao.yaml"),
                "padrao": str(pasta / "missao.yaml"), "cruzeiro20": str(pasta / "missao_cruzeiro20.yaml")}
    cfg_sim = carregar_yaml("config/simulacao.yaml")
    cfg_auto = carregar_yaml("config/autonomo.yaml")
    if mapa == "town10":
        cfg_sim["mapa"] = {**cfg_sim.get("mapa", {}), "arquivo": f"data/mapas/{TOWN10}_carla.xodr", "tipos_faixa": ["driving"]}
        cfg_sim["carla"] = {**cfg_sim.get("carla", {}), "mapa_carla": TOWN10}
        gm = dict(cfg_auto.get("gerar_missao") or {})
        gm["rota"] = {"vias": "auto", "alvo_m": alvo_m, "min_m": 0.6 * alvo_m, "max_m": 1.6 * alvo_m, "largada": None}
        gm["lombadas"], gm["portoes"] = [], []  # o Town10 não tem lombada nem portão; faixas e curvas vêm do mapa
        cfg_auto["gerar_missao"] = gm
    cfg_auto["missao"] = {**cfg_auto.get("missao", {}), "arquivo": caminhos["padrao"]}
    return caminhos, cfg_sim, cfg_auto


def exportar_town10(cfg_sim: dict[str, Any]) -> None:
    """Abre o Town10HD_Opt no CARLA e salva o OpenDRIVE dele em data/mapas (a rota sai desse arquivo)."""
    import carla

    from mover.autonomo.carla_tm import carregar_mapa_instalado

    cc = cfg_sim.get("carla", {})
    cliente = carla.Client(cc.get("host", "localhost"), int(cc.get("porta", 2000)))
    cliente.set_timeout(120.0)
    carregar_mapa_instalado(cliente, TOWN10, caminho("data/mapas"))


def preparar_missoes(mapa: str, cfg_sim: dict[str, Any], cfg_auto: dict[str, Any]) -> tuple[Any, Any, list[Situacao]]:
    """As duas missões (30 e 20 km/h) e as situações; a faixa de pedestre criada para a situação vai nas duas."""
    if mapa == "town10":
        from mover.autonomo.gerar_missao import gerar

        plano = gerar(cfg_sim, cfg_auto["gerar_missao"])
        plano20 = gerar(cfg_sim, {**cfg_auto["gerar_missao"], "cruzeiro_kmh": 20.0})
    else:
        plano = PlanoMissao.carregar(caminho("config/missao.yaml"))
        plano20 = PlanoMissao.carregar(caminho("config/missao_cruzeiro20.yaml"))
    situacoes = escolher_situacoes(plano)
    for z in plano.zonas:
        if z.id == "faixa_cenario" and z.id not in {x.id for x in plano20.zonas}:
            plano20.zonas = sorted([*plano20.zonas, z], key=lambda x: x.s_ini)
    cfg_sim["cenario"] = figurantes_do_cenario(situacoes)
    return plano, plano20, situacoes


def escrever_configs(caminhos: dict[str, str], cfg_sim: dict[str, Any], cfg_auto: dict[str, Any], plano: Any,
                     plano20: Any, situacoes: list[Situacao]) -> None:
    from mover.autonomo.gerar_missao import salvar_figura
    from mover.simulacao.opendrive import ler_xodr

    _salvar_yaml(cfg_sim, Path(caminhos["config_simulacao"]))
    _salvar_yaml(cfg_auto, Path(caminhos["config"]))
    cabecalho = "# Plano de missão da apresentação (gerado por mover.autonomo.apresentacao)."
    plano.salvar(Path(caminhos["padrao"]), cabecalho)
    plano20.salvar(Path(caminhos["cruzeiro20"]), cabecalho)
    (Path(caminhos["padrao"]).parent / "situacoes.json").write_text(
        json.dumps([s.para_dict() for s in situacoes], ensure_ascii=False, indent=2), encoding="utf-8")
    salvar_figura(plano, Path(caminhos["padrao"]).with_suffix(".png"),
                  ler_xodr(caminho(cfg_sim["mapa"]["arquivo"]), 1.0, ("driving",)).pontos)
    log.info("Missão: %.0f m (%s), %d zonas, %d parada(s). Situações: %s.", plano.comprimento, plano.sentido,
             len(plano.zonas), len(plano.paradas), ", ".join(f"{s.id} ({s.s_ini:.0f}-{s.s_fim:.0f} m)" for s in situacoes))


# ---------------------------------------------------------------------------------------------
# Comandos
# ---------------------------------------------------------------------------------------------
def comando_benchmark(modo: Modo, cfgs: dict[str, str], pasta: Path, args: argparse.Namespace) -> list[str]:
    return [sys.executable, "-m", "mover.autonomo.benchmark", "--voltas", str(args.voltas),
            "--config", cfgs["config"], "--config-simulacao", cfgs["config_simulacao"], "--missao", cfgs[modo.missao],
            "--provedor", args.provedor, "--saida", str(pasta / "benchmark"), "--silencioso", *modo.benchmark]


def comando_corrida(modo: Modo, cfgs: dict[str, str], args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, "-m", "mover.autonomo.dirigir", "--voltas", str(args.voltas_video),
           "--config", cfgs["config"], "--config-simulacao", cfgs["config_simulacao"], "--missao", cfgs[modo.missao],
           "--provedor", args.provedor, "--camera", "perseguicao", "--sem-navegador", *modo.corrida]
    return cmd + (["--gravar"] if args.video_hud else [])


# ---------------------------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------------------------
def _ambiente() -> dict[str, str]:
    return {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}


def _processo(cmd: list[str]) -> subprocess.Popen:
    return subprocess.Popen(cmd, cwd=SRC, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", env=_ambiente())


def rodar_e_registrar(cmd: list[str], arquivo_log: Path) -> int:
    arquivo_log.parent.mkdir(parents=True, exist_ok=True)
    proc = _processo(cmd)
    with arquivo_log.open("w", encoding="utf-8") as f:
        for linha in proc.stdout:
            print(linha, end="", flush=True)
            f.write(linha)
    return proc.wait()


def _get_json(url: str, timeout_s: float = 1.0) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resposta:
            return json.loads(resposta.read().decode("utf-8"))
    except (OSError, ValueError):
        return None


def abrir_navegador(url: str, perfil: Path) -> subprocess.Popen | None:
    """Janela de app do Edge/Chrome, maximizada e com perfil próprio (para fechar depois sem afetar outras)."""
    for exe in NAVEGADORES:
        if Path(exe).exists():
            return subprocess.Popen([exe, f"--user-data-dir={perfil}", f"--app={url}", "--start-maximized",
                                     "--no-first-run", "--no-default-browser-check", "--disable-session-crashed-bubble"])
    webbrowser.open(url)
    return None


def encerrar(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()


class GravadorTela:
    """Grava a área de trabalho com o ffmpeg do imageio-ffmpeg (gdigrab, só no Windows)."""

    def __init__(self, destino: Path, fps: int = 30):
        self.destino, self.fps, self.proc = destino, fps, None

    @staticmethod
    def disponivel() -> str | None:
        if os.name != "nt":
            return None
        try:
            import imageio_ffmpeg

            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError):
            return None
        saida = subprocess.run([exe, "-hide_banner", "-devices"], capture_output=True, text=True, errors="replace")
        return exe if "gdigrab" in (saida.stdout + saida.stderr) else None

    def iniciar(self, exe: str) -> None:
        self.destino.parent.mkdir(parents=True, exist_ok=True)
        self._log = (self.destino.parent / "ffmpeg.log").open("w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [exe, "-y", "-f", "gdigrab", "-framerate", str(self.fps), "-draw_mouse", "0", "-i", "desktop",
             "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
             "-pix_fmt", "yuv420p", str(self.destino)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._log)
        log.info("Gravando a tela em %s", self.destino)

    def parar(self) -> None:
        if self.proc is None:
            return
        try:
            self.proc.stdin.write(b"q")  # o ffmpeg fecha o MP4 direito
            self.proc.stdin.flush()
            self.proc.wait(timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            self.proc.kill()
        self._log.close()
        self.proc = None


def rodar_corrida_gravando(cmd: list[str], pasta: Path, url: str, ffmpeg: str | None, prazo_s: float) -> dict[str, Any]:
    """Roda o dirigir; quando a sessão abre, mostra o painel e grava a tela até a volta acabar."""
    pasta.mkdir(parents=True, exist_ok=True)
    proc = _processo(cmd)
    estado: dict[str, Any] = {"fim": False, "corrida": None}

    def ler() -> None:
        with (pasta / "corrida.log").open("w", encoding="utf-8") as f:
            for linha in proc.stdout:
                print(linha, end="", flush=True)
                f.write(linha)
                if "Corrida concluída" in linha:
                    estado["fim"] = True
                achado = re.search(r"Corrida gravada em (.+)$", linha.strip())
                if achado:
                    estado["corrida"] = achado.group(1).strip()

    leitor = threading.Thread(target=ler, daemon=True)
    leitor.start()
    navegador, gravador, aberto, inicio = None, None, False, time.monotonic()
    try:
        while proc.poll() is None:
            if not aberto:
                saude = _get_json(f"{url}/saude")
                if saude and saude.get("sessao"):
                    aberto = True
                    navegador = abrir_navegador(url + "/", pasta.parent / "_navegador")
                    time.sleep(2.5)  # a página carrega e recebe o histórico da sessão
                    if ffmpeg:
                        gravador = GravadorTela(pasta / "video_painel.mp4")
                        estado["t_gravacao"] = time.time()
                        gravador.iniciar(ffmpeg)
            if estado["fim"]:
                time.sleep(6.0)  # o painel com a volta completa e o resumo
                break
            if time.monotonic() - inicio > prazo_s:
                log.warning("A volta passou de %.0f s: encerrando.", prazo_s)
                break
            time.sleep(0.5)
    finally:
        if gravador is not None:
            gravador.parar()
        encerrar(navegador)
        encerrar(proc)
        leitor.join(timeout=5)
    estado["codigo"] = proc.returncode
    video = pasta / "video_painel.mp4"
    estado["video"] = str(video) if video.exists() and video.stat().st_size > 0 else None
    return estado


def gravar_video_hud(corrida: str, cfgs: dict[str, str], missao: str, destino: Path) -> None:
    cmd = [sys.executable, "-m", "mover.video.gravar_video", "--corrida", corrida, "--config", cfgs["config"],
           "--config-simulacao", cfgs["config_simulacao"], "--missao", missao, "--saida", str(destino)]
    rodar_e_registrar(cmd, destino.parent / "video_hud.log")


def trechos_do_video(tel: pd.DataFrame, situacoes: list[Situacao], L: float, t_gravacao: float,
                     margem_s: float = 4.0) -> dict[str, tuple[float, float]]:
    """Início e duração (s do vídeo da tela) de cada situação: a primeira volta em que a janela aparece
    inteira (ou a primeira, até o fim da corrida). Usa o relógio de parede da telemetria."""
    if tel.empty or "t_parede" not in tel:
        return {}
    d, tp = tel["d"].to_numpy(float), tel["t_parede"].to_numpy(float)
    trechos = {}
    for s in situacoes:
        largura = (s.s_fim - s.s_ini) % L
        inis = [k * L + s.s_ini for k in range(int(d.max() // L) + 1) if k * L + s.s_ini <= d.max()]
        if not inis:
            continue
        ini = next((x for x in inis if x + largura <= d.max()), inis[0])
        i0 = int(np.argmax(d >= ini))
        i1 = int(np.argmax(d >= ini + largura)) if d.max() >= ini + largura else len(d) - 1
        a = max(0.0, tp[i0] - t_gravacao - margem_s)
        trechos[s.id] = (round(a, 2), round(tp[i1] - t_gravacao + margem_s - a, 2))
    return trechos


def recortar(ffmpeg: str, video: Path, inicio_s: float, duracao_s: float, destino: Path) -> bool:
    destino.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([ffmpeg, "-y", "-ss", f"{inicio_s:.2f}", "-i", str(video), "-t", f"{duracao_s:.2f}",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-an",
                        str(destino)], capture_output=True, text=True, errors="replace")
    if r.returncode != 0:
        log.warning("Recorte %s falhou: %s", destino.name, r.stderr.strip()[-300:])
    return r.returncode == 0 and destino.exists()


# ---------------------------------------------------------------------------------------------
# Índice
# ---------------------------------------------------------------------------------------------
def _ler(arquivo: Path) -> dict[str, Any]:
    try:
        return json.loads(arquivo.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


COLUNAS = {"tempo_s": "Tempo (s)", "diesel_ml": "Diesel (mL)", "eventos": "Eventos bruscos", "criticos": "Críticos",
           "v_min_kmh": "V mín. (km/h)", "desacel_max_mps2": "Pior desacel. (m/s²)",
           "dist_min_figurante_m": "Menor dist. ao figurante (m)"}


def _limiares(cfg_auto: dict[str, Any]) -> Limiares:
    lim = cfg_auto.get("limiares") or {}
    return Limiares(float(lim.get("atencao_mps3", 2.5)), float(lim.get("critico_mps3", 5.0)),
                    float(lim.get("agrupar_s", 1.0)), float(lim.get("fc_hz", 1.5)))


def matriz(modos: list[Modo], situacoes: list[Situacao], L: float, pastas: dict[str, Path], limiares: Limiares,
           videos: dict[tuple[str, str], str]) -> list[dict[str, Any]]:
    """Uma linha por situação e modo (média por volta do benchmark do modo) e uma da volta inteira por modo."""
    linhas = []
    for m in modos:
        pasta_bench = pastas[m.benchmark_de or m.id] / "benchmark"
        arquivo = pasta_bench / f"telemetria_{m.bateria}_agentes.csv"
        tel = pd.read_csv(arquivo) if arquivo.exists() else pd.DataFrame()
        eventos = eventos_da_corrida(tel, limiares, L) if not tel.empty else None
        for s in situacoes:
            met = metricas_situacao(tel, s, L, limiares, eventos) if not tel.empty else {}
            linhas.append({"situacao": s.id, "modo": m.id, **{c: met.get(c, "") for c in COLUNAS},
                           "voltas": met.get("voltas", ""), "video": videos.get((s.id, m.id), "")})
        pv = (_ler(pasta_bench / "resumo.json").get(f"{m.bateria}_agentes") or {}).get("por_volta", {})
        linhas.append({"situacao": "volta", "modo": m.id, "tempo_s": pv.get("tempo_s", ""),
                       "diesel_ml": pv.get("combustivel_ml", ""), "eventos": pv.get("eventos_total", ""),
                       "criticos": pv.get("eventos_criticos", ""), "v_min_kmh": "", "desacel_max_mps2": "",
                       "dist_min_figurante_m": "", "voltas": "", "video": videos.get(("volta", m.id), "")})
    return linhas


def escrever_indice(saida: Path, linhas: list[dict[str, Any]], situacoes: list[Situacao], modos: list[Modo],
                    mapa: str, voltas: int) -> None:
    if not linhas:
        return
    with (saida / "indice.csv").open("w", encoding="utf-8-sig", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=list(linhas[0]), delimiter=";")
        escritor.writeheader()
        escritor.writerows(linhas)

    nomes = {m.id: m.nome for m in modos}

    def rel(caminho_: str) -> str:
        return Path(os.path.relpath(caminho_, saida)).as_posix()

    def tabela(sid: str) -> list[str]:
        texto = ["| Modo | " + " | ".join(COLUNAS.values()) + " | Vídeo |", "|---" * (len(COLUNAS) + 2) + "|"]
        for l in (x for x in linhas if x["situacao"] == sid):
            video = f"[ver]({rel(l['video'])})" if l["video"] else "-"
            texto.append(f"| {nomes[l['modo']]} | " + " | ".join(str(l[c]) if l[c] != "" else "-" for c in COLUNAS)
                         + f" | {video} |")
        return texto

    texto = [f"# Apresentação: cada situação em cada modo ({mapa}, {datetime.now():%d/%m/%Y %H:%M})", "",
             f"Números: média por volta das {voltas} voltas do benchmark de cada modo, dentro do trecho da situação "
             "(trechos cortados pela chegada ficam de fora). Vídeos: recorte da volta gravada ao vivo com o painel.", ""]
    for i, s in enumerate(situacoes, start=1):
        texto += [f"## {i}. {s.nome}", "", f"Pergunta: {s.pergunta}",
                  f"Trecho: de {s.s_ini:.0f} m a {s.s_fim:.0f} m da volta.", "", *tabela(s.id), ""]
    texto += ["## Volta inteira (benchmark, por volta)", "", *tabela("volta"), "", "## Modos", ""]
    texto += [f"- {m.nome}: {m.explica}" for m in modos]
    (saida / "indice.md").write_text("\n".join(texto) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# Programa
# ---------------------------------------------------------------------------------------------
def conferir_ambiente(args: argparse.Namespace, cfg_sim: dict[str, Any], url: str) -> str | None:
    from mover.autonomo.cliente import servidor_no_ar

    if servidor_no_ar(url):
        raise SystemExit(f"Já há um servidor em {url} (um rodar_servidor ou um dirigir esperando Ctrl+C). Feche-o: "
                         "cada volta sobe o próprio servidor com o plano de missão do modo.")
    try:
        import carla

        cc = cfg_sim.get("carla", {})
        cliente = carla.Client(cc.get("host", "localhost"), int(cc.get("porta", 2000)))
        cliente.set_timeout(10.0)
        log.info("CARLA %s no ar.", cliente.get_server_version())
    except Exception as erro:  # sem CARLA não há o que gravar
        raise SystemExit(f"O CARLA não respondeu ({erro}). Abra C:\\CARLA_0.9.16\\CarlaUE4.exe com janela.") from None
    if args.provedor == "ollama":
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        if _get_json(f"{host}/api/tags", 3.0) is None:
            log.warning("O Ollama não respondeu em %s: nos modos com LLM, as decisões ficam com a regra.", host)
    ffmpeg = GravadorTela.disponivel() if not args.sem_video else None
    if not args.sem_video and ffmpeg is None:
        log.warning("Sem gravação da tela (precisa do Windows e do ffmpeg do imageio-ffmpeg com gdigrab): as voltas "
                    "rodam com o painel, mas sem vídeo.%s", " O --video-hud ainda grava o vídeo do CARLA." if args.video_hud else "")
    return ffmpeg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vídeos e benchmarks da apresentação: cada situação em cada modo.")
    parser.add_argument("--mapa", choices=("town10", "campus"), default="town10")
    parser.add_argument("--modos", help=f"ids separados por vírgula (padrão: todos: {', '.join(m.id for m in MODOS)})")
    parser.add_argument("--situacoes", help="ids separados por vírgula (padrão: todas: curva, pedestre, veiculo, "
                                               "lombada, parada; as que o mapa tiver)")
    parser.add_argument("--voltas", type=int, default=3, help="voltas de cada bateria do benchmark")
    parser.add_argument("--voltas-video", type=int, default=1, help="voltas de cada vídeo")
    parser.add_argument("--provedor", default="ollama", help="LLM dos modos com LLM (falso: sem Ollama)")
    parser.add_argument("--alvo-m", type=float, default=800.0, help="Town10: comprimento aproximado da volta")
    parser.add_argument("--sem-benchmark", action="store_true", help="só os vídeos")
    parser.add_argument("--sem-video", action="store_true", help="só os benchmarks")
    parser.add_argument("--video-hud", action="store_true",
                        help="também o MP4 do CARLA com HUD (gravar_video, renderiza quadro a quadro: lento)")
    parser.add_argument("--saida", help="pasta (padrão: data/apresentacao/<data>_<mapa>)")
    parser.add_argument("--seco", action="store_true", help="só mostra as situações, os modos e os comandos")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    modos = [m for m in MODOS if not args.modos or m.id in args.modos.split(",")]
    if not modos:
        raise SystemExit(f"Nenhum modo com esses ids. Há: {', '.join(m.id for m in MODOS)}")
    por_id = {m.id: m for m in MODOS}
    saida = caminho(args.saida) if args.saida else caminho("data/apresentacao") / f"{datetime.now():%Y%m%d_%H%M}_{args.mapa}"
    pastas = {m.id: saida / "modos" / f"{i:02d}_{m.id}" for i, m in enumerate(MODOS, start=1)}
    cfgs, cfg_sim, cfg_auto = preparar_configs(args.mapa, saida, args.alvo_m)
    cs = cfg_sim.get("servidor", {})
    url = f"http://{cs.get('host', '127.0.0.1')}:{int(cs.get('porta', 8000))}"
    # o benchmark de um modo reaproveitado (so_carla usa a bateria sem agentes do completo) também precisa rodar
    benchmarks = [b for b in por_id if b in {m.benchmark_de or m.id for m in modos}]

    def filtrar(sits: list[Situacao]) -> list[Situacao]:
        sits = [s for s in sits if not args.situacoes or s.id in args.situacoes.split(",")]
        cfg_sim["cenario"] = figurantes_do_cenario(sits)
        return sits

    resumo: dict[str, Any] = {"saida": str(saida), "mapa": args.mapa, "configs": cfgs, "situacoes": [], "modos": []}
    for m in modos:
        resumo["modos"].append({"id": m.id, "nome": m.nome,
                                "benchmark": comando_benchmark(m, cfgs, pastas[m.id], args)[2:] if m.benchmark is not None
                                else f"(bateria sem agentes do {m.benchmark_de})",
                                "corrida": comando_corrida(m, cfgs, args)[2:]})
    for b in benchmarks:
        if b not in {m.id for m in modos}:
            resumo["modos"].append({"id": b, "benchmark": comando_benchmark(por_id[b], cfgs, pastas[b], args)[2:],
                                    "corrida": "(só o benchmark)"})
    if args.seco:
        if args.mapa == "campus" or caminho(cfg_sim["mapa"]["arquivo"]).exists():
            resumo["situacoes"] = [s.para_dict() for s in filtrar(preparar_missoes(args.mapa, cfg_sim, cfg_auto)[2])]
        else:
            resumo["situacoes"] = f"(escolhidas depois de exportar o {TOWN10} do CARLA)"
        print(json.dumps(resumo, ensure_ascii=False, indent=2))
        return 0

    saida.mkdir(parents=True, exist_ok=True)
    ffmpeg = conferir_ambiente(args, cfg_sim, url)
    if args.mapa == "town10":
        log.info("Preparando o %s: OpenDRIVE do CARLA e volta automática de ~%.0f m...", TOWN10, args.alvo_m)
        exportar_town10(cfg_sim)
    plano, plano20, situacoes = preparar_missoes(args.mapa, cfg_sim, cfg_auto)
    situacoes = filtrar(situacoes)
    escrever_configs(cfgs, cfg_sim, cfg_auto, plano, plano20, situacoes)
    resumo["situacoes"] = [s.para_dict() for s in situacoes]
    (saida / "plano.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    L, limiares = plano.comprimento, _limiares(cfg_auto)
    nome_mapa = TOWN10 if args.mapa == "town10" else "campus_maua.xodr"

    inicio = time.monotonic()
    if not args.sem_benchmark:
        for b in benchmarks:
            m = por_id[b]
            log.info("=== Benchmark %s: %s", m.id, m.nome)
            codigo = rodar_e_registrar(comando_benchmark(m, cfgs, pastas[m.id], args), pastas[m.id] / "benchmark.log")
            if codigo != 0:
                log.warning("Benchmark %s terminou com código %d (veja %s).", m.id, codigo, pastas[m.id] / "benchmark.log")
    videos: dict[tuple[str, str], str] = {}
    escrever_indice(saida, matriz(modos, situacoes, L, pastas, limiares, videos), situacoes, modos, nome_mapa, args.voltas)
    for m in modos if not args.sem_video else []:
        log.info("=== Volta ao vivo %s: %s", m.id, m.nome)
        prazo = 600.0 * args.voltas_video + 240.0
        estado = rodar_corrida_gravando(comando_corrida(m, cfgs, args), pastas[m.id], url, ffmpeg, prazo)
        if not estado.get("corrida"):
            log.warning("A volta %s não gravou resultado: veja %s.", m.id, pastas[m.id] / "corrida.log")
        if estado.get("video"):
            videos[("volta", m.id)] = estado["video"]
            tel_corrida = Path(estado.get("corrida") or "-") / "telemetria.csv"
            if tel_corrida.exists() and estado.get("t_gravacao"):
                trechos = trechos_do_video(pd.read_csv(tel_corrida), situacoes, L, estado["t_gravacao"])
                for sid, (a, dur) in trechos.items():
                    destino = saida / "situacoes" / sid / f"{pastas[m.id].name}.mp4"
                    if recortar(ffmpeg, Path(estado["video"]), a, dur, destino):
                        videos[(sid, m.id)] = str(destino)
        if args.video_hud and estado.get("corrida"):
            gravar_video_hud(estado["corrida"], cfgs, cfgs[m.missao], pastas[m.id] / "video_hud.mp4")
        escrever_indice(saida, matriz(modos, situacoes, L, pastas, limiares, videos), situacoes, modos, nome_mapa,
                        args.voltas)
    perfil = saida / "modos" / "_navegador"
    if perfil.exists():
        shutil.rmtree(perfil, ignore_errors=True)
    log.info("Pronto em %.0f min: %s (indice.md e indice.csv).", (time.monotonic() - inicio) / 60, saida)
    return 0


if __name__ == "__main__":
    sys.exit(main())
