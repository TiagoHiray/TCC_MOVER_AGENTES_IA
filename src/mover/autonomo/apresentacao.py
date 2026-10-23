"""Vídeos e benchmarks da apresentação: um cenário para cada pergunta provável da banca.

Para cada cenário, o script roda, em sequência:
1. o benchmark (N voltas sem agentes x N voltas com agentes, o CARLA sem desenhar a cena);
2. uma volta ao vivo com o painel aberto, gravando a tela: o painel mostra a câmera do CARLA embutida,
   os gráficos, o log das decisões dos agentes e o chat.
No fim, escreve indice.md e indice.csv com a pergunta, o vídeo, o benchmark e os números de cada cenário.

Uso, no Windows da VM, com o CARLA aberto COM JANELA, o Ollama no ar e a área de trabalho visível (a
gravação é da tela: com a conexão remota minimizada, o vídeo sai preto), a partir de src/:
    python -m mover.autonomo.apresentacao                       # Town10HD_Opt, todos os cenários
    python -m mover.autonomo.apresentacao --mapa campus         # no campus_maua.xodr
    python -m mover.autonomo.apresentacao --seco                # só mostra o que vai rodar
    python -m mover.autonomo.apresentacao --cenarios completo,so_regra --sem-benchmark
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

import yaml

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
class Cenario:
    id: str
    pergunta: str
    mostra: str
    corrida: tuple[str, ...]              # opções do dirigir
    benchmark: tuple[str, ...] | None     # opções do benchmark (None: usa o de outro cenário)
    missao: str = "padrao"                # padrao | cruzeiro20
    benchmark_de: str | None = None


CENARIOS: tuple[Cenario, ...] = (
    Cenario("so_carla", "Como o caminhão anda sem a camada agêntica, só com o piloto automático do CARLA?",
            "Linha de base: o Traffic Manager segue o limite de cada zona em degrau e freia em cima das curvas, "
            "faixas e lombadas. É a bateria 'sem agentes' dos benchmarks.",
            ("--sem-agentes",), None, benchmark_de="completo"),
    Cenario("completo", "Como fica com a camada agêntica completa (ML + regra + LLM)?",
            "O ML propõe a suavização, o modelo do caminhão confere, a regra escolhe e o LLM supervisiona e "
            "explica cada decisão no log.", (), ()),
    Cenario("so_regra", "Só regras fixas, sem ML e sem LLM, já resolveriam?",
            "A regra escolhe entre as 4 opções fixas (econômica, suave, moderada, firme), conferidas pelo modelo "
            "do caminhão. Sem ML e sem LLM.", ("--sem-ml", "--sem-llm"), ("--sem-ml", "--sem-llm")),
    Cenario("so_ml", "O que o modelo de ML (floresta aleatória) faz sem o LLM?",
            "O ML propõe a suavização zona a zona e a regra só confere. Sem LLM, o log usa o texto-modelo: é "
            "também o que acontece se o LLM cair no meio da corrida.", ("--sem-llm",), ("--sem-llm",)),
    Cenario("so_llm", "E o LLM sem o ML? Ele decide sozinho?",
            "O LLM escolhe entre as opções fixas já simuladas; a guarda recusa opção com mais eventos bruscos que a "
            "da regra e números sem respaldo. Ele nunca mexe no pedal.", ("--sem-ml",), ("--sem-ml",)),
    Cenario("devagar_20", "Não bastava o caminhão andar mais devagar (cruzeiro de 20 km/h)?",
            "Missão com cruzeiro de 20 km/h. A volta é sem agentes; o benchmark compara o caminhão sozinho e com os "
            "agentes nessa mesma missão.", ("--sem-agentes",), (), missao="cruzeiro20"),
)


# ---------------------------------------------------------------------------------------------
# Mapa e plano de missão
# ---------------------------------------------------------------------------------------------
def _salvar_yaml(dados: dict[str, Any], destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(yaml.safe_dump(dados, allow_unicode=True, sort_keys=False), encoding="utf-8")


def configs_campus() -> dict[str, str]:
    return {"config": "config/autonomo.yaml", "config_simulacao": "config/simulacao.yaml",
            "padrao": "config/missao.yaml", "cruzeiro20": "config/missao_cruzeiro20.yaml"}


def configs_town10(saida: Path, alvo_m: float, escrever: bool = True) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    """Configs do Town10HD_Opt, derivadas das do projeto: mapa instalado no CARLA e volta achada no grafo."""
    pasta = saida / "config"
    xodr = f"data/mapas/{TOWN10}_carla.xodr"
    cfg_sim = carregar_yaml("config/simulacao.yaml")
    cfg_sim["mapa"] = {**cfg_sim.get("mapa", {}), "arquivo": xodr, "tipos_faixa": ["driving"]}
    cfg_sim["carla"] = {**cfg_sim.get("carla", {}), "mapa_carla": TOWN10}
    cfg_auto = carregar_yaml("config/autonomo.yaml")
    caminhos = {"config": str(pasta / "autonomo_town10.yaml"), "config_simulacao": str(pasta / "simulacao_town10.yaml"),
                "padrao": str(pasta / "missao_town10.yaml"), "cruzeiro20": str(pasta / "missao_town10_cruzeiro20.yaml")}
    cfg_auto["missao"] = {**cfg_auto.get("missao", {}), "arquivo": caminhos["padrao"]}
    gm = dict(cfg_auto.get("gerar_missao") or {})
    gm["rota"] = {"vias": "auto", "alvo_m": alvo_m, "min_m": 0.6 * alvo_m, "max_m": 1.6 * alvo_m, "largada": None}
    gm["lombadas"], gm["portoes"] = [], []  # o Town10 não tem lombada nem portão; faixas e curvas vêm do mapa
    cfg_auto["gerar_missao"] = gm
    if escrever:
        _salvar_yaml(cfg_sim, Path(caminhos["config_simulacao"]))
        _salvar_yaml(cfg_auto, Path(caminhos["config"]))
    return caminhos, cfg_sim, cfg_auto


def exportar_town10(cfg_sim: dict[str, Any]) -> None:
    """Abre o Town10HD_Opt no CARLA e salva o OpenDRIVE dele em data/mapas (a rota sai desse arquivo)."""
    import carla

    from mover.autonomo.carla_tm import carregar_mapa_instalado

    cc = cfg_sim.get("carla", {})
    cliente = carla.Client(cc.get("host", "localhost"), int(cc.get("porta", 2000)))
    cliente.set_timeout(120.0)
    carregar_mapa_instalado(cliente, TOWN10, caminho("data/mapas"))


def gerar_missoes_town10(caminhos: dict[str, str], cfg_sim: dict[str, Any], cfg_auto: dict[str, Any]) -> None:
    from mover.autonomo.gerar_missao import gerar, salvar_figura
    from mover.simulacao.opendrive import ler_xodr

    cabecalho = "# Plano de missão do Town10HD_Opt (gerado por mover.autonomo.apresentacao)."
    for chave, cruzeiro in (("padrao", None), ("cruzeiro20", 20.0)):
        gm = dict(cfg_auto["gerar_missao"])
        if cruzeiro is not None:
            gm["cruzeiro_kmh"] = cruzeiro
        plano = gerar(cfg_sim, gm)
        plano.salvar(Path(caminhos[chave]), cabecalho)
        log.info("Missão %s: %.0f m (%s), %d zonas, %d parada(s).", chave, plano.comprimento, plano.sentido,
                 len(plano.zonas), len(plano.paradas))
        if chave == "padrao":
            salvar_figura(plano, Path(caminhos[chave]).with_suffix(".png"),
                          ler_xodr(caminho(cfg_sim["mapa"]["arquivo"]), 1.0, ("driving",)).pontos)


# ---------------------------------------------------------------------------------------------
# Comandos
# ---------------------------------------------------------------------------------------------
def comando_benchmark(c: Cenario, cfgs: dict[str, str], pasta: Path, args: argparse.Namespace) -> list[str]:
    return [sys.executable, "-m", "mover.autonomo.benchmark", "--voltas", str(args.voltas),
            "--config", cfgs["config"], "--config-simulacao", cfgs["config_simulacao"], "--missao", cfgs[c.missao],
            "--provedor", args.provedor, "--saida", str(pasta / "benchmark"), "--silencioso", *c.benchmark]


def comando_corrida(c: Cenario, cfgs: dict[str, str], args: argparse.Namespace) -> list[str]:
    cmd = [sys.executable, "-m", "mover.autonomo.dirigir", "--voltas", str(args.voltas_video),
           "--config", cfgs["config"], "--config-simulacao", cfgs["config_simulacao"], "--missao", cfgs[c.missao],
           "--provedor", args.provedor, "--camera", "perseguicao", "--sem-navegador", *c.corrida]
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
                    time.sleep(4.0)  # a página carrega e recebe o histórico da sessão
                    if ffmpeg:
                        gravador = GravadorTela(pasta / "video_painel.mp4")
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


# ---------------------------------------------------------------------------------------------
# Índice
# ---------------------------------------------------------------------------------------------
def _ler(arquivo: Path) -> dict[str, Any]:
    try:
        return json.loads(arquivo.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def linha_do_indice(c: Cenario, pasta: Path, pastas: dict[str, Path], estado: dict[str, Any]) -> dict[str, Any]:
    pasta_bench = pastas[c.benchmark_de or c.id] / "benchmark"
    bench = _ler(pasta_bench / "resumo.json")
    sem, com = (bench.get("sem_agentes") or {}).get("por_volta", {}), (bench.get("com_agentes") or {}).get("por_volta", {})
    dif = bench.get("diferenca_pct") or {}
    corrida = _ler(Path(estado["corrida"]) / "resumo.json") if estado.get("corrida") else {}
    total = corrida.get("total") or {}
    return {
        "cenario": c.id, "pergunta": c.pergunta, "mostra": c.mostra,
        "video": estado.get("video") or "", "corrida": estado.get("corrida") or "",
        "fim_da_volta": corrida.get("motivo_fim", ""),
        "volta_diesel_ml": round(1000 * float(total.get("combustivel_l", 0) or 0), 1) if total else "",
        "volta_tempo_s": total.get("tempo_s", ""),
        "volta_eventos": (total.get("eventos_atencao", 0) + total.get("eventos_criticos", 0)) if total else "",
        "benchmark": str(pasta_bench) if bench else "",
        "sem_agentes_diesel_ml": sem.get("combustivel_ml", ""), "com_agentes_diesel_ml": com.get("combustivel_ml", ""),
        "diesel_pct": dif.get("combustivel_ml", ""), "tempo_pct": dif.get("tempo_s", ""),
        "eventos_pct": dif.get("eventos_total", ""),
        "sem_agentes_eventos": sem.get("eventos_total", ""), "com_agentes_eventos": com.get("eventos_total", ""),
        "decisoes": json.dumps(bench.get("decisoes_por_origem", {}), ensure_ascii=False) if bench else "",
    }


def escrever_indice(saida: Path, linhas: list[dict[str, Any]], mapa: str) -> None:
    if not linhas:
        return
    with (saida / "indice.csv").open("w", encoding="utf-8-sig", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=list(linhas[0]), delimiter=";")
        escritor.writeheader()
        escritor.writerows(linhas)

    def rel(caminho_: str) -> str:
        return Path(os.path.relpath(caminho_, saida)).as_posix() if caminho_ else "-"

    texto = [f"# Apresentação: vídeos e benchmarks ({mapa}, {datetime.now():%d/%m/%Y %H:%M})", ""]
    for i, l in enumerate(linhas, start=1):
        texto += [f"## {i}. {l['pergunta']}", "", l["mostra"], "",
                  f"- Vídeo: [{rel(l['video'])}]({rel(l['video'])})" if l["video"] else "- Vídeo: não gravado",
                  f"- Volta ao vivo: {l['volta_tempo_s']} s, {l['volta_diesel_ml']} mL, {l['volta_eventos']} eventos bruscos "
                  f"(fim: {l['fim_da_volta'] or '-'})"]
        if l["benchmark"]:
            texto += [f"- Benchmark: [{rel(l['benchmark'])}]({rel(l['benchmark'])}) - por volta, diesel "
                      f"{l['sem_agentes_diesel_ml']} -> {l['com_agentes_diesel_ml']} mL ({l['diesel_pct']} %), eventos "
                      f"{l['sem_agentes_eventos']} -> {l['com_agentes_eventos']} ({l['eventos_pct']} %), tempo "
                      f"{l['tempo_pct']} %; decisões {l['decisoes']}",
                      f"- Gráfico: [{rel(l['benchmark'])}/comparativo.png]({rel(l['benchmark'])}/comparativo.png)"]
        texto.append("")
    (saida / "indice.md").write_text("\n".join(texto), encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# Programa
# ---------------------------------------------------------------------------------------------
def conferir_ambiente(args: argparse.Namespace, cfg_sim: dict[str, Any], url: str) -> str | None:
    from mover.autonomo.cliente import servidor_no_ar

    if servidor_no_ar(url):
        raise SystemExit(f"Já há um servidor em {url} (um rodar_servidor ou um dirigir esperando Ctrl+C). Feche-o: "
                         "cada volta sobe o próprio servidor com o plano de missão do cenário.")
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
            log.warning("O Ollama não respondeu em %s: nos cenários com LLM, as decisões ficam com a regra.", host)
    ffmpeg = GravadorTela.disponivel() if not args.sem_video else None
    if not args.sem_video and ffmpeg is None:
        log.warning("Sem gravação da tela (precisa do Windows e do ffmpeg do imageio-ffmpeg com gdigrab): as voltas "
                    "rodam com o painel, mas sem vídeo.%s", " O --video-hud ainda grava o vídeo do CARLA." if args.video_hud else "")
    return ffmpeg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vídeos e benchmarks da apresentação, um cenário por pergunta.")
    parser.add_argument("--mapa", choices=("town10", "campus"), default="town10")
    parser.add_argument("--cenarios", help=f"ids separados por vírgula (padrão: todos: {', '.join(c.id for c in CENARIOS)})")
    parser.add_argument("--voltas", type=int, default=3, help="voltas de cada bateria do benchmark")
    parser.add_argument("--voltas-video", type=int, default=1, help="voltas de cada vídeo")
    parser.add_argument("--provedor", default="ollama", help="LLM dos cenários com LLM (falso: sem Ollama)")
    parser.add_argument("--alvo-m", type=float, default=800.0, help="Town10: comprimento aproximado da volta")
    parser.add_argument("--sem-benchmark", action="store_true", help="só os vídeos")
    parser.add_argument("--sem-video", action="store_true", help="só os benchmarks")
    parser.add_argument("--video-hud", action="store_true",
                        help="também o MP4 do CARLA com HUD (gravar_video, renderiza quadro a quadro: lento)")
    parser.add_argument("--saida", help="pasta (padrão: data/apresentacao/<data>_<mapa>)")
    parser.add_argument("--seco", action="store_true", help="só mostra os cenários e os comandos")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    escolhidos = [c for c in CENARIOS if not args.cenarios or c.id in args.cenarios.split(",")]
    if not escolhidos:
        raise SystemExit(f"Nenhum cenário com esses ids. Há: {', '.join(c.id for c in CENARIOS)}")
    por_id = {c.id: c for c in CENARIOS}
    saida = caminho(args.saida) if args.saida else caminho("data/apresentacao") / f"{datetime.now():%Y%m%d_%H%M}_{args.mapa}"
    pastas = {c.id: saida / f"{i:02d}_{c.id}" for i, c in enumerate(CENARIOS, start=1)}

    if args.mapa == "town10":
        cfgs, cfg_sim, cfg_auto = configs_town10(saida, args.alvo_m, escrever=not args.seco)
    else:
        cfgs, cfg_sim, cfg_auto = configs_campus(), carregar_yaml("config/simulacao.yaml"), carregar_yaml("config/autonomo.yaml")
    cs = cfg_sim.get("servidor", {})
    url = f"http://{cs.get('host', '127.0.0.1')}:{int(cs.get('porta', 8000))}"
    # o benchmark de um cenário reaproveitado (so_carla usa o do completo) também precisa rodar
    benchmarks = {c.benchmark_de or c.id for c in escolhidos}
    plano = {"saida": str(saida), "mapa": args.mapa, "configs": cfgs, "cenarios": []}
    for c in escolhidos:
        plano["cenarios"].append({"id": c.id, "pergunta": c.pergunta,
                                  "benchmark": comando_benchmark(c, cfgs, pastas[c.id], args)[2:] if c.benchmark is not None else
                                  f"(o do cenário {c.benchmark_de})",
                                  "corrida": comando_corrida(c, cfgs, args)[2:]})
    for b in sorted(benchmarks - {c.id for c in escolhidos}):
        plano["cenarios"].append({"id": b, "benchmark": comando_benchmark(por_id[b], cfgs, pastas[b], args)[2:],
                                  "corrida": "(só o benchmark)"})
    if args.seco:
        print(json.dumps(plano, ensure_ascii=False, indent=2))
        return 0

    saida.mkdir(parents=True, exist_ok=True)
    (saida / "plano.json").write_text(json.dumps(plano, ensure_ascii=False, indent=2), encoding="utf-8")
    ffmpeg = conferir_ambiente(args, cfg_sim, url)
    if args.mapa == "town10":
        log.info("Preparando o %s: OpenDRIVE do CARLA e volta automática de ~%.0f m...", TOWN10, args.alvo_m)
        exportar_town10(cfg_sim)
        gerar_missoes_town10(cfgs, cfg_sim, cfg_auto)

    inicio = time.monotonic()
    if not args.sem_benchmark:
        for b in sorted(benchmarks, key=list(por_id).index):
            c = por_id[b]
            log.info("=== Benchmark %s: %s", c.id, c.pergunta)
            codigo = rodar_e_registrar(comando_benchmark(c, cfgs, pastas[c.id], args), pastas[c.id] / "benchmark.log")
            if codigo != 0:
                log.warning("Benchmark %s terminou com código %d (veja %s).", c.id, codigo, pastas[c.id] / "benchmark.log")
    linhas = []
    for c in escolhidos:
        estado: dict[str, Any] = {}
        if not args.sem_video:
            log.info("=== Volta ao vivo %s: %s", c.id, c.pergunta)
            prazo = 600.0 * args.voltas_video + 240.0
            estado = rodar_corrida_gravando(comando_corrida(c, cfgs, args), pastas[c.id], url, ffmpeg, prazo)
            if not estado.get("corrida"):
                log.warning("A volta %s não gravou resultado: veja %s.", c.id, pastas[c.id] / "corrida.log")
            if args.video_hud and estado.get("corrida"):
                gravar_video_hud(estado["corrida"], cfgs, cfgs[c.missao], pastas[c.id] / "video_hud.mp4")
        linhas.append(linha_do_indice(c, pastas[c.id], pastas, estado))
        escrever_indice(saida, linhas, TOWN10 if args.mapa == "town10" else "campus_maua.xodr")
    perfil = saida / "_navegador"
    if perfil.exists():
        shutil.rmtree(perfil, ignore_errors=True)
    log.info("Pronto em %.0f min: %s (indice.md e indice.csv).", (time.monotonic() - inicio) / 60, saida)
    return 0


if __name__ == "__main__":
    sys.exit(main())
