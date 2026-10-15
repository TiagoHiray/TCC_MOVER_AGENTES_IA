"""Vídeo MP4 da corrida autônoma no CARLA: câmera de perseguição em 1080p, 30 fps, H.264, com HUD.

Duas formas de reconstruir a corrida no CARLA:
- "gravacao" (padrão): reproduz o arquivo do gravador do CARLA (client.replay_file), feito pelo
  piloto com --gravar. É a corrida exata, com as rodas e a suspensão do caminhão;
- "poses": se a reprodução falhar (por exemplo, o mapa OpenDRIVE não carregar no reprodutor),
  recria o caminhão sem física e impõe a pose gravada na telemetria a cada quadro.
Em ambas, o CARLA roda em modo síncrono com passo de 1/30 s, uma câmera RGB presa ao caminhão
captura cada quadro, o HUD (velocidade, consumo, decisão dos agentes e minimapa) é desenhado por
cima e o ffmpeg (imageio-ffmpeg) codifica em H.264 (libx264, CRF 18, yuv420p).

Uso, a partir da pasta src/ do repositório, com o CARLA aberto (de preferência com
-quality-level=Epic, que é o que define a qualidade da imagem da câmera):
    python -m mover.autonomo.dirigir --gravar --sair-no-fim            # 1) corrida gravada
    python -m mover.video.gravar_video                                 # 2) vídeo da última corrida gravada
    python -m mover.video.gravar_video --corrida ../data/autonomo/corridas/<id> --modo poses --camera frontal
    python -m mover.video.gravar_video --teste-hud hud.png             # só um quadro do HUD (sem CARLA)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import queue
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # execução direta: python src/mover/video/gravar_video.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from mover.config import caminho, carregar_yaml
from mover.video.hud import DadosCorrida, desenhar_hud

log = logging.getLogger("mover.video")

CAMERAS = {
    # posição em relação ao caminhão (m) e inclinação (graus)
    "perseguicao": {"x": -11.0, "y": 0.0, "z": 4.6, "pitch": -12.0, "fov": 90.0, "preso": "SpringArmGhost"},
    "frontal": {"x": 2.6, "y": 0.0, "z": 2.6, "pitch": -4.0, "fov": 95.0, "preso": "Rigid"},
    "alto": {"x": -18.0, "y": 0.0, "z": 12.0, "pitch": -28.0, "fov": 80.0, "preso": "SpringArmGhost"},
}

ULTIMA = ("", "ultima", "última")


def _tem_gravacao(pasta: Path) -> bool:
    if (pasta / "gravacao_carla.log").is_file():
        return True
    try:
        return bool(json.loads((pasta / "resumo.json").read_text(encoding="utf-8")).get("gravacao_carla"))
    except (OSError, ValueError):
        return False


def resolver_corrida(valor: str | None, pasta_corridas: Path) -> Path:
    """Pasta da corrida do vídeo.

    Um caminho é procurado primeiro a partir da pasta atual e depois da raiz do repositório. Sem
    valor (ou "ultima"), vale a corrida mais recente gravada com --gravar; se nenhuma tiver a
    gravação do CARLA, a corrida mais recente. Os nomes das pastas começam pela data e hora, então
    a ordem alfabética é a ordem no tempo.
    """
    if valor is None or valor.strip().lower() in ULTIMA:
        corridas = sorted(p for p in pasta_corridas.glob("*") if (p / "telemetria.csv").is_file())
        if not corridas:
            raise SystemExit(f"Nenhuma corrida em {pasta_corridas}. Rode antes: python -m mover.autonomo.dirigir --gravar")
        gravadas = [p for p in corridas if _tem_gravacao(p)]
        return (gravadas or corridas)[-1]
    pasta = Path(valor)
    return pasta if pasta.exists() else caminho(valor)


class GravadorMP4:
    """Recebe quadros RGB (numpy HxWx3) e escreve o MP4 com o ffmpeg do imageio-ffmpeg."""

    def __init__(self, destino: Path, largura: int, altura: int, fps: int = 30, crf: int = 18, preset: str = "medium"):
        try:
            import imageio_ffmpeg
        except ImportError:
            raise SystemExit("Instale o imageio-ffmpeg (pip install imageio-ffmpeg): ele traz o ffmpeg para o H.264.") from None
        destino.parent.mkdir(parents=True, exist_ok=True)
        self.destino = destino
        self.tamanho = (int(largura), int(altura))
        self.quadros = 0
        self._gerador = imageio_ffmpeg.write_frames(
            str(destino), self.tamanho, fps=fps, codec="libx264", quality=None, pix_fmt_out="yuv420p",
            macro_block_size=1, output_params=["-crf", str(int(crf)), "-preset", preset, "-movflags", "+faststart"])
        self._gerador.send(None)

    def escrever(self, rgb: np.ndarray) -> None:
        if rgb.shape[1::-1] != self.tamanho:
            raise ValueError(f"quadro {rgb.shape[1]}x{rgb.shape[0]}; o vídeo é {self.tamanho[0]}x{self.tamanho[1]}")
        self._gerador.send(np.ascontiguousarray(rgb, dtype=np.uint8))
        self.quadros += 1

    def fechar(self) -> None:
        if self._gerador is not None:
            self._gerador.close()
            self._gerador = None


def bgra_para_rgb(bruto: bytes, largura: int, altura: int) -> np.ndarray:
    pixels = np.frombuffer(bruto, dtype=np.uint8).reshape(altura, largura, 4)
    return pixels[:, :, 2::-1]


def compor(rgb: np.ndarray, dados: DadosCorrida | None, t: float, hud: bool) -> np.ndarray:
    if not hud or dados is None:
        return rgb
    from PIL import Image

    return np.asarray(desenhar_hud(Image.fromarray(np.ascontiguousarray(rgb)), dados, t))


class CenaCarla:
    """Mundo do CARLA para o vídeo: mapa, modo síncrono a 1/fps, caminhão (reproduzido ou por poses) e câmera."""

    def __init__(self, cfg_sim: dict[str, Any], fps: int, largura: int, altura: int, camera: str):
        self.cc = cfg_sim.get("carla", {})
        self.cv = cfg_sim.get("veiculo", {})
        self.fps, self.largura, self.altura = int(fps), int(largura), int(altura)
        self.ccam = CAMERAS[camera]
        self.carla: Any = None
        self.client: Any = None
        self.world: Any = None
        self.ator: Any = None
        self.ator_proprio = False
        self.sensor: Any = None
        self.fila: queue.Queue = queue.Queue()
        self._config_original: Any = None
        self.reproduzindo = False

    def conectar(self) -> None:
        try:
            import carla
        except ImportError:
            raise SystemExit("O pacote 'carla' não está instalado (cliente do CARLA 0.9.16).") from None
        self.carla = carla
        self.client = carla.Client(self.cc.get("host", "localhost"), int(self.cc.get("porta", 2000)))
        self.client.set_timeout(float(self.cc.get("timeout_s", 60.0)))
        self.world = self.client.get_world()

    def carregar_mapa(self, texto_xodr: str | None) -> None:
        if texto_xodr and self.cc.get("mapa_carla"):
            from mover.autonomo.carla_tm import carregar_mapa_instalado

            self.world = carregar_mapa_instalado(self.client, str(self.cc["mapa_carla"]))
        elif texto_xodr:
            p = self.cc.get("opendrive", {})
            parametros = self.carla.OpendriveGenerationParameters(
                vertex_distance=float(p.get("vertex_distance", 2.0)), max_road_length=float(p.get("max_road_length", 50.0)),
                wall_height=float(p.get("wall_height", 0.0)), additional_width=float(p.get("additional_width", 0.6)),
                smooth_junctions=bool(p.get("smooth_junctions", True)), enable_mesh_visibility=True,
                enable_pedestrian_navigation=bool(p.get("enable_pedestrian_navigation", True)))
            log.info("Gerando o mundo OpenDRIVE no CARLA...")
            # preparos do mapa_final.xodr, desligados com o campus_maua.xodr (o mapa vai ao CARLA como é):
            # from mover.autonomo.xodr_carla import preparar_para_carla
            # texto_xodr, _ = preparar_para_carla(texto_xodr, p)
            self.world = self.client.generate_opendrive_world(texto_xodr, parametros)
        self._config_original = self.world.get_settings()
        config = self.world.get_settings()
        config.synchronous_mode = True
        config.fixed_delta_seconds = 1.0 / self.fps
        config.no_rendering_mode = False
        self.world.apply_settings(config)
        clima = getattr(self.carla.WeatherParameters, "ClearNoon", None)
        if clima is not None:
            self.world.set_weather(clima)

    def reproduzir(self, arquivo: str) -> None:
        """Começa a reprodução da gravação e acha o caminhão reproduzido."""
        self.client.set_replayer_ignore_spectator(True)
        self.client.set_replayer_time_factor(1.0)
        info = self.client.replay_file(str(arquivo), 0.0, 0.0, 0, False)
        log.info("Reprodução: %s", (info or "").splitlines()[0] if info else arquivo)
        self.reproduzindo = True
        for _ in range(3):
            self.world.tick()
        blueprint = self.cv.get("blueprint", "vehicle.carlamotors.european_hgv")
        candidatos = [a for a in self.world.get_actors().filter("vehicle.*") if a.type_id == blueprint]
        preferidos = [a for a in candidatos if a.attributes.get("role_name") == self.cv.get("role_name", "mover_gemeo")] or candidatos
        if not preferidos:
            raise RuntimeError(f"a reprodução não criou o caminhão {blueprint} (o mapa da gravação carregou?)")
        self.ator = preferidos[0]
        log.info("Caminhão reproduzido: %s (id %d)", self.ator.type_id, self.ator.id)

    def criar_caminhao_sem_fisica(self) -> None:
        biblioteca = self.world.get_blueprint_library()
        bp = biblioteca.find(self.cv.get("blueprint", "vehicle.carlamotors.european_hgv"))
        self.ator = self.world.spawn_actor(bp, self.carla.Transform(self.carla.Location(z=200.0)))
        self.ator.set_simulate_physics(False)
        self.ator_proprio = True

    def impor_pose(self, linha: dict[str, Any]) -> None:
        c = self.carla
        caixa = self.ator.bounding_box
        base = float(caixa.location.z - caixa.extent.z)
        self.ator.set_transform(c.Transform(
            c.Location(x=float(linha["x_carla"]), y=float(linha["y_carla"]), z=float(linha.get("z", 0.0)) - base + 0.05),
            c.Rotation(pitch=float(linha.get("pitch_carla", 0.0)), yaw=float(linha["yaw_carla"]), roll=float(linha.get("roll_carla", 0.0)))))

    def criar_camera(self) -> None:
        c = self.carla
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(self.largura))
        bp.set_attribute("image_size_y", str(self.altura))
        bp.set_attribute("fov", str(self.ccam["fov"]))
        bp.set_attribute("sensor_tick", "0.0")  # um quadro por tick
        if bp.has_attribute("motion_blur_intensity"):
            bp.set_attribute("motion_blur_intensity", "0.25")
        posicao = c.Transform(c.Location(x=self.ccam["x"], y=self.ccam["y"], z=self.ccam["z"]), c.Rotation(pitch=self.ccam["pitch"]))
        preso = getattr(c.AttachmentType, self.ccam["preso"], c.AttachmentType.Rigid)
        self.sensor = self.world.spawn_actor(bp, posicao, attach_to=self.ator, attachment_type=preso)
        self.sensor.listen(self.fila.put)

    def quadro(self, frame: int, prazo_s: float = 10.0) -> np.ndarray:
        """O quadro da câmera do tick `frame` (descarta os mais antigos)."""
        limite = time.monotonic() + prazo_s
        while True:
            imagem = self.fila.get(timeout=max(0.1, limite - time.monotonic()))
            if imagem.frame >= frame:
                return bgra_para_rgb(bytes(imagem.raw_data), imagem.width, imagem.height)

    def encerrar(self) -> None:
        if self.sensor is not None:
            for acao in ("stop", "destroy"):
                try:
                    getattr(self.sensor, acao)()
                except RuntimeError:
                    pass
            self.sensor = None
        if self.reproduzindo:
            try:
                self.client.stop_replayer(False)
            except RuntimeError:
                pass
            self.reproduzindo = False
        if self.world is not None and self._config_original is not None:
            try:
                self.world.apply_settings(self._config_original)
            except RuntimeError:
                pass
            self._config_original = None
        if self.ator is not None and self.ator_proprio:
            try:
                self.ator.destroy()
            except RuntimeError:
                pass
        self.ator = None


def renderizar(cena: CenaCarla, dados: DadosCorrida, gravador: GravadorMP4, modo: str, inicio_s: float, duracao_s: float,
               hud: bool) -> int:
    fps = cena.fps
    n = int(round(duracao_s * fps))
    t_tel = dados.telemetria["t"].to_numpy(float)
    colunas = ("x_carla", "y_carla", "z", "yaw_carla", "pitch_carla", "roll_carla")
    if modo == "gravacao" and inicio_s > 0:
        for _ in range(int(round(inicio_s * fps))):  # avança a reprodução até o início pedido
            cena.world.tick()
    inicio = time.monotonic()
    for k in range(n):
        t = inicio_s + k / fps
        if modo == "poses":
            i = int(np.clip(np.searchsorted(t_tel, t), 1, len(t_tel) - 1))
            f = (t - t_tel[i - 1]) / max(1e-9, t_tel[i] - t_tel[i - 1])
            antes, depois = dados.telemetria.iloc[i - 1], dados.telemetria.iloc[i]
            linha = {c: float(antes[c]) + max(0.0, min(1.0, f)) * (float(depois[c]) - float(antes[c])) for c in colunas if c in antes}
            linha["yaw_carla"] = float(antes["yaw_carla"]) + f * (((float(depois["yaw_carla"]) - float(antes["yaw_carla"]) + 180) % 360) - 180)
            cena.impor_pose(linha)
        frame = cena.world.tick()
        rgb = cena.quadro(frame)
        gravador.escrever(compor(rgb, dados, t, hud))
        if k % (fps * 10) == 0:
            decorrido = time.monotonic() - inicio
            log.info("Quadro %d/%d (%.0f s de corrida, %.1f quadros/s)", k, n, t, k / decorrido if decorrido > 0 else 0.0)
    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vídeo MP4 da corrida no CARLA (1080p, 30 fps, H.264, com HUD).")
    parser.add_argument("--corrida", default="ultima",
                        help="pasta da corrida (data/autonomo/corridas/<id>); padrão: a última gravada com --gravar")
    parser.add_argument("--saida", help="arquivo MP4 (padrão: <corrida>/video.mp4)")
    parser.add_argument("--modo", choices=("gravacao", "poses"), default="gravacao",
                        help="reproduzir a gravação do CARLA ou impor as poses da telemetria")
    parser.add_argument("--camera", choices=tuple(CAMERAS), default="perseguicao")
    parser.add_argument("--largura", type=int, default=1920)
    parser.add_argument("--altura", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--crf", type=int, default=18, help="qualidade do H.264 (menor = melhor; 18 é quase sem perdas visíveis)")
    parser.add_argument("--inicio", type=float, default=0.0, help="segundo da corrida em que o vídeo começa")
    parser.add_argument("--duracao", type=float, help="segundos de vídeo (padrão: até o fim da corrida)")
    parser.add_argument("--sem-hud", action="store_true", help="vídeo limpo, sem o HUD")
    parser.add_argument("--manter-mundo", action="store_true", help="usa o mapa já aberto no CARLA")
    parser.add_argument("--config", default="config/autonomo.yaml", help="YAML da corrida autônoma (pasta das corridas)")
    parser.add_argument("--config-simulacao", default="config/simulacao.yaml")
    parser.add_argument("--missao", default=None, help="plano de missão do minimapa (padrão: config/autonomo.yaml)")
    parser.add_argument("--teste-hud", help="só desenha o HUD de um instante num PNG (sem CARLA) e sai")
    parser.add_argument("--instante", type=float, help="instante do --teste-hud (padrão: metade da corrida)")
    args = parser.parse_args(argv)

    from mover.servidor.rodar_servidor import configurar_logs

    configurar_logs()
    cfg = carregar_yaml(args.config)
    pasta = resolver_corrida(args.corrida, caminho((cfg.get("corridas") or {}).get("pasta", "data/autonomo/corridas")))
    if not (pasta / "telemetria.csv").exists():
        raise SystemExit(f"Corrida sem telemetria.csv: {pasta}. Rode antes: python -m mover.autonomo.dirigir --gravar")
    log.info("Corrida: %s", pasta)
    plano = None
    try:
        from mover.autonomo.montagem import montar

        plano = montar(carregar_yaml("config/autonomo.yaml"), args.missao).plano
    except SystemExit:
        log.warning("Sem plano de missão: o minimapa do HUD fica de fora.")
    dados = DadosCorrida.carregar(pasta, plano)

    if args.teste_hud:
        from PIL import Image

        t = args.instante if args.instante is not None else dados.duracao_s / 2
        fundo = Image.new("RGB", (args.largura, args.altura), (70, 86, 96))
        desenhar_hud(fundo, dados, t).save(args.teste_hud)
        log.info("HUD de %.1f s salvo em %s", t, args.teste_hud)
        return 0

    resumo = dados.resumo
    modo = args.modo
    gravacao = resumo.get("gravacao_carla")
    if modo == "gravacao" and not gravacao:
        log.warning("A corrida não tem gravação do CARLA (rode o piloto com --gravar); usando as poses da telemetria.")
        modo = "poses"
    if modo == "poses" and "x_carla" not in dados.telemetria:
        raise SystemExit("A telemetria não tem as poses do CARLA (corrida feita com --sem-carla?).")
    cfg_sim = carregar_yaml(args.config_simulacao) if caminho(args.config_simulacao).exists() else {}
    texto_xodr = None
    if not args.manter_mundo:
        mapa = caminho(resumo.get("mapa") or cfg_sim.get("mapa", {}).get("arquivo", "data/mapas/mapa_final.xodr"))
        texto_xodr = mapa.read_text(encoding="utf-8")
    inicio = max(0.0, float(args.inicio))
    duracao = float(args.duracao) if args.duracao else max(1.0, dados.duracao_s - inicio + 1.0)
    destino = Path(args.saida) if args.saida else pasta / "video.mp4"

    cena = CenaCarla(cfg_sim, args.fps, args.largura, args.altura, args.camera)
    gravador = None
    codigo = 0
    try:
        cena.conectar()
        cena.carregar_mapa(texto_xodr)
        if modo == "gravacao":
            try:
                cena.reproduzir(gravacao)
            except Exception as erro:  # mapa da gravação não carregou etc.: cai nas poses
                log.warning("A reprodução da gravação falhou (%s); usando as poses da telemetria.", erro)
                cena.encerrar()
                cena = CenaCarla(cfg_sim, args.fps, args.largura, args.altura, args.camera)
                cena.conectar()
                cena.carregar_mapa(None)
                modo = "poses"
        if modo == "poses":
            cena.criar_caminhao_sem_fisica()
        cena.criar_camera()
        gravador = GravadorMP4(destino, args.largura, args.altura, args.fps, args.crf)
        n = renderizar(cena, dados, gravador, modo, inicio, duracao, not args.sem_hud)
        log.info("Vídeo: %s (%d quadros, %.1f s, %dx%d a %d fps, modo %s)", destino, n, n / args.fps, args.largura,
                 args.altura, args.fps, modo)
    except KeyboardInterrupt:
        log.warning("Interrompido (Ctrl+C); o vídeo até aqui fica salvo.")
        codigo = 130
    finally:
        if gravador is not None:
            gravador.fechar()
        cena.encerrar()
    return codigo


if __name__ == "__main__":
    sys.exit(main())
