"""Câmera RGB de perseguição para o painel 1 da interface (só com o CARLA).

Um `sensor.camera.rgb` fica preso ao caminhão (attach_to, AttachmentType.Rigid), atrás e acima
dele. Cada quadro chega em BGRA numa thread do cliente do CARLA e só é copiado para uma fila
curta (FilaEnvio). A thread da fila converte o quadro em JPEG (Pillow) e o manda ao servidor
(POST /camera), que o repassa à página como MJPEG. Com a fila cheia, o quadro mais antigo é
descartado: o laço de 20 Hz do replay nunca espera a câmera.

O sensor_tick vale em tempo de simulação: 10 fps com fator de tempo 1 dão ~10 quadros por segundo
de relógio. Se algo falhar ao criar a câmera, o replay segue sem ela e a página mostra o mapa 2D.
"""

from __future__ import annotations

import importlib.util
import io
import logging
from typing import Any, Callable

import numpy as np

from mover.simulacao.fila_envio import FilaEnvio

log = logging.getLogger("mover.simulacao")

PADRAO_CAMERA_PAINEL: dict[str, Any] = {
    "ativa": True,
    "largura": 640,
    "altura": 360,
    "fps": 10.0,
    "fov": 90.0,
    "distancia_m": 12.0,     # atrás do centro do caminhão
    "altura_m": 5.0,
    "inclinacao_graus": -15.0,
    "qualidade_jpeg": 70,
}


def codificar_jpeg(bgra: bytes, largura: int, altura: int, qualidade: int = 70) -> bytes:
    """Quadro BGRA de 8 bits (formato do carla.Image.raw_data) -> JPEG RGB."""
    from PIL import Image

    pixels = np.frombuffer(bgra, dtype=np.uint8).reshape(int(altura), int(largura), 4)
    rgb = np.ascontiguousarray(pixels[:, :, 2::-1])  # BGRA -> RGB
    saida = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(saida, format="JPEG", quality=int(qualidade))
    return saida.getvalue()


class CameraPainel:
    def __init__(self, cfg: dict[str, Any], enviar: Callable[[bytes], None]):
        self.cfg = {**PADRAO_CAMERA_PAINEL, **(cfg or {})}
        self.enviar = enviar
        self.sensor: Any = None
        self.recebidos = 0
        self.enviados = 0
        self._fila: FilaEnvio | None = None
        self._falhas = 0

    def criar(self, carla: Any, world: Any, ator: Any) -> bool:
        """Cria o sensor preso ao caminhão e começa a escutar. Devolve False se a câmera ficar desligada."""
        if importlib.util.find_spec("PIL") is None:
            log.warning("Câmera do painel desligada: instale o Pillow (pip install pillow). A página mostra o mapa 2D.")
            return False
        c = self.cfg
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(int(c["largura"])))
        bp.set_attribute("image_size_y", str(int(c["altura"])))
        bp.set_attribute("fov", str(float(c["fov"])))
        bp.set_attribute("sensor_tick", f"{1.0 / max(float(c['fps']), 0.1):.3f}")
        posicao = carla.Transform(carla.Location(x=-float(c["distancia_m"]), y=0.0, z=float(c["altura_m"])),
                                  carla.Rotation(pitch=float(c["inclinacao_graus"])))
        self.sensor = world.spawn_actor(bp, posicao, attach_to=ator, attachment_type=carla.AttachmentType.Rigid)
        self._fila = FilaEnvio(self._codificar_e_enviar, "camera_painel", tamanho=2)
        self.sensor.listen(self._ao_receber)
        log.info("Câmera do painel: %dx%d a %.0f fps, %.0f m atrás e %.0f m acima do caminhão.",
                 int(c["largura"]), int(c["altura"]), float(c["fps"]), float(c["distancia_m"]), float(c["altura_m"]))
        return True

    @property
    def descartados(self) -> int:
        return self._fila.descartados if self._fila is not None else 0

    def _ao_receber(self, imagem: Any) -> None:
        """Thread do cliente do CARLA: só copia o quadro (o raw_data deixa de valer depois do callback)."""
        self.recebidos += 1
        if self._fila is not None:
            self._fila.colocar((bytes(imagem.raw_data), int(imagem.width), int(imagem.height)))

    def _codificar_e_enviar(self, item: tuple[bytes, int, int]) -> None:
        bgra, largura, altura = item
        try:
            self.enviar(codificar_jpeg(bgra, largura, altura, int(self.cfg["qualidade_jpeg"])))
            self.enviados += 1
            self._falhas = 0
        except Exception as erro:  # o painel é só para ver: o replay continua
            self._falhas += 1
            if self._falhas == 1:
                log.warning("Falha ao mandar o quadro da câmera ao servidor (%s); o replay continua.", erro)

    def encerrar(self) -> None:
        """Para de escutar, destrói o sensor e fecha a fila (idempotente)."""
        if self.sensor is not None:
            for acao in ("stop", "destroy"):
                try:
                    getattr(self.sensor, acao)()
                except RuntimeError as erro:
                    log.warning("Câmera do painel: %s falhou (%s).", acao, erro)
            self.sensor = None
        if self._fila is not None:
            self._fila.fechar()
            self._fila = None
