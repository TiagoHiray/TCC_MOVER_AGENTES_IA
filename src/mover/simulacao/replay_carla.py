"""Replay da volta gravada no CARLA 0.9.16: o gêmeo digital do caminhão, em blocos de 10 s.

O que acontece:
1. A volta é alinhada às vias do mapa .xodr (alinhamento.py), o que dá a pose de cada quadro
   de 20 Hz. O CSV tratado não é alterado. As vias e o trajeto alinhado vão para a página
   (PUT /cena), que desenha com eles o mapa do dashboard.
2. O mundo OpenDRIVE é gerado no CARLA a partir do .xodr e o caminhão (firetruck) é criado sem
   física. A câmera do simulador (spectator) segue o caminhão e uma câmera RGB presa a ele manda
   ~10 quadros/s para o painel 1 da página (POST /camera).
3. O replay abre uma sessão no servidor da camada agêntica (FastAPI), que já analisa o bloco 0.
4. A cada quadro a pose gravada é imposta e o mundo avança 0,05 s (modo síncrono). Ao entrar no
   bloco k, o replay avisa o servidor (POST /sessao/blocos/k): o servidor publica o log e os
   problemas do bloco k e já libera a análise do bloco k+1 (liberação antecipada). Os problemas
   do bloco aparecem escritos na pista, no ponto do pico de jerk, antes de o caminhão chegar lá.
5. A cada 4 quadros (5 Hz) a pose vai para o servidor, que a repassa à página pelo WebSocket.
6. No fim (ou com Ctrl+C) a sessão é encerrada, o CARLA volta ao modo assíncrono e a câmera e o
   caminhão são removidos.

Uso, a partir da pasta src/ do repositório, com o CARLA 0.9.16 aberto:
    python -m mover.servidor.rodar_servidor             # terminal 1: camada agêntica e página
    python -m mover.simulacao.replay_carla              # terminal 2: replay
e, no navegador, http://127.0.0.1:8000/ (simulação, dashboard, log e chat).

Outras formas:
    python -m mover.simulacao.replay_carla --iniciar-servidor            # servidor no mesmo processo
    python -m mover.simulacao.replay_carla --iniciar-servidor --manter-servidor   # página no ar após a volta
    python -m mover.simulacao.replay_carla --iniciar-servidor --injetar-eventos --provedor falso
    python -m mover.simulacao.replay_carla --sem-carla --iniciar-servidor   # sem simulador (teste)
    python -m mover.simulacao.replay_carla --fator-tempo 2 --camera cima
    python -m mover.simulacao.replay_carla --manter-mundo                # usa o mapa já aberto no CARLA
    python -m mover.simulacao.replay_carla --sem-camera-painel           # painel 1 com o mapa 2D
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Protocol

if __package__ in (None, ""):  # execução direta: python src/mover/simulacao/replay_carla.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from mover.agentes.llm import PROVEDORES
from mover.agentes.rodar_agentes import formatar_entrada
from mover.config import caminho, carregar_yaml
from mover.interface.cena import montar_cena
from mover.simulacao.alinhamento import ResultadoAlinhamento
from mover.simulacao.camera_painel import PADRAO_CAMERA_PAINEL, CameraPainel
from mover.simulacao.fila_envio import FilaEnvio
from mover.simulacao.verificar_mapa import alinhar_volta, gravar_saidas

log = logging.getLogger("mover.simulacao")

ROTULO_EVENTO = {"frenagem_brusca": "frenagem brusca", "arrancada_brusca": "arrancada brusca",
                 "soltura_freio": "soltura do freio", "corte_aceleracao": "corte de aceleracao"}


# ---------------------------------------------------------------------------------------------
# Trajeto e relógio
# ---------------------------------------------------------------------------------------------
@dataclass
class Trajeto:
    """Pose imposta ao caminhão em cada quadro, já no referencial do CARLA (x, y em m; yaw em graus)."""

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    correcao_m: np.ndarray
    dist_faixa_m: np.ndarray

    @classmethod
    def de_poses(cls, poses: pd.DataFrame) -> "Trajeto":
        colunas = ("sim_time", "x_carla", "y_carla", "yaw_carla", "correcao_m", "dist_faixa_m")
        return cls(*(poses[c].to_numpy(float) for c in colunas))

    def __len__(self) -> int:
        return len(self.t)

    @cached_property
    def yaw_continuo(self) -> np.ndarray:
        return np.degrees(np.unwrap(np.radians(self.yaw)))

    def pose_em(self, t: float) -> tuple[float, float, float]:
        """(x, y, yaw) no instante t, interpolados entre quadros; yaw em [-180, 180)."""
        yaw = float(np.interp(t, self.t, self.yaw_continuo))
        return float(np.interp(t, self.t, self.x)), float(np.interp(t, self.t, self.y)), (yaw + 180.0) % 360.0 - 180.0

    def quadro_mais_proximo(self, t: float) -> int:
        i = int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))
        if i > 0 and abs(self.t[i - 1] - t) <= abs(self.t[i] - t):
            i -= 1
        return i


class Relogio:
    """Faz a volta andar no ritmo do relógio (vezes fator_tempo) e se reancora depois de esperas.

    Depois de uma espera (análise de um bloco atrasada, CARLA lento) o replay segue do ponto
    em que parou, sem acelerar para "recuperar" o tempo perdido.
    """

    def __init__(self, fator_tempo: float = 1.0, sem_espera: bool = False, tolerancia_s: float = 0.25):
        self.fator = max(float(fator_tempo), 1e-6)
        self.sem_espera = sem_espera
        self.tolerancia_s = tolerancia_s
        self.atrasos = 0
        self._t_sim0 = 0.0
        self._t_real0 = time.monotonic()

    def ancorar(self, t_sim: float) -> None:
        self._t_sim0, self._t_real0 = float(t_sim), time.monotonic()

    def esperar_ate(self, t_sim: float) -> None:
        if self.sem_espera:
            return
        restante = self._t_real0 + (float(t_sim) - self._t_sim0) / self.fator - time.monotonic()
        if restante > 0:
            time.sleep(restante)
        elif restante < -self.tolerancia_s:  # o simulador não acompanhou: segue sem tentar recuperar
            self.atrasos += 1
            self.ancorar(t_sim)


def texto_ascii(texto: str) -> str:
    """O draw_string do CARLA não mostra acentos: 'Atenção' -> 'Atencao'."""
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")


def rotulo_problema(entrada: dict[str, Any]) -> str:
    """Texto curto (ASCII) escrito na pista para um problema do log."""
    nivel = "CRITICO" if entrada.get("nivel") == "critico" else "ATENCAO"
    evento = ROTULO_EVENTO.get(entrada.get("evento", ""), str(entrada.get("evento", "")).replace("_", " "))
    causa = "provavel lombada" if entrada.get("causa") == "irregularidade_via" else "conducao brusca"
    partes = [f"{nivel} {float(entrada.get('t_pico', 0.0)):.1f}s", f"{evento} ({causa})"]
    jerk = (entrada.get("fatos") or {}).get("jerk_max_abs_mps3")
    if jerk is not None:
        partes.append(f"jerk {float(jerk):.1f} m/s3")
    return texto_ascii(" | ".join(partes))


# ---------------------------------------------------------------------------------------------
# Cliente do servidor da camada agêntica
# ---------------------------------------------------------------------------------------------
class ErroServidor(RuntimeError):
    pass


class ClienteAgentes:
    """Fala com o servidor da camada agêntica (httpx.Client, ou o TestClient do FastAPI nos testes).

    Se `http_estado` for informado, as poses vão por uma thread própria com fila curta
    (FilaEnvio): o laço de 20 Hz nunca espera o servidor e, se ele ficar lento, as poses mais
    antigas são descartadas. Sem ele, cada pose é enviada na hora (modo usado nos testes).
    `http_camera` (timeout curto) leva os quadros da câmera do painel; sem ele, vale o `http`.
    """

    def __init__(self, http: Any, http_estado: Any = None, timeout_bloco_s: float = 300.0, proprio: bool = False,
                 http_camera: Any = None):
        self.http = http
        self.timeout_bloco_s = float(timeout_bloco_s)
        self.estados_enviados = 0
        self._falhas_seguidas = 0
        self._proprio = proprio
        self._http_estado = http_estado
        self._http_camera = http_camera
        self._fila_estados: FilaEnvio | None = None
        if http_estado is not None:
            self._fila_estados = FilaEnvio(lambda estado: self._postar_estado(http_estado, estado), "envio_estado", 4)

    @classmethod
    def conectar(cls, url: str, timeout_bloco_s: float = 300.0, prazo_s: float = 10.0) -> "ClienteAgentes":
        import httpx

        # leitura longa: POST /sessao e /sessao/blocos/{k} só voltam quando a análise do bloco termina
        http = httpx.Client(base_url=url, timeout=httpx.Timeout(10.0, read=float(timeout_bloco_s)))
        limite = time.monotonic() + prazo_s
        while True:
            try:
                http.get("/saude").raise_for_status()
                break
            except httpx.HTTPError as erro:
                if time.monotonic() > limite:
                    http.close()
                    raise ErroServidor(f"o servidor da camada agêntica não respondeu em {url} ({erro}). Suba antes "
                                       "com: python -m mover.servidor.rodar_servidor (ou use --iniciar-servidor)") from None
                time.sleep(0.5)
        log.info("Servidor da camada agêntica: %s", url)
        return cls(http, httpx.Client(base_url=url, timeout=httpx.Timeout(2.0)), timeout_bloco_s, proprio=True,
                   http_camera=httpx.Client(base_url=url, timeout=httpx.Timeout(2.0)))

    @property
    def estados_descartados(self) -> int:
        return self._fila_estados.descartados if self._fila_estados is not None else 0

    @staticmethod
    def _conteudo(resposta: Any, acao: str) -> dict[str, Any]:
        if resposta.status_code >= 400:
            try:
                detalhe = resposta.json().get("detail")
            except ValueError:
                detalhe = resposta.text
            if isinstance(detalhe, dict):
                detalhe = detalhe.get("mensagem", detalhe)
            raise ErroServidor(f"{acao}: HTTP {resposta.status_code} - {detalhe}")
        return resposta.json() if resposta.content else {}

    def abrir_sessao(self, opcoes: dict[str, Any]) -> dict[str, Any]:
        """Cria a sessão; volta quando a análise do bloco 0 está pronta."""
        return self._conteudo(self.http.post("/sessao", json=opcoes), "abrir a sessão")

    def entrar_no_bloco(self, k: int) -> dict[str, Any]:
        """Bloqueante: volta com as entradas do bloco k (normalmente já prontas)."""
        return self._conteudo(self.http.post(f"/sessao/blocos/{k}"), f"entrar no bloco {k}")

    def previsao(self) -> dict[str, Any] | None:
        """Próximo bloco já analisado e ainda não percorrido (GET /sessao/previsao), ou None."""
        resposta = self.http.get("/sessao/previsao")
        if resposta.status_code >= 400:
            return None
        return resposta.json().get("previsao")

    def enviar_cena(self, cena: dict[str, Any]) -> bool:
        """Manda à interface as vias e o trajeto alinhado (PUT /cena). Se falhar, a página usa a cena do config."""
        try:
            self._conteudo(self.http.put("/cena", json=cena), "enviar a cena")
            return True
        except Exception as erro:  # a cena é só para a página: o replay continua
            log.warning("Não foi possível mandar a cena à interface (%s); a página usa a do config.", erro)
            return False

    def postar_quadro(self, jpeg: bytes) -> None:
        """Um quadro JPEG da câmera do painel (POST /camera). Chamado pela thread da câmera."""
        resposta = (self._http_camera or self.http).post("/camera", content=jpeg,
                                                         headers={"Content-Type": "image/jpeg"})
        if resposta.status_code >= 400:
            raise ErroServidor(f"HTTP {resposta.status_code} ao mandar o quadro da câmera")

    def enviar_estado(self, estado: dict[str, Any]) -> None:
        if self._fila_estados is None:
            self._postar_estado(self.http, estado)
        else:
            self._fila_estados.colocar(estado)

    def _postar_estado(self, http: Any, estado: dict[str, Any]) -> None:
        try:
            resposta = http.post("/sessao/estado", json=estado)
            if resposta.status_code >= 400:
                raise ErroServidor(f"HTTP {resposta.status_code}")
            self.estados_enviados += 1
            self._falhas_seguidas = 0
        except Exception as erro:  # a pose é só para o dashboard: o replay continua
            self._falhas_seguidas += 1
            if self._falhas_seguidas == 1:
                log.warning("Falha ao enviar a pose ao servidor (%s); o replay continua.", erro)

    def encerrar_sessao(self) -> dict[str, Any] | None:
        try:
            return self._conteudo(self.http.delete("/sessao"), "encerrar a sessão")
        except Exception as erro:
            log.warning("Não foi possível encerrar a sessão: %s", erro)
            return None

    def fechar(self) -> None:
        """Espera as poses que ainda estão na fila e fecha os clientes HTTP criados por `conectar`."""
        if self._fila_estados is not None:
            self._fila_estados.fechar()
            self._fila_estados = None
        if self._proprio:
            for http in (self.http, self._http_estado, self._http_camera):
                if http is not None:
                    http.close()
            self._proprio = False


# ---------------------------------------------------------------------------------------------
# Mundos: CARLA de verdade ou nenhum (--sem-carla)
# ---------------------------------------------------------------------------------------------
class Mundo(Protocol):
    def preparar(self, trajeto: Trajeto, texto_xodr: str | None, telemetria: pd.DataFrame) -> None: ...
    def aplicar(self, i: int) -> None: ...
    def aplicar_em(self, t: float) -> None: ...
    def avancar(self) -> None: ...
    def altura(self, i: int) -> float: ...
    def altura_em(self, t: float) -> float: ...
    def marcar_problemas(self, marcas: list[tuple[int, dict[str, Any]]]) -> None: ...
    def encerrar(self) -> None: ...


def parametros_opendrive(carla: Any, cc: dict[str, Any]) -> Any:
    """Parâmetros da geração do mundo OpenDRIVE (seção carla.opendrive do YAML)."""
    p = cc.get("opendrive", {})
    return carla.OpendriveGenerationParameters(
        vertex_distance=float(p.get("vertex_distance", 2.0)),
        max_road_length=float(p.get("max_road_length", 50.0)),
        wall_height=float(p.get("wall_height", 0.0)),
        additional_width=float(p.get("additional_width", 0.6)),
        smooth_junctions=bool(p.get("smooth_junctions", True)),
        enable_mesh_visibility=bool(p.get("enable_mesh_visibility", True)),
        enable_pedestrian_navigation=bool(p.get("enable_pedestrian_navigation", True)),
    )


class MundoFalso:
    """Sem simulador (--sem-carla): percorre as poses para testar a sessão, o log e o dashboard."""

    def __init__(self) -> None:
        self.trajeto: Trajeto | None = None
        self.quadros = 0
        self.marcas: list[tuple[int, dict[str, Any]]] = []
        self.ultima_pose: tuple[float, float, float] | None = None
        self.encerrado = False

    def preparar(self, trajeto: Trajeto, texto_xodr: str | None, telemetria: pd.DataFrame) -> None:
        self.trajeto = trajeto

    def aplicar(self, i: int) -> None:
        self.ultima_pose = (float(self.trajeto.x[i]), float(self.trajeto.y[i]), float(self.trajeto.yaw[i]))

    def aplicar_em(self, t: float) -> None:
        self.ultima_pose = self.trajeto.pose_em(t)

    def avancar(self) -> None:
        self.quadros += 1

    def altura(self, i: int) -> float:
        return 0.0

    def altura_em(self, t: float) -> float:
        return 0.0

    def marcar_problemas(self, marcas: list[tuple[int, dict[str, Any]]]) -> None:
        self.marcas.extend(marcas)
        for i, entrada in marcas:
            log.debug("Marca na pista (quadro %d): %s", i, rotulo_problema(entrada))

    def encerrar(self) -> None:
        self.encerrado = True


class MundoCarla:
    """Lado CARLA do replay: mundo OpenDRIVE, caminhão sem física, câmeras, marcas e clima."""

    def __init__(self, cfg: dict[str, Any], camera: str | None = None, manter_mundo: bool = False):
        self.cc = cfg.get("carla", {})
        self.cv = cfg.get("veiculo", {})
        self.ccam = dict(cfg.get("camera", {}))
        if camera:
            self.ccam["modo"] = camera
        self.ccam_painel = {**PADRAO_CAMERA_PAINEL, **(cfg.get("camera_painel") or {})}
        self.vida_marcas_s = float(cfg.get("replay", {}).get("vida_marcas_s", 12.0))
        self.manter_mundo = manter_mundo or not self.cc.get("carregar_mapa", True)
        self.carla: Any = None
        self.client: Any = None
        self.world: Any = None
        self.mapa: Any = None
        self.ator: Any = None
        self.spectator: Any = None
        self.camera_painel: CameraPainel | None = None
        self.trajeto: Trajeto | None = None
        self.z_estrada = np.zeros(0)
        self.pitch = np.zeros(0)
        self._enviar_quadro: Callable[[bytes], None] | None = None
        self._config_original: Any = None
        self._base_z = 0.0
        self._altura_ator = 3.0
        self._yaw_camera: float | None = None
        self._camera_posicionada = False

    # ------------------------------------------------------------------ conexão e preparo
    def conectar(self) -> None:
        try:
            import carla
        except ImportError:
            raise SystemExit("O pacote 'carla' não está instalado. Instale o cliente do CARLA 0.9.16 "
                             "(pip install carla==0.9.16, Python 3.12) ou rode com --sem-carla.") from None
        self.carla = carla
        host, porta = self.cc.get("host", "localhost"), int(self.cc.get("porta", 2000))
        self.client = carla.Client(host, porta)
        self.client.set_timeout(float(self.cc.get("timeout_s", 60.0)))
        try:
            versao_servidor = self.client.get_server_version()
        except RuntimeError as erro:
            raise SystemExit(f"O CARLA não respondeu em {host}:{porta} ({erro}). Abra o simulador "
                             "(por exemplo C:\\CARLA_0.9.16\\CarlaUE4.exe) ou rode com --sem-carla.") from None
        versao_cliente = self.client.get_client_version()
        if versao_servidor != versao_cliente:
            log.warning("Versões diferentes: CARLA %s e cliente Python %s.", versao_servidor, versao_cliente)
        log.info("CARLA %s em %s:%d", versao_servidor, host, porta)
        self.world = self.client.get_world()

    def opendrive_atual(self) -> str:
        """O .xodr do mapa já aberto no CARLA (para --manter-mundo sem --mapa)."""
        return self.world.get_map().to_opendrive()

    def preparar(self, trajeto: Trajeto, texto_xodr: str | None, telemetria: pd.DataFrame) -> None:
        carla = self.carla
        self.trajeto = trajeto
        if not self.manter_mundo:
            log.info("Gerando o mundo OpenDRIVE no CARLA (pode levar alguns segundos)...")
            self.world = self.client.generate_opendrive_world(texto_xodr, parametros_opendrive(carla, self.cc))
        self._config_original = self.world.get_settings()
        config = self.world.get_settings()
        config.synchronous_mode = True
        config.fixed_delta_seconds = float(self.cc.get("passo_s", 0.05))
        self.world.apply_settings(config)
        self.mapa = self.world.get_map()
        self._alturas(trajeto)
        if self.cc.get("clima_da_gravacao", True):
            self._clima(telemetria)
        self._criar_caminhao(trajeto)
        self._criar_camera_painel()
        self.spectator = self.world.get_spectator()
        self.aplicar(0)
        self.world.tick()

    def ligar_camera_painel(self, enviar: Callable[[bytes], None]) -> None:
        """Liga a câmera de perseguição do painel 1; `enviar` recebe cada quadro JPEG (chamar antes de preparar)."""
        self._enviar_quadro = enviar

    def _criar_camera_painel(self) -> None:
        if self._enviar_quadro is None or not self.ccam_painel.get("ativa", True):
            return
        painel = CameraPainel(self.ccam_painel, self._enviar_quadro)
        try:
            if painel.criar(self.carla, self.world, self.ator):
                self.camera_painel = painel
        except Exception as erro:  # blueprint ausente, RuntimeError do CARLA...: o replay segue sem a câmera
            log.warning("Câmera do painel desligada (%s); a página mostra o mapa 2D.", erro)
            painel.encerrar()

    def _alturas(self, trajeto: Trajeto) -> None:
        """Altura do asfalto sob cada pose e o pitch correspondente (mapas planos dão zero)."""
        carla = self.carla
        z = np.zeros(len(trajeto))
        for i in range(len(trajeto)):
            local = carla.Location(x=float(trajeto.x[i]), y=float(trajeto.y[i]), z=0.0)
            wp = self.mapa.get_waypoint(local, project_to_road=True, lane_type=carla.LaneType.Any)
            z[i] = wp.transform.location.z if wp is not None else (z[i - 1] if i else 0.0)
        z = uniform_filter1d(z, size=20, mode="nearest")  # ~1 s
        ds = np.hypot(np.gradient(trajeto.x), np.gradient(trajeto.y))
        pitch = np.degrees(np.arctan2(np.gradient(z), np.maximum(ds, 1e-3)))
        pitch[ds < 0.02] = 0.0  # parado
        self.z_estrada, self.pitch = z, np.clip(pitch, -15.0, 15.0)

    def _clima(self, telemetria: pd.DataFrame) -> None:
        clima = self.world.get_weather()
        mudou = []
        for coluna, atributo in (("sun_altitude", "sun_altitude_angle"), ("cloudiness", "cloudiness"),
                                 ("precipitation", "precipitation")):
            if coluna in telemetria and telemetria[coluna].notna().any():
                valor = float(telemetria[coluna].mean())
                setattr(clima, atributo, valor)
                mudou.append(f"{atributo}={valor:.1f}")
        if mudou:
            self.world.set_weather(clima)
            log.info("Clima da gravação: %s", ", ".join(mudou))

    def _criar_caminhao(self, trajeto: Trajeto) -> None:
        carla = self.carla
        biblioteca = self.world.get_blueprint_library()
        nome = self.cv.get("blueprint", "vehicle.carlamotors.firetruck")
        try:
            bp = biblioteca.find(nome)
        except Exception:
            alternativas = list(biblioteca.filter("vehicle.carlamotors.*")) or list(biblioteca.filter("vehicle.*"))
            bp = alternativas[0]
            log.warning("Blueprint %s não encontrado; usando %s.", nome, bp.id)
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", str(self.cv.get("role_name", "mover_gemeo")))
        for extra in (0.5, 2.0, 5.0):  # o asfalto às vezes colide com a caixa do veículo no spawn
            inicio = carla.Transform(carla.Location(x=float(trajeto.x[0]), y=float(trajeto.y[0]),
                                                    z=float(self.z_estrada[0]) + extra),
                                     carla.Rotation(yaw=float(trajeto.yaw[0])))
            self.ator = self.world.try_spawn_actor(bp, inicio)
            if self.ator is not None:
                break
        if self.ator is None:
            raise RuntimeError("não foi possível criar o caminhão no início da volta (local ocupado?)")
        self.ator.set_simulate_physics(False)  # a pose é imposta a cada quadro (replay open-loop)
        caixa = self.ator.bounding_box
        self._base_z = float(caixa.location.z - caixa.extent.z)  # fundo da caixa em relação à origem do ator
        self._altura_ator = float(2 * caixa.extent.z)
        log.info("Caminhão %s criado (id %d), sem física.", self.ator.type_id, self.ator.id)

    # ------------------------------------------------------------------ quadro a quadro
    def altura(self, i: int) -> float:
        return float(self.z_estrada[i])

    def altura_em(self, t: float) -> float:
        return float(np.interp(t, self.trajeto.t, self.z_estrada))

    def aplicar(self, i: int) -> None:
        tr = self.trajeto
        self._impor(float(tr.x[i]), float(tr.y[i]), float(self.z_estrada[i]), float(tr.yaw[i]), float(self.pitch[i]))

    def aplicar_em(self, t: float) -> None:
        """Pose no instante t do trajeto, interpolada entre quadros (o caminhão Y anda no relógio do plano X)."""
        x, y, yaw = self.trajeto.pose_em(t)
        self._impor(x, y, self.altura_em(t), yaw, float(np.interp(t, self.trajeto.t, self.pitch)))

    def _impor(self, x: float, y: float, z_estrada: float, yaw: float, pitch: float) -> None:
        carla = self.carla
        z = z_estrada - self._base_z + float(self.cv.get("altura_extra_m", 0.05))
        transformacao = carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0))
        self.ator.set_transform(transformacao)
        self._camera(x, y, z_estrada, yaw)

    def avancar(self) -> None:
        self.world.tick()

    def _camera(self, x: float, y: float, z: float, yaw: float) -> None:
        carla = self.carla
        modo = self.ccam.get("modo", "perseguicao")
        if modo == "livre" and self._camera_posicionada:
            return
        if modo == "cima":  # norte para cima na tela
            alvo = carla.Transform(carla.Location(x=x, y=y, z=z + float(self.ccam.get("altura_cima_m", 80))),
                                   carla.Rotation(pitch=-90.0, yaw=-90.0))
        else:  # perseguição, com o giro da câmera suavizado (~0,3 s)
            if self._yaw_camera is None:
                self._yaw_camera = yaw
            else:
                self._yaw_camera += 0.15 * ((yaw - self._yaw_camera + 180.0) % 360.0 - 180.0)
            r = math.radians(self._yaw_camera)
            d, h = float(self.ccam.get("distancia_m", 16)), float(self.ccam.get("altura_m", 8))
            alvo = carla.Transform(carla.Location(x=x - d * math.cos(r), y=y - d * math.sin(r), z=z + h),
                                   carla.Rotation(pitch=float(self.ccam.get("inclinacao_graus", -20)), yaw=self._yaw_camera))
        self.spectator.set_transform(alvo)
        self._camera_posicionada = True

    def marcar_problemas(self, marcas: list[tuple[int, dict[str, Any]]]) -> None:
        carla, tr = self.carla, self.trajeto
        for i, entrada in marcas:
            cor = carla.Color(255, 0, 0) if entrada.get("nivel") == "critico" else carla.Color(255, 190, 0)
            base = carla.Location(x=float(tr.x[i]), y=float(tr.y[i]), z=float(self.z_estrada[i]) + 0.3)
            topo = carla.Location(x=base.x, y=base.y, z=base.z + self._altura_ator + 2.0)
            self.world.debug.draw_point(base, size=0.3, color=cor, life_time=self.vida_marcas_s)
            self.world.debug.draw_line(base, topo, thickness=0.06, color=cor, life_time=self.vida_marcas_s)
            self.world.debug.draw_string(topo, rotulo_problema(entrada), draw_shadow=True, color=cor,
                                         life_time=self.vida_marcas_s)

    def encerrar(self) -> None:
        """Devolve o CARLA ao modo assíncrono (senão o simulador fica parado), desliga a câmera do
        painel e remove o caminhão, nesta ordem (a câmera está presa ao caminhão). Idempotente."""
        if self.world is not None and self._config_original is not None:
            try:
                self.world.apply_settings(self._config_original)
            except RuntimeError as erro:
                log.warning("Não foi possível restaurar as configurações do CARLA: %s", erro)
            self._config_original = None
        if self.camera_painel is not None:
            self.camera_painel.encerrar()
            self.camera_painel = None
        if self.ator is not None:
            try:
                self.ator.destroy()
            except RuntimeError:
                pass
            self.ator = None


# ---------------------------------------------------------------------------------------------
# Laço do replay
# ---------------------------------------------------------------------------------------------
@dataclass
class ResultadoReplay:
    quadros: int = 0
    estados: int = 0
    blocos: list[dict[str, Any]] = field(default_factory=list)
    entradas: list[dict[str, Any]] = field(default_factory=list)
    interrompido: bool = False
    duracao_real_s: float = 0.0
    atrasos_relogio: int = 0

    def para_json(self) -> dict[str, Any]:
        dados = asdict(self)
        dados.pop("entradas")
        dados["n_entradas"] = len(self.entradas)
        dados["n_problemas"] = sum(e["tipo"] == "problema" for e in self.entradas)
        dados["maior_espera_s"] = max((b["espera_s"] for b in self.blocos[1:]), default=0.0)
        return dados


def mostrar_bloco(resposta: dict[str, Any], t0: float, duracao: float, imprimir: Callable[[str], None],
                  silencioso: bool) -> None:
    k, entradas = resposta["bloco"], resposta["entradas"]
    n_problemas = sum(e["tipo"] == "problema" for e in entradas)
    imprimir(f"=== Bloco {k:02d} ({t0 + k * duracao:.0f}-{t0 + (k + 1) * duracao:.0f} s): {len(entradas)} entrada(s), "
             f"{n_problemas} problema(s); espera {resposta['espera_s']:.2f} s, análise {resposta['latencia_s']:.2f} s ===")
    if not silencioso:
        for entrada in entradas:
            imprimir(formatar_entrada(entrada))


def _imprimir(texto: str) -> None:
    print(texto, flush=True)  # sem buffer: as linhas do log não se misturam com as do logging


def rodar_replay(trajeto: Trajeto, cliente: ClienteAgentes, mundo: Mundo, sessao: dict[str, Any], *,
                 fator_tempo: float = 1.0, sem_espera: bool = False, estado_a_cada: int = 4,
                 marcar_problemas: bool = True, silencioso: bool = False,
                 imprimir: Callable[[str], None] = _imprimir) -> ResultadoReplay:
    """Percorre a volta quadro a quadro, avisando o servidor a cada bloco novo.

    `sessao` é a resposta de POST /sessao: os blocos seguem t0 e duracao_bloco_s da camada
    agêntica, para o replay e o servidor contarem os blocos do mesmo jeito.
    """
    n_blocos, duracao, t0 = int(sessao["n_blocos"]), float(sessao["duracao_bloco_s"]), float(sessao["t0"])
    blocos = np.clip(np.floor((trajeto.t - t0) / duracao + 1e-9).astype(int), 0, n_blocos - 1)
    relogio = Relogio(fator_tempo, sem_espera)
    resultado = ResultadoReplay()
    inicio = time.monotonic()
    anunciado = -1
    relogio.ancorar(trajeto.t[0])
    try:
        for i in range(len(trajeto)):
            k = int(blocos[i])
            if k > anunciado:
                for kk in range(anunciado + 1, k + 1):  # normalmente só kk = k
                    resposta = cliente.entrar_no_bloco(kk)
                    entradas = resposta["entradas"]
                    resultado.entradas.extend(entradas)
                    resultado.blocos.append({"bloco": kk, "espera_s": resposta["espera_s"],
                                             "latencia_s": resposta["latencia_s"], "n_entradas": len(entradas),
                                             "n_problemas": sum(e["tipo"] == "problema" for e in entradas)})
                    mostrar_bloco(resposta, t0, duracao, imprimir, silencioso)
                    marcas = [(trajeto.quadro_mais_proximo(float(e["t_pico"])), e) for e in entradas
                              if e["tipo"] == "problema" and e.get("t_pico") is not None]
                    if marcar_problemas and marcas:
                        mundo.marcar_problemas(marcas)
                anunciado = k
                relogio.ancorar(trajeto.t[i])  # a espera pela análise não conta como atraso
            relogio.esperar_ate(trajeto.t[i])
            mundo.aplicar(i)
            mundo.avancar()
            resultado.quadros += 1
            if estado_a_cada > 0 and i % estado_a_cada == 0:
                cliente.enviar_estado({
                    "sim_time": float(trajeto.t[i]),
                    "x": round(float(trajeto.x[i]), 3), "y": round(float(trajeto.y[i]), 3),
                    "z": round(mundo.altura(i), 3), "yaw": round(float(trajeto.yaw[i]), 2),
                    "correcao_m": round(float(trajeto.correcao_m[i]), 3),
                    "dist_faixa_m": round(float(trajeto.dist_faixa_m[i]), 3),
                })
                resultado.estados += 1
    except KeyboardInterrupt:
        resultado.interrompido = True
        imprimir("Replay interrompido (Ctrl+C).")
    resultado.duracao_real_s = round(time.monotonic() - inicio, 3)
    resultado.atrasos_relogio = relogio.atrasos
    return resultado


# ---------------------------------------------------------------------------------------------
# Linha de comando
# ---------------------------------------------------------------------------------------------
def mostrar_alinhamento(res: ResultadoAlinhamento) -> None:
    t = res.transformacao.para_json()
    etapa = list(res.estatisticas)[-1]
    est = res.estatisticas[etapa]
    pior = max(res.por_bloco, key=lambda b: b["correcao_max_m"])
    log.info("Alinhamento (%s): rotação %.2f°, translação (%.1f, %.1f) m; distância ao centro da faixa: mediana %.2f m, "
             "máx %.2f m, %.0f%% dos quadros dentro da faixa", res.georreferencia["metodo"], t["rotacao_graus"],
             t["tx_m"], t["ty_m"], est["mediana_m"], est["max_m"], 100 * est["frac_dentro_da_faixa"])
    if "correcao_borda" in res.estatisticas:
        frac = float((res.poses["correcao_m"] > 0.05).mean())
        log.info("Correção de borda: máx %.2f m (bloco %02d), %.0f%% dos quadros corrigidos",
                 pior["correcao_max_m"], pior["bloco"], 100 * frac)
    for aviso in res.avisos:
        log.warning(aviso)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay da volta gravada no CARLA 0.9.16 com a camada agêntica.")
    parser.add_argument("--config", default="config/simulacao.yaml", help="YAML da simulação")
    parser.add_argument("--config-agentes", default="config/agentes.yaml", help="YAML da camada agêntica")
    parser.add_argument("--mapa", help="arquivo .xodr (padrão: YAML; com --manter-mundo, o mapa já aberto no CARLA)")
    parser.add_argument("--telemetria", help="CSV tratado com as poses (padrão: YAML)")
    parser.add_argument("--sem-carla", action="store_true", help="não usa o simulador: percorre as poses e fala com o servidor")
    parser.add_argument("--manter-mundo", action="store_true", help="não gera o mundo OpenDRIVE; usa o que está aberto")
    parser.add_argument("--camera", choices=("perseguicao", "cima", "livre"), help="câmera do simulador (padrão: YAML)")
    parser.add_argument("--servidor", help="URL do servidor da camada agêntica (padrão: host/porta do YAML)")
    parser.add_argument("--iniciar-servidor", action="store_true", help="sobe o servidor da camada agêntica neste processo")
    parser.add_argument("--injetar-eventos", action="store_true", help="soma a frenagem e a arrancada sintéticas")
    parser.add_argument("--provedor", choices=PROVEDORES, help="provedor do LLM (padrão: .env ou YAML)")
    parser.add_argument("--modelo", help="modelo no provedor (padrão: .env ou YAML)")
    parser.add_argument("--sem-ml", action="store_true", help="desliga o especialista de ML")
    parser.add_argument("--fator-tempo", type=float, help="2 = duas vezes mais rápido que o real (padrão: YAML)")
    parser.add_argument("--sem-espera", action="store_true", help="não segue o relógio (o mais rápido possível)")
    parser.add_argument("--offset-x", type=float, default=0.0, help="desloca a volta no mapa (m, leste)")
    parser.add_argument("--offset-y", type=float, default=0.0, help="desloca a volta no mapa (m, norte do mapa)")
    parser.add_argument("--rotacao", type=float, default=0.0, help="gira a volta no mapa (graus, anti-horário)")
    parser.add_argument("--sem-icp", action="store_true", help="só a georreferência, sem o ajuste fino")
    parser.add_argument("--sem-correcao-borda", action="store_true", help="não puxa a pose para dentro da faixa")
    parser.add_argument("--silencioso", action="store_true", help="não imprime cada entrada do log")
    parser.add_argument("--sem-camera-painel", action="store_true",
                        help="não cria a câmera de perseguição do painel 1 (a página mostra o mapa 2D)")
    parser.add_argument("--manter-servidor", action="store_true",
                        help="com --iniciar-servidor: mantém a página e o chat no ar depois da volta, até o Ctrl+C")
    args = parser.parse_args(argv)

    from mover.servidor.rodar_servidor import configurar_logs

    configurar_logs()
    if args.manter_servidor and not args.iniciar_servidor:
        log.warning("--manter-servidor só vale com --iniciar-servidor (o servidor separado já fica no ar).")
    cfg = carregar_yaml(args.config)
    if args.sem_icp:
        cfg.setdefault("alinhamento", {}).setdefault("icp", {})["ativo"] = False
    if args.sem_correcao_borda:
        cfg.setdefault("alinhamento", {}).setdefault("correcao_borda", {})["ativa"] = False
    cr, cs = cfg.get("replay", {}), cfg.get("servidor", {})
    fator_tempo = float(args.fator_tempo or cr.get("fator_tempo", 1.0))
    mundo: Mundo = MundoFalso() if args.sem_carla else MundoCarla(cfg, args.camera, args.manter_mundo)
    servidor = thread = None
    cliente: ClienteAgentes | None = None
    sessao: dict[str, Any] | None = None
    url: str | None = None
    codigo = 0
    try:
        host, porta = cs.get("host", "127.0.0.1"), int(cs.get("porta", 8000))
        if args.iniciar_servidor:
            from mover.servidor.app import criar_app
            from mover.servidor.rodar_servidor import iniciar_em_thread

            servidor, thread = iniciar_em_thread(criar_app(carregar_yaml(args.config_agentes), cfg), host, porta)
        url = args.servidor or f"http://{host}:{porta}"
        cliente = ClienteAgentes.conectar(url, float(cs.get("timeout_bloco_s", 300)))
        log.info("Painel (simulação, dashboard, log e chat): %s/", url)

        texto_xodr = None
        if isinstance(mundo, MundoCarla):
            mundo.conectar()
            if args.manter_mundo and not args.mapa:
                texto_xodr = mundo.opendrive_atual()
                log.info("Usando o mapa já aberto no CARLA (%s).", mundo.world.get_map().name)
        telemetria = pd.read_csv(caminho(args.telemetria or cfg["telemetria"]["arquivo"]))
        ajuste = {"offset_x": args.offset_x, "offset_y": args.offset_y, "rotacao_graus": args.rotacao}
        mapa, alinhamento, texto_xodr = alinhar_volta(cfg, args.mapa, telemetria, ajuste, texto_xodr)
        mostrar_alinhamento(alinhamento)
        gravar_saidas(cfg, mapa, alinhamento, figura=False)
        trajeto = Trajeto.de_poses(alinhamento.poses)
        cliente.enviar_cena(montar_cena(mapa, alinhamento))  # a página desenha as mesmas vias e poses do replay

        if isinstance(mundo, MundoCarla) and not args.sem_camera_painel:
            mundo.ligar_camera_painel(cliente.postar_quadro)
        mundo.preparar(trajeto, texto_xodr, telemetria)
        log.info("Abrindo a sessão na camada agêntica (o bloco 0 é analisado antes da partida)...")
        sessao = cliente.abrir_sessao({"injetar_eventos": args.injetar_eventos, "provedor": args.provedor,
                                       "modelo": args.modelo, "sem_ml": args.sem_ml, "fator_tempo": fator_tempo,
                                       "tempo_real": not args.sem_espera})
        log.info("Sessão %s: %d blocos de %.0f s, LLM %s, especialista de ML %s. Partindo (fator de tempo %.1fx).",
                 sessao["id"], sessao["n_blocos"], sessao["duracao_bloco_s"], sessao["llm"],
                 "ligado" if sessao["especialista_ml"] else "desligado", fator_tempo)
        resultado = rodar_replay(trajeto, cliente, mundo, sessao, fator_tempo=fator_tempo, sem_espera=args.sem_espera,
                                 estado_a_cada=int(cr.get("estado_a_cada_ticks", 4)),
                                 marcar_problemas=bool(cr.get("marcar_problemas", True)), silencioso=args.silencioso)
        relatorio = {**resultado.para_json(), "sessao": sessao["id"], "fator_tempo": fator_tempo,
                     "estados_enviados": cliente.estados_enviados, "estados_descartados": cliente.estados_descartados}
        painel = getattr(mundo, "camera_painel", None)
        if painel is not None:
            relatorio["camera_painel"] = {"recebidos": painel.recebidos, "enviados": painel.enviados,
                                          "descartados": painel.descartados}
        destino = caminho(cfg["alinhamento"].get("saida", {}).get("pasta", "data/simulacao")) / f"replay_{sessao['id']}.json"
        destino.write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("Replay: %d quadros em %.1f s de relógio; maior espera pela análise %.2f s; relatório em %s",
                 resultado.quadros, resultado.duracao_real_s, relatorio["maior_espera_s"], destino)
        if resultado.interrompido:
            codigo = 130
    except KeyboardInterrupt:
        log.warning("Interrompido (Ctrl+C).")
        codigo = 130
    except ErroServidor as erro:
        log.error("%s", erro)
        codigo = 1
    finally:
        mundo.encerrar()  # primeiro o CARLA: em modo síncrono ele fica parado esperando o próximo tick
        if cliente is not None:
            if sessao is not None:
                fim = cliente.encerrar_sessao()
                if fim:
                    print(json.dumps(fim["resumo"], ensure_ascii=False, indent=2))
                    log.info("Log da sessão: %s", fim["sessao"]["arquivo"])
            cliente.fechar()
        if servidor is not None:
            from mover.servidor.rodar_servidor import parar_servidor

            if args.manter_servidor and codigo == 0:
                esperar_ctrl_c(thread, url)
            parar_servidor(servidor, thread)
    return codigo


def esperar_ctrl_c(thread: Any, url: str | None) -> None:
    """Deixa o servidor (thread daemon) no ar até o Ctrl+C: a página e o chat continuam funcionando."""
    log.info("Volta concluída. Painel e chat seguem em %s/ (Ctrl+C para sair).", url)
    try:
        while thread.is_alive():
            thread.join(timeout=0.5)
    except KeyboardInterrupt:
        log.info("Encerrando o servidor...")


if __name__ == "__main__":
    sys.exit(main())
