"""Lado CARLA da corrida autônoma: o caminhão dá a volta sozinho com o Traffic Manager.

O Traffic Manager (TM) do CARLA é o piloto automático: ele esterça e controla acelerador e freio
para seguir um caminho e uma velocidade desejada. Não é aprendizado de máquina: são regras de
navegação e controladores PID. A camada agêntica não mexe nos pedais; ela só muda a velocidade
desejada mandada ao TM a cada tick (TrafficManager.set_desired_speed), por meio do governador.

O que este módulo faz:
1. gera o mundo OpenDRIVE do campus a partir do .xodr (o mesmo mapa_final.xodr), com física;
2. cria o caminhão configurado no início da rota do plano de missão e liga o piloto automático;
3. passa ao TM o caminho da volta (pontos da rota a cada ~10 m, no sentido de tráfego do mapa);
4. a cada tick: manda a velocidade desejada, avança o mundo (modo síncrono, 20 Hz) e lê a
   telemetria (posição, velocidade, aceleração, pedais), projetando a posição na rota para saber
   a distância percorrida;
5. a câmera do simulador (spectator) segue o caminhão: com o CARLA aberto em janela, dá para ver
   o caminhão andando. Uma câmera RGB presa a ele alimenta o painel 1 da página;
6. opcionalmente grava a corrida com o gravador do CARLA (start_recorder), para o vídeo MP4.

Atenção: o servidor da página usa a porta 8000; por isso o TM usa outra (8100, config).
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np

from mover.autonomo.conducao import FimDoSimulador
from mover.autonomo.rota import Rota, de_carla, para_carla
from mover.config import caminho
# from mover.autonomo.xodr_carla import preparar_para_carla  # preparos do mapa_final.xodr (desligados)
from mover.simulacao.camera_painel import PADRAO_CAMERA_PAINEL, CameraPainel

log = logging.getLogger("mover.autonomo")

PADRAO_TM: dict[str, Any] = {
    "porta_tm": 8100,            # a 8000 é do servidor da página
    "semente": 7,
    "passo_caminho_m": 10.0,     # distância entre os pontos do caminho passado ao TM
    "inicio_caminho_m": 8.0,     # o caminho começa um pouco à frente do caminhão
    "horizonte_caminho_m": 6000.0,  # o caminho da corrida inteira vai de uma vez; só corridas maiores são renovadas
    "distancia_lider_m": 4.0,
    "clima": "ClearNoon",
    "acomodar_ticks": 20,        # ticks com freio de mão depois de criar o caminhão (assentar na pista)
    "telemetria_motor": True,    # get_telemetry_data a cada tick: rpm e a marcha de verdade (diagnóstico)
    "max_tick_s": 90.0,          # tick que não volta nesse tempo de relógio: o CARLA ou o TM travou
}


class VigiaTick:
    """Encerra o processo se um tick não voltar: o TM em modo síncrono roda neste processo e, se travar,
    world.tick() espera para sempre (nem o Ctrl+C funciona)."""

    def __init__(self, limite_s: float, ao_travar: Callable[[float], None]):
        self.limite_s = float(limite_s)
        self.ao_travar = ao_travar
        self._inicio: float | None = None
        self._parar = threading.Event()
        self._thread = threading.Thread(target=self._vigiar, name="vigia_tick", daemon=True)
        self._thread.start()

    def comecar(self) -> None:
        self._inicio = time.monotonic()

    def terminar(self) -> None:
        self._inicio = None

    def _vigiar(self) -> None:
        while not self._parar.wait(1.0):
            inicio = self._inicio
            if inicio is not None and time.monotonic() - inicio > self.limite_s:
                self.ao_travar(time.monotonic() - inicio)
                return

    def parar(self) -> None:
        self._parar.set()


def _encerrar_processo_travado(espera_s: float) -> None:
    log.critical("O tick do CARLA não voltou em %.0f s: o simulador ou o Traffic Manager travou. Feche o CARLA "
                 "(Get-Process CarlaUE4* | Stop-Process), abra de novo e rode outra vez.", espera_s)
    logging.shutdown()
    os._exit(3)


def _escalar(a: Any, b: Any) -> float:
    return float(a.x * b.x + a.y * b.y + a.z * b.z)


def carregar_mapa_instalado(client: Any, nome: str, pasta_xodr: Path | None = None) -> Any:
    """Carrega um mapa instalado no CARLA (nome curto ou caminho /Game/...) e salva o .xodr dele em `pasta_xodr`."""
    disponiveis = [str(m) for m in client.get_available_maps()]
    alvo = next((m for m in disponiveis if m == nome or m.rsplit("/", 1)[-1] == nome), None)
    if alvo is None:
        nomes = ", ".join(sorted(m.rsplit("/", 1)[-1] for m in disponiveis))
        raise SystemExit(f"O mapa '{nome}' não está instalado neste CARLA (mapas disponíveis: {nomes}). Instale o pacote do "
                         "mapa (campus_maua_pkg, cozido para o CARLA 0.9.16) ou deixe carla.mapa_carla vazio em "
                         "config/simulacao.yaml para gerar o mundo a partir do .xodr.")
    log.info("Carregando o mapa %s do CARLA (pode levar alguns segundos)...", alvo)
    world = client.load_world(alvo)
    if pasta_xodr is not None:
        texto = world.get_map().to_opendrive()
        destino = Path(pasta_xodr) / f"{alvo.rsplit('/', 1)[-1]}_carla.xodr"
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(texto, encoding="utf-8")
        log.info("OpenDRIVE do mapa do CARLA (%d vias) salvo em %s.", texto.count("<road "), destino)
    return world


class SimuladorCarla:
    """Mesma interface do SimuladorSubstituto: iniciar, reiniciar, passo(v_desejada) e encerrar."""

    nome = "carla"

    def __init__(self, rota: Rota, cfg_simulacao: dict[str, Any], cfg_tm: dict[str, Any] | None = None,
                 dt: float = 0.05, texto_xodr: str | None = None, sem_render: bool = False,
                 manter_mundo: bool = False, voltas: int = 1, camera: str | None = None):
        self.rota = rota
        self.dt = float(dt)
        self.cc = cfg_simulacao.get("carla", {})
        self.cv = cfg_simulacao.get("veiculo", {})
        self.ccam = dict(cfg_simulacao.get("camera", {}))
        if camera:
            self.ccam["modo"] = camera
        self.ccam_painel = {**PADRAO_CAMERA_PAINEL, **(cfg_simulacao.get("camera_painel") or {})}
        self.ctm = {**PADRAO_TM, **(cfg_tm or {})}
        self.texto_xodr = texto_xodr
        self.sem_render = sem_render
        self.manter_mundo = manter_mundo or not self.cc.get("carregar_mapa", True)
        self.voltas = int(voltas)
        self.carla: Any = None
        self.client: Any = None
        self.world: Any = None
        self.mapa: Any = None
        self.tm: Any = None
        self.ator: Any = None
        self.spectator: Any = None
        self.camera_painel: CameraPainel | None = None
        self.gravacao: str | None = None
        self._enviar_quadro: Callable[[bytes], None] | None = None
        self._config_original: Any = None
        self._t_inicio = 0.0
        self._d = 0.0
        self._caminho_ate = 0.0
        self._yaw_camera: float | None = None
        self._ultima: dict[str, Any] = {}
        self.topologia: dict[str, Any] | None = None
        self.ajustes_mapa: dict[str, Any] = {}
        self._alturas: Any = None
        self.fisica: dict[str, Any] = {}
        self._vigia: VigiaTick | None = None
        self._horizonte_caminho = 0.0
        self.frame = 0
        self.hora_inicio = datetime.now().astimezone()

    # ------------------------------------------------------------------ conexão e mundo
    def conectar(self) -> None:
        try:
            import carla
        except ImportError:
            raise SystemExit("O pacote 'carla' não está instalado. Instale o cliente do CARLA 0.9.16 (o .whl de "
                             "PythonAPI\\carla\\dist, Python 3.12) ou rode com --sem-carla.") from None
        self.carla = carla
        host, porta = self.cc.get("host", "localhost"), int(self.cc.get("porta", 2000))
        self.client = carla.Client(host, porta)
        self.client.set_timeout(float(self.cc.get("timeout_s", 60.0)))
        try:
            versao_servidor = self.client.get_server_version()
        except RuntimeError as erro:
            raise SystemExit(f"O CARLA não respondeu em {host}:{porta} ({erro}). Abra o simulador (por exemplo "
                             "C:\\CARLA_0.9.16\\CarlaUE4.exe, em janela para ver o caminhão) ou rode com --sem-carla.") from None
        versao_cliente = self.client.get_client_version()
        if versao_servidor != versao_cliente:
            log.warning("Versões diferentes: CARLA %s e cliente Python %s.", versao_servidor, versao_cliente)
        log.info("CARLA %s em %s:%d", versao_servidor, host, porta)
        self.world = self.client.get_world()

    def opendrive_atual(self) -> str:
        return self.world.get_map().to_opendrive()

    def ligar_camera_painel(self, enviar: Callable[[bytes], None]) -> None:
        """Liga a câmera de perseguição do painel 1 (chamar antes de iniciar)."""
        self._enviar_quadro = enviar

    def iniciar(self) -> None:
        if self.client is None:
            self.conectar()
        carla = self.carla
        if not self.manter_mundo and self.cc.get("mapa_carla"):
            self.world = carregar_mapa_instalado(self.client, str(self.cc["mapa_carla"]),
                                                 caminho(self.cc.get("pasta_xodr_carla") or "data/mapas"))
            if self.texto_xodr and " ".join(self.opendrive_atual().split()) != " ".join(self.texto_xodr.split()):
                log.warning("O .xodr do mapa instalado no CARLA difere do mapa da rota (mapa.arquivo em "
                            "config/simulacao.yaml): a rota pode não bater com as vias. Copie o .xodr do pacote para "
                            "data/mapas e rode python -m mover.autonomo.gerar_missao.")
        elif not self.manter_mundo:
            if not self.texto_xodr:
                raise ValueError("sem o texto do .xodr para gerar o mundo")
            p = self.cc.get("opendrive", {})
            parametros = carla.OpendriveGenerationParameters(
                vertex_distance=float(p.get("vertex_distance", 2.0)),
                max_road_length=float(p.get("max_road_length", 50.0)),
                wall_height=float(p.get("wall_height", 0.0)),
                additional_width=float(p.get("additional_width", 0.6)),
                smooth_junctions=bool(p.get("smooth_junctions", True)),
                enable_mesh_visibility=bool(p.get("enable_mesh_visibility", True)),
                enable_pedestrian_navigation=bool(p.get("enable_pedestrian_navigation", True)),
            )
            log.info("Gerando o mundo OpenDRIVE no CARLA (pode levar alguns segundos)...")
            texto = self.texto_xodr  # o mapa vai ao CARLA como é
            # Preparos do mapa_final.xodr (acostamentos, junções de passagem colapsadas, geoReference ilegível),
            # desligados com o campus_maua.xodr. Para religar, descomente (e as chaves em config/simulacao.yaml):
            # texto, self.ajustes_mapa = preparar_para_carla(self.texto_xodr, p)
            # juncoes = self.ajustes_mapa["juncoes_colapsadas"]
            # log.info("Mapa para o CARLA: acostamento de %.1f m dos dois lados das vias; %d junção(ões) de passagem "
            #          "viraram ligação direta%s.", self.ajustes_mapa["acostamento_m"], len(juncoes),
            #          f" (conectores {', '.join(a['conector'] for a in juncoes)})" if juncoes else "")
            self.world = self.client.generate_opendrive_world(texto, parametros)
        self._config_original = self.world.get_settings()
        config = self.world.get_settings()
        config.synchronous_mode = True
        config.fixed_delta_seconds = self.dt
        config.no_rendering_mode = bool(self.sem_render)
        self.world.apply_settings(config)
        self.mapa = self.world.get_map()
        self._alturas = None
        self.tm = self.client.get_trafficmanager(int(self.ctm["porta_tm"]))
        self.tm.set_synchronous_mode(True)
        self.tm.set_random_device_seed(int(self.ctm["semente"]))
        self.tm.set_hybrid_physics_mode(False)
        clima = getattr(carla.WeatherParameters, str(self.ctm.get("clima") or ""), None)
        if clima is not None:
            self.world.set_weather(clima)
        self.spectator = self.world.get_spectator()
        if self._vigia is None:
            self._vigia = VigiaTick(float(self.ctm["max_tick_s"]), _encerrar_processo_travado)
        self.topologia = self.verificar_topologia()
        self._criar_caminhao()

    def verificar_topologia(self, passo_m: float = 2.0, max_desvio_m: float = 3.0) -> dict[str, Any]:
        """Anda pelos waypoints do CARLA ao longo da rota, como o Traffic Manager vai fazer.

        O TM remove o veículo quando a faixa à frente não tem continuação no mapa do CARLA (beco sem
        saída). Isso aparece aqui antes da largada, com a via e a posição na rota.
        """
        carla = self.carla
        try:
            wp = self.mapa.get_waypoint(self._transform_largada().location, project_to_road=True,
                                        lane_type=carla.LaneType.Driving)
        except (RuntimeError, AttributeError):
            wp = None
        if wp is None or not hasattr(wp, "next"):
            return {"verificada": False}
        L = self.rota.comprimento
        s_atual, andado, problemas = 0.0, 0.0, []
        while andado < L:
            proximos = wp.next(passo_m)
            if not proximos:
                problemas.append({"tipo": "beco sem saída", **self._onde(wp, s_atual)})
                break
            escolhido = None
            for candidato in proximos:  # o TM escolhe o ramo mais perto do caminho (set_path)
                loc = candidato.transform.location
                x, y, _ = de_carla(float(loc.x), float(loc.y), 0.0)
                s, lateral = self.rota.projetar(x, y, s_atual)
                if escolhido is None or abs(lateral) < abs(escolhido[2]):
                    escolhido = (candidato, s, lateral)
            wp, s_atual, lateral = escolhido
            andado += passo_m
            if abs(lateral) > max_desvio_m:
                problemas.append({"tipo": f"a faixa do CARLA sai da rota ({lateral:.1f} m)", **self._onde(wp, s_atual)})
                break
        resultado = {"verificada": True, "percorrido_m": round(andado, 1), "problemas": problemas}
        if problemas:
            p = problemas[0]
            log.warning("Topologia do CARLA: %s em s = %.0f m da volta (via %s, faixa %s%s). O Traffic Manager remove o "
                        "caminhão quando a faixa à frente não tem continuação: confira os links dessa via no .xodr.",
                        p["tipo"], p["s_m"], p["via"], p["faixa"], ", junção" if p["juncao"] else "")
        else:
            log.info("Topologia do CARLA: a rota de %.0f m tem continuação em todas as faixas.", L)
        return resultado

    @staticmethod
    def _onde(wp: Any, s: float) -> dict[str, Any]:
        loc = wp.transform.location
        return {"s_m": round(float(s), 1), "via": getattr(wp, "road_id", None), "faixa": getattr(wp, "lane_id", None),
                "juncao": bool(getattr(wp, "is_junction", False)), "x_carla": round(float(loc.x), 1),
                "y_carla": round(float(loc.y), 1)}

    # ------------------------------------------------------------------ caminhão
    def _z(self, x_carla: float, y_carla: float) -> float:
        """Altura da via mais próxima em planta. Com elevação (campus_maua.xodr: ~750 m), um ponto em z = 0
        casaria em 3D com a via mais baixa, não com a mais próxima."""
        if self._alturas is None:
            try:
                from scipy.spatial import cKDTree

                wps = self.mapa.generate_waypoints(2.0)
                xyz = np.array([(w.transform.location.x, w.transform.location.y, w.transform.location.z) for w in wps], float)
                self._alturas = (cKDTree(xyz[:, :2]), xyz[:, 2]) if len(xyz) else False
            except (AttributeError, RuntimeError, TypeError, ValueError):
                self._alturas = False
        if not self._alturas:
            return 0.0
        arvore, z = self._alturas
        return float(z[arvore.query((x_carla, y_carla))[1]])

    def _transform_largada(self) -> Any:
        carla = self.carla
        p = self.rota.ponto(0.0)
        x, y, yaw = para_carla(p.x, p.y, p.rumo)
        wp = self.mapa.get_waypoint(carla.Location(x=x, y=y, z=self._z(x, y)), project_to_road=True,
                                    lane_type=carla.LaneType.Any)
        z = float(wp.transform.location.z) if wp is not None else 0.0
        return carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw))

    def _criar_caminhao(self) -> None:
        carla = self.carla
        biblioteca = self.world.get_blueprint_library()
        nome = self.cv.get("blueprint", "vehicle.carlamotors.european_hgv")
        bp = biblioteca.find(nome)
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", str(self.cv.get("role_name", "mover_gemeo")))
        largada = self._transform_largada()
        for extra in (0.5, 1.5, 3.0):  # o asfalto às vezes colide com a caixa do veículo no spawn
            local = carla.Location(x=largada.location.x, y=largada.location.y, z=largada.location.z + extra)
            self.ator = self.world.try_spawn_actor(bp, carla.Transform(local, largada.rotation))
            if self.ator is not None:
                break
        if self.ator is None:
            raise RuntimeError("não foi possível criar o caminhão na largada da rota (local ocupado?)")
        log.info("Caminhão %s criado (id %d) na largada, com física.", self.ator.type_id, self.ator.id)
        self._registrar_fisica()
        # assenta na pista com o freio de mão antes de ligar o piloto automático
        self.ator.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
        for _ in range(int(self.ctm["acomodar_ticks"])):
            self._tick()
            self._camera_simulador()
        self.ator.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=False))
        porta = self.tm.get_port()
        self.ator.set_autopilot(True, porta)
        self.tm.auto_lane_change(self.ator, False)
        self.tm.ignore_lights_percentage(self.ator, 100.0)
        self.tm.ignore_signs_percentage(self.ator, 100.0)
        self.tm.distance_to_leading_vehicle(self.ator, float(self.ctm["distancia_lider_m"]))
        self.tm.set_desired_speed(self.ator, 0.0)
        self._d = 0.0
        self._definir_caminho(0.0)
        if self._enviar_quadro is not None and self.ccam_painel.get("ativa", True) and not self.sem_render:
            painel = CameraPainel(self.ccam_painel, self._enviar_quadro)
            try:
                if painel.criar(carla, self.world, self.ator):
                    self.camera_painel = painel
            except Exception as erro:  # o painel é só para ver: a corrida segue sem a câmera
                log.warning("Câmera do painel desligada (%s); a página mostra o mapa 2D.", erro)
                painel.encerrar()
        self._t_inicio = float(self.world.get_snapshot().timestamp.elapsed_seconds)
        self._ultima = {}
        self.frame = 0

    def _registrar_fisica(self) -> None:
        """Loga a física do veículo: troca de marcha lenta e freio-motor forte fazem o caminhão oscilar com acelerador parcial."""
        try:
            f = self.ator.get_physics_control()
            self.fisica = {"massa_kg": round(float(f.mass)), "marchas": len(f.forward_gears),
                           "troca_marcha_s": round(float(f.gear_switch_time), 2), "cambio_automatico": bool(f.use_gear_autobox),
                           "rpm_max": round(float(f.max_rpm)),
                           "amortecimento_sem_acelerador": round(float(f.damping_rate_zero_throttle_clutch_engaged), 2)}
        except (RuntimeError, AttributeError, TypeError):
            return
        log.info("Física do caminhão no CARLA: %s", ", ".join(f"{k} {v}" for k, v in self.fisica.items()))

    def _tick(self) -> None:
        if self._vigia is not None:
            self._vigia.comecar()
        try:
            self.world.tick()
        except RuntimeError as erro:  # o CARLA não respondeu dentro do timeout do cliente
            raise FimDoSimulador("sem_resposta", f"o CARLA não respondeu ao tick ({erro}). Feche o simulador, abra "
                                                 "de novo e rode outra vez.") from None
        finally:
            if self._vigia is not None:
                self._vigia.terminar()

    def _definir_caminho(self, d: float) -> None:
        """Caminho do TM: os pontos da rota à frente do caminhão, até 30 m depois da chegada.

        Vai de uma vez só: renovar no meio da corrida esvazia o buffer do TM, que relocaliza o caminhão pelo
        waypoint mais próximo e, perto de uma junção, pode pegar outra faixa e sair da rota.
        """
        carla = self.carla
        L = self.rota.comprimento
        passo = float(self.ctm["passo_caminho_m"])
        self._horizonte_caminho = float(self.ctm["horizonte_caminho_m"])
        fim = min(self.voltas * L + 30.0, d + self._horizonte_caminho)  # passa da chegada: o governador para antes
        pontos = []
        dist = d + float(self.ctm["inicio_caminho_m"])
        while dist < fim:
            p = self.rota.ponto(dist)
            x, y, _ = para_carla(p.x, p.y, p.rumo)
            local = carla.Location(x=x, y=y, z=self._z(x, y))
            if not self._dentro_de_juncao(local):
                pontos.append(local)
            dist += passo
        self.tm.set_path(self.ator, pontos, True)
        self._caminho_ate = fim
        log.debug("Caminho do TM: %d pontos de d = %.0f a %.0f m.", len(pontos), d, fim)

    def _dentro_de_juncao(self, local: Any) -> bool:
        """Pontos dentro de jun\u00e7\u00f5es ficam fora do caminho do TM.

        Numa bifurca\u00e7\u00e3o, o TM escolhe o ramo cuja sa\u00edda est\u00e1 na mesma via do pr\u00f3ximo ponto do caminho; se esse
        ponto ainda est\u00e1 dentro da jun\u00e7\u00e3o, ele escolhe pela dist\u00e2ncia e pode errar (na VM, a 751 m da volta,
        ele pegou o conector 306 em vez do 305 e saiu da rota).
        """
        try:
            wp = self.mapa.get_waypoint(local, project_to_road=True, lane_type=self.carla.LaneType.Driving)
        except (RuntimeError, AttributeError):
            return False
        return bool(getattr(wp, "is_junction", False)) if wp is not None else False

    def _remover_caminhao(self) -> None:
        if self.camera_painel is not None:
            self.camera_painel.encerrar()
            self.camera_painel = None
        if self.ator is not None:
            try:
                self.ator.set_autopilot(False, self.tm.get_port())
            except RuntimeError:
                pass
            try:
                self.ator.destroy()
            except RuntimeError:
                pass
            self.ator = None

    def reiniciar(self) -> None:
        """Volta à largada com um caminhão novo (entre as baterias do benchmark)."""
        self._remover_caminhao()
        self._tick()
        self._criar_caminhao()

    # ------------------------------------------------------------------ gravação
    def gravar(self, arquivo: Path) -> str:
        """Liga o gravador do CARLA (o arquivo fica na máquina do simulador; use caminho absoluto)."""
        arquivo = Path(arquivo).resolve()
        arquivo.parent.mkdir(parents=True, exist_ok=True)
        self.gravacao = self.client.start_recorder(str(arquivo), True)
        log.info("Gravador do CARLA ligado: %s", arquivo)
        return str(arquivo)

    def parar_gravacao(self) -> None:
        if self.gravacao is not None:
            try:
                self.client.stop_recorder()
            except RuntimeError as erro:
                log.warning("Não foi possível parar o gravador: %s", erro)
            self.gravacao = None

    # ------------------------------------------------------------------ plano do TM
    def acoes_do_tm(self) -> list[tuple[str, float, float]]:
        """As próximas ações que o Traffic Manager já decidiu para o caminhão (get_all_actions).

        Cada ação é (manobra, x, y) no referencial do mapa: 'Left', 'Right', 'Straight', 'LaneFollow'...
        O planejador do caminhão usa isso para dizer aos agentes onde ele vai virar o volante.
        """
        if self.tm is None or self.ator is None:
            return []
        try:
            acoes = self.tm.get_all_actions(self.ator)
        except (RuntimeError, AttributeError) as erro:
            log.debug("get_all_actions indisponível: %s", erro)
            return []
        saida = []
        for acao in acoes or []:
            try:
                opcao, waypoint = acao[0], acao[1]
                loc = waypoint.transform.location
                x, y, _ = de_carla(float(loc.x), float(loc.y), 0.0)
                saida.append((str(opcao), x, y))
            except (AttributeError, IndexError, TypeError, ValueError):
                continue
        return saida

    # ------------------------------------------------------------------ tick
    def passo(self, v_desejada: float) -> dict[str, Any]:
        """Um tick: velocidade desejada (m/s) ao TM, avança o mundo e devolve a telemetria."""
        self.tm.set_desired_speed(self.ator, max(0.0, float(v_desejada)) * 3.6)
        self._tick()
        self.frame += 1
        if not getattr(self.ator, "is_alive", True):
            raise FimDoSimulador("removido", self._diagnostico_remocao())
        snap = self.world.get_snapshot()
        t = float(snap.timestamp.elapsed_seconds) - self._t_inicio
        try:
            tf = self.ator.get_transform()
        except RuntimeError:  # destruído entre o tick e a leitura
            raise FimDoSimulador("removido", self._diagnostico_remocao()) from None
        vel, acc, ang = self.ator.get_velocity(), self.ator.get_acceleration(), self.ator.get_angular_velocity()
        frente, direita = tf.get_forward_vector(), tf.get_right_vector()
        v = max(0.0, _escalar(vel, frente))
        x, y, rumo = de_carla(float(tf.location.x), float(tf.location.y), float(tf.rotation.yaw))
        s, lateral = self.rota.projetar(x, y, self._d % self.rota.comprimento)
        self._d = self.rota.avancar_distancia(self._d, s)
        chegada = self.voltas * self.rota.comprimento + 30.0
        if self._caminho_ate < chegada - 1e-6 and self._d > self._caminho_ate - 0.5 * self._horizonte_caminho:
            self._definir_caminho(self._d)  # só em corridas mais longas que o horizonte
        controle = self.ator.get_control()
        linha = {
            "frame": self.frame,
            "sim_time": round(t, 4),
            "hora_local": (self.hora_inicio + timedelta(seconds=t)).isoformat(timespec="milliseconds"),
            "x": x, "y": y, "z": float(tf.location.z),
            "rumo_graus": math.degrees(rumo),
            "x_carla": float(tf.location.x), "y_carla": float(tf.location.y),
            "yaw_carla": float(tf.rotation.yaw), "pitch_carla": float(tf.rotation.pitch), "roll_carla": float(tf.rotation.roll),
            "speed_mps": v,
            "acc_long": _escalar(acc, frente),
            "acc_lat": _escalar(acc, direita),
            "yaw_rate_dps": float(ang.z),  # convenção do CARLA: positivo à direita
            "throttle": float(controle.throttle), "brake": float(controle.brake), "steer": float(controle.steer),
            "gear": int(controle.gear), "rpm": 0.0,
            "d": self._d, "s": s, "desvio_m": lateral,
            "volta": int(self._d // self.rota.comprimento),
            "odom_m": self._d,
            "rampa": math.tan(math.radians(float(tf.rotation.pitch))),
            "v_desejada": float(v_desejada),
            "fonte": self.nome,
        }
        if self.ctm.get("telemetria_motor"):
            try:
                dados = self.ator.get_telemetry_data()
                linha["rpm"], linha["gear"] = float(dados.engine_rpm), int(dados.gear)
            except (RuntimeError, AttributeError):
                pass
        if not self.sem_render:
            self._camera_simulador(tf)
        self._ultima = linha
        return linha

    def _diagnostico_remocao(self) -> str:
        u = self._ultima
        texto = (f"o CARLA removeu o caminhão no tick {self.frame}. Último estado: d = {u.get('d', 0.0):.1f} m "
                 f"(s = {u.get('s', 0.0):.1f} m na volta), {3.6 * u.get('speed_mps', 0.0):.0f} km/h, "
                 f"z = {u.get('z', 0.0):.2f} m, desvio da rota {u.get('desvio_m', 0.0):.1f} m")
        try:
            loc = self.carla.Location(x=u["x_carla"], y=u["y_carla"], z=u["z"])
            wp = self.mapa.get_waypoint(loc, project_to_road=True, lane_type=self.carla.LaneType.Driving)
            texto += f", via {wp.road_id} faixa {wp.lane_id}{' (junção)' if wp.is_junction else ''}"
        except (KeyError, RuntimeError, AttributeError):
            pass
        problemas = (self.topologia or {}).get("problemas") or []
        if problemas:
            texto += f". A verificação da topologia achou {problemas[0]['tipo']} em s = {problemas[0]['s_m']:.0f} m"
        return (texto + ". Causas prováveis: a faixa à frente não tem continuação no mapa do CARLA (o Traffic Manager "
                "remove o veículo num beco sem saída) ou o caminhão caiu da pista (z muito baixo).")

    def _camera_simulador(self, tf: Any = None) -> None:
        """A câmera da janela do CARLA (spectator) segue o caminhão, com o giro suavizado."""
        if self.spectator is None or self.ator is None:
            return
        carla = self.carla
        tf = tf or self.ator.get_transform()
        x, y, z, yaw = float(tf.location.x), float(tf.location.y), float(tf.location.z), float(tf.rotation.yaw)
        if self.ccam.get("modo", "perseguicao") == "cima":
            alvo = carla.Transform(carla.Location(x=x, y=y, z=z + float(self.ccam.get("altura_cima_m", 80))),
                                   carla.Rotation(pitch=-90.0, yaw=-90.0))
        elif self.ccam.get("modo") == "livre":
            return
        else:
            if self._yaw_camera is None:
                self._yaw_camera = yaw
            else:
                self._yaw_camera += 0.15 * ((yaw - self._yaw_camera + 180.0) % 360.0 - 180.0)
            r = math.radians(self._yaw_camera)
            dist, altura = float(self.ccam.get("distancia_m", 16)), float(self.ccam.get("altura_m", 8))
            alvo = carla.Transform(carla.Location(x=x - dist * math.cos(r), y=y - dist * math.sin(r), z=z + altura),
                                   carla.Rotation(pitch=float(self.ccam.get("inclinacao_graus", -20)), yaw=self._yaw_camera))
        self.spectator.set_transform(alvo)

    # ------------------------------------------------------------------ fim
    def encerrar(self) -> None:
        """Para o gravador, devolve o CARLA ao modo assíncrono (senão o simulador fica parado), tira o TM
        do modo síncrono e remove a câmera e o caminhão. Idempotente."""
        self.parar_gravacao()
        if self._vigia is not None:
            self._vigia.parar()
            self._vigia = None
        if self.world is not None and self._config_original is not None:
            try:
                self.world.apply_settings(self._config_original)
            except RuntimeError as erro:
                log.warning("Não foi possível restaurar as configurações do CARLA: %s", erro)
            self._config_original = None
        if self.tm is not None:
            try:
                self.tm.set_synchronous_mode(False)
            except RuntimeError:
                pass
        self._remover_caminhao()


__all__ = ["PADRAO_TM", "SimuladorCarla"]
