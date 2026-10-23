"""Figurantes do CARLA para as situações da apresentação: pedestre atravessando e veículo lento à frente.

Cada figurante é disparado pela posição do caminhão na volta (uma vez por volta) e removido quando a
situação passa. O Traffic Manager do caminhão reage a eles sozinho (freia para o pedestre, segue o veículo);
os agentes só conhecem o plano de missão. Configuração (config/simulacao.yaml, seção cenario):

    cenario:
      pedestres: [{s_m: 410, gatilho_m: 30, lado_m: 6, atravessar_m: 14, velocidade_mps: 1.0}]
      veiculos:  [{s_m: 150, a_frente_m: 35, percurso_m: 180, velocidade_kmh: 12}]
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable

from mover.autonomo.rota import Rota, para_carla

log = logging.getLogger("mover.autonomo")


class Figurantes:
    def __init__(self, carla: Any, world: Any, tm: Any, rota: Rota, cfg: dict[str, Any] | None,
                 z_da_via: Callable[[float, float], float], caminho_tm: Callable[[float, float], list[Any]]):
        cfg = cfg or {}
        self.carla, self.world, self.tm, self.rota = carla, world, tm, rota
        self.pedestres = [dict(p) for p in cfg.get("pedestres") or []]
        self.veiculos = [dict(v) for v in cfg.get("veiculos") or []]
        self.z_da_via, self.caminho_tm = z_da_via, caminho_tm
        self.ativos: list[dict[str, Any]] = []
        self._disparados: set[tuple[str, int, int]] = set()

    @property
    def configurados(self) -> bool:
        return bool(self.pedestres or self.veiculos)

    # ------------------------------------------------------------------ ciclo
    def atualizar(self, d: float, x_carla: float, y_carla: float) -> float | None:
        """Dispara e remove figurantes pela distância percorrida; devolve a menor distância a um deles (m)."""
        L = self.rota.comprimento
        s = d % L
        for i, p in enumerate(self.pedestres):
            falta = (float(p["s_m"]) - s) % L
            volta = int((d + falta) // L)
            if falta <= float(p.get("gatilho_m", 35.0)) and ("pedestre", i, volta) not in self._disparados:
                self._disparados.add(("pedestre", i, volta))
                self._criar_pedestre(p, d + falta)
        for i, v in enumerate(self.veiculos):
            falta = (float(v["s_m"]) - s) % L
            volta = int((d + falta) // L)
            if falta <= 2.0 and ("veiculo", i, volta) not in self._disparados:
                self._disparados.add(("veiculo", i, volta))
                self._criar_veiculo(v, d)
        perto = None
        for f in list(self.ativos):
            try:
                loc = f["ator"].get_location()
            except RuntimeError:
                self.ativos.remove(f)
                continue
            dist = math.hypot(loc.x - x_carla, loc.y - y_carla)
            perto = dist if perto is None else min(perto, dist)
            chegou = f.get("destino") is not None and math.hypot(loc.x - f["destino"][0], loc.y - f["destino"][1]) < 1.0
            if chegou or d > f["d_fim"]:
                self._remover(f)
        return perto

    def limpar(self) -> None:
        for f in list(self.ativos):
            self._remover(f)
        self._disparados.clear()

    def _remover(self, f: dict[str, Any]) -> None:
        if f in self.ativos:
            self.ativos.remove(f)
        try:
            if f["tipo"] == "veiculo":
                f["ator"].set_autopilot(False, self.tm.get_port())
            f["ator"].destroy()
        except RuntimeError:
            pass

    # ------------------------------------------------------------------ criação
    def _ponto(self, d: float, lateral_m: float = 0.0) -> tuple[float, float, float]:
        """(x, y, yaw) no CARLA do ponto da rota em d, deslocado para a direita do sentido de tráfego."""
        p = self.rota.ponto(d)
        x = p.x + lateral_m * math.sin(p.rumo)
        y = p.y - lateral_m * math.cos(p.rumo)
        return para_carla(x, y, p.rumo)

    def _blueprint(self, padrao: str, filtro: str) -> Any:
        biblioteca = self.world.get_blueprint_library()
        try:
            return biblioteca.find(padrao)
        except (IndexError, RuntimeError):
            return biblioteca.filter(filtro)[0]

    def _criar_pedestre(self, p: dict[str, Any], d_faixa: float) -> None:
        carla = self.carla
        lado, atravessar = float(p.get("lado_m", 6.0)), float(p.get("atravessar_m", 14.0))
        bp = self._blueprint(str(p.get("blueprint", "walker.pedestrian.0001")), "walker.pedestrian.*")
        ator = None
        for recuo in (0.0, 1.5, 3.0):  # calçada ocupada (poste, prédio): tenta mais perto da via
            x0, y0, _ = self._ponto(d_faixa, lado - recuo)
            x1, y1, _ = self._ponto(d_faixa, lado - atravessar)
            yaw = math.degrees(math.atan2(y1 - y0, x1 - x0))
            local = carla.Location(x=x0, y=y0, z=self.z_da_via(x0, y0) + 1.0)
            ator = self.world.try_spawn_actor(bp, carla.Transform(local, carla.Rotation(yaw=yaw)))
            if ator is not None:
                break
        if ator is None:
            log.warning("Pedestre não criado em s = %.0f m (local ocupado).", d_faixa % self.rota.comprimento)
            return
        n = math.hypot(x1 - x0, y1 - y0) or 1.0
        controle = carla.WalkerControl()
        controle.direction = carla.Vector3D((x1 - x0) / n, (y1 - y0) / n, 0.0)
        controle.speed = float(p.get("velocidade_mps", 1.0))
        ator.apply_control(controle)
        self.ativos.append({"tipo": "pedestre", "ator": ator, "destino": (x1, y1), "d_fim": d_faixa + 25.0})
        log.info("Pedestre atravessando a faixa em s = %.0f m da volta.", d_faixa % self.rota.comprimento)

    def _criar_veiculo(self, v: dict[str, Any], d: float) -> None:
        carla = self.carla
        d_ini = d + float(v.get("a_frente_m", 35.0))
        percurso = float(v.get("percurso_m", 180.0))
        x, y, yaw = self._ponto(d_ini)
        bp = self._blueprint(str(v.get("blueprint", "vehicle.tesla.model3")), "vehicle.*")
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "figurante")
        local = carla.Location(x=x, y=y, z=self.z_da_via(x, y) + 0.5)
        ator = self.world.try_spawn_actor(bp, carla.Transform(local, carla.Rotation(yaw=yaw)))
        if ator is None:
            log.warning("Veículo à frente não criado em s = %.0f m (local ocupado).", d_ini % self.rota.comprimento)
            return
        porta = self.tm.get_port()
        ator.set_autopilot(True, porta)
        self.tm.auto_lane_change(ator, False)
        self.tm.ignore_lights_percentage(ator, 100.0)
        self.tm.ignore_signs_percentage(ator, 100.0)
        self.tm.set_desired_speed(ator, float(v.get("velocidade_kmh", 12.0)))
        self.tm.set_path(ator, self.caminho_tm(d_ini + 8.0, d_ini + percurso + 40.0), True)
        self.ativos.append({"tipo": "veiculo", "ator": ator, "destino": None, "d_fim": d_ini + percurso + 30.0})
        log.info("Veículo a %.0f km/h à frente do caminhão em s = %.0f m da volta, por %.0f m.",
                 float(v.get("velocidade_kmh", 12.0)), d_ini % self.rota.comprimento, percurso)
