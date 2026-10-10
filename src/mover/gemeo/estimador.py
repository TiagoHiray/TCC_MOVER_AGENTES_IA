"""Estimador causal do estado do veículo (EKF da Etapa 1 sem o suavizador RTS), passo a passo.

Mesmo modelo do tratamento/fusao.py: estado [leste, norte, psi, v, viés do giro, viés do acel],
entradas do IMU na grade de 20 Hz e medições do GPS (posição, velocidade, rumo). Diferenças:
  * só usa o passado: cada passo de 50 ms sai assim que o IMU cobre o passo;
  * o fix do GPS chega atrasado (defasagem da calibração + lote do app): o filtro volta ao passo
    do fix, aplica a medição e repropaga até o presente com as entradas guardadas;
  * a orientação do celular vem de uma volta de calibração (CalibracaoFixa).
Só depende de numpy.
"""

from __future__ import annotations

import bisect
import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from mover.gemeo.calibracao_fixa import CalibracaoFixa

IE, IN, IPSI, IV, IBG, IBA = range(6)
_H_POS = np.zeros((2, 6)); _H_POS[0, IE] = _H_POS[1, IN] = 1
_H_V = np.zeros((1, 6)); _H_V[0, IV] = 1
_H_PSI = np.zeros((1, 6)); _H_PSI[0, IPSI] = 1


def _embrulhar(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _atualizar(x, P, z, H, R, angular=False):
    y = z - H @ x
    if angular:
        y = np.array([_embrulhar(float(v)) for v in y])
    S = H @ P @ H.T + R
    K = np.linalg.solve(S.T, (P @ H.T).T).T
    x = x + K @ y
    I_KH = np.eye(len(x)) - K @ H
    return x, I_KH @ P @ I_KH.T + K @ R @ K.T


@dataclass
class Fix:
    t: float  # já corrigido pela defasagem (s, época Unix)
    leste: float
    norte: float
    sigma_pos: float
    vel: float  # NaN = inválida
    sigma_vel: float
    psi: float  # NaN = inválido
    sigma_psi: float


@dataclass
class EstadoVeiculo:
    t_unix: float
    leste: float
    norte: float
    psi: float
    v: float
    yaw_rate_dps: float  # positivo virando à direita (convenção CARLA)
    a_long: float
    sigma_pos_m: float
    x: float  # plano local da Etapa 1 (x = leste, y = -norte)
    y: float
    yaw: float
    x_carla: float | None = None  # plano do mapa (com o alinhamento)
    y_carla: float | None = None
    yaw_carla: float | None = None

    def para_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Passo:
    k: int
    x: np.ndarray
    P: np.ndarray
    u: tuple[float, float]  # (w, a_long) usados para chegar a este passo


class EstimadorOnline:
    def __init__(self, cal: CalibracaoFixa, cfg_fusao: dict, hz: float = 20.0, historico_s: float = 5.0):
        self.cal = cal
        self.cfg = cfg_fusao
        self.dt = 1.0 / hz
        rp = cfg_fusao["ruido_processo"]
        self.Q = np.diag(np.array([rp["posicao_m"], rp["posicao_m"], rp["rumo_rad_s"], rp["acel_mps2"],
                                   rp["vies_giro_rad_s"], rp["vies_acel_mps2"]]) ** 2 * self.dt)
        self.porta = float(cfg_fusao.get("porta_chi2_pos", math.inf))
        self.vel_min_rumo = float(cfg_fusao["vel_min_rumo_mps"])
        self.modo_rumo = cfg_fusao.get("usar_rumo_gps", "frescos")
        self.n_hist = int(round(historico_s * hz))

        self._imu: dict[str, tuple[list[float], list[np.ndarray]]] = {"accelerometer": ([], []), "gyroscope": ([], [])}
        self._fixes_espera: list[Fix] = []  # antes da inicialização
        self._fixes_por_passo: dict[int, list[Fix]] = {}
        self._hist: deque[_Passo] = deque(maxlen=self.n_hist)
        self._ultimo_loc: dict[str, float] | None = None
        self._seguidos = 0
        self.t0: float | None = None
        self.k = -1
        self.contagem = {"fixes": 0, "fixes_usados": 0, "fixes_rejeitados": 0, "fixes_antigos": 0, "retrocessos": 0}

    # ------------------------------------------------------------------ entrada
    def adicionar(self, nome: str, t_ns: int, valores: dict[str, Any]) -> None:
        nome = nome.lower()
        t = t_ns / 1e9
        if nome in self._imu:
            try:
                vetor = np.array([float(valores["x"]), float(valores["y"]), float(valores["z"])])
            except (KeyError, TypeError, ValueError):
                return
            ts, vs = self._imu[nome]
            i = bisect.bisect(ts, t)
            ts.insert(i, t)
            vs.insert(i, vetor)
        elif nome == "location":
            fix = self._fix(t, valores)
            if fix is not None:
                self.contagem["fixes"] += 1
                self._novo_fix(fix)

    def _fix(self, t: float, v: dict[str, Any]) -> Fix | None:
        """Mesmas regras de _preparar_fixes da Etapa 1, um fix por vez."""
        def num(chave: str) -> float:
            try:
                return float(v.get(chave))
            except (TypeError, ValueError):
                return math.nan

        lat, lon, hacc = num("latitude"), num("longitude"), num("horizontalAccuracy")
        if not (hacc > 0 and math.isfinite(lat) and math.isfinite(lon)):
            return None
        vel, vel_acc, rumo, rumo_acc = num("speed"), num("speedAccuracy"), num("bearing"), num("bearingAccuracy")
        vel = vel if vel >= 0 else math.nan
        rumo = rumo if (rumo >= 0 and not rumo_acc < 0) else math.nan
        anterior = self._ultimo_loc
        self._ultimo_loc = {"vel": vel, "rumo": rumo}
        # o celular às vezes repete o rumo/velocidade do fix anterior: valor velho
        rumo_fresco = math.isfinite(rumo) and (anterior is None or not abs(rumo - anterior["rumo"]) <= 1e-9)
        vel_velha = anterior is not None and abs(vel - anterior["vel"]) < 1e-9 and vel > 0.5
        c = self.cfg
        leste, norte = self.cal.ref.para_enu(lat, lon)
        psi = math.radians(90.0 - rumo) if math.isfinite(rumo) else math.nan
        if not (vel >= self.vel_min_rumo) or self.modo_rumo == "nenhum" or (self.modo_rumo == "frescos" and not rumo_fresco):
            psi = math.nan
        return Fix(
            t=t - self.cal.defasagem_s, leste=float(leste), norte=float(norte),
            sigma_pos=max(c["fator_sigma_pos"] * hacc, c["sigma_pos_min_m"]),
            vel=math.nan if vel_velha else vel,
            sigma_vel=max(c["fator_sigma_vel"] * (vel_acc if vel_acc > 0 else 1.0), c["sigma_vel_min_mps"]),
            psi=psi,
            sigma_psi=max(math.radians(rumo_acc if rumo_acc > 0 else 30.0), math.radians(c["sigma_rumo_min_graus"])),
        )

    # ------------------------------------------------------------------ inicialização
    def _tentar_iniciar(self) -> bool:
        if not self._fixes_espera:
            return False
        ultimo = self._fixes_espera[-1]
        if math.isfinite(ultimo.psi):
            psi0 = ultimo.psi
            v0 = ultimo.vel if math.isfinite(ultimo.vel) else 0.0
        else:
            primeiro = self._fixes_espera[0]
            d = math.hypot(ultimo.leste - primeiro.leste, ultimo.norte - primeiro.norte)
            if d <= 5.0 or ultimo.t <= primeiro.t:
                return False
            psi0 = math.atan2(ultimo.norte - primeiro.norte, ultimo.leste - primeiro.leste)
            v0 = ultimo.vel if math.isfinite(ultimo.vel) else d / (ultimo.t - primeiro.t)
        x = np.array([ultimo.leste, ultimo.norte, psi0, v0, 0.0, 0.0])
        s = max(ultimo.sigma_pos, 3.0)
        P = np.diag([s**2, s**2, math.radians(30) ** 2, 2.0**2, math.radians(0.5) ** 2, 0.2**2])
        self.t0 = math.ceil(ultimo.t / self.dt) * self.dt
        self.k = 0
        self._hist.append(_Passo(0, x, P, (0.0, 0.0)))
        self._fixes_espera.clear()
        return True

    # ------------------------------------------------------------------ filtro
    def _prever(self, x: np.ndarray, P: np.ndarray, u: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        dt = self.dt
        psi, v = x[IPSI], x[IV]
        c, s = math.cos(psi), math.sin(psi)
        F = np.eye(6)
        F[IE, IPSI], F[IE, IV] = -v * s * dt, c * dt
        F[IN, IPSI], F[IN, IV] = v * c * dt, s * dt
        F[IPSI, IBG] = -dt
        F[IV, IBA] = -dt
        x = x + dt * np.array([v * c, v * s, u[0] - x[IBG], u[1] - x[IBA], 0.0, 0.0])
        return x, F @ P @ F.T + self.Q

    def _aplicar_fix(self, x: np.ndarray, P: np.ndarray, f: Fix) -> tuple[np.ndarray, np.ndarray]:
        z = np.array([f.leste, f.norte])
        R = np.eye(2) * f.sigma_pos**2
        y = z - _H_POS @ x
        d2 = float(y @ np.linalg.solve(_H_POS @ P @ _H_POS.T + R, y))
        if d2 > self.porta and self._seguidos < 3:
            self.contagem["fixes_rejeitados"] += 1
            self._seguidos += 1
        else:
            self._seguidos = 0
            self.contagem["fixes_usados"] += 1
            x, P = _atualizar(x, P, z, _H_POS, R)
        if math.isfinite(f.vel):
            x, P = _atualizar(x, P, np.array([f.vel]), _H_V, np.array([[f.sigma_vel**2]]))
        if math.isfinite(f.psi):
            x, P = _atualizar(x, P, np.array([f.psi]), _H_PSI, np.array([[f.sigma_psi**2]]), angular=True)
        return x, P

    def _novo_fix(self, f: Fix) -> None:
        if self.t0 is None:
            self._fixes_espera.append(f)
            return
        j = int(round((f.t - self.t0) / self.dt))
        if j > self.k:
            self._fixes_por_passo.setdefault(j, []).append(f)
            return
        if not self._hist or j < self._hist[0].k:
            self.contagem["fixes_antigos"] += 1
            return
        # retrocesso: aplica no passo j e repropaga até o presente
        self.contagem["retrocessos"] += 1
        self._fixes_por_passo.setdefault(j, []).append(f)
        base = j - self._hist[0].k
        passo = self._hist[base]
        passo.x, passo.P = self._aplicar_fix(passo.x, passo.P, f)
        for i in range(base + 1, len(self._hist)):
            p = self._hist[i]
            x, P = self._prever(self._hist[i - 1].x, self._hist[i - 1].P, p.u)
            for g in self._fixes_por_passo.get(p.k, []):
                x, P = self._aplicar_fix(x, P, g)
            p.x, p.P = x, P

    def _media(self, nome: str, t_ini: float, t_fim: float) -> np.ndarray | None:
        ts, vs = self._imu[nome]
        i, j = bisect.bisect_right(ts, t_ini), bisect.bisect_right(ts, t_fim)
        if j > i:
            return np.mean(vs[i:j], axis=0)
        return vs[j - 1] if j > 0 else None

    def _descartar_imu_antigo(self, t_limite: float) -> None:
        for ts, vs in self._imu.values():
            i = bisect.bisect_left(ts, t_limite)
            del ts[:i], vs[:i]

    def processar(self) -> list[EstadoVeiculo]:
        """Avança enquanto o IMU cobre o próximo passo; devolve os estados novos."""
        if self.t0 is None and not self._tentar_iniciar():
            return []
        novos = []
        cal = self.cal
        while True:
            t_fim = self.t0 + (self.k + 1) * self.dt
            ultimo_imu = min((ts[-1] if ts else -math.inf) for ts, _ in self._imu.values())
            if ultimo_imu < t_fim:
                break
            acc = self._media("accelerometer", t_fim - self.dt, t_fim)
            giro = self._media("gyroscope", t_fim - self.dt, t_fim)
            if acc is None or giro is None:
                break
            w = float(giro @ cal.R[2])
            a_long = float((cal.sinal * acc) @ cal.R[0]) * cal.ganho_long
            anterior = self._hist[-1]
            x, P = self._prever(anterior.x, anterior.P, (w, a_long))
            self.k += 1
            for f in self._fixes_por_passo.get(self.k, []):
                x, P = self._aplicar_fix(x, P, f)
            self._hist.append(_Passo(self.k, x, P, (w, a_long)))
            novos.append(self._estado(self._hist[-1]))
        if self._hist:
            k_min = self._hist[0].k
            for k in [k for k in self._fixes_por_passo if k < k_min]:
                del self._fixes_por_passo[k]
            self._descartar_imu_antigo(self.t0 + (self.k - 1) * self.dt)
        return novos

    def _estado(self, p: _Passo) -> EstadoVeiculo:
        x, P = p.x, p.P
        psi = _embrulhar(float(x[IPSI]))
        e = EstadoVeiculo(
            t_unix=round(self.t0 + p.k * self.dt, 3), leste=float(x[IE]), norte=float(x[IN]), psi=psi,
            v=max(0.0, float(x[IV])), yaw_rate_dps=-math.degrees(p.u[0] - float(x[IBG])),
            a_long=p.u[1] - float(x[IBA]), sigma_pos_m=math.sqrt(0.5 * (P[IE, IE] + P[IN, IN])),
            x=float(x[IE]), y=-float(x[IN]), yaw=_yaw_carla(psi),
        )
        if self.cal.mapa is not None:
            xm, ym = self.cal.mapa.aplicar(e.leste, e.norte)
            e.x_carla, e.y_carla, e.yaw_carla = xm, -ym, _yaw_carla(psi + self.cal.mapa.rotacao_rad)
        return e

    def atual(self) -> EstadoVeiculo | None:
        return self._estado(self._hist[-1]) if self._hist else None


def _yaw_carla(psi: float) -> float:
    return (-math.degrees(psi) + 180.0) % 360.0 - 180.0
