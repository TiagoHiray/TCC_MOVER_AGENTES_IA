"""Fusão GPS + IMU: filtro de Kalman estendido (EKF) com suavizador RTS.

Estado: [leste, norte, psi, v, viés do giroscópio, viés do acelerômetro]
  psi: rumo do veículo no plano ENU, anti-horário a partir do leste (rad, contínuo)
Entradas (a cada passo da grade de 20 Hz): taxa de guinada w_cima e aceleração longitudinal.
Medições (1 Hz): posição, velocidade e rumo do GPS, já com a defasagem corrigida.
O suavizador RTS usa o passado e o futuro, o que é possível porque a volta já foi gravada.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mover.tratamento.sinais import embrulhar

IE, IN, IPSI, IV, IBG, IBA = range(6)


@dataclass
class EntradaGPS:
    """Fixes do GPS prontos para a fusão (arrays do mesmo tamanho)."""

    t: np.ndarray  # instante corrigido pela defasagem (s)
    leste: np.ndarray
    norte: np.ndarray
    sigma_pos: np.ndarray  # m, por eixo
    vel: np.ndarray  # m/s (NaN = inválida)
    sigma_vel: np.ndarray
    psi: np.ndarray  # rad (NaN = inválido)
    sigma_psi: np.ndarray


@dataclass
class ResultadoFusao:
    """Trajetória suavizada na grade e diagnóstico dos resíduos."""

    t: np.ndarray
    leste: np.ndarray
    norte: np.ndarray
    psi: np.ndarray
    v: np.ndarray
    vies_giro: np.ndarray
    vies_acel: np.ndarray
    sigma_pos: np.ndarray
    sigma_psi: np.ndarray
    sigma_v: np.ndarray
    residuo_m: np.ndarray  # por fix (NaN fora da grade)
    usado: np.ndarray  # por fix
    rms_residuo_m: float
    max_residuo_m: float
    n_rejeitados: int


def _estado_inicial(gps: EntradaGPS, t0: float) -> tuple[np.ndarray, np.ndarray]:
    i0 = int(np.argmin(np.abs(gps.t - t0)))
    validos_v = np.flatnonzero(np.isfinite(gps.vel))
    validos_psi = np.flatnonzero(np.isfinite(gps.psi))
    v0 = gps.vel[validos_v[0]] if len(validos_v) else 0.0
    if len(validos_psi):
        psi0 = gps.psi[validos_psi[0]]
    else:  # sem rumo do GPS: usa o deslocamento entre fixes
        d = np.hypot(gps.leste - gps.leste[i0], gps.norte - gps.norte[i0])
        j = int(np.argmax(d > 5)) if np.any(d > 5) else min(i0 + 1, len(d) - 1)
        psi0 = np.arctan2(gps.norte[j] - gps.norte[i0], gps.leste[j] - gps.leste[i0])
    x = np.array([gps.leste[i0], gps.norte[i0], psi0, v0, 0.0, 0.0])
    s = max(gps.sigma_pos[i0], 3.0)
    P = np.diag([s**2, s**2, np.radians(30) ** 2, 2.0**2, np.radians(0.5) ** 2, 0.2**2])
    return x, P


def _atualizar(x, P, z, H, R, angular=False):
    y = z - H @ x
    if angular:
        y = embrulhar(y)
    S = H @ P @ H.T + R
    K = np.linalg.solve(S.T, (P @ H.T).T).T
    x = x + K @ y
    I_KH = np.eye(len(x)) - K @ H
    P = I_KH @ P @ I_KH.T + K @ R @ K.T
    return x, P


def fundir(t: np.ndarray, w_cima: np.ndarray, a_long: np.ndarray, gps: EntradaGPS, cfg: dict) -> ResultadoFusao:
    """Roda o EKF para frente e o suavizador RTS para trás na grade `t` (passo constante)."""
    n = len(t)
    dt = float(t[1] - t[0])
    rp = cfg["ruido_processo"]
    q = np.array([rp["posicao_m"], rp["posicao_m"], rp["rumo_rad_s"], rp["acel_mps2"],
                  rp["vies_giro_rad_s"], rp["vies_acel_mps2"]]) ** 2 * dt
    Q = np.diag(q)
    porta = float(cfg.get("porta_chi2_pos", np.inf))

    k_fix = np.round((gps.t - t[0]) / dt).astype(int)
    dentro = (k_fix >= 0) & (k_fix < n)
    por_passo: dict[int, list[int]] = {}
    for i in np.flatnonzero(dentro):
        por_passo.setdefault(int(k_fix[i]), []).append(int(i))

    H_pos = np.zeros((2, 6)); H_pos[0, IE] = H_pos[1, IN] = 1
    H_v = np.zeros((1, 6)); H_v[0, IV] = 1
    H_psi = np.zeros((1, 6)); H_psi[0, IPSI] = 1

    xp = np.zeros((n, 6)); Pp = np.zeros((n, 6, 6))
    xf = np.zeros((n, 6)); Pf = np.zeros((n, 6, 6))
    Fs = np.zeros((n, 6, 6))
    usado = np.zeros(len(gps.t), dtype=bool)
    rejeitados, seguidos = 0, 0

    x, P = _estado_inicial(gps, t[0])
    for k in range(n):
        if k > 0:
            psi, v = x[IPSI], x[IV]
            c, s = np.cos(psi), np.sin(psi)
            F = np.eye(6)
            F[IE, IPSI], F[IE, IV] = -v * s * dt, c * dt
            F[IN, IPSI], F[IN, IV] = v * c * dt, s * dt
            F[IPSI, IBG] = -dt
            F[IV, IBA] = -dt
            x = x + dt * np.array([v * c, v * s, w_cima[k - 1] - x[IBG], a_long[k - 1] - x[IBA], 0, 0])
            P = F @ P @ F.T + Q
            Fs[k] = F
        else:
            Fs[k] = np.eye(6)
        xp[k], Pp[k] = x, P

        for i in por_passo.get(k, []):
            z = np.array([gps.leste[i], gps.norte[i]])
            R = np.eye(2) * gps.sigma_pos[i] ** 2
            y = z - H_pos @ x
            d2 = float(y @ np.linalg.solve(H_pos @ P @ H_pos.T + R, y))
            if d2 > porta and seguidos < 3:
                rejeitados += 1
                seguidos += 1
            else:
                seguidos = 0
                usado[i] = True
                x, P = _atualizar(x, P, z, H_pos, R)
            if np.isfinite(gps.vel[i]):
                x, P = _atualizar(x, P, np.array([gps.vel[i]]), H_v, np.array([[gps.sigma_vel[i] ** 2]]))
            if np.isfinite(gps.psi[i]):
                x, P = _atualizar(x, P, np.array([gps.psi[i]]), H_psi, np.array([[gps.sigma_psi[i] ** 2]]),
                                  angular=True)
        xf[k], Pf[k] = x, P

    # Suavizador Rauch-Tung-Striebel
    xs, Ps = xf.copy(), Pf.copy()
    for k in range(n - 2, -1, -1):
        C = np.linalg.solve(Pp[k + 1], Fs[k + 1] @ Pf[k]).T
        dx = xs[k + 1] - xp[k + 1]
        dx[IPSI] = embrulhar(dx[IPSI])
        xs[k] = xf[k] + C @ dx
        Ps[k] = Pf[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ C.T

    residuo = np.full(len(gps.t), np.nan)
    idx = np.flatnonzero(dentro)
    residuo[idx] = np.hypot(xs[k_fix[idx], IE] - gps.leste[idx], xs[k_fix[idx], IN] - gps.norte[idx])
    r_ok = residuo[usado]
    return ResultadoFusao(
        t=t,
        leste=xs[:, IE], norte=xs[:, IN], psi=xs[:, IPSI], v=xs[:, IV],
        vies_giro=xs[:, IBG], vies_acel=xs[:, IBA],
        sigma_pos=np.sqrt(0.5 * (Ps[:, IE, IE] + Ps[:, IN, IN])),
        sigma_psi=np.sqrt(Ps[:, IPSI, IPSI]),
        sigma_v=np.sqrt(Ps[:, IV, IV]),
        residuo_m=residuo,
        usado=usado,
        rms_residuo_m=float(np.sqrt(np.mean(r_ok**2))) if len(r_ok) else float("nan"),
        max_residuo_m=float(np.max(r_ok)) if len(r_ok) else float("nan"),
        n_rejeitados=rejeitados,
    )
