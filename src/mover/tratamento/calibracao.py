"""Calibração da montagem do celular no veículo e da defasagem do GPS.

O celular fica preso numa posição desconhecida. Descobrimos os eixos do veículo no
referencial do aparelho assim:
  * cima: média da direção da gravidade (montagem rígida, desvio de poucos graus);
  * frente: rotação no plano horizontal que melhor explica, ao mesmo tempo, a aceleração
    longitudinal (dv/dt do GPS) e a lateral (v * taxa de guinada do giroscópio).
Referencial do veículo usado internamente: ISO 8855 (x frente, y esquerda, z cima).

Defasagem do GPS: o fix chega atrasado em relação ao IMU. O método padrão ("velocidade")
procura o atraso que maximiza a correlação entre dv/dt do GPS e a aceleração longitudinal
do IMU. O método "rumo" usa a taxa de rumo do GPS, mas o iOS repete o rumo antigo em
muitos fixes, o que infla o atraso estimado.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from mover.tratamento.sinais import correlacao, interpolar, passa_baixa, r2, unitario


@dataclass
class Defasagem:
    """Atraso do GPS em relação ao IMU: t_real = t_gps - segundos."""

    segundos: float
    correlacao: float
    correlacao_sem_defasagem: float
    metodo: str
    curva_lags: list[float] = field(default_factory=list)
    curva_corr: list[float] = field(default_factory=list)


@dataclass
class Montagem:
    """Orientação do aparelho no veículo. `R @ v_aparelho` = v no veículo (frente, esquerda, cima)."""

    cima: np.ndarray
    frente: np.ndarray
    esquerda: np.ndarray
    R: np.ndarray
    theta_graus: float
    theta_long_graus: float
    theta_lat_graus: float
    r2_long: float
    r2_lat: float
    ganho_long: float  # dv/dt do GPS ~= ganho_long * a_long do IMU
    ganho_lat: float
    n_amostras: int
    desvio_gravidade_graus: dict


def vetor_cima(gravidade: np.ndarray, sinal: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Direção "cima" média e instantânea a partir do sensor Gravity.

    Com sinal = -1 (iOS cru) a gravidade aponta para baixo; com +1 aponta para cima.
    """
    cima_t = sinal * unitario(gravidade)
    cima = unitario(cima_t.mean(axis=0))
    ang = np.degrees(np.arccos(np.clip(cima_t @ cima, -1, 1)))
    desvio = {"mediana": float(np.median(ang)), "p95": float(np.percentile(ang, 95)), "max": float(ang.max())}
    return cima, cima_t, desvio


def base_horizontal(cima: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dois vetores unitários ortogonais no plano perpendicular a `cima` (e2 = cima x e1)."""
    ref = np.eye(3)[np.argmin(np.abs(cima))]
    e1 = unitario(ref - (ref @ cima) * cima)
    return e1, np.cross(cima, e1)


def _alvos(t_imu, fs, t_vel, vel, w_lp, fc):
    """Velocidade do GPS na grade do IMU, dv/dt e aceleração centrípeta (esquerda +)."""
    v = passa_baixa(np.interp(t_imu, t_vel, vel), fs, fc)
    return v, np.gradient(v, 1 / fs), v * w_lp


def _rotacao(alvo_long, alvo_esq, m1, m2) -> float:
    """Ângulo theta que minimiza o erro de [a_long, a_esq] = rot(theta) aplicada a [m1, m2]."""
    return float(np.arctan2(np.sum(alvo_long * m2 - alvo_esq * m1), np.sum(alvo_long * m1 + alvo_esq * m2)))


def _lags(cfg_cal: dict) -> np.ndarray:
    passo, lim = cfg_cal["defasagem_passo_s"], cfg_cal["defasagem_max_s"]
    return np.round(np.arange(-lim, lim + passo / 2, passo), 6)


def _melhor(lags: np.ndarray, corrs: np.ndarray, metodo: str) -> Defasagem:
    i = int(np.nanargmax(corrs))
    melhor = float(lags[i])
    if 0 < i < len(lags) - 1:  # refinamento parabólico
        c0, c1, c2 = corrs[i - 1 : i + 2]
        den = c0 - 2 * c1 + c2
        if den < 0:
            melhor += float(0.5 * (c0 - c2) / den * (lags[1] - lags[0]))
    return Defasagem(melhor, float(corrs[i]), float(corrs[np.argmin(np.abs(lags))]), metodo,
                     lags.tolist(), corrs.tolist())


def estimar_defasagem_velocidade(
    t_imu: np.ndarray, fs: float, acc: np.ndarray, w_cima: np.ndarray, cima: np.ndarray,
    t_vel: np.ndarray, vel: np.ndarray, cfg_filtros: dict, cfg_cal: dict,
) -> Defasagem:
    """Atraso que maximiza a correlação entre dv/dt do GPS e a aceleração longitudinal do IMU.

    Para cada atraso candidato refaz o ajuste do eixo frente, então não depende da montagem.
    """
    fc = cfg_filtros["calibracao_corte_hz"]
    lags = _lags(cfg_cal)
    w_lp = passa_baixa(w_cima, fs, fc)
    e1, e2 = base_horizontal(cima)
    a_lp = passa_baixa(acc, fs, fc)
    m1, m2 = a_lp @ e1, a_lp @ e2
    v0 = np.interp(t_imu, t_vel, vel)
    lim = cfg_cal["defasagem_max_s"]
    msk = (v0 > cfg_cal["vel_min_mps"]) & (t_imu > t_vel[0] + lim + 1) & (t_imu < t_vel[-1] - lim - 1)
    corrs = []
    for lag in lags:
        _, tl, te = _alvos(t_imu, fs, t_vel - lag, vel, w_lp, fc)
        th = _rotacao(tl[msk], te[msk], m1[msk], m2[msk])
        corrs.append(correlacao(tl[msk], m1[msk] * np.cos(th) + m2[msk] * np.sin(th)))
    return _melhor(lags, np.array(corrs), "velocidade")


def estimar_defasagem_rumo(
    t_imu: np.ndarray, fs: float, w_cima: np.ndarray, fixes: pd.DataFrame, cfg_filtros: dict, cfg_cal: dict,
    vel_min_rumo: float,
) -> Defasagem:
    """Correlação cruzada entre a taxa de rumo do GPS e a taxa de guinada do giroscópio."""
    ok = fixes["bearing"].notna() & (fixes["speed"] > vel_min_rumo)
    tf = fixes.loc[ok, "t"].to_numpy()
    if len(tf) < 20:
        return Defasagem(0.0, float("nan"), float("nan"), "rumo (dados insuficientes)")
    rumo = np.unwrap(np.radians(fixes.loc[ok, "bearing"].to_numpy()))
    hz = 10.0
    t10 = np.arange(tf[0], tf[-1], 1 / hz)
    # psi = pi/2 - rumo  ->  d(psi)/dt = -d(rumo)/dt (anti-horário positivo, como w_cima)
    taxa_gps = np.gradient(passa_baixa(-np.interp(t10, tf, rumo), hz, cfg_filtros["defasagem_corte_hz"]), 1 / hz)
    perto = np.min(np.abs(t10[:, None] - tf[None, :]), axis=1) < 1.5  # evita lacunas do GPS
    w_lp = passa_baixa(w_cima, fs, cfg_filtros["defasagem_corte_hz"])
    lags = _lags(cfg_cal)
    corrs = []
    for lag in lags:
        tt = t10 - lag  # o GPS no instante t corresponde ao IMU em t - lag
        m = perto & (tt > t_imu[0] + 2) & (tt < t_imu[-1] - 2)
        corrs.append(correlacao(taxa_gps[m], np.interp(tt[m], t_imu, w_lp)))
    return _melhor(lags, np.array(corrs), "rumo")


def ajustar_frente(
    t_imu: np.ndarray, fs: float, acc: np.ndarray, w_cima: np.ndarray, cima: np.ndarray, desvio: dict,
    t_vel: np.ndarray, vel: np.ndarray, cfg_filtros: dict, cfg_cal: dict,
) -> tuple[Montagem, pd.DataFrame]:
    """Ajusta o eixo frente (forma fechada) e os ganhos por eixo contra o GPS.

    acc: aceleração cinemática no aparelho (sem gravidade, sinal físico), (n, 3).
    t_vel/vel: velocidades do GPS já com a defasagem corrigida.
    Devolve a montagem e amostras (10 Hz) para a figura de calibração.
    """
    fc = cfg_filtros["calibracao_corte_hz"]
    v, alvo_long, alvo_esq = _alvos(t_imu, fs, t_vel, vel, passa_baixa(w_cima, fs, fc), fc)
    e1, e2 = base_horizontal(cima)
    a_lp = passa_baixa(acc, fs, fc)
    m1, m2 = a_lp @ e1, a_lp @ e2
    msk = (v > cfg_cal["vel_min_mps"]) & (t_imu > t_vel[0] + 1) & (t_imu < t_vel[-1] - 1)
    if msk.sum() < fs * 20:
        raise ValueError("Poucas amostras em movimento (< 20 s) para calibrar a montagem do celular")

    tl, te, x1, x2 = alvo_long[msk], alvo_esq[msk], m1[msk], m2[msk]
    theta = _rotacao(tl, te, x1, x2)
    theta_long = np.arctan2(np.sum(tl * x2), np.sum(tl * x1))
    theta_lat = np.arctan2(-np.sum(te * x1), np.sum(te * x2))
    c, s = np.cos(theta), np.sin(theta)
    a_long_t, a_esq_t = m1 * c + m2 * s, -m1 * s + m2 * c
    a_long, a_esq = a_long_t[msk], a_esq_t[msk]

    frente = c * e1 + s * e2
    esquerda = np.cross(cima, frente)
    montagem = Montagem(
        cima=cima, frente=frente, esquerda=esquerda, R=np.vstack([frente, esquerda, cima]),
        theta_graus=float(np.degrees(theta)),
        theta_long_graus=float(np.degrees(theta_long)),
        theta_lat_graus=float(np.degrees(theta_lat)),
        r2_long=r2(tl, a_long),
        r2_lat=r2(te, a_esq),
        ganho_long=float(np.sum(a_long * tl) / np.sum(a_long**2)),
        ganho_lat=float(np.sum(a_esq * te) / np.sum(a_esq**2)),
        n_amostras=int(msk.sum()),
        desvio_gravidade_graus=desvio,
    )

    t10 = np.arange(t_imu[0], t_imu[-1], 0.1)
    amostras = pd.DataFrame(
        interpolar(t10, t_imu, np.column_stack([alvo_long, a_long_t, -alvo_esq, -a_esq_t, v])),
        columns=["dvdt_gps", "acc_long_imu", "acc_lat_gps", "acc_lat_imu", "vel_gps"],
    )
    amostras.insert(0, "t", t10)
    amostras["usada"] = np.interp(t10, t_imu, msk.astype(float)) > 0.5
    return montagem, amostras


def ganhos_aplicaveis(montagem: Montagem, cfg_cal: dict) -> tuple[float, float, list[str]]:
    """Ganhos (long, lat) a aplicar, respeitando R² mínimo e limites; devolve avisos."""
    avisos: list[str] = []
    if not cfg_cal.get("corrigir_ganho", False):
        return 1.0, 1.0, avisos
    ganhos = []
    for nome, g, qual in (("longitudinal", montagem.ganho_long, montagem.r2_long),
                          ("lateral", montagem.ganho_lat, montagem.r2_lat)):
        if not (qual >= cfg_cal["r2_min_ganho"]):
            avisos.append(f"Ganho {nome} não aplicado: R² = {qual:.2f} abaixo do mínimo.")
            ganhos.append(1.0)
        else:
            g_lim = float(np.clip(g, cfg_cal["ganho_min"], cfg_cal["ganho_max"]))
            if g_lim != g:
                avisos.append(f"Ganho {nome} {g:.2f} limitado a {g_lim:.2f}.")
            ganhos.append(g_lim)
    return ganhos[0], ganhos[1], avisos


def atitude_por_gravidade(cima_veiculo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Arfagem (nariz para cima +) e rolagem (lado direito para baixo +) em rad.

    `cima_veiculo` é o vetor "cima" do mundo escrito nos eixos do veículo (frente, esquerda, cima).
    """
    arfagem = np.arcsin(np.clip(cima_veiculo[:, 0], -1, 1))
    rolagem = np.arcsin(np.clip(cima_veiculo[:, 1], -1, 1))
    return arfagem, rolagem
