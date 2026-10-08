"""Utilitários de processamento de sinais usados no tratamento."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt

G0 = 9.80665  # gravidade padrão (m/s²)


def passa_baixa(x, fs: float, fc: float | None, ordem: int = 4) -> np.ndarray:
    """Butterworth passa-baixa de fase zero (sosfiltfilt) ao longo do eixo 0.

    O preenchimento das bordas cresce com 1/fc para evitar transientes em filtros lentos.
    """
    x = np.asarray(x, dtype=float)
    if fc is None or fc <= 0 or fc >= 0.5 * fs or x.shape[0] < 8:
        return x.copy()
    sos = butter(ordem, fc, btype="low", fs=fs, output="sos")
    padlen = min(x.shape[0] - 1, max(3 * (2 * sos.shape[0] + 1), int(2 * fs / fc)))
    return sosfiltfilt(sos, x, axis=0, padlen=padlen)


def interpolar(t_novo, t, valores) -> np.ndarray:
    """Interpolação linear de uma série 1D ou de cada coluna de uma matriz (n, k)."""
    t = np.asarray(t, dtype=float)
    v = np.asarray(valores, dtype=float)
    if v.ndim == 1:
        return np.interp(t_novo, t, v)
    return np.column_stack([np.interp(t_novo, t, v[:, j]) for j in range(v.shape[1])])


def unitario(v) -> np.ndarray:
    """Normaliza um vetor (ou cada linha de uma matriz)."""
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def embrulhar(angulo):
    """Leva ângulos (rad) para o intervalo [-pi, pi)."""
    return (np.asarray(angulo) + np.pi) % (2 * np.pi) - np.pi


def correlacao(a, b) -> float:
    """Correlação de Pearson ignorando NaN; devolve NaN se não houver variância."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def r2(alvo, previsto) -> float:
    """Coeficiente de determinação de `previsto` em relação a `alvo`."""
    alvo = np.asarray(alvo, dtype=float)
    previsto = np.asarray(previsto, dtype=float)
    sst = np.sum((alvo - alvo.mean()) ** 2)
    return float(1 - np.sum((alvo - previsto) ** 2) / sst) if sst > 0 else float("nan")


def moda_movel(rotulos, janela: int) -> np.ndarray:
    """Filtro de moda deslizante para rótulos categóricos (janela em amostras)."""
    rotulos = np.asarray(rotulos)
    if janela <= 1 or rotulos.size == 0:
        return rotulos.copy()
    categorias, indices = np.unique(rotulos, return_inverse=True)
    um_quente = np.eye(len(categorias))[indices]
    suavizado = uniform_filter1d(um_quente, size=janela, axis=0, mode="nearest")
    return categorias[np.argmax(suavizado, axis=1)]
