"""Projeções cartográficas para ler o geoReference dos mapas (.xodr), sem depender do pyproj.

Cobre o que os conversores de mapa usados no projeto escrevem no geoReference:
- Osm2Odr do CARLA: "+proj=tmerc" (padrão: lat_0 = lon_0 = 0) ou um tmerc com lat_0/lon_0;
- netconvert do SUMO com proj.utm (mapa do Eduardo): "+proj=utm +zone=23 ...". Sem +south,
  a coordenada norte fica negativa no hemisfério sul, exatamente como o PROJ faz.

A transversa de Mercator usa a série de Krüger até n^6 (Karney, 2011), o mesmo algoritmo do
"tmerc" do PROJ >= 6 (e do PROJ 7 que o CARLA embute no Osm2Odr). Para outras projeções o
pyproj é usado, se estiver instalado.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

# (semieixo maior, achatamento) dos elipsoides aceitos
ELIPSOIDES = {
    "WGS84": (6378137.0, 1 / 298.257223563),
    "GRS80": (6378137.0, 1 / 298.257222101),  # padrão do PROJ quando nada é informado
}

Projetar = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def ler_proj4(texto: str) -> dict[str, str | None]:
    """'+proj=utm +zone=23 +south' -> {'proj': 'utm', 'zone': '23', 'south': None}."""
    parametros: dict[str, str | None] = {}
    for parte in texto.replace("\n", " ").split():
        if not parte.startswith("+"):
            continue
        chave, _, valor = parte[1:].partition("=")
        parametros[chave.strip().lower()] = valor.strip() if valor else None
    return parametros


@dataclass(frozen=True)
class TransversaMercator:
    """Transversa de Mercator elipsoidal (série de Krüger até n^6)."""

    a: float = ELIPSOIDES["WGS84"][0]
    f: float = ELIPSOIDES["WGS84"][1]
    lat0_graus: float = 0.0
    lon0_graus: float = 0.0
    k0: float = 1.0
    x0: float = 0.0
    y0: float = 0.0

    def __post_init__(self) -> None:
        n = self.f / (2 - self.f)
        n2, n3, n4, n5, n6 = n**2, n**3, n**4, n**5, n**6
        alfas = (
            n / 2 - 2 * n2 / 3 + 5 * n3 / 16 + 41 * n4 / 180 - 127 * n5 / 288 + 7891 * n6 / 37800,
            13 * n2 / 48 - 3 * n3 / 5 + 557 * n4 / 1440 + 281 * n5 / 630 - 1983433 * n6 / 1935360,
            61 * n3 / 240 - 103 * n4 / 140 + 15061 * n5 / 26880 + 167603 * n6 / 181440,
            49561 * n4 / 161280 - 179 * n5 / 168 + 6601661 * n6 / 7257600,
            34729 * n5 / 80640 - 3418889 * n6 / 1995840,
            212378941 * n6 / 319334400,
        )
        raio_retificante = self.a / (1 + n) * (1 + n2 / 4 + n4 / 64 + n6 / 256)
        object.__setattr__(self, "_alfas", alfas)
        object.__setattr__(self, "_A", raio_retificante)
        object.__setattr__(self, "_e", math.sqrt(self.f * (2 - self.f)))
        xi0, _ = self._xi_eta(np.array([math.radians(self.lat0_graus)]), np.array([0.0]))
        object.__setattr__(self, "_xi0", float(xi0[0]))

    def _xi_eta(self, phi: np.ndarray, lam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        e = self._e
        tau = np.tan(phi)
        sigma = np.sinh(e * np.arctanh(e * tau / np.sqrt(1 + tau**2)))
        tau_c = tau * np.sqrt(1 + sigma**2) - sigma * np.sqrt(1 + tau**2)  # tangente da latitude conforme
        xi_l = np.arctan2(tau_c, np.cos(lam))
        eta_l = np.arcsinh(np.sin(lam) / np.sqrt(tau_c**2 + np.cos(lam) ** 2))
        xi, eta = xi_l.copy(), eta_l.copy()
        for j, alfa in enumerate(self._alfas, start=1):
            xi += alfa * np.sin(2 * j * xi_l) * np.cosh(2 * j * eta_l)
            eta += alfa * np.cos(2 * j * xi_l) * np.sinh(2 * j * eta_l)
        return xi, eta

    def __call__(self, lat_graus, lon_graus) -> tuple[np.ndarray, np.ndarray]:
        """(lat, lon) em graus -> (leste, norte) em metros."""
        phi = np.radians(np.asarray(lat_graus, dtype=float))
        dlon = (np.asarray(lon_graus, dtype=float) - self.lon0_graus + 180.0) % 360.0 - 180.0
        xi, eta = self._xi_eta(phi, np.radians(dlon))
        leste = self.x0 + self.k0 * self._A * eta
        norte = self.y0 + self.k0 * self._A * (xi - self._xi0)
        return leste, norte


def _elipsoide(p: dict[str, str | None]) -> tuple[float, float]:
    if p.get("a"):
        a = float(p["a"])
        if p.get("rf"):
            return a, 1 / float(p["rf"])
        if p.get("f"):
            return a, float(p["f"])
        if p.get("b"):
            return a, 1 - float(p["b"]) / a
        return a, 0.0
    nome = (p.get("ellps") or p.get("datum") or "GRS80").upper()
    if nome not in ELIPSOIDES:
        raise ValueError(f"elipsoide/datum não suportado: {nome}")
    return ELIPSOIDES[nome]


def transversa_de_proj4(texto: str) -> TransversaMercator:
    """Monta a projeção a partir de um proj4 com +proj=tmerc ou +proj=utm."""
    p = ler_proj4(texto)
    tipo = (p.get("proj") or "").lower()
    if p.get("units") not in (None, "m"):
        raise ValueError(f"unidade não suportada: {p['units']} (só metros)")
    a, f = _elipsoide(p)
    if tipo == "utm":
        if not p.get("zone"):
            raise ValueError("+proj=utm sem +zone")
        zona = int(p["zone"])
        return TransversaMercator(a, f, 0.0, -183.0 + 6.0 * zona, 0.9996, 500000.0,
                                  10_000_000.0 if "south" in p else 0.0)
    if tipo in ("tmerc", "etmerc"):
        k0 = float(p.get("k_0") or p.get("k") or 1.0)
        return TransversaMercator(a, f, float(p.get("lat_0") or 0.0), float(p.get("lon_0") or 0.0), k0,
                                  float(p.get("x_0") or 0.0), float(p.get("y_0") or 0.0))
    raise ValueError(f"projeção '{tipo}' não implementada aqui")


def criar_projecao(proj4: str) -> tuple[Projetar, str]:
    """Função (lat, lon) -> (leste, norte) do proj4 e o nome do método usado."""
    try:
        return transversa_de_proj4(proj4), "interno (Krüger n^6)"
    except ValueError as erro_interno:
        try:
            from pyproj import Transformer
        except ImportError:
            raise ValueError(f"{erro_interno}; instale o pyproj para usar esta projeção") from None
        transf = Transformer.from_crs("EPSG:4326", proj4, always_xy=True)

        def projetar(lat, lon):
            return transf.transform(np.asarray(lon, float), np.asarray(lat, float))

        return projetar, "pyproj"
