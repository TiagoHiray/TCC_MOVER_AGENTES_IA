"""Geodésia local e convenções de coordenadas do CARLA.

Convenções:
  * ENU local: leste (E), norte (N), plano tangente ao elipsoide WGS84 em (lat0, lon0).
  * psi: ângulo do veículo no plano ENU, anti-horário a partir do leste (rad).
  * rumo: 0 = norte, horário (graus), como o `bearing` do GPS e a bússola.
  * CARLA (Unreal, levógiro): x = leste, y = -norte (sul), z = cima;
    yaw = rumo - 90 (graus), positivo virando à direita.

Não usamos as funções de geolocalização do CARLA para posicionar o veículo: no 0.9.16
`GeoLocation::Transform` deixou de inverter o eixo y (a latitude do GNSS cresce com +y),
enquanto o importador de OpenDRIVE trata o norte como -y. Fazemos a conversão aqui e
geramos o .xodr com `proj_string(...)` centrado na mesma referência.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WGS84_A = 6378137.0
WGS84_F = 1 / 298.257223563
WGS84_E2 = WGS84_F * (2 - WGS84_F)


@dataclass(frozen=True)
class ReferenciaGeo:
    """Origem do plano local; também deve ser a origem do mapa CARLA."""

    lat0: float
    lon0: float
    origem: str = "manual"

    def _raios(self) -> tuple[float, float]:
        """Raios de curvatura meridiano (M) e do primeiro vertical (N) em lat0."""
        s = math.sin(math.radians(self.lat0))
        w = math.sqrt(1 - WGS84_E2 * s * s)
        return WGS84_A * (1 - WGS84_E2) / w**3, WGS84_A / w

    def para_enu(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        """WGS84 -> (leste, norte) em metros. Erro < 1 cm a 1 km da origem."""
        m, n = self._raios()
        leste = np.radians(np.asarray(lon, float) - self.lon0) * n * math.cos(math.radians(self.lat0))
        norte = np.radians(np.asarray(lat, float) - self.lat0) * m
        return leste, norte

    def de_enu(self, leste, norte) -> tuple[np.ndarray, np.ndarray]:
        """(leste, norte) em metros -> (lat, lon) WGS84."""
        m, n = self._raios()
        lat = self.lat0 + np.degrees(np.asarray(norte, float) / m)
        lon = self.lon0 + np.degrees(np.asarray(leste, float) / (n * math.cos(math.radians(self.lat0))))
        return lat, lon

    def proj_string(self) -> str:
        """Projeção para o Osm2Odr/OpenDRIVE com origem nesta referência (sem offsets)."""
        return (
            f"+proj=tmerc +lat_0={self.lat0:.9f} +lon_0={self.lon0:.9f} +k=1 "
            "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        )


def enu_para_carla(leste, norte):
    """(leste, norte) -> (x, y) do CARLA."""
    return np.asarray(leste, float), -np.asarray(norte, float)


def carla_para_enu(x, y):
    """(x, y) do CARLA -> (leste, norte)."""
    return np.asarray(x, float), -np.asarray(y, float)


def psi_para_rumo(psi):
    """Ângulo ENU anti-horário a partir do leste (rad) -> rumo em graus [0, 360)."""
    return np.mod(90.0 - np.degrees(psi), 360.0)


def rumo_para_psi(rumo_graus):
    """Rumo (0 = norte, horário, graus) -> ângulo ENU anti-horário a partir do leste (rad)."""
    return np.radians(90.0 - np.asarray(rumo_graus, float))


def psi_para_yaw_carla(psi):
    """Ângulo ENU (rad) -> yaw do CARLA em graus no intervalo [-180, 180)."""
    return np.mod(-np.degrees(psi) + 180.0, 360.0) - 180.0


def ler_osm(caminho: str | Path) -> dict:
    """Lê limites e vias (`highway`) de um arquivo .osm.

    Devolve {"limites": (minlat, minlon, maxlat, maxlon), "vias": [{"tipo", "nome", "lat", "lon"}]}.
    """
    raiz = ET.parse(caminho).getroot()
    nos = {n.get("id"): (float(n.get("lat")), float(n.get("lon"))) for n in raiz.iter("node")}
    b = raiz.find("bounds")
    if b is not None:
        limites = tuple(float(b.get(k)) for k in ("minlat", "minlon", "maxlat", "maxlon"))
    else:
        lats, lons = zip(*nos.values())
        limites = (min(lats), min(lons), max(lats), max(lons))
    vias = []
    for w in raiz.iter("way"):
        tags = {t.get("k"): t.get("v") for t in w.iter("tag")}
        if "highway" not in tags:
            continue
        pts = [nos[nd.get("ref")] for nd in w.iter("nd") if nd.get("ref") in nos]
        if len(pts) >= 2:
            lat, lon = map(np.array, zip(*pts))
            vias.append({"tipo": tags["highway"], "nome": tags.get("name", ""), "lat": lat, "lon": lon})
    return {"limites": limites, "vias": vias}


def referencia_do_osm(caminho: str | Path) -> ReferenciaGeo:
    """Referência no centro do retângulo <bounds> do arquivo OSM."""
    minlat, minlon, maxlat, maxlon = ler_osm(caminho)["limites"]
    return ReferenciaGeo((minlat + maxlat) / 2, (minlon + maxlon) / 2, origem=f"centro do OSM {Path(caminho).name}")


def altitude_solar(lat: float, lon: float, t_unix) -> np.ndarray:
    """Altitude do Sol em graus (algoritmo de baixa precisão do Astronomical Almanac, ~0,1°)."""
    n = np.asarray(t_unix, float) / 86400.0 + 2440587.5 - 2451545.0
    l_media = np.radians(np.mod(280.460 + 0.9856474 * n, 360))
    g = np.radians(np.mod(357.528 + 0.9856003 * n, 360))
    lam = l_media + np.radians(1.915 * np.sin(g) + 0.020 * np.sin(2 * g))
    eps = np.radians(23.439 - 4e-7 * n)
    dec = np.arcsin(np.sin(eps) * np.sin(lam))
    ar = np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam))
    tsmg = np.radians(np.mod(280.46061837 + 360.98564736629 * n, 360))
    h = tsmg + math.radians(lon) - ar
    phi = math.radians(lat)
    return np.degrees(np.arcsin(np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(h)))
