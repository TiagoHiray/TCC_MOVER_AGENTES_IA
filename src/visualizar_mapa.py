# visualizar_mapa.py
# Plota as vias do .osm e sobrepoe o track GPS (Sensor Logger) para conferir
# geometria e alinhamento antes de converter para .xodr no CARLA.
# Uso: python visualizar_mapa.py

import csv
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OSM_PATH = DATA_DIR / "openstreetviewmap.osm"
GPS_SENSORLOGGER = DATA_DIR / "csv_maua" / "Location.csv"
OUT_PNG = DATA_DIR / "mapa_check.png"

# Tipos de via que representam ruas navegaveis
HIGHWAY_DRIVEL = {
    "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
    "residential", "service", "motorway_link", "trunk_link", "primary_link",
    "secondary_link", "tertiary_link", "living_street", "road",
}


def parse_osm(path):
    """Retorna (nos: id->(lon,lat), vias: lista de listas de (lon,lat)), bounds."""
    tree = ET.parse(path)
    root = tree.getroot()

    nodes = {}
    for n in root.findall("node"):
        nodes[n.get("id")] = (float(n.get("lon")), float(n.get("lat")))

    bounds = None
    b = root.find("bounds")
    if b is not None:
        bounds = (float(b.get("minlon")), float(b.get("minlat")),
                  float(b.get("maxlon")), float(b.get("maxlat")))

    vias = []
    for w in root.findall("way"):
        tags = {t.get("k"): t.get("v") for t in w.findall("tag")}
        if tags.get("highway") not in HIGHWAY_DRIVEL:
            continue
        pts = [nodes[nd.get("ref")] for nd in w.findall("nd") if nd.get("ref") in nodes]
        if len(pts) >= 2:
            vias.append(pts)
    return nodes, vias, bounds


def _to_float(s):
    s = (s or "").strip().replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def parse_gps_sensorlogger(path):
    """Location.csv (Sensor Logger): colunas ... longitude, latitude no fim."""
    lons, lats = [], []
    if not path.exists():
        return lons, lats
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            lon = _to_float(row.get("longitude"))
            lat = _to_float(row.get("latitude"))
            if lon is not None and lat is not None:
                lons.append(lon)
                lats.append(lat)
    return lons, lats


def dentro(bounds, lons, lats):
    if not bounds or not lons:
        return 0.0
    minlon, minlat, maxlon, maxlat = bounds
    n = sum(1 for lo, la in zip(lons, lats)
            if minlon <= lo <= maxlon and minlat <= la <= maxlat)
    return 100.0 * n / len(lons)


def main():
    nodes, vias, bounds = parse_osm(OSM_PATH)
    print(f"[OSM] {len(nodes)} nos, {len(vias)} vias navegaveis")
    if bounds:
        print(f"[OSM] bounds lon[{bounds[0]:.5f},{bounds[2]:.5f}] "
              f"lat[{bounds[1]:.5f},{bounds[3]:.5f}]")

    sl_lon, sl_lat = parse_gps_sensorlogger(GPS_SENSORLOGGER)
    print(f"[GPS SensorLogger] {len(sl_lon)} pontos, {dentro(bounds, sl_lon, sl_lat):.0f}% dentro do mapa")

    fig, ax = plt.subplots(figsize=(11, 9))
    for pts in vias:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color="#888888", linewidth=0.8, zorder=1)

    if bounds:
        minlon, minlat, maxlon, maxlat = bounds
        ax.plot([minlon, maxlon, maxlon, minlon, minlon],
                [minlat, minlat, maxlat, maxlat, minlat],
                color="#1f77b4", linewidth=1.2, linestyle="--",
                label="OSM bounds", zorder=2)

    if sl_lon:
        ax.plot(sl_lon, sl_lat, color="#d62728", linewidth=1.6,
                label="GPS (Sensor Logger)", zorder=3)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Mapa OSM x track GPS (csv_maua)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    print(f"[OK] Figura salva em {OUT_PNG}")
    plt.show()


if __name__ == "__main__":
    main()
