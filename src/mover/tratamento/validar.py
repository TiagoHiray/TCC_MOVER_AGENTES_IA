"""Figuras de validação do tratamento, geradas a partir dos arquivos de saída.

Uso, a partir da pasta src/:
    python -m mover.tratamento.validar
    python -m mover.tratamento.validar --saida data/tratado
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # execução direta
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from mover.config import caminho, carregar_yaml  # noqa: E402
from mover.tratamento import geo  # noqa: E402

CORES_MANOBRA = {
    "acelerar": "#2ca02c", "cruzeiro": "#d9d9d9", "frear": "#d62728",
    "curva_esq": "#1f77b4", "curva_dir": "#ff7f0e", "parar": "#000000",
}


def _faixas_manobra(ax, t: np.ndarray, manobra: np.ndarray) -> None:
    mudancas = np.flatnonzero(manobra[1:] != manobra[:-1]) + 1
    for i, f in zip(np.r_[0, mudancas], np.r_[mudancas, len(manobra)]):
        ax.axvspan(t[i], t[min(f, len(t) - 1)], color=CORES_MANOBRA.get(manobra[i], "white"), lw=0)
    ax.set_yticks([])
    presentes = [m for m in CORES_MANOBRA if m in set(manobra)]
    ax.legend(handles=[Patch(color=CORES_MANOBRA[m], label=m) for m in presentes],
              ncol=len(presentes), loc="upper center", bbox_to_anchor=(0.5, -0.6), fontsize=8, frameon=False)


def figura_trajetoria(df: pd.DataFrame, fixes: pd.DataFrame, rel: dict, arquivo_osm: Path | None, destino: Path) -> None:
    ref = geo.ReferenciaGeo(rel["referencia_geo"]["lat0"], rel["referencia_geo"]["lon0"])
    leste, norte = df["x"].to_numpy(), -df["y"].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 8))
    if arquivo_osm is not None and arquivo_osm.exists():
        for via in geo.ler_osm(arquivo_osm)["vias"]:
            e, n = ref.para_enu(via["lat"], via["lon"])
            principal = via["tipo"] in ("trunk", "primary", "secondary", "tertiary")
            ax.plot(e, n, color="#9a9a9a" if principal else "#cfcfcf", lw=4 if principal else 2, zorder=1,
                    solid_capstyle="round")
    ax.scatter(fixes["leste"], fixes["norte"], s=16, facecolors="none", edgecolors="crimson", lw=0.8, zorder=2,
               label="fixes do GPS")
    sc = ax.scatter(leste, norte, c=df["speed_kmh"], s=4, cmap="viridis", zorder=3)
    ax.plot(leste[0], norte[0], "k^", ms=10, zorder=4, label="início")
    ax.plot(leste[-1], norte[-1], "ks", ms=7, zorder=4, label="fim")
    margem = 40
    ax.set_xlim(min(leste.min(), fixes["leste"].min()) - margem, max(leste.max(), fixes["leste"].max()) + margem)
    ax.set_ylim(min(norte.min(), fixes["norte"].min()) - margem, max(norte.max(), fixes["norte"].max()) + margem)
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, shrink=0.8, label="velocidade (km/h)")
    ax.set_xlabel("leste (m) = x do CARLA")
    ax.set_ylabel("norte (m) = -y do CARLA")
    ax.set_title(f"Trajetória fundida sobre as vias do OSM (RMS dos resíduos {rel['fusao']['rms_residuo_pos_m']:.1f} m)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(destino, dpi=130)
    plt.close(fig)


def figura_series(df: pd.DataFrame, fixes: pd.DataFrame, destino: Path) -> None:
    t = df["sim_time"].to_numpy()
    ft = fixes["sim_time"].to_numpy()
    fig, axs = plt.subplots(6, 1, figsize=(12, 14), sharex=True,
                            gridspec_kw={"height_ratios": [3, 3, 2, 3, 2.5, 0.5]})
    axs[0].plot(t, df["speed_kmh"], lw=1.2, label="fundida GPS+IMU")
    axs[0].plot(ft, fixes["speed"] * 3.6, ".", ms=4, label="GPS")
    axs[0].set_ylabel("velocidade (km/h)")

    axs[1].plot(t, df["acc_long"], lw=0.9, label="longitudinal")
    axs[1].plot(t, df["acc_lat"], lw=0.9, alpha=0.7, label="lateral (direita +)")
    axs[1].set_ylabel("aceleração (m/s²)")

    axs[2].plot(t, df["jerk_long"], lw=0.8, color="tab:red")
    axs[2].set_ylabel("jerk long. (m/s³)")

    ok = fixes["bearing"].notna().to_numpy()
    rumo = np.degrees(np.unwrap(np.radians(fixes.loc[ok, "bearing"].to_numpy())))
    taxa_gps = np.diff(rumo) / np.diff(ft[ok])
    axs[3].plot(t, df["yaw_rate_dps"], lw=1, label="giroscópio (viés removido)")
    axs[3].plot(0.5 * (ft[ok][1:] + ft[ok][:-1]), taxa_gps, ".", ms=4, label="GPS (Δrumo/Δt)")
    axs[3].set_ylabel("guinada (°/s, direita +)")

    axs[4].plot(t, df["alt_m"], lw=1.2, label="barômetro ancorado no GPS")
    axs[4].plot(ft, fixes["alt_msl"], ".", ms=4, label="GPS")
    axs[4].set_ylabel("altitude (m)")
    ax_r = axs[4].twinx()
    ax_r.plot(t, df["grade_pct"], color="tab:purple", lw=0.9, alpha=0.7)
    ax_r.set_ylabel("rampa (%)", color="tab:purple")

    _faixas_manobra(axs[5], t, df["manobra"].to_numpy())
    axs[5].set_ylabel("manobra", rotation=0, ha="right", va="center")
    axs[5].set_xlabel("tempo (s)")
    for ax in axs[:5]:
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Séries tratadas (20 Hz)")
    fig.tight_layout()
    fig.savefig(destino, dpi=120)
    plt.close(fig)


def figura_calibracao(cal: pd.DataFrame, rel: dict, destino: Path) -> None:
    m = rel["montagem"]
    usada = cal["usada"].astype(bool)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, xcol, ycol, ganho, r2, titulo in (
        (axs[0], "acc_long_imu", "dvdt_gps", m["ganho_longitudinal_ajustado"], m["r2_longitudinal"], "Longitudinal"),
        (axs[1], "acc_lat_imu", "acc_lat_gps", m["ganho_lateral_ajustado"], m["r2_lateral"], "Lateral (direita +)"),
    ):
        x, y = cal.loc[usada, xcol], cal.loc[usada, ycol]
        ax.scatter(x, y, s=3, alpha=0.35)
        lim = np.nanmax(np.abs(np.r_[x, y])) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="y = x")
        ax.plot([-lim, lim], [-ganho * lim, ganho * lim], "r-", lw=1, label=f"ganho {ganho:.2f}")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("IMU (m/s²)")
        ax.set_ylabel("referência do GPS (m/s²)")
        ax.set_title(f"{titulo}: R² = {r2:.2f}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    d = rel["defasagem_gps"]
    if d["curva"]["lags_s"]:
        axs[2].plot(d["curva"]["lags_s"], d["curva"]["correlacao"])
        axs[2].axvline(d["segundos"], color="r", ls="--", label=f"defasagem {d['segundos']:.2f} s")
        axs[2].legend(fontsize=8)
    axs[2].set_xlabel("defasagem do GPS (s)")
    metodo = d.get("metodo", "")
    axs[2].set_ylabel("correlação dv/dt GPS x acel. IMU" if metodo == "velocidade" else "correlação guinada GPS x giroscópio")
    axs[2].set_title(f"Defasagem do GPS (método: {metodo})")
    axs[2].grid(alpha=0.3)
    fig.suptitle(f"Calibração da montagem (θ = {m['theta_graus']:.1f}°, banda 0,3 Hz)")
    fig.tight_layout()
    fig.savefig(destino, dpi=120)
    plt.close(fig)


def figura_jerk(df: pd.DataFrame, destino: Path) -> None:
    j = df["jerk_long"].to_numpy()
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.5), gridspec_kw={"width_ratios": [1, 2]})
    axs[0].hist(j, bins=80, color="tab:red", alpha=0.75, log=True)
    for p, estilo in ((1, ":"), (5, "--"), (95, "--"), (99, ":")):
        v = np.percentile(j, p)
        axs[0].axvline(v, color="k", ls=estilo, lw=0.9)
        axs[0].text(v, axs[0].get_ylim()[1] * 0.5, f"p{p}\n{v:.2f}", fontsize=7, ha="center")
    axs[0].set_xlabel("jerk longitudinal (m/s³)")
    axs[0].set_ylabel("amostras (log)")
    axs[0].set_title("Distribuição do jerk")
    axs[1].plot(df["sim_time"], j, lw=0.8, color="tab:red")
    for p in (1, 99):
        axs[1].axhline(np.percentile(j, p), color="k", ls=":", lw=0.8)
    axs[1].set_xlabel("tempo (s)")
    axs[1].set_ylabel("jerk longitudinal (m/s³)")
    axs[1].set_title("Jerk ao longo da volta (linhas: p1 e p99)")
    for ax in axs:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(destino, dpi=120)
    plt.close(fig)


def gerar_figuras(saida: Path, cfg: dict) -> list[Path]:
    """Lê os arquivos tratados em `saida` e grava as figuras em `saida/<figuras>`."""
    cs = cfg["saida"]
    df = pd.read_csv(saida / cs["telemetria"])
    fixes = pd.read_csv(saida / cs["gps"])
    cal = pd.read_csv(saida / cs["calibracao"])
    with (saida / cs["relatorio"]).open(encoding="utf-8") as f:
        rel = json.load(f)
    pasta = saida / cs["figuras"]
    pasta.mkdir(parents=True, exist_ok=True)
    arquivo_osm = caminho(cfg["referencia_geo"]["arquivo_osm"]) if cfg["referencia_geo"].get("arquivo_osm") else None
    destinos = [pasta / n for n in ("trajetoria.png", "series_temporais.png", "calibracao.png", "distribuicao_jerk.png")]
    figura_trajetoria(df, fixes, rel, arquivo_osm, destinos[0])
    figura_series(df, fixes, destinos[1])
    figura_calibracao(cal, rel, destinos[2])
    figura_jerk(df, destinos[3])
    return destinos


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Gera as figuras de validação do tratamento.")
    parser.add_argument("--config", default="config/tratamento.yaml")
    parser.add_argument("--saida", help="pasta com os arquivos tratados (padrão: YAML)")
    args = parser.parse_args(argv)
    cfg = carregar_yaml(args.config)
    for p in gerar_figuras(caminho(args.saida or cfg["saida"]["pasta"]), cfg):
        print(p)


if __name__ == "__main__":
    main()
