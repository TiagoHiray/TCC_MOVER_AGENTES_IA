"""Confere, sem abrir o CARLA, se a volta gravada cabe nas vias do mapa .xodr.

Uso, a partir da pasta src/ do repositório:
    python -m mover.simulacao.verificar_mapa                        # mapa do config/simulacao.yaml
    python -m mover.simulacao.verificar_mapa --mapa outro_mapa.xodr
    python -m mover.simulacao.verificar_mapa --offset-x 2 --rotacao -1   # ajuste manual extra

Imprime o relatório do alinhamento (georreferência, ICP, correção de borda por bloco) e grava
em data/simulacao/: alinhamento.json, poses_carla.csv (a pose imposta ao caminhão em cada
quadro) e alinhamento.png.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # execução direta: python src/mover/simulacao/verificar_mapa.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from mover.config import caminho, carregar_yaml
from mover.simulacao.alinhamento import ResultadoAlinhamento, alinhar, carregar_parametros
from mover.simulacao.opendrive import MapaOpenDrive, ler_xodr

log = logging.getLogger("mover.simulacao")


def alinhar_volta(cfg: dict[str, Any], arquivo_mapa: str | Path | None = None, telemetria: pd.DataFrame | None = None,
                  ajuste_manual: dict[str, float] | None = None, texto_xodr: str | None = None,
                  ) -> tuple[MapaOpenDrive, ResultadoAlinhamento, str]:
    """Lê o mapa e a telemetria do YAML (ou os informados) e devolve mapa, alinhamento e texto do .xodr."""
    if texto_xodr is None:
        arquivo = caminho(arquivo_mapa or cfg["mapa"]["arquivo"])
        if not arquivo.exists():
            raise SystemExit(f"Mapa não encontrado: {arquivo}. Copie o mapa_final.xodr do Eduardo para data/mapas/ "
                             "ou informe outro com --mapa.")
        texto_xodr = arquivo.read_text(encoding="utf-8")
        log.info("Mapa: %s", arquivo)
    mapa = ler_xodr(texto_xodr, float(cfg["mapa"].get("passo_m", 0.5)), tuple(cfg["mapa"].get("tipos_faixa", ("driving",))))
    if telemetria is None:
        arquivo_tel = caminho(cfg["telemetria"]["arquivo"])
        if not arquivo_tel.exists():
            raise SystemExit(f"Telemetria não encontrada: {arquivo_tel}. Rode antes: python -m mover.tratamento.tratar_dados")
        telemetria = pd.read_csv(arquivo_tel)
    taxa = 1.0 / float(np.median(np.diff(telemetria["sim_time"].to_numpy(float))))
    resultado = alinhar(mapa, telemetria, carregar_parametros(cfg), ajuste_manual, taxa_hz=taxa)
    return mapa, resultado, texto_xodr


def gravar_saidas(cfg: dict[str, Any], mapa: MapaOpenDrive, resultado: ResultadoAlinhamento, figura: bool = True) -> Path:
    pasta = caminho(cfg["alinhamento"].get("saida", {}).get("pasta", "data/simulacao"))
    pasta.mkdir(parents=True, exist_ok=True)
    with (pasta / "alinhamento.json").open("w", encoding="utf-8") as f:
        json.dump(resultado.para_json(), f, ensure_ascii=False, indent=2)
    resultado.poses.round(4).to_csv(pasta / "poses_carla.csv", index=False)
    if figura:
        desenhar(mapa, resultado, pasta / "alinhamento.png")
    return pasta


def desenhar(mapa: MapaOpenDrive, resultado: ResultadoAlinhamento, destino: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    poses = resultado.poses
    fig, eixos = plt.subplots(1, 2, figsize=(16, 7.5), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = eixos[0]
    for via in np.unique(mapa.vias):
        sel = mapa.vias == via
        ax.plot(mapa.pontos[sel, 0], mapa.pontos[sel, 1], color="0.78", lw=6, solid_capstyle="round", zorder=1)
    ax.plot(poses["x_mapa_rigido"], poses["y_mapa_rigido"], color="0.35", lw=0.9, ls="--", zorder=2,
            label="volta alinhada (sem correção de borda)")
    ax.plot(poses["x_mapa"], poses["y_mapa"], color="tab:blue", lw=1.2, zorder=3, label="pose imposta ao caminhão")
    corrigidos = (poses["correcao_m"] > 0.05).to_numpy()
    sc = ax.scatter(poses["x_mapa"].to_numpy()[corrigidos], poses["y_mapa"].to_numpy()[corrigidos],
                    c=poses["correcao_m"].to_numpy()[corrigidos], cmap="autumn_r", s=6, zorder=4,
                    vmin=0, vmax=max(1.0, float(poses["correcao_m"].max())))
    for t in range(0, int(poses["sim_time"].max()) + 1, 10):
        i = int(np.argmin(np.abs(poses["sim_time"].to_numpy() - t)))
        ax.annotate(f"{t}s", (poses["x_mapa"].iat[i], poses["y_mapa"].iat[i]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("x do mapa (m)")
    ax.set_ylabel("y do mapa (m)")
    ax.set_title("Vias do mapa (cinza) e pose imposta no CARLA\npontos coloridos = quadros com correção de borda")
    fig.colorbar(sc, ax=ax, shrink=0.7, label="correção de borda (m)")
    ax.legend(loc="lower right")

    ax = eixos[1]
    ax.plot(poses["sim_time"], poses["dist_faixa_sem_correcao_m"], color="0.5", lw=1, label="depois do ICP (sem correção)")
    ax.plot(poses["sim_time"], poses["dist_faixa_m"], color="tab:blue", lw=1.2, label="pose imposta")
    ax.plot(poses["sim_time"], poses["correcao_m"], color="tab:red", lw=1, label="correção aplicada")
    meia = float(np.median(mapa.larguras)) / 2
    ax.axhline(meia, color="k", ls="--", lw=0.8, label=f"borda da faixa ({meia:.1f} m)")
    for k in range(0, int(poses["sim_time"].max()) + 1, 10):
        ax.axvline(k, color="0.85", lw=0.6, zorder=0)
    ax.set_xlabel("tempo (s)")
    ax.set_ylabel("metros")
    ax.set_title("Distância ao centro da faixa e correção por quadro")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(destino, dpi=110)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Confere o alinhamento da volta gravada com o mapa .xodr (sem CARLA).")
    parser.add_argument("--config", default="config/simulacao.yaml")
    parser.add_argument("--mapa", help="arquivo .xodr (padrão: YAML)")
    parser.add_argument("--telemetria", help="CSV tratado (padrão: YAML)")
    parser.add_argument("--offset-x", type=float, default=0.0, help="desloca a volta no mapa (m, leste)")
    parser.add_argument("--offset-y", type=float, default=0.0, help="desloca a volta no mapa (m, norte do mapa)")
    parser.add_argument("--rotacao", type=float, default=0.0, help="gira a volta no mapa (graus, anti-horário)")
    parser.add_argument("--sem-icp", action="store_true")
    parser.add_argument("--sem-correcao-borda", action="store_true")
    parser.add_argument("--sem-figura", action="store_true")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    cfg = carregar_yaml(args.config)
    if args.sem_icp:
        cfg.setdefault("alinhamento", {}).setdefault("icp", {})["ativo"] = False
    if args.sem_correcao_borda:
        cfg.setdefault("alinhamento", {}).setdefault("correcao_borda", {})["ativa"] = False
    telemetria = pd.read_csv(caminho(args.telemetria)) if args.telemetria else None
    ajuste = {"offset_x": args.offset_x, "offset_y": args.offset_y, "rotacao_graus": args.rotacao}
    mapa, resultado, _ = alinhar_volta(cfg, args.mapa, telemetria, ajuste)
    print(f"Faixas do mapa por tipo: {dict(mapa.tipos_faixa)}; {mapa.n_vias} vias; geoReference: {mapa.georeferencia}")
    print(resultado.resumo())
    pasta = gravar_saidas(cfg, mapa, resultado, figura=not args.sem_figura)
    print(f"Saídas gravadas em {pasta}")


if __name__ == "__main__":
    main()
