"""Diagnóstico de uma corrida: onde o caminhão andou devagar e o que o governador pedia ali.

Lê a telemetria de uma corrida (telemetria.csv) ou de uma bateria do benchmark
(telemetria_com_agentes.csv / telemetria_sem_agentes.csv) e mostra, por trechos de 10 m da volta,
quanto tempo o caminhão passou, a velocidade medida, a desejada (o que o TM recebeu), o alvo do
governador, quem limitava, o modo do tick, os pedais e quantas vezes a aceleração trocou de sinal
(oscilação entre acelerar e frear).

Uso, a partir da pasta src/:
    python -m mover.autonomo.diagnostico ../data/benchmark/<data>/telemetria_com_agentes.csv
    python -m mover.autonomo.diagnostico ../data/autonomo/corridas/<id>/telemetria.csv --trechos 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # execução direta
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd


def resumo_por_trecho(tel: pd.DataFrame, passo_m: float = 10.0) -> pd.DataFrame:
    dt = float(np.median(np.diff(tel["sim_time"]))) if len(tel) > 1 else 0.05
    trecho = (np.floor(tel["d"].to_numpy(float) / passo_m) * passo_m).astype(int)
    a = tel["acc_long"].to_numpy(float)
    trocas = np.r_[0, (np.sign(a[1:]) != np.sign(a[:-1])) & (np.abs(a[1:]) > 0.3) & (np.abs(a[:-1]) > 0.3)]
    marchas = np.r_[0, np.diff(tel["gear"].to_numpy(float)) != 0] if "gear" in tel else np.zeros(len(tel))
    dados = tel.assign(trecho_m=trecho, trocas=trocas, marchas=marchas)
    colunas = {"tempo_s": ("sim_time", lambda s: len(s) * dt), "v_kmh": ("speed_kmh", "mean"),
               "v_min_kmh": ("speed_kmh", "min"), "v_desejada_kmh": ("v_desejada_kmh", "mean"),
               "v_alvo_kmh": ("v_alvo_kmh", "mean"), "acelerador": ("throttle", "mean"), "freio": ("brake", "mean"),
               "trocas_acel": ("trocas", "sum"), "trocas_marcha": ("marchas", "sum")}
    if "desvio_m" in tel:
        colunas["desvio_m"] = ("desvio_m", lambda s: float(np.abs(s).mean()))
    tabela = dados.groupby("trecho_m").agg(**colunas)
    for nome in ("limitado_por", "modo_tick"):
        if nome in tel:
            tabela[nome] = dados.groupby("trecho_m")[nome].agg(lambda s: s.mode().iloc[0] if s.notna().any() else "-")
    return tabela.round(2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Onde o caminhão andou devagar e o que o governador pedia ali.")
    parser.add_argument("telemetria", help="telemetria.csv da corrida ou telemetria_*_agentes.csv do benchmark")
    parser.add_argument("--trechos", type=int, default=12, help="quantos trechos mais lentos mostrar")
    parser.add_argument("--passo", type=float, default=10.0, help="tamanho do trecho (m)")
    args = parser.parse_args(argv)

    tel = pd.read_csv(args.telemetria)
    tabela = resumo_por_trecho(tel, args.passo)
    dt = float(np.median(np.diff(tel["sim_time"]))) if len(tel) > 1 else 0.05
    print(f"{args.telemetria}: {len(tel)} linhas, {len(tel) * dt:.0f} s, {tel['d'].iloc[-1]:.0f} m, "
          f"{tel['speed_kmh'].mean():.1f} km/h de média, modo {tel['modo'].iloc[0] if 'modo' in tel else '?'}")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print(f"\nOs {args.trechos} trechos de {args.passo:.0f} m em que o caminhão passou mais tempo:")
    print(tabela.sort_values("tempo_s", ascending=False).head(args.trechos).sort_index().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
