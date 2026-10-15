"""Diagnóstico de uma corrida: onde o caminhão andou devagar e o que o governador pedia ali.

Lê a telemetria de uma corrida (telemetria.csv) ou de uma bateria do benchmark
(telemetria_com_agentes.csv / telemetria_sem_agentes.csv) e mostra, por trechos de 10 m da volta,
quanto tempo o caminhão passou, a velocidade medida, a desejada (o que o TM recebeu), o alvo do
governador, quem limitava, o modo do tick, os pedais e quantas vezes a aceleração trocou de sinal
(oscilação entre acelerar e frear).

Uso, a partir da pasta src/:
    python -m mover.autonomo.diagnostico                   # a telemetria com agentes mais recente
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

from mover.autonomo.consumo import Limiares, eventos_jerk, jerk, suavizar
from mover.config import caminho, carregar_yaml


def telemetria_mais_recente() -> Path | None:
    """A telemetria com agentes mais recente: do benchmark ou de uma corrida (também as interrompidas)."""
    cfg = carregar_yaml("config/autonomo.yaml") if caminho("config/autonomo.yaml").exists() else {}
    pasta_bench = caminho((cfg.get("benchmark") or {}).get("pasta", "data/benchmark"))
    pasta_corridas = caminho((cfg.get("corridas") or {}).get("pasta", "data/autonomo/corridas"))
    candidatos = list(pasta_bench.glob("*/telemetria_com_agentes.csv")) + list(pasta_corridas.glob("*_agentes/telemetria.csv"))
    return max(candidatos, key=lambda p: p.stat().st_mtime) if candidatos else None


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


def eventos_da_corrida(tel: pd.DataFrame, limiares: Limiares, comprimento_m: float, janela_s: float = 1.0) -> pd.DataFrame:
    """Os eventos bruscos (mesma conta do benchmark) com o contexto de cada um.

    `dz_m` e `dpitch_graus` (maior variação na janela) apontam degraus do mapa; `troca_marcha`, o câmbio.
    """
    t = tel["sim_time"].to_numpy(float)
    if len(t) < 3:
        return pd.DataFrame()
    dt = float(np.median(np.diff(t)))
    a_suave = suavizar(tel["acc_long"].to_numpy(float), dt, limiares.fc_hz)
    linhas = []
    for ev in eventos_jerk(t, jerk(a_suave, dt), limiares):
        k = int(np.searchsorted(t, ev["t_pico"]))
        perto = (t >= ev["t_pico"] - janela_s) & (t <= ev["t_pico"] + janela_s)
        trecho = tel[perto]
        d = float(tel["d"].iloc[k])
        volta = int(tel["volta"].iloc[k]) + 1 if "volta" in tel else int(d // comprimento_m) + 1
        s = float(tel["s"].iloc[k]) if "s" in tel else d % comprimento_m
        linha = {"t_s": round(ev["t_pico"], 2), "volta": volta, "s_m": round(s, 1),
                 "nivel": ev["nivel"], "jerk_mps3": round(ev["jerk_mps3"], 1), "acel_mps2": round(float(a_suave[k]), 2),
                 "v_kmh": round(float(tel["speed_kmh"].iloc[k]), 1),
                 "acelerador": round(float(trecho["throttle"].max()), 2), "freio": round(float(trecho["brake"].max()), 2)}
        if "gear" in tel:
            linha["troca_marcha"] = int(trecho["gear"].nunique() > 1)
        if "z" in tel:
            linha["dz_m"] = round(float(np.abs(np.diff(trecho["z"].to_numpy(float))).max(initial=0.0)), 3)
        if "pitch_carla" in tel:
            linha["dpitch_graus"] = round(float(trecho["pitch_carla"].max() - trecho["pitch_carla"].min()), 2)
        for nome in ("limitado_por", "modo_tick"):
            if nome in tel:
                linha[nome] = tel[nome].iloc[k]
        linhas.append(linha)
    return pd.DataFrame(linhas)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Onde o caminhão andou devagar e o que o governador pedia ali.")
    parser.add_argument("telemetria", nargs="?", help="telemetria.csv de uma corrida ou telemetria_*_agentes.csv do benchmark "
                                                     "(padrão: a com agentes mais recente)")
    parser.add_argument("--trechos", type=int, default=12, help="quantos trechos mais lentos mostrar")
    parser.add_argument("--passo", type=float, default=10.0, help="tamanho do trecho (m)")
    args = parser.parse_args(argv)

    arquivo = Path(args.telemetria) if args.telemetria else telemetria_mais_recente()
    if arquivo is None or not arquivo.is_file():
        raise SystemExit("Nenhuma telemetria com agentes encontrada. Rode uma volta antes (ela grava mesmo se for "
                         "interrompida com Ctrl+C):\n    python -m mover.autonomo.dirigir --sem-painel --sem-render --sem-espera "
                         "--provedor falso")
    tel = pd.read_csv(arquivo)
    tabela = resumo_por_trecho(tel, args.passo)
    dt = float(np.median(np.diff(tel["sim_time"]))) if len(tel) > 1 else 0.05
    print(f"{arquivo}: {len(tel)} linhas, {len(tel) * dt:.0f} s, {tel['d'].iloc[-1]:.0f} m, "
          f"{tel['speed_kmh'].mean():.1f} km/h de média, modo {tel['modo'].iloc[0] if 'modo' in tel else '?'}")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print(f"\nOs {args.trechos} trechos de {args.passo:.0f} m em que o caminhão passou mais tempo:")
    print(tabela.sort_values("tempo_s", ascending=False).head(args.trechos).sort_index().to_string())

    cfg = carregar_yaml("config/autonomo.yaml") if caminho("config/autonomo.yaml").exists() else {}
    lim = cfg.get("limiares") or {}
    limiares = Limiares(float(lim.get("atencao_mps3", 2.5)), float(lim.get("critico_mps3", 5.0)),
                        float(lim.get("agrupar_s", 1.0)), float(lim.get("fc_hz", 1.5)))
    comprimento = float(tel["d"].iloc[-1] / max(1, int(tel["volta"].max()) + 1)) if "volta" in tel else float(tel["d"].iloc[-1])
    eventos = eventos_da_corrida(tel, limiares, comprimento)
    print(f"\nEventos bruscos (|jerk| >= {limiares.atencao_mps3} m/s³; crítico >= {limiares.critico_mps3}): {len(eventos)}")
    if len(eventos):
        print(eventos.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
