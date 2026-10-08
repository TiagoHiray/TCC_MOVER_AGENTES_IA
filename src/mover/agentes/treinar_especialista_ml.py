"""Retreina o IsolationForest da POC com a telemetria tratada do csv_maua.

Uso, a partir da pasta src/ do repositório:
    python -m mover.agentes.treinar_especialista_ml
    python -m mover.agentes.treinar_especialista_ml --entrada data/tratado/telemetria_tratada.csv

Grava data/modelos/especialista_ml.joblib (modelo + lista de features) e um JSON de
metadados ao lado. Não mexe no data/modelo_especialista.joblib da POC.

Diferenças em relação ao treinar_ml.py da POC:
- features no referencial do veículo (speed_mps, acc_long, acc_lat, acc_vert, jerk_long,
  yaw_rate_dps), em vez de acc_x/acc_y, que estão no referencial do mundo e mudam com o rumo;
- throttle/brake/steer ficam de fora porque aqui são estimativas derivadas da própria aceleração;
- só linhas com fonte = real entram no treino.

Atenção: com uma única volta, o modelo é treinado e avaliado nos mesmos dados. Ele aponta os
~5 % de quadros mais atípicos desta volta (contaminação), não anomalias em sentido absoluto.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # execução direta: python src/mover/agentes/treinar_especialista_ml.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from mover.config import RAIZ, caminho, carregar_yaml

log = logging.getLogger("mover.agentes")


def treinar(cfg: dict[str, Any], entrada: Path | None = None, saida: Path | None = None) -> dict[str, Any]:
    """Treina, grava o modelo e devolve os metadados."""
    import joblib
    import sklearn
    from sklearn.ensemble import IsolationForest

    cm = cfg["especialista_ml"]
    entrada = entrada or caminho(cfg["entrada"]["telemetria"])
    saida = saida or caminho(cm["modelo"])
    features = list(cm["features"])

    df = pd.read_csv(entrada)
    if "fonte" in df.columns:
        df = df[df["fonte"] == "real"]
    x = df[features].dropna()
    modelo = IsolationForest(
        n_estimators=int(cm.get("n_arvores", 200)),
        contamination=float(cm["contaminacao"]),
        random_state=int(cm.get("semente", 42)),
        n_jobs=-1,
    )
    modelo.fit(x)

    rotulos = modelo.predict(x)
    pontuacao = modelo.decision_function(x)
    try:
        origem = str(Path(entrada).resolve().relative_to(RAIZ))
    except ValueError:
        origem = str(entrada)
    metadados = {
        "treinado_em": datetime.now().astimezone().isoformat(timespec="seconds"),
        "origem": origem,
        "amostras": int(len(x)),
        "features": features,
        "contaminacao": float(cm["contaminacao"]),
        "n_arvores": int(cm.get("n_arvores", 200)),
        "semente": int(cm.get("semente", 42)),
        "versao_sklearn": sklearn.__version__,
        "fracao_atipica_no_treino": float(np.mean(rotulos == -1)),
        "pontuacao_percentis": {f"p{p}": float(np.percentile(pontuacao, p)) for p in (1, 5, 50, 95)},
        "estatisticas_features": {
            f: {"media": float(x[f].mean()), "desvio": float(x[f].std()), "min": float(x[f].min()), "max": float(x[f].max())}
            for f in features
        },
    }
    saida.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modelo": modelo, "features": features, "metadados": metadados}, saida)
    with saida.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(metadados, f, ensure_ascii=False, indent=2)
    log.info("Modelo gravado em %s (%d amostras, %.1f %% atípicas no treino)",
             saida, metadados["amostras"], 100 * metadados["fracao_atipica_no_treino"])
    return metadados


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Retreina o IsolationForest do especialista de ML.")
    parser.add_argument("--config", default="config/agentes.yaml", help="YAML da camada agêntica")
    parser.add_argument("--entrada", help="CSV de telemetria tratada (padrão: YAML)")
    parser.add_argument("--saida", help="arquivo .joblib de saída (padrão: YAML)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    cfg = carregar_yaml(args.config)
    treinar(cfg, caminho(args.entrada) if args.entrada else None, caminho(args.saida) if args.saida else None)


if __name__ == "__main__":
    main()
