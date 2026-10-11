"""Treina o especialista de ML da camada de logs (IsolationForest) com a telemetria de uma corrida.

O modelo aprende o padrão da condução na corrida (data/autonomo/corridas/<id>/telemetria.csv) e,
na camada de logs, marca como atípicas as janelas que fogem dele. Não confundir com o especialista
de ML de suavização (mover.agentes.suavizador_ml), que propõe as alterações do plano do caminhão.

Uso, a partir da pasta src/ do repositório:
    python -m mover.agentes.treinar_especialista_ml                       # a corrida mais recente
    python -m mover.agentes.treinar_especialista_ml --corrida ../data/autonomo/corridas/<id>
    python -m mover.agentes.treinar_especialista_ml --entrada <csv>       # outro CSV (no legado, a tratada)

Grava data/modelos/especialista_ml.joblib (modelo + lista de features) e um JSON de metadados ao
lado, com a corrida usada. A análise offline (rodar_agentes) retreina sozinha quando a corrida muda.
Não mexe no data/modelo_especialista.joblib da POC.

Diferenças em relação ao treinar_ml.py da POC:
- features no referencial do veículo (speed_mps, acc_long, acc_lat, acc_vert, jerk_long,
  yaw_rate_dps), em vez de acc_x/acc_y, que estão no referencial do mundo e mudam com o rumo;
- throttle/brake/steer ficam de fora: o modelo olha a dinâmica medida, não os comandos;
- linhas com fonte = sintetico (eventos de teste do legado) não entram no treino.

Atenção: o modelo é treinado e avaliado na mesma corrida. Ele aponta os ~5 % de quadros mais
atípicos dela (contaminação), não anomalias em sentido absoluto.
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


def ler_metadados(modelo: Path) -> dict[str, Any]:
    """Os metadados gravados ao lado do modelo (vazio se não houver)."""
    arquivo = Path(modelo).with_suffix(".json")
    try:
        return json.loads(arquivo.read_text(encoding="utf-8")) if arquivo.is_file() else {}
    except ValueError:
        return {}


def dados_de_treino(cfg: dict[str, Any], entrada: Path | None = None,
                    corrida: str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """A telemetria de treino, no formato da camada de logs, e a sua origem (a corrida mais recente, por padrão)."""
    from mover.agentes.rodar_agentes import carregar_telemetria

    return carregar_telemetria(cfg, str(entrada) if entrada else None, False, corrida)


def treinar(cfg: dict[str, Any], entrada: Path | None = None, saida: Path | None = None,
            dados: pd.DataFrame | None = None, origem: dict[str, Any] | None = None,
            corrida: str | None = None) -> dict[str, Any]:
    """Treina, grava o modelo e devolve os metadados.

    Sem `dados`, lê a telemetria de `entrada` ou, por padrão, a da corrida (a dada em `corrida` ou a
    mais recente).
    """
    import joblib
    import sklearn
    from sklearn.ensemble import IsolationForest

    cm = cfg["especialista_ml"]
    saida = saida or caminho(cm["modelo"])
    features = list(cm["features"])
    if dados is None:
        dados, origem = dados_de_treino(cfg, entrada, corrida)
    origem = dict(origem or {})

    df = dados
    if "fonte" in df.columns:
        df = df[df["fonte"].astype(str) != "sintetico"]
    faltam = [f for f in features if f not in df.columns]
    if faltam:
        raise SystemExit(f"Telemetria sem as colunas do especialista de ML: {', '.join(faltam)}")
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
    arquivo = origem.get("arquivo")
    if arquivo:
        try:
            arquivo = str(Path(arquivo).resolve().relative_to(RAIZ))
        except ValueError:
            pass
    metadados = {
        "treinado_em": datetime.now().astimezone().isoformat(timespec="seconds"),
        "origem": origem.get("tipo", "dados"),
        "corrida": origem.get("corrida"),
        "simulador": origem.get("simulador"),
        "arquivo": arquivo,
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
    log.info("Modelo gravado em %s (%s; %d amostras, %.1f %% atípicas no treino)", saida,
             f"corrida {metadados['corrida']}" if metadados["corrida"] else metadados["origem"],
             metadados["amostras"], 100 * metadados["fracao_atipica_no_treino"])
    return metadados


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Treina o especialista de ML da camada de logs com uma corrida.")
    parser.add_argument("--config", default="config/agentes.yaml", help="YAML da camada agêntica")
    parser.add_argument("--corrida", help="pasta da corrida (padrão: a mais recente em data/autonomo/corridas)")
    parser.add_argument("--entrada", help="em vez de uma corrida, um CSV (no legado, a telemetria tratada)")
    parser.add_argument("--saida", help="arquivo .joblib de saída (padrão: YAML)")
    args = parser.parse_args(argv)
    if args.corrida and args.entrada:
        parser.error("use --corrida ou --entrada, não os dois")

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    cfg = carregar_yaml(args.config)
    treinar(cfg, caminho(args.entrada) if args.entrada else None, caminho(args.saida) if args.saida else None,
            corrida=args.corrida)


if __name__ == "__main__":
    main()
