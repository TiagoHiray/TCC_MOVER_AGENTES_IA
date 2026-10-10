"""Roda a camada agêntica sobre a telemetria tratada e grava o log em JSONL.

Uso, a partir da pasta src/ do repositório:
    python -m mover.agentes.rodar_agentes                      # provedor do .env/YAML (padrão: Ollama gemma2:2b)
    python -m mover.agentes.rodar_agentes --provedor falso     # sem LLM (texto-modelo), para testes
    python -m mover.agentes.rodar_agentes --injetar-eventos    # soma frenagem e arrancada sintéticas
    python -m mover.agentes.rodar_agentes --tempo-real --fator-tempo 5   # cada bloco de 10 s dura 2 s

Cada linha do JSONL é uma entrada: tipo (log | problema), bloco, t_ini/t_fim, hora_local,
nivel (info | atencao | critico), causa, evento, texto, diagnostico, acao, fatos, autor,
texto_origem (llm | modelo), provedor, latencia_s, observacao, fonte (real | sintetico | misto),
continuacao, mesclado_desde e, nos problemas, ajuste (zona de velocidade para o caminhão).
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # execução direta: python src/mover/agentes/rodar_agentes.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from mover.agentes.eventos_sinteticos import injetar
from mover.agentes.executor import ExecutorBlocos, ResultadoBloco
from mover.agentes.grafo import CamadaAgentica
from mover.agentes.llm import PROVEDORES, criar_llm
from mover.agentes.supervisor import Supervisor
from mover.agentes.textos import num
from mover.config import caminho, carregar_yaml

log = logging.getLogger("mover.agentes")

ROTULO_NIVEL = {"info": "INFO", "atencao": "ATENÇÃO", "critico": "CRÍTICO"}


def formatar_entrada(e: dict[str, Any]) -> str:
    """Uma linha legível por entrada, para o terminal."""
    if e["tipo"] == "log":
        quando, tipo = f"{num(e['t_ini'], 1)}-{num(e['t_fim'], 1)} s", "LOG"
    else:
        quando, tipo = f"{num(e['t_pico'], 1)} s", "PROBLEMA"
    origem = "LLM" if e["texto_origem"] == "llm" else "modelo"
    sintetico = " [SINTÉTICO]" if e["fonte"] != "real" else ""
    return (f"#{e['id']:03d} bloco {e['bloco']:02d} {e['hora_local']} {quando:>13} "
            f"{ROTULO_NIVEL[e['nivel']]:<8} {tipo:<8} {e['texto']}{sintetico} ({origem})")


def _json_padrao(valor: Any) -> Any:
    if isinstance(valor, np.generic):
        return valor.item()
    raise TypeError(f"tipo não serializável: {type(valor).__name__}")


def aplicar_eventos_sinteticos(cfg: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """Devolve uma cópia da telemetria com os eventos sintéticos do YAML (fonte=sintetico)."""
    trat = carregar_yaml("config/tratamento.yaml") if caminho("config/tratamento.yaml").exists() else {}
    return injetar(
        df, cfg["eventos_sinteticos"], trat.get("veiculo", {}),
        taxa_hz=float(cfg["entrada"]["taxa_hz"]),
        corte_jerk_hz=float(trat.get("filtros", {}).get("jerk_corte_hz", 1.5)),
        acel_limiar_manobra=float(trat.get("manobras", {}).get("acel_limiar_mps2", 0.5)),
    )


def carregar_telemetria(cfg: dict[str, Any], entrada: str | None, injetar_eventos: bool) -> pd.DataFrame:
    arquivo = caminho(entrada or cfg["entrada"]["telemetria"])
    if not arquivo.exists():
        raise SystemExit(f"Telemetria não encontrada: {arquivo}. Rode antes: python -m mover.tratamento.tratar_dados")
    df = pd.read_csv(arquivo)
    log.info("Telemetria: %s (%d linhas, %.1f s)", arquivo, len(df), df["sim_time"].iloc[-1] - df["sim_time"].iloc[0])
    if not injetar_eventos:
        return df
    df = aplicar_eventos_sinteticos(cfg, df)
    destino = caminho(cfg["saida"]["pasta"]) / cfg["saida"]["telemetria_com_eventos"]
    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destino, index=False)
    log.info("Telemetria com eventos sintéticos gravada em %s", destino)
    return df


def carregar_especialista_ml(cfg: dict[str, Any], desligar: bool):
    cm = cfg["especialista_ml"]
    if desligar or not cm.get("ativo", True):
        return None
    from mover.agentes.especialistas import EspecialistaML

    arquivo = caminho(cm["modelo"])
    if not arquivo.exists():
        from mover.agentes.treinar_especialista_ml import treinar

        log.info("Modelo de ML não encontrado; treinando agora com a telemetria real.")
        treinar(cfg)
    return EspecialistaML.carregar(arquivo, float(cm["fracao_atipica"]))


def criar_supervisor(cfg: dict[str, Any], provedor: str | None, modelo: str | None) -> Supervisor:
    try:
        llm = criar_llm(cfg["llm"], provedor, modelo)
        log.info("Supervisor com LLM %s", llm.nome)
    except Exception as erro:  # pacote ausente, chave faltando...
        log.warning("Não foi possível criar o LLM (%s). O Supervisor vai usar só o texto-modelo.", erro)
        llm = None
    return Supervisor(llm, cfg)


def resumir(resultados: list[ResultadoBloco], supervisor: Supervisor, duracao_bloco: float,
            tempo_real: bool) -> dict[str, Any]:
    entradas = [e for r in resultados for e in r.entradas]
    logs = [e for e in entradas if e["tipo"] == "log"]
    problemas = [e for e in entradas if e["tipo"] == "problema"]
    latencias = [r.latencia_s for r in resultados]
    resumo = {
        "blocos": len(resultados),
        "entradas": len(entradas),
        "logs": len(logs),
        "logs_continuacao": sum(e["continuacao"] for e in logs),
        "logs_mesclados": sum((e["n_janelas"] or 1) > 1 for e in logs),
        "problemas": len(problemas),
        "problemas_por_nivel": dict(Counter(e["nivel"] for e in problemas)),
        "problemas_por_causa": dict(Counter(e["causa"] for e in problemas)),
        "problemas_por_evento": dict(Counter(e["evento"] for e in problemas)),
        "logs_atipicos_ml": sum("especialista_ml" in e["detectado_por"] for e in logs),
        "texto_origem": dict(Counter(e["texto_origem"] for e in entradas)),
        "motivos_texto_modelo": dict(Counter(e["observacao"] for e in entradas if e["texto_origem"] == "modelo")),
        "chamadas_llm": dict(supervisor.estatisticas),
        "latencia_bloco_media_s": round(statistics.fmean(latencias), 3) if latencias else 0.0,
        "latencia_bloco_max_s": round(max(latencias), 3) if latencias else 0.0,
        "duracao_bloco_s": duracao_bloco,
    }
    if tempo_real:  # no modo rápido o caminhão não espera o relógio, então a espera não diz nada
        resumo["maior_espera_s"] = round(max((r.espera_s for r in resultados[1:]), default=0.0), 3)
    return resumo


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Roda a camada agêntica (logs + problemas de jerk).")
    parser.add_argument("--config", default="config/agentes.yaml", help="YAML da camada agêntica")
    parser.add_argument("--entrada", help="CSV de telemetria tratada (padrão: YAML)")
    parser.add_argument("--saida", help="arquivo JSONL do log (padrão: YAML)")
    parser.add_argument("--provedor", choices=PROVEDORES, help="provedor do LLM (padrão: .env ou YAML)")
    parser.add_argument("--modelo", help="nome do modelo no provedor (padrão: .env ou YAML)")
    parser.add_argument("--injetar-eventos", action="store_true", help="soma os eventos sintéticos do YAML")
    parser.add_argument("--sem-ml", action="store_true", help="desliga o especialista de ML")
    parser.add_argument("--tempo-real", action="store_true", help="cada bloco dura o seu tempo de relógio")
    parser.add_argument("--fator-tempo", type=float, default=1.0, help="acelera o tempo real (2 = 2x mais rápido)")
    parser.add_argument("--silencioso", action="store_true", help="não imprime cada entrada")
    parser.add_argument("--depurar", action="store_true", help="mostra os prompts e as respostas do Supervisor")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # terminais sem UTF-8 (Windows) não quebram com "³"
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    for ruidoso in ("httpx", "httpcore"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)
    if args.depurar:
        log.setLevel(logging.DEBUG)

    cfg = carregar_yaml(args.config)
    telemetria = carregar_telemetria(cfg, args.entrada, args.injetar_eventos)
    supervisor = criar_supervisor(cfg, args.provedor, args.modelo)
    camada = CamadaAgentica(cfg, telemetria, supervisor, carregar_especialista_ml(cfg, args.sem_ml))

    cs = cfg["saida"]
    nome_log = cs["log_com_eventos"] if args.injetar_eventos else cs["log"]
    destino = caminho(args.saida) if args.saida else caminho(cs["pasta"]) / nome_log
    destino.parent.mkdir(parents=True, exist_ok=True)
    log.info("%d blocos de %.0f s; log em %s", camada.n_blocos, camada.duracao_bloco, destino)

    with destino.open("w", encoding="utf-8") as arquivo:
        def publicar(resultado: ResultadoBloco) -> None:
            for entrada in resultado.entradas:
                arquivo.write(json.dumps(entrada, ensure_ascii=False, default=_json_padrao) + "\n")
                if not args.silencioso:
                    print(formatar_entrada(entrada), flush=True)
            arquivo.flush()

        resultados = ExecutorBlocos(camada, ao_publicar=publicar).rodar(args.tempo_real, args.fator_tempo)

    resumo = resumir(resultados, supervisor, camada.duracao_bloco, args.tempo_real)
    with destino.with_name(destino.stem + "_resumo.json").open("w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)
    print(json.dumps(resumo, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
