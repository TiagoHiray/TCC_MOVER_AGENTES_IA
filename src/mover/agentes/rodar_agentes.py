"""Análise offline da camada de logs sobre uma corrida já feita, gravada em JSONL.

Lê o telemetria.csv de uma corrida (data/autonomo/corridas/<id>/, gravada por
python -m mover.autonomo.dirigir) e roda a camada de logs (especialista de jerk, especialista de ML e
Supervisor com o LLM) bloco a bloco de 10 s, sem abrir o simulador: os mesmos logs e problemas que a
página publica ao vivo, para estudar a corrida e testar o LLM. Numa corrida sem agentes
(--sem-agentes), aparecem as decisões bruscas do caminhão sozinho; com agentes, elas devem sumir.

Uso, a partir da pasta src/ do repositório:
    python -m mover.agentes.rodar_agentes                      # a corrida mais recente
    python -m mover.agentes.rodar_agentes --corrida ../data/autonomo/corridas/<id>
    python -m mover.agentes.rodar_agentes --provedor falso     # sem LLM (texto-modelo), para testes
    python -m mover.agentes.rodar_agentes --tempo-real --fator-tempo 5   # cada bloco de 10 s dura 2 s

Legado, fora do fluxo do gêmeo digital: --entrada data/tratado/telemetria_tratada.csv lê a gravação
tratada do Sensor Logger, e só com ela vale --injetar-eventos (eventos sintéticos de teste).

Cada linha do JSONL é uma entrada: tipo (log | problema), bloco, t_ini/t_fim, hora_local,
nivel (info | atencao | critico), causa, evento, texto, diagnostico, acao, fatos, autor,
texto_origem (llm | modelo), provedor, latencia_s, observacao, fonte (carla | substituto, ou
real | sintetico | misto no legado), continuacao e mesclado_desde.
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
from mover.agentes.telemetria_viva import para_agentes
from mover.agentes.textos import num
from mover.config import caminho, carregar_yaml

log = logging.getLogger("mover.agentes")

ROTULO_NIVEL = {"info": "INFO", "atencao": "ATENÇÃO", "critico": "CRÍTICO"}
FONTES_SINTETICAS = ("sintetico", "misto")  # legado: trechos com eventos injetados


def formatar_entrada(e: dict[str, Any]) -> str:
    """Uma linha legível por entrada, para o terminal."""
    if e["tipo"] == "log":
        quando, tipo = f"{num(e['t_ini'], 1)}-{num(e['t_fim'], 1)} s", "LOG"
    else:
        quando, tipo = f"{num(e['t_pico'], 1)} s", "PROBLEMA"
    origem = "LLM" if e["texto_origem"] == "llm" else "modelo"
    sintetico = " [SINTÉTICO]" if e["fonte"] in FONTES_SINTETICAS else ""
    return (f"#{e['id']:03d} bloco {e['bloco']:02d} {e['hora_local']} {quando:>13} "
            f"{ROTULO_NIVEL[e['nivel']]:<8} {tipo:<8} {e['texto']}{sintetico} ({origem})")


def _json_padrao(valor: Any) -> Any:
    if isinstance(valor, np.generic):
        return valor.item()
    raise TypeError(f"tipo não serializável: {type(valor).__name__}")


def aplicar_eventos_sinteticos(cfg: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    """Legado: cópia da telemetria tratada do Sensor Logger com os eventos sintéticos do YAML."""
    trat = carregar_yaml("config/tratamento.yaml") if caminho("config/tratamento.yaml").exists() else {}
    return injetar(
        df, cfg["eventos_sinteticos"], trat.get("veiculo", {}),
        taxa_hz=float(cfg["entrada"]["taxa_hz"]),
        corte_jerk_hz=float(trat.get("filtros", {}).get("jerk_corte_hz", 1.5)),
        acel_limiar_manobra=float(trat.get("manobras", {}).get("acel_limiar_mps2", 0.5)),
    )


PASTA_CORRIDAS = "data/autonomo/corridas"


def usa_corrida(cfg: dict[str, Any], entrada: str | None, corrida: str | None) -> bool:
    """A análise lê uma corrida (padrão: a mais recente), a não ser que --entrada ou o YAML apontem um CSV."""
    if corrida:
        return True
    valor = entrada or str((cfg.get("entrada") or {}).get("telemetria", "corrida"))
    return valor.strip().lower() == "corrida"


def achar_corrida(explicita: str | None = None, pasta_corridas: str = PASTA_CORRIDAS) -> Path:
    """A pasta da corrida dada ou a da mais recente em data/autonomo/corridas."""
    if explicita:
        pasta = Path(explicita) if Path(explicita).exists() else caminho(explicita)
        if not (pasta / "telemetria.csv").is_file():
            raise SystemExit(f"Corrida sem telemetria.csv: {pasta}")
        return pasta
    raiz = Path(pasta_corridas) if Path(pasta_corridas).is_absolute() else caminho(pasta_corridas)
    corridas = sorted(p for p in raiz.glob("*") if (p / "telemetria.csv").is_file())
    if not corridas:
        raise SystemExit(f"Nenhuma corrida em {raiz}. Rode antes uma volta: python -m mover.autonomo.dirigir --sem-painel")
    return corridas[-1]


def carregar_telemetria(cfg: dict[str, Any], entrada: str | None = None, injetar_eventos: bool = False,
                        corrida: str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """A telemetria da análise, no formato da camada de logs, e a sua origem."""
    taxa_hz = float(cfg["entrada"]["taxa_hz"])
    if usa_corrida(cfg, entrada, corrida):
        if injetar_eventos:
            raise SystemExit("--injetar-eventos só vale com --entrada (telemetria tratada do Sensor Logger, legado). "
                             "Uma corrida sem agentes já traz as decisões bruscas do caminhão sozinho.")
        pasta = achar_corrida(corrida, str((cfg.get("entrada") or {}).get("pasta_corridas", PASTA_CORRIDAS)))
        df = para_agentes(pd.read_csv(pasta / "telemetria.csv"), taxa_hz)
        arquivo_resumo = pasta / "resumo.json"
        resumo = json.loads(arquivo_resumo.read_text(encoding="utf-8")) if arquivo_resumo.is_file() else {}
        origem = {"tipo": "corrida", "corrida": pasta.name, "simulador": resumo.get("simulador"),
                  "modo": resumo.get("modo"), "voltas": resumo.get("voltas")}
        log.info("Corrida: %s (%s, modo %s, %d linhas, %.1f s)", pasta.name, resumo.get("simulador", "?"),
                 resumo.get("modo", "?"), len(df), df["sim_time"].iloc[-1] - df["sim_time"].iloc[0])
        return df, origem

    arquivo = caminho(entrada or cfg["entrada"]["telemetria"])
    if not arquivo.exists():
        raise SystemExit(f"Telemetria não encontrada: {arquivo}")
    df = pd.read_csv(arquivo)
    convertida = "jerk_long" not in df.columns  # telemetria crua de uma corrida do simulador
    if convertida:
        if injetar_eventos:
            raise SystemExit("--injetar-eventos só vale para a telemetria tratada do Sensor Logger (legado).")
        df = para_agentes(df, taxa_hz)
    origem = {"tipo": "corrida" if convertida else "csv", "arquivo": str(arquivo)}
    log.info("Telemetria: %s (%d linhas, %.1f s)", arquivo, len(df), df["sim_time"].iloc[-1] - df["sim_time"].iloc[0])
    if not injetar_eventos:
        return df, origem
    df = aplicar_eventos_sinteticos(cfg, df)
    destino = caminho(cfg["saida"]["pasta"]) / cfg["saida"]["telemetria_com_eventos"]
    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destino, index=False)
    log.info("Telemetria com eventos sintéticos gravada em %s", destino)
    return df, {**origem, "eventos_sinteticos": True}


def carregar_especialista_ml(cfg: dict[str, Any], desligar: bool, telemetria: pd.DataFrame | None = None,
                             origem: dict[str, Any] | None = None):
    """O especialista de ML, treinado com os mesmos dados analisados (retreina se a corrida mudou)."""
    cm = cfg["especialista_ml"]
    if desligar or not cm.get("ativo", True):
        return None
    from mover.agentes.especialistas import EspecialistaML
    from mover.agentes.treinar_especialista_ml import ler_metadados, treinar

    arquivo = caminho(cm["modelo"])
    alvo = (origem or {}).get("corrida")
    treinado_com = ler_metadados(arquivo).get("corrida")
    if not arquivo.exists() or (alvo and treinado_com != alvo):
        log.info("Treinando o especialista de ML com %s.", f"a corrida {alvo}" if alvo else "a telemetria analisada")
        treinar(cfg, saida=arquivo, dados=telemetria, origem=origem)
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
    parser = argparse.ArgumentParser(description="Análise offline da camada de logs sobre uma corrida já feita.")
    parser.add_argument("--config", default="config/agentes.yaml", help="YAML da camada agêntica")
    parser.add_argument("--corrida", help="pasta da corrida (padrão: a mais recente em data/autonomo/corridas)")
    parser.add_argument("--entrada", help="em vez de uma corrida, um CSV (no legado, a telemetria tratada)")
    parser.add_argument("--saida", help="arquivo JSONL do log (padrão: YAML)")
    parser.add_argument("--provedor", choices=PROVEDORES, help="provedor do LLM (padrão: .env ou YAML)")
    parser.add_argument("--modelo", help="nome do modelo no provedor (padrão: .env ou YAML)")
    parser.add_argument("--injetar-eventos", action="store_true",
                        help="legado: soma os eventos sintéticos do YAML à telemetria tratada do Sensor Logger")
    parser.add_argument("--sem-ml", action="store_true", help="desliga o especialista de ML")
    parser.add_argument("--tempo-real", action="store_true", help="cada bloco dura o seu tempo de relógio")
    parser.add_argument("--fator-tempo", type=float, default=1.0, help="acelera o tempo real (2 = 2x mais rápido)")
    parser.add_argument("--silencioso", action="store_true", help="não imprime cada entrada")
    parser.add_argument("--depurar", action="store_true", help="mostra os prompts e as respostas do Supervisor")
    args = parser.parse_args(argv)
    if args.corrida and args.entrada:
        parser.error("use --corrida ou --entrada, não os dois")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # terminais sem UTF-8 (Windows) não quebram com "³"
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    for ruidoso in ("httpx", "httpcore"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)
    if args.depurar:
        log.setLevel(logging.DEBUG)

    cfg = carregar_yaml(args.config)
    telemetria, origem = carregar_telemetria(cfg, args.entrada, args.injetar_eventos, args.corrida)
    supervisor = criar_supervisor(cfg, args.provedor, args.modelo)
    camada = CamadaAgentica(cfg, telemetria, supervisor, carregar_especialista_ml(cfg, args.sem_ml, telemetria, origem))

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

    resumo = {"entrada": origem, **resumir(resultados, supervisor, camada.duracao_bloco, args.tempo_real)}
    with destino.with_name(destino.stem + "_resumo.json").open("w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)
    print(json.dumps(resumo, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
