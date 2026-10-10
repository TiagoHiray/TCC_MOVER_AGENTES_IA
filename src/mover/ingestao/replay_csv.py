"""Reenvia um export CSV do Sensor Logger no formato do HTTP Push, no ritmo da gravação.

Uso, a partir da pasta src/ (com o servidor no ar):
    python -m mover.ingestao.replay_csv                                  # data/csv_maua, tempo real
    python -m mover.ingestao.replay_csv --duracao 30 --sensores location accelerometer
    python -m mover.ingestao.replay_csv --url http://<IP>:8000/ingestao/sensorlogger --fator-tempo 2
"""

from __future__ import annotations

import argparse
import csv
import heapq
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

if __package__ in (None, ""):  # execução direta: python src/mover/ingestao/replay_csv.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mover.config import caminho

log = logging.getLogger("mover.ingestao")

IGNORADOS = {"metadata"}


def _numero(v: str) -> Any:
    try:
        return float(v)
    except ValueError:
        return v


def _tempo_ns(v: str | None) -> int | None:
    try:
        return int(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        try:
            return int(float(v))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None


def ler_sensor(arquivo: Path) -> Iterator[tuple[int, str, dict[str, Any]]]:
    nome = arquivo.stem.lower()
    with open(arquivo, newline="", encoding="utf-8") as f:
        leitor = csv.DictReader(f)
        if not leitor.fieldnames or "time" not in leitor.fieldnames:
            return
        for linha in leitor:
            t = _tempo_ns(linha.pop("time", None))
            if t is None:
                continue
            linha.pop("seconds_elapsed", None)
            yield t, nome, {k: _numero(v) for k, v in linha.items() if k is not None and v not in (None, "")}


def arquivos_da_gravacao(pasta: Path, sensores: Iterable[str] | None = None) -> list[Path]:
    desejados = {s.lower() for s in sensores} if sensores else None
    return [p for p in sorted(pasta.glob("*.csv"))
            if p.stem.lower() not in IGNORADOS and (desejados is None or p.stem.lower() in desejados)]


def lotes(arquivos: list[Path], periodo_s: float,
          duracao_s: float | None = None) -> Iterator[tuple[int, int, list[dict[str, Any]]]]:
    """Agrupa as leituras em janelas de `periodo_s`; devolve (t0_ns, fim_da_janela_ns, payload)."""
    periodo_ns = int(periodo_s * 1e9)
    # cada CSV já vem em ordem de tempo; o merge mantém a memória baixa
    fluxo = heapq.merge(*(ler_sensor(p) for p in arquivos), key=lambda r: r[0])
    t0 = fim = None
    lote: list[dict[str, Any]] = []
    for t, nome, valores in fluxo:
        if t0 is None:
            t0, fim = t, t + periodo_ns
        if duracao_s is not None and t - t0 > duracao_s * 1e9:
            break
        while t >= fim:
            if lote:
                yield t0, fim, lote
                lote = []
            fim += periodo_ns
        lote.append({"name": nome, "time": t, "values": valores})
    if lote:
        yield t0, fim, lote


def reenviar(arquivos: list[Path], postar: Callable[[dict[str, Any]], bool], periodo_s: float = 0.2,
             fator_tempo: float = 1.0, duracao_s: float | None = None, rebase: bool = True,
             sem_espera: bool = False) -> dict[str, Any]:
    """Envia os lotes com `postar(mensagem) -> ok`. Com `rebase`, os tempos passam a ser os de agora."""
    sessao = f"replay-{uuid.uuid4().hex[:8]}"
    enviadas = falhas = leituras = 0
    inicio = offset = None
    for msg_id, (t0, fim, payload) in enumerate(lotes(arquivos, periodo_s, duracao_s)):
        if inicio is None:
            inicio = time.monotonic()
            offset = time.time_ns() - t0 if rebase else 0
        if not sem_espera:
            atraso = inicio + (fim - t0) / 1e9 / fator_tempo - time.monotonic()
            if atraso > 0:
                time.sleep(atraso)
        for leitura in payload:
            leitura["time"] += offset
        ok = postar({"messageId": msg_id, "sessionId": sessao, "deviceId": "replay-csv", "payload": payload})
        enviadas += ok
        falhas += not ok
        leituras += len(payload)
        if msg_id % max(1, round(5 / periodo_s)) == 0:
            log.info("Replay t=%6.1f s: %d msgs, %d falhas, %d leituras", (fim - t0) / 1e9, enviadas, falhas, leituras)
    return {"sessao": sessao, "mensagens": enviadas, "falhas": falhas, "leituras": leituras}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay de um export do Sensor Logger via HTTP Push.")
    parser.add_argument("--pasta", default="data/csv_maua", help="pasta com os CSVs do Sensor Logger")
    parser.add_argument("--url", default="http://127.0.0.1:8000/ingestao/sensorlogger")
    parser.add_argument("--token", help="padrão: MOVER_INGESTAO_TOKEN do .env")
    parser.add_argument("--periodo", type=float, default=0.2, help="período do lote em s (batch period do app)")
    parser.add_argument("--fator-tempo", type=float, default=1.0, help="2 = duas vezes mais rápido")
    parser.add_argument("--duracao", type=float, help="segundos de gravação a reenviar")
    parser.add_argument("--sensores", nargs="*", help="ex.: location accelerometer gyroscope")
    parser.add_argument("--sem-rebase", action="store_true", help="mantém os timestamps originais")
    parser.add_argument("--sem-espera", action="store_true", help="envia o mais rápido possível")
    args = parser.parse_args(argv)

    if not args.url.startswith(("http://", "https://")):
        parser.error("--url deve começar com http:// ou https://")
    if args.periodo <= 0 or args.fator_tempo <= 0:
        parser.error("--periodo e --fator-tempo devem ser > 0")
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    # o httpx loga cada POST (com o token na URL)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    import httpx

    from mover.agentes.llm import carregar_env

    carregar_env()
    token = args.token or os.getenv("MOVER_INGESTAO_TOKEN")
    arquivos = arquivos_da_gravacao(caminho(args.pasta), args.sensores)
    if not arquivos:
        parser.error(f"nenhum CSV em {caminho(args.pasta)}")
    log.info("Sensores: %s", ", ".join(p.stem.lower() for p in arquivos))

    with httpx.Client(timeout=5.0, params={"token": token} if token else None) as cliente:
        def postar(msg: dict[str, Any]) -> bool:
            try:
                r = cliente.post(args.url, json=msg)
            except httpx.HTTPError as erro:
                log.warning("Falha no envio: %s", erro)
                return False
            if r.status_code != 200:
                log.warning("Servidor respondeu %d: %s", r.status_code, r.text[:200])
            return r.status_code == 200

        try:
            resumo = reenviar(arquivos, postar, args.periodo, args.fator_tempo, args.duracao,
                              rebase=not args.sem_rebase, sem_espera=args.sem_espera)
        except KeyboardInterrupt:
            log.info("Interrompido.")
            return
    log.info("Fim: %s", resumo)


if __name__ == "__main__":
    main()
