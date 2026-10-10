"""Grava as mensagens do HTTP Push do Sensor Logger: uma pasta por gravação (sessionId).

Cada pasta tem `raw.jsonl` (mensagem inteira + recv_ns) e um CSV por sensor com as colunas
`time` (ns UTC do celular), `recv_ns` (ns UTC do servidor) e os campos de `values`.
Schema: https://github.com/tszheichoi/awesome-sensor-logger/blob/main/PUSHING.md
"""

from __future__ import annotations

import csv
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("mover.ingestao")


class MensagemInvalida(ValueError):
    pass


def nome_seguro(nome: str, padrao: str = "x") -> str:
    return "".join(c for c in nome if c.isalnum() or c in "-_")[:40] or padrao


@dataclass
class _Sensor:
    colunas: list[str]
    n: int = 0
    t_primeiro: int | None = None
    t_ultimo: int | None = None
    ultima: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Gravacao:
    pasta: Path
    mensagens: int = 0
    fora_de_ordem: int = 0
    ultimo_id: int | None = None
    sensores: dict[str, _Sensor] = field(default_factory=dict)


class ArmazemIngestao:
    def __init__(self, pasta: Path, intervalo_log_s: float = 5.0):
        self.pasta = pasta
        self.intervalo_log_s = intervalo_log_s
        self._trava = threading.Lock()
        self._gravacoes: dict[str, _Gravacao] = {}
        self._ultima_rx_ns: int | None = None
        self._latencia_ms: float | None = None
        self._proximo_log = 0.0

    def _gravacao(self, sessao: str) -> _Gravacao:
        g = self._gravacoes.get(sessao)
        if g is None:
            pasta = self.pasta / f"{datetime.now():%Y%m%d-%H%M%S}_{nome_seguro(sessao, 'anon')[:8]}"
            pasta.mkdir(parents=True, exist_ok=True)
            g = self._gravacoes[sessao] = _Gravacao(pasta)
            log.info("Ingestão: nova gravação %s -> %s", sessao, pasta)
        return g

    def registrar(self, msg: dict[str, Any]) -> int:
        """Grava uma mensagem e devolve quantas leituras foram aceitas."""
        payload = msg.get("payload", [])
        if not isinstance(payload, list):
            raise MensagemInvalida("payload deve ser uma lista")
        recv_ns = time.time_ns()
        sessao = str(msg.get("sessionId") or "anon")

        linhas: dict[str, list[dict[str, Any]]] = {}
        mais_recente: int | None = None
        for leitura in payload:
            if not isinstance(leitura, dict) or not isinstance(leitura.get("time"), (int, float)):
                continue
            valores = leitura.get("values") if isinstance(leitura.get("values"), dict) else {}
            t = int(leitura["time"])
            linhas.setdefault(str(leitura.get("name") or "desconhecido"), []).append(
                {"time": t, "recv_ns": recv_ns, **valores})
            mais_recente = t if mais_recente is None else max(mais_recente, t)

        with self._trava:
            g = self._gravacao(sessao)
            with open(g.pasta / "raw.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"recv_ns": recv_ns, **msg}, ensure_ascii=False) + "\n")
            g.mensagens += 1
            mid = msg.get("messageId")
            if isinstance(mid, int):
                if g.ultimo_id is not None and mid <= g.ultimo_id:
                    g.fora_de_ordem += 1
                else:
                    g.ultimo_id = mid

            for nome, rows in linhas.items():
                s = g.sensores.get(nome)
                arquivo = g.pasta / f"{nome_seguro(nome)}.csv"
                novo = s is None
                if novo:
                    s = g.sensores[nome] = _Sensor(colunas=list(rows[0].keys()))
                with open(arquivo, "a", newline="", encoding="utf-8") as f:
                    # colunas fixadas pela primeira leitura; campos novos continuam no raw.jsonl
                    w = csv.DictWriter(f, fieldnames=s.colunas, extrasaction="ignore")
                    if novo:
                        w.writeheader()
                    w.writerows(rows)
                s.n += len(rows)
                s.t_primeiro = rows[0]["time"] if s.t_primeiro is None else min(s.t_primeiro, rows[0]["time"])
                s.t_ultimo = rows[-1]["time"] if s.t_ultimo is None else max(s.t_ultimo, rows[-1]["time"])
                s.ultima = {k: v for k, v in rows[-1].items() if k != "recv_ns"}

            self._ultima_rx_ns = recv_ns
            if mais_recente is not None:
                # só tem sentido com os relógios do celular e do servidor sincronizados (NTP)
                self._latencia_ms = (recv_ns - mais_recente) / 1e6
            agora = time.monotonic()
            if agora >= self._proximo_log:
                self._proximo_log = agora + self.intervalo_log_s
                log.info("Ingestão %s: %d msgs, latência ~%s ms | %s", sessao, g.mensagens,
                         "?" if self._latencia_ms is None else f"{self._latencia_ms:.0f}",
                         ", ".join(f"{k}={v.n}" for k, v in sorted(g.sensores.items())))
        return sum(len(r) for r in linhas.values())

    def status(self) -> dict[str, Any]:
        with self._trava:
            agora = time.time_ns()
            gravacoes = {}
            for sessao, g in self._gravacoes.items():
                sensores = {}
                for nome, s in g.sensores.items():
                    dur = (s.t_ultimo - s.t_primeiro) / 1e9 if s.t_primeiro is not None and s.t_ultimo is not None else 0
                    sensores[nome] = {"amostras": s.n, "taxa_hz": round((s.n - 1) / dur, 2) if dur > 0 else None,
                                      "ultima": s.ultima}
                gravacoes[sessao] = {"pasta": str(g.pasta), "mensagens": g.mensagens,
                                     "fora_de_ordem": g.fora_de_ordem, "sensores": sensores}
            return {
                "gravacoes": gravacoes,
                "latencia_ms": None if self._latencia_ms is None else round(self._latencia_ms, 1),
                "s_desde_ultima_mensagem": None if self._ultima_rx_ns is None
                else round((agora - self._ultima_rx_ns) / 1e9, 2),
            }
