"""Leitura e validação das exportações CSV do app Sensor Logger (iOS e Android).

Cada sensor vem num CSV com as colunas `time` (época em ns), `seconds_elapsed` e os
valores. O relógio mestre é a coluna `time`, mais robusta que `seconds_elapsed` quando o
arquivo passou por planilhas em pt-BR (que removem o ponto decimal de números longos).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SENSORES_OBRIGATORIOS = ("Accelerometer", "Gravity", "Gyroscope", "Location")
SENSORES_OPCIONAIS = ("Barometer", "Compass", "Orientation")

COLUNAS_ESPERADAS = {
    "Accelerometer": ("x", "y", "z"),
    "Gravity": ("x", "y", "z"),
    "Gyroscope": ("x", "y", "z"),
    "Location": ("latitude", "longitude", "horizontalAccuracy", "speed"),
    "Barometer": ("relativeAltitude",),
    "Compass": ("magneticBearing",),
    "Orientation": ("qw", "qx", "qy", "qz"),
}

# Limite físico (percentil 99,9 do valor absoluto) para detectar arquivos corrompidos.
LIMITES_PLAUSIVEIS = {
    "Accelerometer": 80.0,  # m/s²
    "Gravity": 12.0,  # m/s²
    "Gyroscope": 35.0,  # rad/s
    "Barometer": 2000.0,  # m de altitude relativa
    "Compass": 360.0,  # graus
    "Orientation": 7.0,  # quaternion e radianos
}


@dataclass
class Metadados:
    """Conteúdo relevante do Metadata.csv do Sensor Logger."""

    plataforma: str  # "ios", "android" ou "desconhecida"
    padronizado: bool  # opção "standardisation" do app
    dispositivo: str
    versao_app: str
    epoca_ns: int  # início da gravação (época, ns)
    fuso: str
    hora_gravacao_utc: str  # campo "recording time": o app grava em UTC (confere com a época)
    taxas_ms: dict[str, str] = field(default_factory=dict)


@dataclass
class Gravacao:
    """Gravação carregada: metadados e um DataFrame por sensor (coluna `t` em segundos)."""

    pasta: Path
    meta: Metadados
    sensores: dict[str, pd.DataFrame]
    ignorados: dict[str, str] = field(default_factory=dict)
    avisos: list[str] = field(default_factory=list)


def _ler_csv(caminho: Path) -> pd.DataFrame:
    """Lê um CSV detectando o separador (`,` padrão ou `;` com vírgula decimal, típico do pt-BR)."""
    with caminho.open("r", encoding="utf-8-sig") as f:
        cabecalho = f.readline()
    if cabecalho.count(";") > cabecalho.count(","):
        return pd.read_csv(caminho, sep=";", decimal=",", encoding="utf-8-sig")
    return pd.read_csv(caminho, encoding="utf-8-sig")


def ler_metadados(pasta: Path) -> Metadados | None:
    """Lê o Metadata.csv; devolve None se o arquivo não existir."""
    caminho = pasta / "Metadata.csv"
    if not caminho.exists():
        return None
    linha = pd.read_csv(caminho, dtype=str, encoding="utf-8-sig").iloc[0].to_dict()
    sensores = str(linha.get("sensors", "")).split("|")
    taxas = str(linha.get("sampleRateMs", "")).split("|")
    return Metadados(
        plataforma=str(linha.get("platform", "desconhecida")).strip().lower(),
        padronizado=str(linha.get("standardisation", "false")).strip().lower() == "true",
        dispositivo=str(linha.get("device name", "")),
        versao_app=str(linha.get("appVersion", "")),
        epoca_ns=int(float(linha["recording epoch time"]) * 1_000_000),
        fuso=str(linha.get("recording timezone", "UTC")),
        hora_gravacao_utc=str(linha.get("recording time", "")),
        taxas_ms=dict(zip(sensores, taxas)),
    )


def _preparar_sensor(nome: str, df: pd.DataFrame, epoca_ns: int, avisos: list[str]) -> pd.DataFrame:
    """Valida colunas, converte o tempo e checa faixas plausíveis de um sensor."""
    faltando = [c for c in ("time", *COLUNAS_ESPERADAS.get(nome, ())) if c not in df.columns]
    if faltando:
        raise ValueError(f"{nome}.csv sem as colunas {faltando}")

    df = df.copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["time"])
    time_ns = df["time"].to_numpy(dtype=np.float64).round().astype(np.int64)
    df["t"] = (time_ns - epoca_ns) / 1e9
    df["t_unix"] = time_ns / 1e9

    if "seconds_elapsed" in df.columns:
        desvio = np.nanmedian(np.abs(df["seconds_elapsed"].to_numpy() - df["t"].to_numpy()))
        if not desvio < 0.05:
            avisos.append(
                f"{nome}: seconds_elapsed não bate com a coluna time (desvio mediano {desvio:.3g} s). "
                "Possível corrupção do separador decimal; o tempo foi reconstruído pela coluna time."
            )

    limite = LIMITES_PLAUSIVEIS.get(nome)
    if limite is not None:
        valores = df[list(COLUNAS_ESPERADAS[nome])].to_numpy(dtype=float)
        p999 = np.nanpercentile(np.abs(valores), 99.9)
        if not np.isfinite(p999) or p999 > limite:
            raise ValueError(
                f"{nome}.csv tem valores implausíveis (|valor| p99,9 = {p999:.3g} > {limite}). "
                "Provavelmente o ponto decimal foi removido por uma ferramenta em pt-BR; "
                "exporte de novo direto do Sensor Logger."
            )
    if nome == "Location":
        lat, lon = df["latitude"], df["longitude"]
        if not (lat.between(-90, 90).all() and lon.between(-180, 180).all()):
            raise ValueError("Location.csv com latitude/longitude fora da faixa válida")

    df = df.sort_values("t").drop_duplicates(subset="t", keep="first").reset_index(drop=True)
    return df


def carregar(pasta: str | Path) -> Gravacao:
    """Carrega uma gravação do Sensor Logger a partir da pasta com os CSVs."""
    pasta = Path(pasta)
    if not pasta.is_dir():
        raise FileNotFoundError(f"Pasta de entrada não encontrada: {pasta}")
    avisos: list[str] = []

    brutos: dict[str, pd.DataFrame] = {}
    for nome in (*SENSORES_OBRIGATORIOS, *SENSORES_OPCIONAIS):
        arquivo = pasta / f"{nome}.csv"
        if arquivo.exists() and arquivo.stat().st_size > 0:
            brutos[nome] = _ler_csv(arquivo)
        elif nome in SENSORES_OBRIGATORIOS:
            raise FileNotFoundError(f"Sensor obrigatório ausente: {arquivo}")

    meta = ler_metadados(pasta)
    if meta is None:
        epoca = min(int(df["time"].iloc[0]) for df in brutos.values())
        meta = Metadados("desconhecida", False, "", "", epoca, "UTC", "")
        avisos.append("Metadata.csv ausente: plataforma desconhecida, início = primeira amostra.")

    sensores = {nome: _preparar_sensor(nome, df, meta.epoca_ns, avisos) for nome, df in brutos.items()}

    ignorados = {}
    for arquivo in sorted(pasta.glob("*.csv")):
        nome = arquivo.stem
        if nome in sensores or nome == "Metadata":
            continue
        if nome.endswith("Uncalibrated"):
            ignorados[nome] = "redundante com a versão calibrada"
        elif arquivo.stat().st_size == 0:
            ignorados[nome] = "arquivo vazio"
        else:
            ignorados[nome] = "não usado pelo tratamento"
    for nome, df in sensores.items():
        log.info("%-14s %6d amostras, %.1f s", nome, len(df), df["t"].iloc[-1] - df["t"].iloc[0])
    return Gravacao(pasta=pasta, meta=meta, sensores=sensores, ignorados=ignorados, avisos=avisos)
