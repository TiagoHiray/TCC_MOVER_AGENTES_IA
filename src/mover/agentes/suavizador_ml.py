"""Especialista de ML de suavização: propõe quanto suavizar cada frenagem ou arrancada brusca.

O caminhão manda à camada agêntica o plano dos próximos 10 s. Para cada zona do plano em que ele vai
frear ou arrancar bruscamente, este modelo propõe uma alteração leve da aceleração planejada:
- desacel_mps2: com que desaceleração frear antes da zona (no lugar da frenagem tardia e forte);
- acel_mps2: com que aceleração retomar depois dela (no lugar da arrancada em degrau);
- teto_kmh: até que velocidade retomar quando a zona seguinte está perto (sem teto = cruzeiro).

As entradas vêm do plano do caminhão: velocidade agora, limite e tamanho da zona, distância até ela,
zona seguinte e a frenagem e a arrancada que o próprio plano traz.

É uma regressão (floresta aleatória com três saídas) treinada com milhares de situações simuladas
pelo modelo do caminhão. Em cada situação, o rótulo é a suavização de menor custo (diesel + tempo)
sem eventos bruscos e sem alongar demais o trecho. Na corrida, a proposta é conferida pelo modelo
do caminhão antes de valer, e o Supervisor (LLM) aprova ou troca.

Uso, a partir da pasta src/ (sem modelo, a corrida com agentes treina um sozinha na primeira vez):
    python -m mover.agentes.suavizador_ml                  # treina e grava data/modelos/suavizador_ml.joblib
    python -m mover.agentes.suavizador_ml --situacoes 5000
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # execução direta: python src/mover/agentes/suavizador_ml.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from mover.config import caminho, carregar_yaml

log = logging.getLogger("mover.agentes")

ARQUIVO_PADRAO = "data/modelos/suavizador_ml.joblib"
FEATURES = ("v_atual_kmh", "v_alvo_kmh", "cruzeiro_kmh", "distancia_m", "comprimento_m", "duracao_s", "parada",
            "gap_seguinte_m", "v_alvo_seguinte_kmh", "desacel_plano_mps2", "acel_plano_mps2")
ALVOS = ("desacel_mps2", "acel_mps2", "teto_kmh")
LIMITES = {"desacel_mps2": (0.4, 1.8), "acel_mps2": (0.3, 1.2)}
GAP_MAX_M = 200.0
DESACEL_CANDIDATAS = (0.5, 0.7, 0.9, 1.2, 1.6)
ACEL_CANDIDATAS = (0.4, 0.6, 0.9)
TETO_CANDIDATOS = (18.0, 22.0)


def caracteristicas(v_atual_kmh: float, v_alvo_kmh: float, cruzeiro_kmh: float, distancia_m: float,
                    comprimento_m: float, duracao_s: float, parada: bool, gap_seguinte_m: float | None,
                    v_alvo_seguinte_kmh: float | None, desacel_plano_mps2: float, acel_plano_mps2: float) -> dict[str, float]:
    """As entradas do modelo para um elemento do plano (as mesmas no treino e na corrida)."""
    return {
        "v_atual_kmh": float(v_atual_kmh), "v_alvo_kmh": float(v_alvo_kmh), "cruzeiro_kmh": float(cruzeiro_kmh),
        "distancia_m": max(0.0, float(distancia_m)), "comprimento_m": max(0.0, float(comprimento_m)),
        "duracao_s": max(0.0, float(duracao_s)), "parada": 1.0 if parada else 0.0,
        "gap_seguinte_m": GAP_MAX_M if gap_seguinte_m is None else min(GAP_MAX_M, max(0.0, float(gap_seguinte_m))),
        "v_alvo_seguinte_kmh": float(cruzeiro_kmh if v_alvo_seguinte_kmh is None else v_alvo_seguinte_kmh),
        "desacel_plano_mps2": abs(float(desacel_plano_mps2)), "acel_plano_mps2": max(0.0, float(acel_plano_mps2)),
    }


class SuavizadorML:
    """O modelo treinado: devolve a suavização proposta para um elemento do plano."""

    def __init__(self, modelo: Any, metadados: dict[str, Any] | None = None):
        self.modelo = modelo
        self.metadados = dict(metadados or {})

    @classmethod
    def carregar(cls, arquivo: str | Path) -> "SuavizadorML":
        import joblib

        pacote = joblib.load(arquivo)
        if tuple(pacote.get("features", ())) != FEATURES or tuple(pacote.get("alvos", ())) != ALVOS:
            raise ValueError("modelo de outra versão (entradas ou saídas diferentes)")
        return cls(pacote["modelo"], pacote.get("metadados"))

    @property
    def nome(self) -> str:
        return f"floresta aleatória ({self.metadados.get('amostras', '?')} situações simuladas)"

    def propor(self, x: dict[str, float]) -> tuple[float, float, float | None]:
        """(desaceleração antecipada, aceleração da retomada, teto em km/h ou None)."""
        vetor = np.array([[float(x[f]) for f in FEATURES]])
        desacel, acel, teto = (float(v) for v in self.modelo.predict(vetor)[0])
        desacel = float(np.clip(desacel, *LIMITES["desacel_mps2"]))
        acel = float(np.clip(acel, *LIMITES["acel_mps2"]))
        cruzeiro = float(x["cruzeiro_kmh"])
        teto_final = None if teto >= cruzeiro - 1.0 else round(max(teto, float(x["v_alvo_kmh"]) + 2.0, 12.0), 1)
        return round(desacel, 2), round(acel, 2), teto_final


# ---------------------------------------------------------------------- situações simuladas
def situacao_aleatoria(rng: np.random.Generator) -> dict[str, Any]:
    """Uma situação de treino: um elemento do plano à frente do caminhão (e talvez o seguinte)."""
    cruzeiro = float(rng.uniform(25.0, 35.0))
    tipo = str(rng.choice(["lombada", "faixa", "curva", "parada", "partida"], p=[0.4, 0.12, 0.25, 0.13, 0.1]))
    sit: dict[str, Any] = {"cruzeiro_kmh": cruzeiro, "tipo": tipo, "v_alvo_kmh": cruzeiro, "comprimento_m": 0.0,
                           "duracao_s": 0.0, "distancia_m": 0.0, "v0_kmh": 0.0}
    if tipo == "lombada":
        sit.update(v_alvo_kmh=10.0, comprimento_m=float(rng.uniform(12.0, 18.0)))
    elif tipo == "faixa":
        sit.update(v_alvo_kmh=15.0, comprimento_m=float(rng.uniform(10.0, 14.0)))
    elif tipo == "curva":
        sit.update(v_alvo_kmh=float(rng.uniform(12.0, min(24.0, cruzeiro - 3.0))), comprimento_m=float(rng.uniform(15.0, 60.0)))
    elif tipo == "parada":
        sit.update(v_alvo_kmh=0.0, duracao_s=float(rng.choice([0.0, rng.uniform(2.0, 6.0)])))
    if tipo != "partida":
        sit["distancia_m"] = float(rng.uniform(20.0, 140.0))
        sit["v0_kmh"] = float(cruzeiro * rng.uniform(0.6, 1.0))
    if rng.random() < 0.55:
        tipo2 = str(rng.choice(["lombada", "faixa", "curva", "parada"], p=[0.5, 0.15, 0.25, 0.1]))
        v2 = {"lombada": 10.0, "faixa": 15.0, "parada": 0.0}.get(tipo2, float(rng.uniform(12.0, min(24.0, cruzeiro - 3.0))))
        sit["seguinte"] = {"tipo": tipo2, "v_alvo_kmh": v2, "gap_m": float(rng.uniform(20.0, 150.0)),
                           "comprimento_m": 0.0 if tipo2 == "parada" else float(rng.uniform(12.0, 40.0)),
                           "duracao_s": float(rng.uniform(2.0, 5.0)) if tipo2 == "parada" else 0.0}
    return sit


_ROTA_TREINO = None


def _rota_treino():
    """Um círculo de 800 m de raio: longo e sem fim de linha (a curvatura não afeta o modelo longitudinal)."""
    global _ROTA_TREINO
    if _ROTA_TREINO is None:
        from mover.autonomo.rota import Rota

        teta = np.linspace(0.0, 2.0 * math.pi, 2400, endpoint=False)
        _ROTA_TREINO = Rota(800.0 * np.cos(teta), 800.0 * np.sin(teta), passo_m=1.0, suavizar_m=0.0)
    return _ROTA_TREINO


def plano_da_situacao(sit: dict[str, Any]) -> tuple[Any, str, float, float, Any]:
    """(plano de missão, chave do elemento-alvo, d0, d_fim da comparação, instância do seguinte ou None).

    Abaixo do cruzeiro, o caminhão vem de uma zona anterior (já passada, com retomada suave): assim a
    única decisão brusca do plano é a do elemento-alvo, como na corrida.
    """
    from mover.autonomo.missao import Parada, PlanoMissao, Zona

    d0 = 0.0 if sit["tipo"] == "partida" else 300.0
    zonas, paradas = [], []
    if sit["tipo"] != "partida" and sit["v0_kmh"] < sit["cruzeiro_kmh"] - 0.5:
        zonas.append(Zona("anterior", "zona", d0 - 20.0, d0 - 0.5, sit["v0_kmh"], "anterior"))
    ini = d0 + sit["distancia_m"]
    if sit["tipo"] in ("lombada", "faixa", "curva"):
        fim = ini + sit["comprimento_m"]
        zonas.append(Zona("alvo", sit["tipo"], ini, fim, sit["v_alvo_kmh"], "alvo"))
        chave = "alvo@0"
    elif sit["tipo"] == "parada":
        fim = ini
        paradas.append(Parada("alvo", ini, sit["duracao_s"], "alvo"))
        chave = "alvo@0"
    else:
        fim = 0.0
        chave = "partida@0"
    seg = sit.get("seguinte")
    d_fim = fim + 45.0
    if seg:
        ini2 = fim + seg["gap_m"]
        if seg["tipo"] == "parada":
            paradas.append(Parada("seguinte", ini2, seg["duracao_s"], "seguinte"))
        else:
            zonas.append(Zona("seguinte", seg["tipo"], ini2, ini2 + seg["comprimento_m"], seg["v_alvo_kmh"], "seguinte"))
        d_fim = max(fim + 5.0, min(d_fim, ini2 - 12.0))
    plano = PlanoMissao(_rota_treino(), sit["cruzeiro_kmh"], zonas, paradas)
    return plano, chave, d0, d_fim, seg


def _montagem(cfg: dict[str, Any], plano: Any):
    from mover.autonomo.montagem import montar

    return montar(cfg, plano=plano)


def _simular(m: Any, chave: str, d0: float, v0: float, d_fim: float, interv: Any) -> dict[str, Any]:
    from mover.autonomo.planejador import ate_distancia

    from mover.autonomo.governador import Intervencao

    gov = m.governador("agentes", 1)
    gov.definir("anterior@0", Intervencao("ml", 0.8, 0.6, None, "ml"))  # retomada suave da zona anterior, se houver
    if interv is not None:
        gov.definir(chave, interv)
    traj = ate_distancia(m.modelo(), gov, d0, v0, 0.0, 0.0, d_fim, max_s=90.0)
    met = m.modelo().metricas(traj)
    return {"traj": traj, "tempo_s": float(met["tempo_s"]), "custo_r": float(met["custo_r"]),
            "eventos": int(met["eventos_atencao"]) + int(met["eventos_criticos"])}


def _melhor(resultados: list[tuple[tuple, dict[str, Any]]], tempo_bruto: float, valor_tempo_r_s: float,
            max_extra_pct: float) -> tuple[tuple, dict[str, Any]]:
    def custo(r: dict[str, Any]) -> float:
        return r["custo_r"] + valor_tempo_r_s * r["tempo_s"]

    def extra(r: dict[str, Any]) -> float:
        return 100.0 * (r["tempo_s"] - tempo_bruto) / tempo_bruto if tempo_bruto > 0 else 0.0

    validos = [(c, r) for c, r in resultados if r["eventos"] == 0 and extra(r) <= max_extra_pct]
    if validos:
        return min(validos, key=lambda cr: custo(cr[1]))
    return min(resultados, key=lambda cr: (cr[1]["eventos"], custo(cr[1])))


def rotular(cfg: dict[str, Any], sit: dict[str, Any]) -> dict[str, Any] | None:
    """Simula a situação sem alteração e com as candidatas e devolve entradas + melhor suavização."""
    from mover.autonomo.governador import Intervencao
    from mover.autonomo.planejador import decisoes_bruscas

    plano, chave, d0, d_fim, seg = plano_da_situacao(sit)
    m = _montagem(cfg, plano)
    dec = m.cfg_decisao
    valor_tempo_r_s = float(dec.get("valor_tempo_r_h", 30.0)) / 3600.0
    max_extra = float(dec.get("max_tempo_extra_pct", 60.0))
    v0 = sit["v0_kmh"] / 3.6
    bruto = _simular(m, chave, d0, v0, d_fim, None)
    inst = next((i for i in plano.instancias(max(0.0, d0 - 1.0), d_fim + 1.0, 1) if i.chave == chave), None)
    if inst is None:
        return None
    brutas = [b for b in decisoes_bruscas(bruto["traj"], [inst], 0.0, m.modelo()) if b["elemento"] == chave]
    if not brutas:
        return None  # o plano não tem decisão brusca nesse elemento: nada a suavizar
    desacel_plano = min((b["acel_mps2"] for b in brutas if b["tipo"] == "frenagem_brusca"), default=0.0)
    acel_plano = max((b["acel_mps2"] for b in brutas if b["tipo"] == "arrancada_brusca"), default=0.0)

    def interv(d: float, a: float, teto: float | None) -> Any:
        return Intervencao("ml", d, a, teto, "ml")

    # etapa 1: a frenagem antecipada (retomada padrão); etapa 2: a retomada e o teto
    if sit["tipo"] == "partida":
        melhor_d = 0.8
    else:
        etapa1 = [((d, 0.6, None), _simular(m, chave, d0, v0, d_fim, interv(d, 0.6, None))) for d in DESACEL_CANDIDATAS]
        melhor_d = _melhor(etapa1, bruto["tempo_s"], valor_tempo_r_s, max_extra)[0][0]
    tetos: list[float | None] = [None]
    if seg and seg["gap_m"] < 120.0:
        tetos += [t for t in TETO_CANDIDATOS if t < sit["cruzeiro_kmh"] - 1.0 and t > seg["v_alvo_kmh"] + 2.0]
    etapa2 = [((melhor_d, a, t), _simular(m, chave, d0, v0, d_fim, interv(melhor_d, a, t)))
              for a in ACEL_CANDIDATAS for t in tetos]
    (d_ok, a_ok, teto_ok), r_ok = _melhor(etapa2, bruto["tempo_s"], valor_tempo_r_s, max_extra)
    x = caracteristicas(sit["v0_kmh"], sit["v_alvo_kmh"], sit["cruzeiro_kmh"], sit["distancia_m"], sit["comprimento_m"],
                        sit["duracao_s"], sit["tipo"] == "parada", seg["gap_m"] if seg else None,
                        seg["v_alvo_kmh"] if seg else None, desacel_plano, acel_plano)
    return {**x, "desacel_mps2": d_ok, "acel_mps2": a_ok, "teto_kmh": teto_ok if teto_ok else sit["cruzeiro_kmh"],
            "tempo_bruto_s": bruto["tempo_s"], "custo_otimo_r": r_ok["custo_r"] + valor_tempo_r_s * r_ok["tempo_s"],
            "eventos_bruto": bruto["eventos"], "eventos_otimo": r_ok["eventos"], "situacao": sit}


def _rotular_lote(args: tuple[dict[str, Any], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    cfg, situacoes = args
    saida = []
    for sit in situacoes:
        linha = rotular(cfg, sit)
        if linha is not None:
            saida.append(linha)
    return saida


def gerar_dados(cfg: dict[str, Any], situacoes: int, semente: int = 7, processos: int = 0) -> list[dict[str, Any]]:
    """Sorteia e rotula as situações (em vários processos, se der)."""
    rng = np.random.default_rng(semente)
    lista = [situacao_aleatoria(rng) for _ in range(int(situacoes))]
    processos = processos or max(1, min(8, (os.cpu_count() or 2) - 1))
    if processos <= 1 or len(lista) < 40:
        return _rotular_lote((cfg, lista))
    tamanho = max(10, math.ceil(len(lista) / (processos * 4)))
    lotes = [(cfg, lista[i:i + tamanho]) for i in range(0, len(lista), tamanho)]
    try:
        with futures.ProcessPoolExecutor(max_workers=processos) as pool:
            return [linha for parte in pool.map(_rotular_lote, lotes) for linha in parte]
    except (OSError, RuntimeError, futures.process.BrokenProcessPool) as erro:
        log.warning("Sem processos paralelos (%s); rotulando num processo só.", erro)
        return _rotular_lote((cfg, lista))


# ---------------------------------------------------------------------- treino
def treinar(cfg: dict[str, Any], situacoes: int = 2000, saida: Path | None = None, semente: int = 7,
            n_arvores: int = 200, processos: int = 0) -> dict[str, Any]:
    """Gera as situações, treina a floresta, avalia numa parte separada e grava o modelo."""
    import joblib
    import sklearn
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score

    inicio = time.monotonic()
    cs = cfg.get("suavizador_ml") or {}
    saida = saida or caminho(cs.get("modelo", ARQUIVO_PADRAO))
    linhas = gerar_dados(cfg, situacoes, semente, processos)
    if len(linhas) < 10:
        raise RuntimeError(f"poucas situações com decisão brusca para treinar ({len(linhas)})")
    x = np.array([[linha[f] for f in FEATURES] for linha in linhas])
    y = np.array([[linha[a] for a in ALVOS] for linha in linhas])
    rng = np.random.default_rng(semente + 1)
    ordem = rng.permutation(len(linhas))
    n_teste = max(1, int(round(0.2 * len(linhas))))
    teste, treino = ordem[:n_teste], ordem[n_teste:]
    modelo = RandomForestRegressor(n_estimators=int(n_arvores), min_samples_leaf=2, random_state=int(semente), n_jobs=-1)
    modelo.fit(x[treino], y[treino])
    previsto = modelo.predict(x[teste])
    r2 = {a: round(float(r2_score(y[teste, k], previsto[:, k])), 3) if len(teste) > 1 else None for k, a in enumerate(ALVOS)}
    validacao = avaliar_propostas(cfg, SuavizadorML(modelo), [linhas[i] for i in teste[:150]])
    modelo.fit(x, y)  # o modelo final usa todas as situações
    metadados = {
        "treinado_em": datetime.now().astimezone().isoformat(timespec="seconds"),
        "tipo": "RandomForestRegressor (3 saídas)",
        "situacoes_sorteadas": int(situacoes),
        "amostras": int(len(linhas)),
        "features": list(FEATURES),
        "alvos": list(ALVOS),
        "n_arvores": int(n_arvores),
        "semente": int(semente),
        "versao_sklearn": sklearn.__version__,
        "r2_teste": r2,
        "validacao": validacao,
        "importancia": {f: round(float(v), 3) for f, v in zip(FEATURES, modelo.feature_importances_)},
        "duracao_s": round(time.monotonic() - inicio, 1),
    }
    saida.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modelo": modelo, "features": FEATURES, "alvos": ALVOS, "metadados": metadados}, saida)
    saida.with_suffix(".json").write_text(json.dumps(metadados, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Modelo de ML de suavização gravado em %s: %d situações, R² %s, %s %% das propostas de teste sem "
             "eventos bruscos (%.0f s).", saida, len(linhas), r2, validacao.get("sem_eventos_pct"), metadados["duracao_s"])
    return metadados


def avaliar_propostas(cfg: dict[str, Any], suavizador: SuavizadorML, linhas: list[dict[str, Any]]) -> dict[str, Any]:
    """Simula a proposta do modelo em situações de teste: sem eventos bruscos? quanto acima do ótimo?"""
    from mover.autonomo.governador import Intervencao

    sem_eventos, excesso = 0, []
    for linha in linhas:
        sit = linha["situacao"]
        plano, chave, d0, d_fim, _ = plano_da_situacao(sit)
        m = _montagem(cfg, plano)
        valor_tempo_r_s = float(m.cfg_decisao.get("valor_tempo_r_h", 30.0)) / 3600.0
        d, a, teto = suavizador.propor({f: linha[f] for f in FEATURES})
        r = _simular(m, chave, d0, sit["v0_kmh"] / 3.6, d_fim, Intervencao("ml", d, a, teto, "ml"))
        sem_eventos += r["eventos"] == 0
        custo = r["custo_r"] + valor_tempo_r_s * r["tempo_s"]
        if linha["custo_otimo_r"] > 0:
            excesso.append(100.0 * (custo - linha["custo_otimo_r"]) / linha["custo_otimo_r"])
    n = max(1, len(linhas))
    return {"situacoes": len(linhas), "sem_eventos_pct": round(100.0 * sem_eventos / n, 1),
            "custo_acima_do_otimo_pct": round(float(np.median(excesso)), 2) if excesso else None}


def carregar_suavizador(cfg: dict[str, Any], treinar_se_faltar: bool = True, arquivo: str | Path | None = None) -> SuavizadorML | None:
    """O modelo de suavização do config (seção suavizador_ml); treina um se faltar. None se estiver desligado."""
    cs = cfg.get("suavizador_ml") or {}
    if not cs.get("ativo", True):
        return None
    destino = Path(arquivo) if arquivo else caminho(cs.get("modelo", ARQUIVO_PADRAO))
    try:
        if destino.exists():
            try:
                return SuavizadorML.carregar(destino)
            except Exception as erro:  # versão diferente do scikit-learn ou do modelo
                log.warning("Modelo de ML de suavização inválido (%s): treinando de novo.", erro)
        if not treinar_se_faltar:
            return None
        log.info("Treinando o modelo de ML de suavização (uma vez; %d situações simuladas, ~1-2 min)...",
                 int(cs.get("situacoes", 2000)))
        treinar(cfg, int(cs.get("situacoes", 2000)), destino, int(cs.get("semente", 7)), int(cs.get("n_arvores", 200)),
                int(cs.get("processos", 0)))
        return SuavizadorML.carregar(destino)
    except ImportError as erro:
        log.warning("Sem o scikit-learn (%s): os agentes seguem só com as opções fixas de suavização.", erro)
        return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Treina o modelo de ML que propõe quanto suavizar cada decisão brusca.")
    parser.add_argument("--config", default="config/autonomo.yaml")
    parser.add_argument("--situacoes", type=int, help="situações simuladas (padrão: YAML, 2000)")
    parser.add_argument("--processos", type=int, help="processos em paralelo (padrão: núcleos - 1)")
    parser.add_argument("--saida", help="arquivo .joblib (padrão: YAML)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    cfg = carregar_yaml(args.config)
    cs = cfg.get("suavizador_ml") or {}
    meta = treinar(cfg, int(args.situacoes or cs.get("situacoes", 2000)), caminho(args.saida) if args.saida else None,
                   int(cs.get("semente", 7)), int(cs.get("n_arvores", 200)), int(args.processos or cs.get("processos", 0)))
    print(json.dumps({k: meta[k] for k in ("amostras", "r2_teste", "validacao", "importancia", "duracao_s")},
                     ensure_ascii=False, indent=2))


__all__ = ["ALVOS", "ARQUIVO_PADRAO", "FEATURES", "SuavizadorML", "avaliar_propostas", "caracteristicas",
           "carregar_suavizador", "gerar_dados", "rotular", "situacao_aleatoria", "treinar"]

if __name__ == "__main__":
    main()
