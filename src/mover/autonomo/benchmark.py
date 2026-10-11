"""Benchmark: N voltas sem os agentes e N voltas com os agentes, no mesmo simulador.

Mede o efeito prático da camada agêntica na volta autônoma do campus:
1. bateria "sem agentes": o caminhão segue o próprio plano sozinho (frenagens tardias e arrancadas
   em degrau perto das zonas);
2. bateria "com agentes": a cada bloco, o caminhão manda aos agentes o plano dos próximos 10 s,
   montado a partir do estado atual da corrida, e os agentes o alteram antes de ele ser executado
   (a simulação espera cada decisão, então o resultado não depende da velocidade do computador).
Cada bateria sai parada da largada e termina parada na chegada. Entre as duas, o caminhão volta à
largada (no CARLA, um caminhão novo).

O consumo vem do mesmo modelo (VT-CPFM, diesel) aplicado à velocidade e à aceleração medidas nas
duas baterias. Saída em data/benchmark/<data>/:
- resultados.csv: uma linha por volta de cada bateria, as médias, a diferença e a extrapolação
  para 100 voltas (L, R$, kg de CO2, eventos bruscos e tempo);
- resumo.json (lido pela página), decisoes.jsonl, telemetria_sem_agentes.csv,
  telemetria_com_agentes.csv e comparativo.png.

Uso, a partir da pasta src/ do repositório:
    python -m mover.autonomo.benchmark --voltas 3               # CARLA sem desenhar a cena (mais rápido)
    python -m mover.autonomo.benchmark --voltas 3 --com-render  # CARLA desenhando (para acompanhar)
    python -m mover.autonomo.benchmark --voltas 3 --sem-carla   # simulador substituto
    python -m mover.autonomo.benchmark --voltas 3 --sem-llm     # agentes só com a regra
    python -m mover.autonomo.benchmark --voltas 3 --sem-ml      # agentes sem o modelo de ML (opções fixas)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # execução direta: python src/mover/autonomo/benchmark.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from mover.agentes.llm import PROVEDORES
from mover.autonomo.conducao import Conducao, OpcoesConducao, ResultadoConducao, metricas_da_corrida
from mover.autonomo.montagem import Montagem, montar
from mover.config import caminho, carregar_yaml

log = logging.getLogger("mover.autonomo")

COLUNAS = ("tempo_s", "distancia_m", "vel_media_kmh", "combustivel_ml", "custo_r", "co2_g", "eventos_atencao",
           "eventos_criticos", "eventos_total", "jerk_max_mps3", "acel_min_mps2", "acel_max_mps2", "energia_freio_wh")
NOMES_MODO = {"bruto": "sem_agentes", "agentes": "com_agentes"}


def linha_da_volta(modo: str, volta: Any, m: dict[str, Any], preco: float, co2_kg_l: float) -> dict[str, Any]:
    litros = float(m["combustivel_l"])
    return {
        "tipo": "volta", "bateria": NOMES_MODO.get(modo, modo), "volta": volta,
        "tempo_s": round(float(m["tempo_s"]), 2), "distancia_m": round(float(m["distancia_m"]), 1),
        "vel_media_kmh": round(float(m["vel_media_kmh"]), 2), "combustivel_ml": round(1000 * litros, 2),
        "custo_r": round(litros * preco, 4), "co2_g": round(1000 * litros * co2_kg_l, 1),
        "eventos_atencao": int(m["eventos_atencao"]), "eventos_criticos": int(m["eventos_criticos"]),
        "eventos_total": int(m["eventos_atencao"]) + int(m["eventos_criticos"]),
        "jerk_max_mps3": round(float(m["jerk_max_abs_mps3"]), 2), "acel_min_mps2": round(float(m["acel_min_mps2"]), 2),
        "acel_max_mps2": round(float(m["acel_max_mps2"]), 2),
        "energia_freio_wh": round(1000 * float(m["energia_freio_kwh"]), 1),
    }


def extrapolar(voltas: list[dict[str, Any]], coluna: str, alvo: int) -> float:
    """Total em `alvo` voltas: 1ª volta (saída parada) + última (parada final) + as do meio pela média.

    Com menos de 3 voltas medidas, é a média por volta vezes `alvo` (aproximação).
    """
    valores = [float(v[coluna]) for v in voltas]
    if not valores:
        return float("nan")
    if len(valores) >= 3:
        return valores[0] + valores[-1] + float(np.mean(valores[1:-1])) * (alvo - 2)
    return float(np.mean(valores)) * alvo


def montar_resultados(voltas_bruto: list[dict[str, Any]], voltas_agentes: list[dict[str, Any]], alvo: int) -> pd.DataFrame:
    linhas = list(voltas_bruto) + list(voltas_agentes)
    medias = {}
    for nome, voltas in (("sem_agentes", voltas_bruto), ("com_agentes", voltas_agentes)):
        media = {c: round(float(np.mean([v[c] for v in voltas])), 3) for c in COLUNAS} if voltas else {}
        medias[nome] = media
        linhas.append({"tipo": "media_por_volta", "bateria": nome, "volta": "", **media})
    if voltas_bruto and voltas_agentes:
        sem, com = medias["sem_agentes"], medias["com_agentes"]
        linhas.append({"tipo": "diferenca_por_volta", "bateria": "com - sem", "volta": "",
                       **{c: round(com[c] - sem[c], 3) for c in COLUNAS}})
        linhas.append({"tipo": "diferenca_pct", "bateria": "(com - sem) / sem", "volta": "",
                       **{c: round(100.0 * (com[c] - sem[c]) / sem[c], 2) if sem[c] else None for c in COLUNAS}})
        for nome, voltas in (("sem_agentes", voltas_bruto), ("com_agentes", voltas_agentes)):
            linhas.append({"tipo": f"total_{alvo}_voltas", "bateria": nome, "volta": alvo,
                           **{c: round(extrapolar(voltas, c, alvo), 2) for c in COLUNAS if c not in
                              ("vel_media_kmh", "jerk_max_mps3", "acel_min_mps2", "acel_max_mps2")}})
    return pd.DataFrame(linhas)


def resumo_da_comparacao(voltas_bruto: list[dict[str, Any]], voltas_agentes: list[dict[str, Any]], alvo: int,
                         preco: float, co2_kg_l: float) -> dict[str, Any]:
    def bateria(voltas: list[dict[str, Any]]) -> dict[str, Any]:
        return {"voltas": len(voltas),
                "por_volta": {c: round(float(np.mean([v[c] for v in voltas])), 3) for c in COLUNAS} if voltas else {},
                "total": {c: round(float(np.sum([v[c] for v in voltas])), 3) for c in ("tempo_s", "combustivel_ml", "custo_r",
                                                                                    "co2_g", "eventos_total")}}

    sem, com = bateria(voltas_bruto), bateria(voltas_agentes)
    resumo: dict[str, Any] = {"sem_agentes": sem, "com_agentes": com}
    if voltas_bruto and voltas_agentes:
        ps, pc = sem["por_volta"], com["por_volta"]
        resumo["diferenca_pct"] = {c: round(100.0 * (pc[c] - ps[c]) / ps[c], 2) if ps[c] else None
                                   for c in ("combustivel_ml", "tempo_s", "eventos_total", "energia_freio_wh")}
        litros_sem = extrapolar(voltas_bruto, "combustivel_ml", alvo) / 1000.0
        litros_com = extrapolar(voltas_agentes, "combustivel_ml", alvo) / 1000.0
        tempo_sem = extrapolar(voltas_bruto, "tempo_s", alvo)
        tempo_com = extrapolar(voltas_agentes, "tempo_s", alvo)
        resumo["extrapolacao"] = {
            "voltas": alvo,
            "litros_sem_agentes": round(litros_sem, 2), "litros_com_agentes": round(litros_com, 2),
            "economia_l": round(litros_sem - litros_com, 2),
            "economia_r": round((litros_sem - litros_com) * preco, 2),
            "economia_co2_kg": round((litros_sem - litros_com) * co2_kg_l, 2),
            "eventos_sem_agentes": round(extrapolar(voltas_bruto, "eventos_total", alvo)),
            "eventos_com_agentes": round(extrapolar(voltas_agentes, "eventos_total", alvo)),
            "tempo_extra_min": round((tempo_com - tempo_sem) / 60.0, 1),
        }
    return resumo


def desenhar(pasta: Path, res_bruto: ResultadoConducao, res_agentes: ResultadoConducao, comprimento: float,
             tabela: pd.DataFrame) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [3, 2]})
    for res, cor, rotulo in ((res_bruto, "#d9534f", "sem agentes"), (res_agentes, "#2e8b57", "com agentes")):
        d, v = res.coluna("d"), res.coluna("speed_mps") * 3.6
        volta = (d >= 0) & (d < comprimento)  # 1ª volta
        ax1.plot(d[volta], v[volta], color=cor, lw=1.6, label=rotulo)
    if res_agentes.linhas:
        d, lim = res_agentes.coluna("d"), res_agentes.coluna("v_limite_kmh")
        volta = (d >= 0) & (d < comprimento)
        ax1.plot(d[volta], lim[volta], color="#888888", lw=1, ls="--", label="limite do plano")
    ax1.set_xlabel("distância na 1ª volta (m)")
    ax1.set_ylabel("velocidade (km/h)")
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax1.set_title("Volta autônoma no campus: caminhão sozinho x camada agêntica")
    voltas = tabela[tabela["tipo"] == "volta"]
    largura = 0.38
    for i, (bateria, cor) in enumerate((("sem_agentes", "#d9534f"), ("com_agentes", "#2e8b57"))):
        dados = voltas[voltas["bateria"] == bateria]
        x = np.arange(len(dados)) + (i - 0.5) * largura
        ax2.bar(x, dados["combustivel_ml"], width=largura, color=cor, label=bateria.replace("_", " "))
    ax2.set_xticks(np.arange(int(voltas["volta"].astype(int).max()) if len(voltas) else 0))
    ax2.set_xticklabels([f"volta {k + 1}" for k in range(len(ax2.get_xticks()))])
    ax2.set_ylabel("diesel (mL)")
    ax2.grid(alpha=0.3, axis="y")
    ax2.legend()
    fig.tight_layout()
    destino = pasta / "comparativo.png"
    fig.savefig(destino, dpi=130)
    plt.close(fig)
    return destino


def rodar_bateria(m: Montagem, sim: Any, modo: str, voltas: int, llm: Any, imprimir: Any,
                  suavizador: Any = None) -> ResultadoConducao:
    governador = m.governador(modo, voltas)
    decisor = m.decisor(voltas, llm, suavizador) if modo == "agentes" else None
    opcoes = OpcoesConducao(voltas=voltas, usar_llm=llm is not None, assincrona=False, sem_espera=True,
                            duracao_bloco_s=m.duracao_bloco_s)
    inicio = time.monotonic()
    resultado = Conducao(m.plano, sim, governador, decisor, opcoes, m.caminhao, m.consumo, imprimir=imprimir).rodar()
    if decisor is not None:
        decisor.encerrar()
    log.info("Bateria %s: %s em %.1f s de relógio (%d linhas, %d decisões).", NOMES_MODO[modo], resultado.motivo_fim,
             time.monotonic() - inicio, len(resultado.linhas), len(resultado.decisoes))
    return resultado


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark: N voltas sem agentes e N voltas com agentes.")
    parser.add_argument("--voltas", type=int, help="voltas de cada bateria (padrão: YAML, 3)")
    parser.add_argument("--config", default="config/autonomo.yaml")
    parser.add_argument("--config-simulacao", default="config/simulacao.yaml")
    parser.add_argument("--config-agentes", default="config/agentes.yaml")
    parser.add_argument("--missao", help="plano de missão (padrão: YAML)")
    parser.add_argument("--sem-carla", action="store_true", help="usa o simulador substituto")
    parser.add_argument("--com-render", action="store_true", help="o CARLA desenha a cena (mais lento)")
    parser.add_argument("--manter-mundo", action="store_true", help="usa o mapa já aberto no CARLA")
    parser.add_argument("--sem-llm", action="store_true", help="os agentes decidem só pela regra")
    parser.add_argument("--sem-ml", action="store_true", help="sem o modelo de ML de suavização (só as opções fixas)")
    parser.add_argument("--provedor", choices=PROVEDORES, help="provedor do LLM (padrão: .env ou YAML)")
    parser.add_argument("--modelo", help="modelo no provedor")
    parser.add_argument("--extrapolar", type=int, help="voltas da extrapolação (padrão: YAML, 100)")
    parser.add_argument("--saida", help="pasta de saída (padrão: data/benchmark/<data>)")
    parser.add_argument("--silencioso", action="store_true", help="não imprime cada decisão")
    args = parser.parse_args(argv)

    from mover.autonomo.dirigir import criar_llm_decisoes, preparar_simulador
    from mover.servidor.rodar_servidor import configurar_logs

    configurar_logs()
    cfg = carregar_yaml(args.config)
    cfg_sim = carregar_yaml(args.config_simulacao) if caminho(args.config_simulacao).exists() else {}
    cfg_agentes = carregar_yaml(args.config_agentes)
    cb = cfg.get("benchmark") or {}
    m = montar(cfg, args.missao)
    voltas = int(args.voltas or cb.get("voltas", 3))
    alvo = int(args.extrapolar or cb.get("extrapolar_voltas", 100))
    llm = None if args.sem_llm else criar_llm_decisoes(cfg_agentes, args.provedor, args.modelo,
                                                       float(m.cfg_decisao.get("prazo_llm_s", 6.0)))
    suavizador = None if args.sem_ml else m.suavizador()
    pasta = caminho(args.saida) if args.saida else caminho(cb.get("pasta", "data/benchmark")) / datetime.now().strftime("%Y%m%d_%H%M%S")
    pasta.mkdir(parents=True, exist_ok=True)
    imprimir = (lambda texto: None) if args.silencioso else (lambda texto: print(texto, flush=True))
    args_sim = argparse.Namespace(sem_carla=args.sem_carla, mapa=None, manter_mundo=args.manter_mundo,
                                  sem_render=not args.com_render, camera=None)
    sim, _ = preparar_simulador(args_sim, m, cfg_sim, voltas)
    codigo = 0
    try:
        sim.iniciar()
        log.info("Benchmark no %s: %d volta(s) por bateria; plano de %.0f m por volta.", sim.nome, voltas, m.plano.comprimento)
        res_bruto = rodar_bateria(m, sim, "bruto", voltas, None, imprimir)
        sim.reiniciar()
        res_agentes = rodar_bateria(m, sim, "agentes", voltas, llm, imprimir, suavizador)
    except KeyboardInterrupt:
        log.warning("Benchmark interrompido (Ctrl+C).")
        return 130
    finally:
        sim.encerrar()

    preco, co2 = m.consumo.preco_r_l, m.consumo.co2_kg_l
    met_bruto = metricas_da_corrida(res_bruto, m.plano.comprimento, m.caminhao, m.consumo, m.limiares)
    met_agentes = metricas_da_corrida(res_agentes, m.plano.comprimento, m.caminhao, m.consumo, m.limiares)
    v_bruto = [linha_da_volta("bruto", v["volta"], v, preco, co2) for v in met_bruto["voltas"]]
    v_agentes = [linha_da_volta("agentes", v["volta"], v, preco, co2) for v in met_agentes["voltas"]]
    tabela = montar_resultados(v_bruto, v_agentes, alvo)
    tabela.to_csv(pasta / "resultados.csv", index=False)
    pd.DataFrame(res_bruto.linhas).to_csv(pasta / "telemetria_sem_agentes.csv", index=False)
    pd.DataFrame(res_agentes.linhas).to_csv(pasta / "telemetria_com_agentes.csv", index=False)
    with (pasta / "decisoes.jsonl").open("w", encoding="utf-8") as f:
        for entrada in res_agentes.decisoes:
            entrada = {k: v for k, v in entrada.items() if k != "curvas"}  # as curvas só servem para a página
            f.write(json.dumps(entrada, ensure_ascii=False) + "\n")
    figura = desenhar(pasta, res_bruto, res_agentes, m.plano.comprimento, tabela)
    resumo = {
        "data": datetime.now().isoformat(timespec="seconds"), "simulador": sim.nome, "voltas": voltas,
        "combustivel": m.consumo.combustivel, "preco_r_l": preco, "co2_kg_l": co2,
        "modelo_consumo": "VT-CPFM (Willans + termo quadrático) sobre a velocidade e a aceleração medidas",
        "llm_decisoes": llm.nome if llm else None,
        "suavizador_ml": suavizador.nome if suavizador is not None else None,
        "decisoes": len(res_agentes.decisoes),
        "decisoes_por_origem": {o: sum(d.get("texto_origem") == o for d in res_agentes.decisoes) for o in ("llm", "regra")},
        "opcoes": {o: sum(d.get("opcao") == o for d in res_agentes.decisoes)
                   for o in sorted({d.get("opcao") for d in res_agentes.decisoes})},
        "fim": {"sem_agentes": res_bruto.motivo_fim, "com_agentes": res_agentes.motivo_fim},
        **resumo_da_comparacao(v_bruto, v_agentes, alvo, preco, co2),
        "arquivos": {"resultados": str(pasta / "resultados.csv"), "figura": str(figura) if figura else None},
    }
    (pasta / "resumo.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")

    print(tabela[tabela["tipo"] != "volta"].to_string(index=False), flush=True)
    if "extrapolacao" in resumo:
        e, dif = resumo["extrapolacao"], resumo["diferenca_pct"]
        print(f"\nCom os agentes, cada volta gastou {dif['combustivel_ml']:+.1f} % de diesel, {dif['tempo_s']:+.1f} % de tempo "
              f"e {dif['eventos_total']:+.0f} % de eventos bruscos. Em {e['voltas']} voltas: {e['litros_sem_agentes']:.1f} L "
              f"sem agentes x {e['litros_com_agentes']:.1f} L com agentes = {e['economia_l']:.1f} L "
              f"(R$ {e['economia_r']:.2f}; {e['economia_co2_kg']:.1f} kg de CO2), "
              f"{e['eventos_sem_agentes']} x {e['eventos_com_agentes']} eventos bruscos, {e['tempo_extra_min']:+.1f} min.",
              flush=True)
    log.info("Resultados em %s", pasta)
    if not (res_bruto.completa and res_agentes.completa):
        log.error("Alguma bateria não chegou ao fim (%s / %s): confira antes de usar os números.",
                  res_bruto.motivo_fim, res_agentes.motivo_fim)
        codigo = 2
    return codigo


if __name__ == "__main__":
    sys.exit(main())
