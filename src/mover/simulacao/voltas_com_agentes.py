"""Fase Y: as voltas da Fase X refeitas no CARLA com a camada agêntica em malha fechada.

Para cada volta gravada pela Fase X (voltas_autonomas.py):

1. O servidor abre uma sessão com a telemetria da volta (o plano do caminhão autônomo, X) e
   analisa o bloco 0 antes da partida.
2. O caminhão Y percorre o mesmo caminho de X, com a pose imposta a 20 Hz. Ao entrar no bloco k,
   recebe o log e os problemas do bloco k, e o servidor libera a análise do bloco k+1: os agentes
   leem os 10 s seguintes do plano antes de Y chegar lá.
3. Os ajustes dos problemas (zonas de suavização e tetos de velocidade) voltam para Y assim que a
   análise termina (GET /sessao/previsao) e mudam a velocidade dali em diante (plano_velocidade.py).
   Y nunca anda mais rápido que X no mesmo ponto do caminho.
4. Y só passa do ponto de decisão de cada bloco (`ajustes.antecedencia_s` antes do bloco seguinte)
   com a análise desse bloco pronta. Se ela atrasar, o caminhão espera, como já acontecia na
   entrada de cada bloco.
5. No fim, a telemetria executada (Y) é gravada ao lado da de X, com a comparação dos eventos de
   jerk, e cada volta vira uma linha de data/resultados_pesquisa/voltas_agentes_<data>.csv.

Uso, a partir da pasta src/, com o CarlaUE4 0.9.16 aberto:
    python -m mover.simulacao.voltas_com_agentes --iniciar-servidor                  # todas as voltas
    python -m mover.simulacao.voltas_com_agentes --iniciar-servidor --volta volta_001 --manter-servidor
    python -m mover.simulacao.voltas_com_agentes --sem-carla --iniciar-servidor --sem-espera --provedor falso
e a página em http://127.0.0.1:8000/ (simulação, dashboard, log e chat).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

if __package__ in (None, ""):  # execução direta: python src/mover/simulacao/voltas_com_agentes.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from mover.agentes.especialistas import EspecialistaJerk
from mover.agentes.llm import PROVEDORES
from mover.config import caminho, carregar_yaml
from mover.simulacao.plano_velocidade import PlanoVelocidade
from mover.simulacao.replay_carla import (ClienteAgentes, ErroServidor, Mundo, MundoCarla, MundoFalso, Relogio, Trajeto,
                                          esperar_ctrl_c, mostrar_bloco)
from mover.simulacao.voltas_autonomas import (ARQUIVO_TELEMETRIA, ARQUIVO_VOLTA, cfg_voltas, derivar_sinais,
                                              horas_locais)

log = logging.getLogger("mover.simulacao")

ARQUIVO_TELEMETRIA_Y = "telemetria_y.csv"
ARQUIVO_COMPARACAO = "comparacao.json"


# ---------------------------------------------------------------------------------------------
# Voltas gravadas
# ---------------------------------------------------------------------------------------------
def listar_voltas(pasta: Path, nomes: list[str] | None = None, quantidade: int | None = None) -> list[Path]:
    voltas = sorted(p.parent for p in pasta.glob(f"*/{ARQUIVO_TELEMETRIA}"))
    if nomes:
        voltas = [p for p in voltas if p.name in set(nomes)]
    return voltas[:quantidade] if quantidade else voltas


def trajeto_da_volta(tel: pd.DataFrame) -> Trajeto:
    """Poses de X no referencial do CARLA (a volta já foi gravada no mapa: não há alinhamento)."""
    zeros = np.zeros(len(tel))
    return Trajeto(tel["sim_time"].to_numpy(float), tel["x"].to_numpy(float), tel["y"].to_numpy(float),
                   tel["yaw"].to_numpy(float), zeros, zeros)


def poses_no_mapa(tel: pd.DataFrame) -> pd.DataFrame:
    """Poses de X no referencial do OpenDRIVE (y para cima), para a cena da página."""
    return pd.DataFrame({"sim_time": tel["sim_time"], "x_mapa": tel["x"], "y_mapa": -tel["y"], "yaw_carla": tel["yaw"]})


def ajustes_das_entradas(entradas: list[dict[str, Any]]) -> list[tuple[int, dict[str, Any]]]:
    return [(e["id"], e["ajuste"]) for e in entradas if e.get("tipo") == "problema" and e.get("ajuste")]


# ---------------------------------------------------------------------------------------------
# Laço da Fase Y
# ---------------------------------------------------------------------------------------------
@dataclass
class ResultadoVoltaY:
    ticks: int = 0
    estados: int = 0
    blocos: list[dict[str, Any]] = field(default_factory=list)
    entradas: list[dict[str, Any]] = field(default_factory=list)
    tau: list[float] = field(default_factory=list)          # relógio do plano X em cada passo de Y
    esperas_previsao: list[dict[str, Any]] = field(default_factory=list)
    ajustes: list[dict[str, Any]] = field(default_factory=list)
    interrompido: bool = False
    duracao_real_s: float = 0.0
    atrasos_relogio: int = 0

    def para_json(self) -> dict[str, Any]:
        return {"ticks": self.ticks, "estados": self.estados, "blocos": self.blocos, "n_entradas": len(self.entradas),
                "n_problemas": sum(e["tipo"] == "problema" for e in self.entradas), "ajustes": self.ajustes,
                "esperas_previsao": self.esperas_previsao, "interrompido": self.interrompido,
                "duracao_real_s": self.duracao_real_s, "atrasos_relogio": self.atrasos_relogio,
                "maior_espera_bloco_s": max((b["espera_s"] for b in self.blocos[1:]), default=0.0),
                "maior_espera_previsao_s": max((e["espera_s"] for e in self.esperas_previsao), default=0.0)}


def _imprimir(texto: str) -> None:
    print(texto, flush=True)


def rodar_volta_y(trajeto: Trajeto, v_x: np.ndarray, cliente: ClienteAgentes, mundo: Mundo, sessao: dict[str, Any], *,
                  antecedencia_s: float = 6.0, rampa_replanejamento_s: float = 1.5, passo_s: float = 0.05,
                  consulta_s: float = 0.5, fator_tempo: float = 1.0, sem_espera: bool = False, estado_a_cada: int = 4,
                  marcar_problemas: bool = True, silencioso: bool = False, timeout_previsao_s: float = 300.0,
                  imprimir: Callable[[str], None] = _imprimir) -> ResultadoVoltaY:
    """Percorre a volta X com a velocidade do plano ajustado pelos agentes; devolve o relógio de X por passo de Y."""
    n_blocos, duracao, t0 = int(sessao["n_blocos"]), float(sessao["duracao_bloco_s"]), float(sessao["t0"])
    plano = PlanoVelocidade(trajeto.t, v_x, rampa_replanejamento_s)
    relogio = Relogio(fator_tempo, sem_espera)
    res = ResultadoVoltaY()
    conhecidos: set[int] = set()  # blocos cujos ajustes Y já recebeu
    consulta_a_cada = max(1, int(round(consulta_s / passo_s)))
    tau, t_y, anunciado, i = float(trajeto.t[0]), 0.0, -1, 0
    t_fim = float(trajeto.t[-1])

    def bloco_de(t: float) -> int:
        return int(np.clip(math.floor((t - t0) / duracao + 1e-9), 0, n_blocos - 1))

    def incorporar(entradas: list[dict[str, Any]], bloco: int) -> None:
        conhecidos.add(bloco)
        for id_entrada, ajuste in plano.atualizar(ajustes_das_entradas(entradas), tau):
            if not silencioso:
                imprimir(f"    ajuste #{id_entrada:03d} (bloco {bloco:02d}): {ajuste['descricao']}")

    def esperar_previsao(bloco: int) -> None:
        inicio, limite = time.monotonic(), time.monotonic() + timeout_previsao_s
        while True:
            previsao = cliente.previsao()
            if previsao and previsao.get("bloco") == bloco:
                incorporar(previsao["entradas"], bloco)
                break
            if time.monotonic() > limite:
                log.warning("A análise do bloco %d não chegou em %.0f s; o caminhão segue sem os ajustes dele.",
                            bloco, timeout_previsao_s)
                conhecidos.add(bloco)
                break
            time.sleep(0.02)
        res.esperas_previsao.append({"bloco": bloco, "tau_s": round(tau, 2), "espera_s": round(time.monotonic() - inicio, 3)})

    inicio = time.monotonic()
    relogio.ancorar(t_y)
    try:
        while tau <= t_fim + 1e-9:
            k = bloco_de(tau)
            if k > anunciado:
                for kk in range(anunciado + 1, k + 1):
                    resposta = cliente.entrar_no_bloco(kk)
                    entradas = resposta["entradas"]
                    res.entradas.extend(entradas)
                    res.blocos.append({"bloco": kk, "espera_s": resposta["espera_s"], "latencia_s": resposta["latencia_s"],
                                       "tau_s": round(tau, 2), "t_y_s": round(t_y, 2), "n_entradas": len(entradas),
                                       "n_problemas": sum(e["tipo"] == "problema" for e in entradas)})
                    mostrar_bloco(resposta, t0, duracao, imprimir, silencioso)
                    incorporar(entradas, kk)
                    marcas = [(trajeto.quadro_mais_proximo(float(e["t_pico"])), e) for e in entradas
                              if e["tipo"] == "problema" and e.get("t_pico") is not None]
                    if marcar_problemas and marcas:
                        mundo.marcar_problemas(marcas)
                anunciado = k
                relogio.ancorar(t_y)  # a espera pela análise não conta como atraso
            proximo = anunciado + 1
            if proximo < n_blocos and proximo not in conhecidos:
                if tau >= t0 + proximo * duracao - antecedencia_s:
                    esperar_previsao(proximo)
                    relogio.ancorar(t_y)
                elif i % consulta_a_cada == 0:
                    previsao = cliente.previsao()
                    if previsao and previsao.get("bloco") == proximo:
                        incorporar(previsao["entradas"], proximo)
            relogio.esperar_ate(t_y)
            mundo.aplicar_em(tau)
            mundo.avancar()
            res.tau.append(tau)
            res.ticks += 1
            taxa = plano.taxa(tau)
            if estado_a_cada > 0 and i % estado_a_cada == 0:
                x, y, yaw = trajeto.pose_em(tau)
                v_plano = plano.velocidade_x(tau)
                cliente.enviar_estado({
                    "sim_time": round(tau, 3), "x": round(x, 3), "y": round(y, 3), "z": round(mundo.altura_em(tau), 3),
                    "yaw": round(yaw, 2), "speed_kmh": round(taxa * v_plano * 3.6, 2),
                    "vel_plano_kmh": round(v_plano * 3.6, 2), "t_y": round(t_y, 2), "correcao_m": 0.0, "dist_faixa_m": 0.0,
                })
                res.estados += 1
            tau += taxa * passo_s
            t_y += passo_s
            i += 1
    except KeyboardInterrupt:
        res.interrompido = True
        imprimir("Volta interrompida (Ctrl+C).")
    res.ajustes = plano.chegadas
    res.duracao_real_s = round(time.monotonic() - inicio, 3)
    res.atrasos_relogio = relogio.atrasos
    return res


# ---------------------------------------------------------------------------------------------
# Telemetria executada e comparação X x Y
# ---------------------------------------------------------------------------------------------
def telemetria_executada(tau: np.ndarray, tel_x: pd.DataFrame, cfg_tratamento: dict[str, Any], inicio: datetime,
                         passo_s: float = 0.05) -> pd.DataFrame:
    """Telemetria do caminhão que percorreu as poses de X no relógio `tau` (um valor por passo de `passo_s`).

    Com tau = sim_time de X, sai a própria volta X pelo mesmo cálculo, o que deixa a comparação justa.
    """
    tau = np.asarray(tau, float)
    t = np.arange(len(tau)) * passo_s
    tx = tel_x["sim_time"].to_numpy(float)
    s = np.interp(tau, tx, tel_x["odom_m"].to_numpy(float))
    yaw = np.interp(tau, tx, np.degrees(np.unwrap(np.radians(tel_x["yaw"].to_numpy(float)))))
    v = np.gradient(s, passo_s) if len(tau) > 1 else np.zeros_like(s)
    sinais = derivar_sinais(t, v, yaw, cfg_tratamento, acc_vert=np.interp(tau, tx, tel_x["acc_vert"].to_numpy(float)))
    return pd.DataFrame({
        "sim_time": t, "tau_x": tau,
        "x": np.interp(tau, tx, tel_x["x"].to_numpy(float)), "y": np.interp(tau, tx, tel_x["y"].to_numpy(float)),
        "yaw": (yaw + 180.0) % 360.0 - 180.0,
        "hora_local": horas_locais(inicio, t), **sinais, "fonte": "real",
    }).round(5)


def resumo_de_conducao(tel: pd.DataFrame, especialista: EspecialistaJerk) -> dict[str, Any]:
    eventos = especialista.detectar(tel, -math.inf, math.inf)
    niveis = Counter(e.nivel for e in eventos)
    return {"duracao_s": round(float(tel["sim_time"].iloc[-1]), 2), "vel_media_kmh": round(float(tel["speed_kmh"].mean()), 1),
            "eventos": len(eventos), "criticos": niveis.get("critico", 0), "atencao": niveis.get("atencao", 0),
            "jerk_max_abs_mps3": round(float(tel["jerk_long"].abs().max()), 2),
            "acel_long_min_mps2": round(float(tel["acc_long"].min()), 2),
            "acel_long_max_mps2": round(float(tel["acc_long"].max()), 2)}


def comparar(tel_x: pd.DataFrame, tel_y: pd.DataFrame, cfg_agentes: dict[str, Any]) -> dict[str, Any]:
    especialista = EspecialistaJerk(cfg_agentes["limiares"])
    x, y = resumo_de_conducao(tel_x, especialista), resumo_de_conducao(tel_y, especialista)
    return {"x": x, "y": y, "acrescimo_tempo_s": round(y["duracao_s"] - x["duracao_s"], 2),
            "eventos_evitados": x["eventos"] - y["eventos"]}


# ---------------------------------------------------------------------------------------------
# Linha de comando
# ---------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fase Y: voltas da Fase X refeitas no CARLA com a camada agêntica.")
    parser.add_argument("--config", default="config/simulacao.yaml", help="YAML da simulação")
    parser.add_argument("--config-agentes", default="config/agentes.yaml", help="YAML da camada agêntica")
    parser.add_argument("--config-tratamento", default="config/tratamento.yaml", help="filtros e manobras")
    parser.add_argument("--pasta", help="pasta das voltas da Fase X (padrão: YAML, data/voltas)")
    parser.add_argument("--volta", nargs="*", help="só estas voltas (nomes das pastas, ex.: volta_001)")
    parser.add_argument("--quantidade", type=int, help="no máximo N voltas")
    parser.add_argument("--mapa", help="arquivo .xodr (padrão: YAML)")
    parser.add_argument("--sem-carla", action="store_true", help="não usa o simulador: percorre as poses e fala com o servidor")
    parser.add_argument("--manter-mundo", action="store_true", help="não gera o mundo OpenDRIVE; usa o que está aberto")
    parser.add_argument("--camera", choices=("perseguicao", "cima", "livre"), help="câmera do simulador (padrão: YAML)")
    parser.add_argument("--sem-camera-painel", action="store_true", help="sem a câmera RGB do painel 1")
    parser.add_argument("--servidor", help="URL do servidor da camada agêntica (padrão: host/porta do YAML)")
    parser.add_argument("--iniciar-servidor", action="store_true", help="sobe o servidor da camada agêntica neste processo")
    parser.add_argument("--manter-servidor", action="store_true", help="com --iniciar-servidor: página no ar até o Ctrl+C")
    parser.add_argument("--injetar-eventos", action="store_true", help="soma a frenagem e a arrancada sintéticas ao plano")
    parser.add_argument("--provedor", choices=PROVEDORES, help="provedor do LLM (padrão: .env ou YAML)")
    parser.add_argument("--modelo", help="modelo no provedor (padrão: .env ou YAML)")
    parser.add_argument("--sem-ml", action="store_true", help="desliga o especialista de ML")
    parser.add_argument("--fator-tempo", type=float, help="2 = duas vezes mais rápido que o real (padrão: YAML)")
    parser.add_argument("--sem-espera", action="store_true", help="não segue o relógio (o mais rápido possível)")
    parser.add_argument("--silencioso", action="store_true", help="não imprime cada entrada do log")
    args = parser.parse_args(argv)

    from mover.servidor.rodar_servidor import configurar_logs

    configurar_logs()
    cfg = carregar_yaml(args.config)
    cfg_ag = carregar_yaml(args.config_agentes)
    cfg_trat = carregar_yaml(args.config_tratamento)
    cvol, cr, cs = cfg_voltas(cfg), cfg.get("replay", {}), cfg.get("servidor", {})
    pasta = caminho(args.pasta or cvol["pasta"])
    cfg["voltas"] = {**cvol, "pasta": str(pasta)}  # o servidor embutido abre as sessões na mesma pasta
    voltas = listar_voltas(pasta, args.volta, args.quantidade)
    if not voltas:
        raise SystemExit(f"Nenhuma volta em {pasta}. Rode antes: python -m mover.simulacao.voltas_autonomas")
    fator_tempo = float(args.fator_tempo or cr.get("fator_tempo", 1.0))
    passo_s = float(cfg.get("carla", {}).get("passo_s", 0.05))
    antecedencia_s = float((cfg_ag.get("ajustes") or {}).get("antecedencia_s", 6.0))

    texto_xodr, mapa = None, None
    arquivo_mapa = caminho(args.mapa or cfg["mapa"]["arquivo"])
    if arquivo_mapa.exists():
        from mover.simulacao.opendrive import ler_xodr

        texto_xodr = arquivo_mapa.read_text(encoding="utf-8")
        mapa = ler_xodr(texto_xodr, float(cfg["mapa"].get("passo_m", 0.5)), tuple(cfg["mapa"].get("tipos_faixa", ("driving",))))
    elif not (args.sem_carla or args.manter_mundo):
        raise SystemExit(f"Mapa não encontrado: {arquivo_mapa}")

    mundo: Mundo = MundoFalso() if args.sem_carla else MundoCarla(cfg, args.camera, args.manter_mundo)
    servidor = thread = None
    cliente: ClienteAgentes | None = None
    url: str | None = None
    linhas: list[dict[str, Any]] = []
    codigo = 0
    try:
        host, porta = cs.get("host", "127.0.0.1"), int(cs.get("porta", 8000))
        if args.iniciar_servidor:
            from mover.servidor.app import criar_app
            from mover.servidor.rodar_servidor import iniciar_em_thread

            servidor, thread = iniciar_em_thread(criar_app(cfg_ag, cfg), host, porta)
        url = args.servidor or f"http://{host}:{porta}"
        cliente = ClienteAgentes.conectar(url, float(cs.get("timeout_bloco_s", 300)))
        log.info("Painel (simulação, dashboard, log e chat): %s/", url)
        if isinstance(mundo, MundoCarla):
            mundo.conectar()
            if args.manter_mundo and texto_xodr is None:
                texto_xodr = mundo.opendrive_atual()

        for n, pasta_volta in enumerate(voltas, start=1):
            tel_x = pd.read_csv(pasta_volta / ARQUIVO_TELEMETRIA)
            meta_x = json.loads((pasta_volta / ARQUIVO_VOLTA).read_text(encoding="utf-8")) \
                if (pasta_volta / ARQUIVO_VOLTA).exists() else {}
            trajeto = trajeto_da_volta(tel_x)
            log.info("=== %s (%d/%d): %.0f s de plano, semente %s ===", pasta_volta.name, n, len(voltas),
                     trajeto.t[-1] - trajeto.t[0], meta_x.get("semente", "?"))
            if mapa is not None:
                from mover.interface.cena import montar_cena_poses

                cliente.enviar_cena(montar_cena_poses(mapa, poses_no_mapa(tel_x), origem=pasta_volta.name))
            if isinstance(mundo, MundoCarla) and not args.sem_camera_painel:
                mundo.ligar_camera_painel(cliente.postar_quadro)
            mundo.preparar(trajeto, texto_xodr, tel_x)
            sessao = cliente.abrir_sessao({"volta": pasta_volta.name, "injetar_eventos": args.injetar_eventos,
                                           "provedor": args.provedor, "modelo": args.modelo, "sem_ml": args.sem_ml,
                                           "fator_tempo": fator_tempo, "tempo_real": not args.sem_espera})
            inicio = datetime.now().astimezone()
            try:
                res = rodar_volta_y(trajeto, tel_x["speed_mps"].to_numpy(float), cliente, mundo, sessao,
                                    antecedencia_s=antecedencia_s,
                                    rampa_replanejamento_s=float(cvol.get("rampa_replanejamento_s", 1.5)),
                                    passo_s=passo_s, consulta_s=float(cvol.get("consulta_previsao_s", 0.5)),
                                    fator_tempo=fator_tempo, sem_espera=args.sem_espera,
                                    estado_a_cada=int(cr.get("estado_a_cada_ticks", 4)),
                                    marcar_problemas=bool(cr.get("marcar_problemas", True)), silencioso=args.silencioso,
                                    timeout_previsao_s=float(cs.get("timeout_bloco_s", 300)))
            finally:
                mundo.encerrar()  # primeiro o CARLA: em modo síncrono ele fica parado esperando o próximo tick
                fim = cliente.encerrar_sessao()
            if isinstance(mundo, MundoCarla):
                mundo.manter_mundo = True  # o mundo do campus já foi gerado na primeira volta
            if not res.tau:
                break
            tel_y = telemetria_executada(np.asarray(res.tau), tel_x, cfg_trat, inicio, passo_s)
            tel_y.to_csv(pasta_volta / ARQUIVO_TELEMETRIA_Y, index=False, encoding="utf-8", lineterminator="\n")
            comparacao = comparar(telemetria_executada(tel_x["sim_time"].to_numpy(float), tel_x, cfg_trat, inicio, passo_s),
                                  tel_y, cfg_ag)
            relatorio = {"volta": pasta_volta.name, "semente": meta_x.get("semente"), "sessao": sessao["id"],
                         "llm": sessao.get("llm"), "fator_tempo": fator_tempo, "completa": not res.interrompido,
                         "comparacao": comparacao, "execucao": res.para_json(),
                         "resumo_agentes": (fim or {}).get("resumo")}
            (pasta_volta / ARQUIVO_COMPARACAO).write_text(json.dumps(relatorio, ensure_ascii=False, indent=2),
                                                          encoding="utf-8")
            cx, cy = comparacao["x"], comparacao["y"]
            linhas.append({"volta": pasta_volta.name, "semente": meta_x.get("semente"), "completa": not res.interrompido,
                           "ajustes": len(res.ajustes), "ajustes_atrasados": sum(a["atrasado"] for a in res.ajustes),
                           "duracao_x_s": cx["duracao_s"], "duracao_y_s": cy["duracao_s"],
                           "eventos_x": cx["eventos"], "eventos_y": cy["eventos"],
                           "criticos_x": cx["criticos"], "criticos_y": cy["criticos"],
                           "jerk_max_x_mps3": cx["jerk_max_abs_mps3"], "jerk_max_y_mps3": cy["jerk_max_abs_mps3"],
                           "maior_espera_previsao_s": res.para_json()["maior_espera_previsao_s"]})
            log.info("%s: %d ajuste(s); eventos de jerk X %d -> Y %d (críticos %d -> %d); jerk máx %.1f -> %.1f m/s³; "
                     "+%.1f s de volta", pasta_volta.name, len(res.ajustes), cx["eventos"], cy["eventos"], cx["criticos"],
                     cy["criticos"], cx["jerk_max_abs_mps3"], cy["jerk_max_abs_mps3"], comparacao["acrescimo_tempo_s"])
            if res.interrompido:
                codigo = 130
                break
    except KeyboardInterrupt:
        log.warning("Interrompido (Ctrl+C).")
        codigo = 130
    except ErroServidor as erro:
        log.error("%s", erro)
        codigo = 1
    finally:
        mundo.encerrar()
        if linhas:
            destino = caminho("data/resultados_pesquisa") / f"voltas_agentes_{datetime.now():%Y%m%d-%H%M%S}.csv"
            destino.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(linhas).to_csv(destino, index=False, encoding="utf-8", lineterminator="\n")
            log.info("Resumo de %d volta(s) em %s", len(linhas), destino)
        if cliente is not None:
            cliente.fechar()
        if servidor is not None:
            from mover.servidor.rodar_servidor import parar_servidor

            if args.manter_servidor and codigo == 0:
                esperar_ctrl_c(thread, url)
            parar_servidor(servidor, thread)
    return codigo


if __name__ == "__main__":
    sys.exit(main())
