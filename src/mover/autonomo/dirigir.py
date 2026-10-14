"""Corrida ao vivo do caminhão autônomo no campus, com a camada agêntica intervindo em tempo real.

O que acontece:
1. O plano de missão (config/missao.yaml, tirado do mapa do CARLA) dá a rota da volta no sentido
   horário, as zonas de velocidade (lombadas, faixa de pedestre, curvas) e a parada na portaria.
2. O caminhão sai parado da largada e anda sozinho: no CARLA, com o piloto automático (Traffic
   Manager); sem o CARLA (--sem-carla), num simulador substituto com o mesmo controle.
3. A cada bloco de 10 s, o caminhão monta, a partir do estado atual da corrida, o plano do bloco
   seguinte (a velocidade e a aceleração que vai aplicar, as manobras e as decisões bruscas) e o
   manda aos agentes por HTTP (POST /agentes/plano) antes de executá-lo, como um caminhão real
   ligado a um servidor. As alterações voltam também por HTTP (GET /agentes/decisoes).
4. Os agentes, no servidor, alteram levemente esse plano quando ele tem decisões bruscas: o modelo
   de ML propõe quanto suavizar, o modelo do caminhão confere, a regra recomenda e o Supervisor
   (LLM local, com prazo) aprova ou troca. A alteração muda de verdade a velocidade desejada
   mandada ao piloto.
5. A página (http://127.0.0.1:8000/) mostra a simulação, os gráficos que se montam ao vivo, o
   log (alterações e o que foi medido), o consumo comparado com o plano sem agentes e o chat.
6. No fim, a corrida fica em data/autonomo/corridas/<id>/: telemetria.csv, decisoes.jsonl (com o
   plano que o caminhão mandou em cada bloco), resumo.json e, com --gravar, a gravação do CARLA
   para o vídeo (mover.video.gravar_video).

Uso, a partir da pasta src/ do repositório:
    python -m mover.autonomo.dirigir                     # CARLA aberto + painel no navegador
    python -m mover.autonomo.dirigir --sem-carla         # sem o simulador (substituto)
    python -m mover.autonomo.dirigir --voltas 2 --gravar # grava a corrida para o vídeo MP4
    python -m mover.autonomo.dirigir --sem-agentes       # o caminhão sozinho, para comparar
    python -m mover.autonomo.dirigir --sem-llm           # agentes sem o Supervisor (sem Ollama)
    python -m mover.autonomo.dirigir --sem-ml            # agentes só com as opções fixas de suavização
    python -m mover.autonomo.dirigir --provedor falso    # LLM falso (sem modelo instalado)
    python -m mover.autonomo.dirigir --servidor http://<ip>:8000   # agentes e painel em outra máquina
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # execução direta: python src/mover/autonomo/dirigir.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from mover.agentes.llm import PROVEDORES
from mover.autonomo.canal_agentes import CanalHttp, ErroAgentes
from mover.autonomo.cliente import ClientePainel, ErroServidor, servidor_no_ar
from mover.autonomo.conducao import Conducao, FimDoSimulador, OpcoesConducao, ResultadoConducao, metricas_da_corrida
from mover.autonomo.montagem import Montagem, montar
from mover.config import caminho, carregar_yaml

log = logging.getLogger("mover.autonomo")


def abrir_agentes(url: str, m: Montagem, voltas: int, usar_llm: bool, usar_ml: bool, provedor: str | None,
                  modelo: str | None) -> tuple[CanalHttp, dict[str, Any]]:
    """Conecta o caminhão ao servidor dos agentes e abre a sessão (o servidor monta o LLM e o ML)."""
    canal = CanalHttp.conectar(url)
    log.info("Abrindo a sessão dos agentes em %s (na primeira vez o modelo de ML é treinado, ~1-2 min)...", url)
    try:
        info = canal.abrir({"voltas": voltas, "usar_llm": usar_llm, "usar_ml": usar_ml, "provedor": provedor,
                            "modelo": modelo, "comprimento_volta_m": m.plano.comprimento})
    except ErroAgentes:
        canal.http.close()
        raise
    log.info("Agentes por HTTP: LLM %s, especialista de ML %s.", info.get("llm") or "- (só a regra)", info.get("ml") or "-")
    return canal, info


def gravar_corrida(pasta: Path, resultado: ResultadoConducao, metricas: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """telemetria.csv, decisoes.jsonl e resumo.json da corrida."""
    pasta.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(resultado.linhas).to_csv(pasta / "telemetria.csv", index=False)
    with (pasta / "decisoes.jsonl").open("w", encoding="utf-8") as f:
        for entrada in resultado.decisoes:
            f.write(json.dumps(entrada, ensure_ascii=False) + "\n")
    resumo = {
        "modo": resultado.modo, "simulador": resultado.simulador, "voltas": resultado.voltas,
        "motivo_fim": resultado.motivo_fim, "duracao_real_s": resultado.duracao_real_s,
        "atrasos_relogio": resultado.atrasos_relogio, "decisoes": len(resultado.decisoes),
        "decisoes_llm": sum(d.get("texto_origem") == "llm" for d in resultado.decisoes),
        "aplicacoes": resultado.aplicacoes, **extra,
        "total": metricas.get("total", {}), "por_volta": metricas.get("voltas", []),
    }
    (pasta / "resumo.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return resumo


def texto_do_resumo(resumo: dict[str, Any], preco: float) -> str:
    t = resumo.get("total") or {}
    if not t:
        return "Corrida sem telemetria."
    litros = float(t.get("combustivel_l", 0.0))
    return (f"Corrida {resumo['modo']} ({resumo['simulador']}, {resumo['voltas']} volta(s), fim: {resumo['motivo_fim']}): "
            f"{t.get('tempo_s', 0):.1f} s, {t.get('distancia_m', 0):.0f} m, {1000 * litros:.0f} mL "
            f"(R$ {litros * preco:.2f}), eventos bruscos {t.get('eventos_atencao', 0)} de atenção e "
            f"{t.get('eventos_criticos', 0)} críticos; {resumo['decisoes']} decisões ({resumo['decisoes_llm']} do LLM).")


def preparar_simulador(args: argparse.Namespace, m: Montagem, cfg_sim: dict[str, Any], voltas: int):
    if args.sem_carla:
        return m.simulador_substituto(), None
    from mover.autonomo.carla_tm import SimuladorCarla

    arquivo_mapa = caminho(args.mapa or m.plano.mapa or cfg_sim.get("mapa", {}).get("arquivo", "data/mapas/mapa_final.xodr"))
    texto_xodr = None
    if not args.manter_mundo:
        if not arquivo_mapa.exists():
            raise SystemExit(f"Mapa não encontrado: {arquivo_mapa}")
        texto_xodr = arquivo_mapa.read_text(encoding="utf-8")
    sim = SimuladorCarla(m.plano.rota, cfg_sim, m.cfg.get("carla"), m.dt, texto_xodr, sem_render=args.sem_render,
                         manter_mundo=args.manter_mundo, voltas=voltas, camera=args.camera)
    return sim, arquivo_mapa


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Corrida ao vivo do caminhão autônomo com a camada agêntica.")
    parser.add_argument("--config", default="config/autonomo.yaml", help="YAML da corrida autônoma")
    parser.add_argument("--config-simulacao", default="config/simulacao.yaml", help="YAML do CARLA e do servidor")
    parser.add_argument("--config-agentes", default="config/agentes.yaml", help="YAML da camada agêntica (LLM, logs, chat)")
    parser.add_argument("--missao", help="plano de missão (padrão: YAML)")
    parser.add_argument("--mapa", help="arquivo .xodr (padrão: o do plano de missão)")
    parser.add_argument("--voltas", type=int, help="número de voltas (padrão: YAML)")
    parser.add_argument("--sem-carla", action="store_true", help="usa o simulador substituto em vez do CARLA")
    parser.add_argument("--sem-agentes", action="store_true", help="o caminhão sozinho (plano sem alteração), para comparar")
    parser.add_argument("--sem-llm", action="store_true", help="os agentes decidem só pela regra")
    parser.add_argument("--provedor", choices=PROVEDORES, help="provedor do LLM (padrão: .env ou config/agentes.yaml)")
    parser.add_argument("--modelo", help="modelo no provedor (padrão: gemma2:2b no Ollama)")
    parser.add_argument("--fator-tempo", type=float, default=1.0, help="2 = duas vezes mais rápido que o real")
    parser.add_argument("--sem-espera", action="store_true",
                        help="não segue o relógio (o mais rápido possível; a simulação espera cada decisão dos agentes)")
    parser.add_argument("--gravar", action="store_true", help="grava a corrida no CARLA (para o vídeo MP4)")
    parser.add_argument("--sem-render", action="store_true", help="CARLA sem desenhar a cena (mais rápido; sem câmeras)")
    parser.add_argument("--manter-mundo", action="store_true", help="usa o mapa já aberto no CARLA")
    parser.add_argument("--camera", choices=("perseguicao", "cima", "livre"), help="câmera da janela do CARLA")
    parser.add_argument("--sem-camera-painel", action="store_true", help="o painel 1 mostra o mapa 2D")
    parser.add_argument("--servidor", help="URL de um servidor do painel e dos agentes já no ar (padrão: sobe um neste processo)")
    parser.add_argument("--sem-painel", action="store_true",
                        help="não usa a página (os agentes seguem por HTTP, num servidor só desta corrida)")
    parser.add_argument("--sem-navegador", action="store_true", help="não abre o painel no navegador")
    parser.add_argument("--sair-no-fim", action="store_true", help="encerra o servidor assim que a corrida terminar")
    parser.add_argument("--sem-ml", action="store_true", help="sem o modelo de ML de suavização (só as opções fixas)")
    args = parser.parse_args(argv)

    from mover.servidor.rodar_servidor import configurar_logs, iniciar_em_thread, parar_servidor, porta_do_servidor

    configurar_logs()
    cfg = carregar_yaml(args.config)
    cfg_sim = carregar_yaml(args.config_simulacao) if caminho(args.config_simulacao).exists() else {}
    cfg_agentes = carregar_yaml(args.config_agentes)
    m = montar(cfg, args.missao)
    voltas = int(args.voltas or (cfg.get("missao") or {}).get("voltas", 1))
    modo = "bruto" if args.sem_agentes else "agentes"
    log.info("Plano de missão: rota de %.0f m (%s), %d zonas, %d parada(s); %d volta(s), modo %s.",
             m.plano.comprimento, m.plano.sentido or "?", len(m.plano.zonas), len(m.plano.paradas), voltas, modo)

    servidor = thread = None
    cliente: ClientePainel | None = None
    canal: CanalHttp | None = None
    info_agentes: dict[str, Any] = {}
    sim = None
    codigo = 0
    url = None
    sessao_aberta = False
    try:
        if not args.sem_painel or modo == "agentes":
            cs = cfg_sim.get("servidor", {})
            host, porta = cs.get("host", "127.0.0.1"), int(cs.get("porta", 8000))
            url = args.servidor or f"http://{host}:{porta}"
            if not args.servidor and (args.sem_painel or not servidor_no_ar(url)):
                from mover.servidor.app import criar_app

                if args.sem_painel:  # só os agentes: numa porta livre, sem mexer num painel que esteja no ar
                    host, porta = "127.0.0.1", 0
                servidor, thread = iniciar_em_thread(criar_app(cfg_agentes, cfg_sim, cfg_autonomo=cfg, montagem=m),
                                                     host, porta)
                url = f"http://{host}:{porta_do_servidor(servidor, porta)}"
        if not args.sem_painel:
            cliente = ClientePainel.conectar(url)
            log.info("Painel (simulação, gráficos, log e chat): %s/", url)
            if servidor is not None and not args.sem_navegador:
                webbrowser.open(f"{url}/")
        if modo == "agentes":
            canal, info_agentes = abrir_agentes(url, m, voltas, not args.sem_llm, not args.sem_ml, args.provedor, args.modelo)
        llm_nome, ml_nome = info_agentes.get("llm"), info_agentes.get("ml")

        sim, arquivo_mapa = preparar_simulador(args, m, cfg_sim, voltas)
        if cliente is not None:
            from mover.interface.cena import montar_cena
            from mover.simulacao.opendrive import ler_xodr

            mapa = None
            destino = arquivo_mapa or caminho(args.mapa or m.plano.mapa or cfg_sim.get("mapa", {}).get("arquivo", ""))
            if destino.is_file():
                cm = cfg_sim.get("mapa", {})
                mapa = ler_xodr(destino, float(cm.get("passo_m", 0.5)),
                                tuple(cm.get("tipos_faixa", ("driving", "restricted", "bidirectional", "parking"))))
            cliente.enviar_cena(montar_cena(m.plano, mapa))
            if not args.sem_carla and not args.sem_camera_painel:
                sim.ligar_camera_painel(cliente.postar_quadro)
        sim.iniciar()
        if args.sem_render and not args.sem_carla:
            log.info("--sem-render: o CARLA não desenha a cena (a janela fica preta).")

        id_corrida = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{modo}"
        pasta = caminho((cfg.get("corridas") or {}).get("pasta", "data/autonomo/corridas")) / id_corrida
        gravacao = None
        if args.gravar:
            if args.sem_carla:
                log.warning("--gravar só vale com o CARLA; ignorado.")
            else:
                gravacao = sim.gravar(pasta / "gravacao_carla.log")

        linha_base = m.linha_de_base(voltas) if modo == "agentes" else None  # o plano sem agentes, previsto
        prevista = m.simular_corrida(modo, voltas, "economica")
        if cliente is not None:
            info = cliente.abrir_sessao({
                "modo": modo, "voltas": voltas, "simulador": sim.nome, "provedor": args.provedor, "modelo": args.modelo,
                "llm_decisoes": (llm_nome or "regra") if modo == "agentes" else None,
                "fator_tempo": args.fator_tempo, "tempo_real": not args.sem_espera,
                "duracao_bloco_s": m.duracao_bloco_s, "comprimento_volta_m": m.plano.comprimento,
                "duracao_prevista_s": round(float(prevista.t[-1]), 1), "linha_de_base": linha_base,
                "combustivel": m.consumo.combustivel, "preco_r_l": m.consumo.preco_r_l, "co2_kg_l": m.consumo.co2_kg_l,
            })
            sessao_aberta = True
            log.info("Sessão %s aberta. Partindo!", info["id"])

        governador = m.governador(modo, voltas)
        # sem o relógio, a simulação correria na frente dos agentes: então ela espera cada decisão
        opcoes = OpcoesConducao(voltas=voltas, usar_llm=llm_nome is not None, assincrona=not args.sem_espera,
                                fator_tempo=args.fator_tempo,
                                sem_espera=args.sem_espera, duracao_bloco_s=m.duracao_bloco_s)
        conducao = Conducao(m.plano, sim, governador, canal, opcoes, m.caminhao, m.consumo,
                            ao_lote=cliente.enviar_telemetria if cliente else None,
                            ao_decisao=cliente.enviar_decisao if cliente else None,
                            imprimir=lambda texto: print(texto, flush=True),
                            planejador=m.planejador(voltas) if canal is not None else None)
        resultado = conducao.rodar()
        metricas = metricas_da_corrida(resultado, m.plano.comprimento, m.caminhao, m.consumo, m.limiares)
        extra = {"id": id_corrida, "gravacao_carla": gravacao, "mapa": str(arquivo_mapa) if arquivo_mapa else None,
                 "llm_decisoes": llm_nome, "linha_de_base": (linha_base or {}).get("total"),
                 "fonte_linha_de_base": (linha_base or {}).get("fonte"), "suavizador_ml": ml_nome,
                 "agentes_http": url if canal is not None else None,
                 "requisicoes_agentes": canal.requisicoes if canal is not None else 0,
                 "topologia_carla": getattr(sim, "topologia", None),
                 "ajustes_mapa_carla": getattr(sim, "ajustes_mapa", None),
                 "preco_r_l": m.consumo.preco_r_l, "combustivel": m.consumo.combustivel}
        if hasattr(sim, "parar_gravacao"):
            sim.parar_gravacao()
        resumo = gravar_corrida(pasta, resultado, metricas, extra)
        if linha_base is not None:  # para o HUD do vídeo comparar com o plano sem agentes na mesma distância
            (pasta / "linha_de_base.json").write_text(json.dumps(linha_base, ensure_ascii=False), encoding="utf-8")
        print(texto_do_resumo(resumo, m.consumo.preco_r_l), flush=True)
        log.info("Corrida gravada em %s", pasta)
        if gravacao:
            log.info("Vídeo: python -m mover.video.gravar_video --corrida %s", pasta)
        if cliente is not None:
            fim = cliente.encerrar_sessao(resumo)
            sessao_aberta = False
            if fim:
                log.info("Log da sessão: %s", fim["sessao"]["arquivo"])
        if resultado.motivo_fim == "interrompido":
            codigo = 130
        elif not resultado.completa:
            codigo = 2
    except KeyboardInterrupt:
        log.warning("Interrompido (Ctrl+C).")
        codigo = 130
    except (ErroServidor, ErroAgentes, FimDoSimulador) as erro:
        log.error("%s", erro)
        codigo = 1
    finally:
        if sim is not None:
            sim.encerrar()  # primeiro o CARLA: em modo síncrono ele fica parado esperando o próximo tick
        if canal is not None:
            canal.encerrar()
        if cliente is not None:
            if sessao_aberta:
                cliente.encerrar_sessao(None)
            cliente.fechar()
        if servidor is not None:
            if not args.sair_no_fim and not args.sem_painel and codigo in (0, 2):
                esperar_ctrl_c(thread, url)
            parar_servidor(servidor, thread)
    return codigo


def esperar_ctrl_c(thread: Any, url: str | None) -> None:
    """Deixa o servidor no ar até o Ctrl+C: a página e o chat continuam funcionando."""
    log.info("Corrida concluída. Painel e chat seguem em %s/ (Ctrl+C para sair).", url)
    try:
        while thread.is_alive():
            thread.join(timeout=0.5)
    except KeyboardInterrupt:
        log.info("Encerrando o servidor...")


if __name__ == "__main__":
    sys.exit(main())
