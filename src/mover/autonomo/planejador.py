"""Plano futuro do caminhão autônomo (X): o que ele vai fazer nos próximos segundos, decidido antes.

Um carro autônomo decide antes de agir. A cada bloco, a partir do estado medido agora na corrida
(posição, velocidade e aceleração no CARLA), o caminhão monta o plano dos próximos 10 s e o manda
à camada agêntica, antes de executá-lo:
- a velocidade desejada que ele vai pedir ao piloto automático em cada zona do plano de missão;
- a velocidade, a aceleração e o jerk que isso vai gerar (modelo do controle do Traffic Manager:
  o TM não expõe a aceleração que vai aplicar);
- o trajeto à frente: as manobras (no CARLA, as do próprio Traffic Manager, por get_all_actions;
  sem ele, pela curvatura da rota);
- as decisões bruscas desse plano (frenagens e arrancadas bruscas), com a zona que causa cada uma.

Tudo vem da corrida atual: nenhum dado de corridas anteriores entra no plano. A camada agêntica
pode alterar levemente o plano, e o caminhão executa o plano alterado (Y).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from mover.autonomo.consumo import eventos_jerk, jerk, suavizar
from mover.autonomo.governador import Governador, Intervencao
from mover.autonomo.missao import Instancia, PlanoMissao
from mover.autonomo.previsao import ModeloInterno, Trajetoria

NOMES_MANOBRA = {"Left": "esquerda", "Right": "direita", "Straight": "reto", "LaneFollow": "seguir a faixa",
                 "ChangeLaneLeft": "troca de faixa à esquerda", "ChangeLaneRight": "troca de faixa à direita"}


@dataclass
class PlanoFuturo:
    """O plano dos próximos segundos que o caminhão manda à camada agêntica (ainda não executado)."""

    bloco: int
    janela: tuple[float, float]
    estado: dict[str, float]                  # t, d, v, a medidos agora, na corrida
    pendentes: list[Instancia]                # elementos do plano no trecho, ainda sem decisão
    d_fim: float                              # até onde o plano vai (distância na corrida)
    trajetoria: Trajetoria                    # o plano do caminhão: t, d, v, a e velocidade desejada
    projecao: Trajetoria                      # onde ele estará no fim da janela (com as decisões já tomadas)
    decisoes_bruscas: list[dict[str, Any]] = field(default_factory=list)
    manobras: list[dict[str, Any]] = field(default_factory=list)
    fonte_manobras: str = "rota"              # traffic_manager | rota

    def resumo(self) -> dict[str, Any]:
        """O plano em JSON (para o log e a página): amostrado a 2 Hz, com as decisões e as manobras."""
        traj = self.trajetoria
        idx = np.arange(0, len(traj.t), 10)
        return {
            "t_ini": round(float(self.estado["t"]), 2),
            "d_ini": round(float(self.estado["d"]), 1),
            "d_fim": round(float(self.d_fim), 1),
            "t": np.round(traj.t[idx], 2).tolist(),
            "v_kmh": np.round(traj.v[idx] * 3.6, 1).tolist(),
            "a_mps2": np.round(traj.a[idx], 2).tolist(),
            "v_desejada_kmh": np.round(traj.v_desejada[idx] * 3.6, 1).tolist(),
            "decisoes_bruscas": list(self.decisoes_bruscas),
            "manobras": list(self.manobras),
            "fonte_manobras": self.fonte_manobras,
        }


# ---------------------------------------------------------------------- simulação do plano
def ate_distancia(modelo: ModeloInterno, gov: Governador, d0: float, v0: float, a0: float, t0: float,
                  d_alvo: float, max_s: float = 120.0) -> Trajetoria:
    """Simula até passar de d_alvo (ou o caminhão terminar a corrida), em pedaços de 10 s."""
    pedacos: list[Trajetoria] = []
    d, v, a, t = d0, v0, a0, t0
    for _ in range(max(1, int(math.ceil(max_s / 10.0)))):
        traj = modelo.simular(gov, d, v, a, t, 10.0)
        pedacos.append(traj)
        if not len(traj.t):
            break
        d, v, a, t = float(traj.d[-1]), float(traj.v[-1]), float(traj.a[-1]), float(traj.t[-1])
        if d >= d_alvo or (gov.terminou and v <= 0.01):
            break
    junta = Trajetoria(*(np.concatenate([getattr(p, campo) for p in pedacos]) for campo in
                         ("t", "d", "v", "a", "v_desejada", "parado")))
    fim = int(np.searchsorted(junta.d, d_alvo)) + 1
    return Trajetoria(junta.t[:fim], junta.d[:fim], junta.v[:fim], junta.a[:fim], junta.v_desejada[:fim], junta.parado[:fim])


def elemento_da_decisao(pendentes: list[Instancia], d: float, frenagem: bool) -> Instancia | None:
    """O elemento que provoca a decisão: o que vem à frente (frenagem) ou o que acabou (arrancada)."""
    if not pendentes:
        return None
    if frenagem:
        a_frente = [i for i in pendentes if i.d_ini >= d - 3.0]
        return min(a_frente, key=lambda i: i.d_ini - d) if a_frente else pendentes[-1]
    atras = [i for i in pendentes if (i.d_fim if not i.e_parada else i.d_ini) <= d + 3.0]
    return max(atras, key=lambda i: i.d_fim) if atras else pendentes[0]


def decisoes_bruscas(traj: Trajetoria, pendentes: list[Instancia], t0: float, modelo: ModeloInterno) -> list[dict[str, Any]]:
    """Frenagens e arrancadas bruscas de um plano, com o elemento que causa cada uma."""
    if len(traj.t) < 3:
        return []
    dt = modelo.dt
    a_suave = suavizar(traj.a, dt, modelo.limiares.fc_hz)
    j = jerk(a_suave, dt)
    saida = []
    for ev in eventos_jerk(traj.t, j, modelo.limiares):
        k = int(np.argmin(np.abs(traj.t - ev["t_pico"])))
        janela = (traj.t >= ev["t_ini"] - 0.7) & (traj.t <= ev["t_fim"] + 0.7)
        a_min, a_max = float(a_suave[janela].min()), float(a_suave[janela].max())
        frenagem = abs(a_min) >= abs(a_max)
        d_ev = float(traj.d[k])
        causa = elemento_da_decisao(pendentes, d_ev, frenagem)
        saida.append({
            "tipo": "frenagem_brusca" if frenagem else "arrancada_brusca",
            "t_s": round(float(traj.t[k]), 2),
            "em_s": round(float(traj.t[k]) - t0, 1),
            "d_m": round(d_ev, 1),
            "acel_mps2": round(a_min if frenagem else a_max, 1),
            "jerk_mps3": round(float(ev["jerk_mps3"]), 1),
            "nivel": ev["nivel"],
            "vel_kmh": round(float(traj.v[k]) * 3.6, 0),
            "elemento": causa.chave if causa else None,
            "nome_elemento": causa.nome if causa else None,
            "fonte": "plano",
        })
    return saida


# ---------------------------------------------------------------------- manobras
def _tempo_ate(traj: Trajetoria, d: float, t0: float) -> float | None:
    if len(traj.d) and traj.d[0] <= d <= traj.d[-1]:
        return round(float(np.interp(d, traj.d, traj.t)) - t0, 1)
    return None


def manobras_da_rota(plano: PlanoMissao, d0: float, d_fim: float, traj: Trajetoria, t0: float,
                     raio_max_m: float = 80.0, giro_min_graus: float = 20.0) -> list[dict[str, Any]]:
    """Curvas do trajeto à frente (onde o volante vai virar, para que lado e quanto), pela rota."""
    rota = plano.rota
    passo = 1.0
    d = np.arange(d0, max(d_fim, d0 + passo), passo)
    curvatura = np.array([rota.ponto(x).curvatura for x in d])
    em_curva = np.abs(curvatura) > 1.0 / raio_max_m
    saida = []
    i = 0
    while i < len(d):
        if not em_curva[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(d) and em_curva[j + 1] and np.sign(curvatura[j + 1]) == np.sign(curvatura[i]):
            j += 1
        giro = math.degrees(float(np.sum(curvatura[i:j + 1]) * passo))
        if abs(giro) >= giro_min_graus:
            saida.append({"tipo": "esquerda" if giro > 0 else "direita", "d_m": round(float(d[i]), 1),
                          "distancia_m": round(float(d[i] - d0), 0), "em_s": _tempo_ate(traj, float(d[i]), t0),
                          "angulo_graus": round(abs(giro), 0), "fonte": "rota"})
        i = j + 1
    return saida


def manobras_do_tm(plano: PlanoMissao, acoes: Iterable[tuple[str, float, float]], d0: float, d_fim: float,
                   traj: Trajetoria, t0: float) -> list[dict[str, Any]]:
    """Manobras que o Traffic Manager informa (get_all_actions), localizadas na rota pela distância."""
    rota = plano.rota
    L = rota.comprimento
    s0 = d0 % L
    saida = []
    for opcao, x, y in acoes:
        if opcao == "LaneFollow":
            continue
        s, lateral = rota.projetar(float(x), float(y))
        if abs(lateral) > 15.0:
            continue
        d = d0 + ((s - s0) % L)
        if d > d_fim:
            continue
        saida.append({"tipo": NOMES_MANOBRA.get(opcao, opcao), "d_m": round(d, 1), "distancia_m": round(d - d0, 0),
                      "em_s": _tempo_ate(traj, d, t0), "fonte": "traffic_manager"})
    saida.sort(key=lambda m: m["d_m"])
    unicas: list[dict[str, Any]] = []
    for m in saida:  # o TM repete a ação em vários waypoints seguidos: fica a primeira de cada trecho
        if not unicas or m["tipo"] != unicas[-1]["tipo"] or m["d_m"] - unicas[-1]["d_m"] > 15.0:
            unicas.append(m)
    return unicas


# ---------------------------------------------------------------------- planejador
class PlanejadorCaminhao:
    """O planejador do caminhão autônomo: monta, a partir do estado atual, o plano do próximo bloco."""

    def __init__(self, plano: PlanoMissao, modelo: ModeloInterno, n_voltas: int | None = None, margem_m: float = 20.0):
        self.plano = plano
        self.modelo = modelo
        self.n_voltas = n_voltas
        # elementos logo depois da janela também entram: a frenagem suave para eles começa dentro dela,
        # e o caminhão para ~1 m antes da chegada (a distância no fim da janela nunca a alcança)
        self.margem_m = float(margem_m)

    def instancias_pendentes(self, governador: Governador, d_ate: float, d_atual: float) -> list[Instancia]:
        candidatas = self.plano.instancias(max(0.0, d_atual - 60.0), d_ate + self.margem_m, self.n_voltas)
        pendentes = []
        for inst in candidatas:
            if governador.decidida(inst.chave) or governador.ja_passou(inst.chave):
                continue
            if inst.tipo == "partida":
                if d_atual > 5.0:  # a partida só se decide com o caminhão ainda na largada
                    continue
            elif not inst.e_parada and inst.d_fim <= d_atual:
                continue
            pendentes.append(inst)
        return pendentes

    def planejar(self, governador: Governador, estado: dict[str, float], bloco: int, janela: tuple[float, float],
                 acoes_tm: Iterable[tuple[str, float, float]] | None = None) -> PlanoFuturo | None:
        """O plano do caminhão para os elementos que caem na janela. None se não houver nada a decidir.

        O governador recebido não é alterado (as simulações usam clones).
        """
        t0, d0, v0, a0 = (float(estado[k]) for k in ("t", "d", "v", "a"))
        dt = self.modelo.dt
        # onde o caminhão vai estar no fim da janela, com as alterações já combinadas
        projecao = self.modelo.simular(governador.clonar(), d0, v0, a0, t0, max(janela[1] - t0, dt) + 1.0)
        d_ate = float(np.interp(janela[1], projecao.t, projecao.d)) if len(projecao.t) else d0
        pendentes = self.instancias_pendentes(governador, d_ate, d0)
        if not pendentes:
            return None
        d_fim = max(max(i.d_fim for i in pendentes) + 25.0, d_ate)
        # o plano para antes da frenagem do próximo elemento que fica para a decisão seguinte
        chaves = {i.chave for i in pendentes}
        seguintes = [i for i in self.plano.instancias(max(i.d_ini for i in pendentes), d_fim + 60.0, self.n_voltas)
                     if i.chave not in chaves and not governador.decidida(i.chave) and i.tipo != "partida"]
        if seguintes:
            d_fim = max(d_ate, min(d_fim, min(i.d_ini for i in seguintes) - 12.0))
        gov = governador.clonar()
        sem_alteracao = Intervencao.da_opcao("nenhuma")
        for inst in pendentes:
            gov.definir(inst.chave, sem_alteracao)
        trajetoria = ate_distancia(self.modelo, gov, d0, v0, a0, t0, d_fim)
        brutas = decisoes_bruscas(trajetoria, pendentes, t0, self.modelo)
        acoes = list(acoes_tm or [])
        manobras = manobras_do_tm(self.plano, acoes, d0, d_fim, trajetoria, t0) if acoes else []
        fonte = "traffic_manager" if acoes else "rota"
        if not acoes:
            manobras = manobras_da_rota(self.plano, d0, d_fim, trajetoria, t0)
        return PlanoFuturo(bloco=bloco, janela=janela, estado={"t": t0, "d": d0, "v": v0, "a": a0}, pendentes=pendentes,
                           d_fim=d_fim, trajetoria=trajetoria, projecao=projecao, decisoes_bruscas=brutas,
                           manobras=manobras, fonte_manobras=fonte)


__all__ = ["NOMES_MANOBRA", "PlanejadorCaminhao", "PlanoFuturo", "ate_distancia", "decisoes_bruscas",
           "elemento_da_decisao", "manobras_da_rota", "manobras_do_tm"]
