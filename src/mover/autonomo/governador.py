"""Governador: a velocidade desejada mandada ao piloto automático a cada tick (20 Hz).

Dois modos:
- "bruto": o plano de missão como o piloto automático o seguiria sozinho. A velocidade desejada
  cai em degrau perto de cada zona ou parada (frenagem tardia, a `desacel_bruta_mps2`) e volta ao
  cruzeiro em degrau quando a zona acaba. O controle do piloto satura: frenagens e arrancadas
  bruscas.
- "agentes": para cada elemento do plano com uma intervenção decidida pela camada agêntica, a
  velocidade desejada segue um envelope de frenagem antecipada (desaceleração da opção escolhida)
  e uma retomada com aceleração limitada, sempre com jerk limitado. Elementos sem intervenção
  (opção "nenhuma" ou ainda sem decisão) continuam no comportamento bruto.

No CARLA a saída vira TrafficManager.set_desired_speed(caminhao, km/h) a cada tick.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any

from mover.autonomo.missao import Instancia, PlanoMissao
from mover.autonomo.veiculo import distancia_para_frear

# opção -> (desaceleração alvo, aceleração da retomada em m/s², teto de velocidade do trecho seguinte em km/h)
OPCOES_PADRAO: dict[str, tuple[float, float, float | None]] = {
    "economica": (0.6, 0.5, 22.0),
    "suave": (0.8, 0.6, None),
    "moderada": (1.2, 0.9, None),
    "firme": (1.8, 1.2, None),
}
NOMES_OPCAO = {"economica": "econômica", "suave": "suave", "moderada": "moderada", "firme": "firme",
               "ml": "ajuste do ML", "nenhuma": "sem alteração"}


@dataclass(frozen=True)
class Intervencao:
    opcao: str                      # economica | suave | moderada | firme | ml | nenhuma
    desacel_mps2: float = 0.0
    acel_mps2: float = 0.0
    teto_kmh: float | None = None   # depois do elemento, não passa desta velocidade até o próximo
    origem: str = "regra"           # llm | regra (| ml no treino do modelo de suavização)

    @property
    def ativa(self) -> bool:
        return self.opcao != "nenhuma"

    @classmethod
    def da_opcao(cls, opcao: str, opcoes: dict[str, Any] | None = None, origem: str = "regra") -> "Intervencao":
        if opcao == "nenhuma":
            return cls("nenhuma", 0.0, 0.0, None, origem)
        valores = tuple((opcoes or OPCOES_PADRAO)[opcao])
        desacel, acel = float(valores[0]), float(valores[1])
        teto = float(valores[2]) if len(valores) > 2 and valores[2] is not None else None
        return cls(opcao, desacel, acel, teto, origem)


@dataclass
class ConfigGovernador:
    desacel_bruta_mps2: float = 4.5     # o piloto sozinho só começa a frear a esta desaceleração
    margem_bruta_m: float = 1.0
    margem_suave_m: float = 2.0         # o envelope termina antes da zona
    folga_parada_m: float = 0.5
    raio_parada_m: float = 8.0          # parado a até esta distância do ponto conta como parada feita
    passou_parada_m: float = 10.0
    jerk_max_mps3: float = 1.6          # limite de jerk da velocidade desejada suavizada
    ganho_referencia: float = 1.2       # 1/s
    fator_envelope: float = 0.85        # o envelope usa 85 % da desaceleração (folga para a rampa de jerk)
    avanco_s: float = 0.35              # adianta a referência: compensa o atraso do controle do piloto
    tau_avanco_s: float = 0.6           # o avanço é filtrado para não virar degrau
    ganho_parada: float = 0.8           # 1/s: no fim de uma parada a desaceleração some com a velocidade
    v_parar_mps: float = 0.15
    abaixo_do_limite_mps: float = 0.3   # o envelope mira ~1 km/h abaixo do limite da zona
    ressincronizar_mps: float = 1.5
    desacel_recuperacao_mps2: float = 2.5
    horizonte_m: float = 250.0
    olhar_atras_m: float = 80.0

    @classmethod
    def de_dict(cls, dados: dict[str, Any] | None) -> "ConfigGovernador":
        nomes = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in (dados or {}).items() if k in nomes})


def _envelope(v_alvo: float, distancia: float, desacel: float) -> float:
    """Maior velocidade da qual ainda dá para chegar a v_alvo em `distancia` com `desacel`."""
    return math.sqrt(v_alvo * v_alvo + 2.0 * desacel * max(0.0, distancia))


@dataclass(frozen=True)
class _Limite:
    v: float
    chave: str | None
    modo: str                      # cruzeiro | bruto | suave
    intervencao: Intervencao | None
    parada: bool = False


class Governador:
    def __init__(self, plano: PlanoMissao, cfg: ConfigGovernador | None = None, modo: str = "bruto",
                 n_voltas: int | None = None, opcoes: dict[str, tuple[float, float]] | None = None):
        if modo not in ("bruto", "agentes"):
            raise ValueError(f"modo desconhecido: {modo}")
        self.plano = plano
        self.cfg = cfg or ConfigGovernador()
        self.modo = modo
        self.n_voltas = n_voltas
        self.opcoes = opcoes or OPCOES_PADRAO
        self.intervencoes: dict[str, Intervencao] = {}
        self._disparadas: set[str] = set()
        self._feitas: set[str] = set()
        self._parada: tuple[str, float] | None = None
        self._retomada: Intervencao | None = None
        self._v_ref = 0.0
        self._a_ref = 0.0
        self._a_avanco = 0.0
        self.terminou = False
        self.ultimo: dict[str, Any] = {}

    # ------------------------------------------------------------------ decisões
    def definir(self, chave: str, intervencao: Intervencao) -> None:
        self.intervencoes[chave] = intervencao

    def decidida(self, chave: str) -> bool:
        return chave in self.intervencoes

    def ja_passou(self, chave: str) -> bool:
        return chave in self._feitas

    def pode_intervir(self, chave: str) -> bool:
        """Uma intervenção para o elemento ainda muda algo: ele não passou, o caminhão não está parado
        nele e o piloto ainda não começou a frenagem bruta (uma decisão atrasada não interrompe a frenagem)."""
        parado_nele = self._parada is not None and self._parada[0] == chave
        return chave not in self._feitas and chave not in self._disparadas and not parado_nele

    def _intervencao_ativa(self, chave: str) -> Intervencao | None:
        if self.modo != "agentes":
            return None
        interv = self.intervencoes.get(chave)
        return interv if interv is not None and interv.ativa else None

    def clonar(self) -> "Governador":
        """Cópia do estado (o plano é compartilhado): usada nas previsões."""
        novo = Governador.__new__(Governador)
        novo.__dict__.update(self.__dict__)
        novo.intervencoes = dict(self.intervencoes)
        novo._disparadas = set(self._disparadas)
        novo._feitas = set(self._feitas)
        novo.ultimo = dict(self.ultimo)
        return novo

    # ------------------------------------------------------------------ tick
    def velocidade_desejada(self, d: float, v: float, a: float, dt: float) -> float:
        """Velocidade desejada (m/s) para o próximo tick; detalhes em self.ultimo."""
        cfg = self.cfg
        if self.terminou:
            return self._parado("chegada")
        if self._parada is not None:
            chave, resta = self._parada
            resta -= dt
            if resta > 0.0 or v > 0.3:
                self._parada = (chave, resta)
                return self._parado(chave)
            self._parada = None
            self._feitas.add(chave)
            self._retomada = self._intervencao_ativa(chave)
            if chave.startswith("chegada"):
                self.terminou = True
                return self._parado(chave)

        v_cruzeiro = self.plano.cruzeiro_kmh / 3.6
        if self._retomada is not None and self._retomada.teto_kmh:
            v_cruzeiro = min(v_cruzeiro, self._retomada.teto_kmh / 3.6)
        limites = [_Limite(v_cruzeiro, None, "cruzeiro", None)]
        for inst in self._instancias(d):
            if inst.chave in self._feitas:
                continue
            interv = self._intervencao_ativa(inst.chave)
            if inst.tipo == "partida":
                self._feitas.add(inst.chave)
                self._retomada = interv
                continue
            if inst.e_parada:
                dist = inst.d_ini - d
                if dist < -cfg.passou_parada_m:
                    self._feitas.add(inst.chave)
                    if inst.tipo == "chegada":  # passou da chegada sem parar (piloto freou menos): encerra assim mesmo
                        self.terminou = True
                        return self._parado(inst.chave)
                    continue
                if abs(dist) <= cfg.raio_parada_m and v <= 0.3:
                    if inst.duracao_s <= 0.0 and inst.tipo != "chegada":
                        self._feitas.add(inst.chave)
                        self._retomada = interv
                        continue
                    self._parada = (inst.chave, inst.duracao_s)
                    return self._parado(inst.chave)
                limite = self._limite(inst, 0.0, dist, v, interv, cfg.folga_parada_m)
            else:
                if d >= inst.d_fim:
                    self._feitas.add(inst.chave)
                    self._retomada = interv
                    continue
                limite = self._limite(inst, inst.v_kmh / 3.6, inst.d_ini - d, v, interv, cfg.margem_suave_m)
            if limite is not None:
                limites.append(limite)

        lim = min(limites, key=lambda item: item.v)
        bruto = lim.modo == "bruto" or (lim.modo == "cruzeiro" and self._retomada is None)
        if bruto:
            # passo bruto: o alvo vai direto ao piloto; a referência acompanha o caminhão
            self._v_ref, self._a_ref = v, max(-3.0, min(3.0, a))
            self._a_avanco = 0.0
            v_des = lim.v
        else:
            v_des = self._referencia_suave(lim, v, dt)
        self.ultimo = {"v_alvo_kmh": round(lim.v * 3.6, 2), "v_desejada_kmh": round(v_des * 3.6, 2),
                       "limitado_por": lim.chave, "modo_tick": "bruto" if bruto else "suave", "parado": False}
        return v_des

    # ------------------------------------------------------------------ auxiliares
    def _referencia_suave(self, lim: _Limite, v: float, dt: float) -> float:
        """Referência de velocidade com aceleração limitada e jerk limitado que persegue o alvo."""
        cfg = self.cfg
        interv, retomada = lim.intervencao, self._retomada
        desacel = interv.desacel_mps2 if interv is not None else (retomada.desacel_mps2 if retomada else 1.2)
        acel = retomada.acel_mps2 if retomada is not None else (interv.acel_mps2 if interv else 0.9)
        if abs(self._v_ref - v) > cfg.ressincronizar_mps:
            self._v_ref = v
        a_desejada = max(-desacel, min(acel, cfg.ganho_referencia * (lim.v - self._v_ref)))
        if self._v_ref > lim.v + 0.5:  # atrasado em relação ao envelope: freia um pouco mais
            a_desejada = max(-cfg.desacel_recuperacao_mps2, cfg.ganho_referencia * (lim.v - self._v_ref))
        if lim.parada and lim.v < 0.5:
            # fim de parada: a desaceleração some junto com a velocidade (sem tranco ao parar)
            a_desejada = max(a_desejada, -cfg.ganho_parada * self._v_ref)
        if v < 0.5 and self._v_ref - v > 0.25 and a_desejada > 0.0:
            # saindo da imobilidade: o freio ainda está soltando; não acumula pedido de aceleração
            self._v_ref = v + 0.25
            a_desejada = min(a_desejada, 0.2)
        passo = cfg.jerk_max_mps3 * dt
        self._a_ref += max(-passo, min(passo, a_desejada - self._a_ref))
        self._v_ref = max(0.0, self._v_ref + self._a_ref * dt)
        if lim.parada and lim.v < 0.5 and self._v_ref < cfg.v_parar_mps and abs(self._a_ref) < 0.15:
            self._v_ref, self._a_ref = 0.0, 0.0
        if self._v_ref <= 0.0 and self._a_ref < 0.0:
            self._a_ref = 0.0
        # avanço filtrado: compensa o atraso do controle do piloto sem dar degraus
        self._a_avanco += (self._a_ref - self._a_avanco) * min(1.0, dt / cfg.tau_avanco_s)
        return max(0.0, self._v_ref + cfg.avanco_s * self._a_avanco)

    def _instancias(self, d: float) -> list[Instancia]:
        cfg = self.cfg
        return self.plano.instancias(max(0.0, d - cfg.olhar_atras_m), d + cfg.horizonte_m, self.n_voltas)

    def _limite(self, inst: Instancia, v_alvo: float, dist: float, v: float, interv: Intervencao | None,
                folga: float) -> _Limite | None:
        cfg = self.cfg
        if interv is None:  # comportamento bruto: frenagem tardia, travada até o fim do elemento
            gatilho = distancia_para_frear(v, v_alvo, cfg.desacel_bruta_mps2) + cfg.margem_bruta_m
            if inst.chave in self._disparadas or dist <= gatilho:
                self._disparadas.add(inst.chave)
                return _Limite(v_alvo, inst.chave, "bruto", None, inst.e_parada)
            return None
        if dist <= 0.0:
            return _Limite(v_alvo, inst.chave, "suave", interv, inst.e_parada)
        # mira um pouco abaixo do limite e desconta a distância gasta nas rampas de jerk (curva em S)
        alvo = max(0.0, v_alvo - cfg.abaixo_do_limite_mps) if v_alvo > 0 else 0.0
        desacel = cfg.fator_envelope * interv.desacel_mps2
        rampa_jerk = (v + alvo) * interv.desacel_mps2 / (2.0 * cfg.jerk_max_mps3)
        v_env = _envelope(alvo, dist - folga - rampa_jerk, desacel)
        return _Limite(v_env, inst.chave, "suave", interv, inst.e_parada)

    def _parado(self, chave: str) -> float:
        self._v_ref, self._a_ref, self._a_avanco = 0.0, 0.0, 0.0
        self.ultimo = {"v_alvo_kmh": 0.0, "v_desejada_kmh": 0.0, "limitado_por": chave, "modo_tick": "parado",
                       "parado": True}
        return 0.0
