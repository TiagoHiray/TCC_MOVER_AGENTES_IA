"""Plano de missão da volta: velocidade de cruzeiro, zonas de velocidade e paradas.

O piloto automático segue o plano "em degrau": a velocidade desejada só cai quando ele já está
perto da zona (frenagem tardia e forte) e volta ao cruzeiro assim que a zona acaba (arrancada
forte). Essas são as decisões que a camada agêntica lê 10 s antes e suaviza.

Cada elemento (zona ou parada) se repete a cada volta. Uma *instância* é o elemento numa volta
específica, identificada por "<id>@<volta>" e localizada pela distância percorrida d.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from mover.autonomo.rota import Rota

TIPOS_ZONA = ("lombada", "faixa", "curva", "portaria", "zona")


def _puro(valor: Any) -> Any:
    """Tipos do numpy -> tipos do Python (o YAML seguro não grava np.float64)."""
    if isinstance(valor, dict):
        return {k: _puro(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_puro(v) for v in valor]
    if hasattr(valor, "item") and not isinstance(valor, (str, bytes)):
        return valor.item()
    return valor


@dataclass(frozen=True)
class Zona:
    id: str
    tipo: str            # lombada | faixa | curva | portaria | zona
    s_ini: float         # início da zona dentro da volta (m)
    s_fim: float         # fim (m); pode ser menor que s_ini se a zona cruzar a largada
    v_max_kmh: float
    nome: str = ""
    origem: str = ""

    @property
    def comprimento(self) -> float:
        return self.s_fim - self.s_ini if self.s_fim >= self.s_ini else math.nan


@dataclass(frozen=True)
class Parada:
    id: str
    s: float             # ponto de parada dentro da volta (m)
    duracao_s: float
    nome: str = ""
    origem: str = ""


@dataclass(frozen=True)
class Instancia:
    """Um elemento do plano numa volta: zona (d_ini..d_fim) ou parada (d_ini == d_fim)."""

    chave: str           # "<id>@<volta>"
    tipo: str            # tipo da zona ou "parada" / "chegada"
    d_ini: float
    d_fim: float
    v_kmh: float         # limite da zona; 0 para paradas
    duracao_s: float     # tempo parado (paradas)
    nome: str
    volta: int

    @property
    def e_parada(self) -> bool:
        return self.v_kmh <= 0.0


@dataclass
class PlanoMissao:
    rota: Rota
    cruzeiro_kmh: float = 30.0
    zonas: list[Zona] = field(default_factory=list)
    paradas: list[Parada] = field(default_factory=list)
    mapa: str = ""
    sentido: str = ""
    observacoes: list[str] = field(default_factory=list)

    @property
    def comprimento(self) -> float:
        return self.rota.comprimento

    # ------------------------------------------------------------------ consultas
    def limite_kmh(self, d: float) -> float:
        """Limite de velocidade do plano no ponto d (cruzeiro ou a menor zona que cobre o ponto)."""
        s = d % self.comprimento
        limite = self.cruzeiro_kmh
        for z in self.zonas:
            if self._dentro(z, s):
                limite = min(limite, z.v_max_kmh)
        return limite

    def _dentro(self, z: Zona, s: float) -> bool:
        if z.s_fim >= z.s_ini:
            return z.s_ini - 1e-9 <= s < z.s_fim
        return s >= z.s_ini - 1e-9 or s < z.s_fim  # cruza a largada

    def instancias(self, d_ini: float, d_fim: float, n_voltas: int | None = None) -> list[Instancia]:
        """Instâncias cujo início cai em [d_ini, d_fim), em ordem de distância.

        Com `n_voltas`, a corrida termina com uma parada de chegada na largada da volta seguinte à
        última; instâncias depois dela são descartadas.
        """
        L = self.comprimento
        saida: list[Instancia] = []
        v0 = max(0, int(math.floor(d_ini / L)) - 1)
        v1 = int(math.floor(d_fim / L)) + 1
        for volta in range(v0, v1 + 1):
            base = volta * L
            for z in self.zonas:
                ini = base + z.s_ini
                fim = base + (z.s_fim if z.s_fim >= z.s_ini else z.s_fim + L)
                saida.append(Instancia(f"{z.id}@{volta}", z.tipo, ini, fim, z.v_max_kmh, 0.0, z.nome or z.tipo, volta))
            for p in self.paradas:
                d = base + p.s
                saida.append(Instancia(f"{p.id}@{volta}", "parada", d, d, 0.0, p.duracao_s, p.nome or "parada", volta))
        if n_voltas is not None:
            chegada = n_voltas * L
            saida = [i for i in saida if i.d_ini < chegada - 1e-6]
            saida.append(Instancia(f"chegada@{n_voltas}", "chegada", chegada, chegada, 0.0, 0.0, "chegada", n_voltas))
        # a largada parada também é uma decisão: com que aceleração sair
        saida.append(Instancia("partida@0", "partida", 0.0, 0.0, self.cruzeiro_kmh, 0.0, "partida", 0))
        saida = [i for i in saida if d_ini - 1e-9 <= i.d_ini < d_fim and i.d_ini >= 0.0]
        return sorted(saida, key=lambda i: (i.d_ini, i.tipo != "partida", i.chave))

    # ------------------------------------------------------------------ arquivo
    def para_dict(self, passo_rota_m: float = 5.0) -> dict[str, Any]:
        return _puro({
            "mapa": self.mapa,
            "sentido": self.sentido,
            "velocidade_cruzeiro_kmh": self.cruzeiro_kmh,
            "observacoes": list(self.observacoes),
            "rota": self.rota.para_dict(passo_rota_m),
            "zonas": [asdict(z) for z in self.zonas],
            "paradas": [asdict(p) for p in self.paradas],
        })

    def salvar(self, caminho: Path, cabecalho: str = "") -> None:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        texto = yaml.safe_dump(self.para_dict(), allow_unicode=True, sort_keys=False, width=200)
        caminho.write_text((cabecalho + "\n" if cabecalho else "") + texto, encoding="utf-8")

    @classmethod
    def de_dict(cls, dados: dict[str, Any]) -> "PlanoMissao":
        rota = Rota.de_dict(dados["rota"])
        zonas = [Zona(**{**z, "s_ini": float(z["s_ini"]), "s_fim": float(z["s_fim"]), "v_max_kmh": float(z["v_max_kmh"])})
                 for z in dados.get("zonas") or []]
        paradas = [Parada(**{**p, "s": float(p["s"]), "duracao_s": float(p["duracao_s"])}) for p in dados.get("paradas") or []]
        return cls(rota=rota, cruzeiro_kmh=float(dados.get("velocidade_cruzeiro_kmh", 30.0)), zonas=zonas, paradas=paradas,
                   mapa=str(dados.get("mapa", "")), sentido=str(dados.get("sentido", "")),
                   observacoes=list(dados.get("observacoes") or []))

    @classmethod
    def carregar(cls, caminho: Path) -> "PlanoMissao":
        with Path(caminho).open("r", encoding="utf-8") as f:
            return cls.de_dict(yaml.safe_load(f))

    def resumo(self) -> dict[str, Any]:
        return {"comprimento_m": round(self.comprimento, 1), "cruzeiro_kmh": self.cruzeiro_kmh,
                "zonas": len(self.zonas), "paradas": len(self.paradas),
                "zonas_por_tipo": {t: sum(z.tipo == t for z in self.zonas) for t in sorted({z.tipo for z in self.zonas})}}
