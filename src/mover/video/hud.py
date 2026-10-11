"""HUD dos vídeos da corrida: velocidade, consumo, decisão dos agentes e minimapa (Pillow).

Desenhado por cima de cada quadro da câmera do CARLA, a partir da telemetria e das decisões que a
corrida gravou em data/autonomo/corridas/<id>/. Tudo em coordenadas de pixel de um quadro de
largura x altura (1920 x 1080 por padrão); as medidas escalam com a altura.
"""

from __future__ import annotations

import json
import math
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

NOMES_OPCAO = {"economica": "econômica", "suave": "suave", "moderada": "moderada", "firme": "firme", "nenhuma": "sem intervenção"}
FONTES = ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "arial.ttf", "segoeui.ttf", "Arial.ttf", "LiberationSans-Regular.ttf")
FONTES_NEGRITO = ("DejaVuSans-Bold.ttf", "arialbd.ttf", "segoeuib.ttf", "Arial Bold.ttf", "LiberationSans-Bold.ttf")
DECISAO_NA_TELA_S = 9.0


def fonte(tamanho: int, negrito: bool = False):
    from PIL import ImageFont

    for nome in (FONTES_NEGRITO if negrito else FONTES):
        try:
            return ImageFont.truetype(nome, tamanho)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=tamanho)
    except TypeError:  # Pillow antigo
        return ImageFont.load_default()


def num(valor: float, casas: int = 0) -> str:
    texto = f"{float(valor):.{casas}f}"
    if texto.lstrip("-").strip("0.") == "":
        texto = texto.lstrip("-")
    return texto.replace(".", ",")


@dataclass
class DadosCorrida:
    """Telemetria, decisões e resumo de uma corrida, com consulta por instante."""

    telemetria: pd.DataFrame
    decisoes: list[dict[str, Any]] = field(default_factory=list)
    resumo: dict[str, Any] = field(default_factory=dict)
    rota_xy: np.ndarray | None = None
    base: dict[str, Any] | None = None   # linha de base (plano sem agentes, previsto) pela distância: d, combustivel_ml

    @classmethod
    def carregar(cls, pasta: Path, plano: Any = None) -> "DadosCorrida":
        telemetria = pd.read_csv(pasta / "telemetria.csv")
        decisoes = []
        if (pasta / "decisoes.jsonl").exists():
            with (pasta / "decisoes.jsonl").open(encoding="utf-8") as f:
                decisoes = [json.loads(linha) for linha in f if linha.strip()]
        resumo = json.loads((pasta / "resumo.json").read_text(encoding="utf-8")) if (pasta / "resumo.json").exists() else {}
        base = json.loads((pasta / "linha_de_base.json").read_text(encoding="utf-8")) if (pasta / "linha_de_base.json").exists() else None
        rota = None
        if plano is not None:
            pontos = plano.rota.amostrar(0.0, plano.comprimento, 2.0)
            rota = np.array([[p.x, p.y] for p in pontos])
        return cls(telemetria, sorted(decisoes, key=lambda d: d.get("t_aplicada", d.get("t_pedido", 0.0))), resumo, rota, base)

    def base_ml(self, d: float) -> float | None:
        """Consumo do plano sem agentes na mesma distância percorrida (previsto pelo modelo do caminhão)."""
        if not self.base or not self.base.get("d"):
            return None
        return float(np.interp(d, self.base["d"], self.base["combustivel_ml"]))

    @property
    def duracao_s(self) -> float:
        return float(self.telemetria["t"].iloc[-1]) if len(self.telemetria) else 0.0

    def no_instante(self, t: float) -> dict[str, Any]:
        tempos = self.telemetria["t"].to_numpy(float)
        i = int(np.clip(np.searchsorted(tempos, t), 0, len(tempos) - 1))
        return self.telemetria.iloc[i].to_dict()

    def decisao_em(self, t: float) -> dict[str, Any] | None:
        atual = None
        for d in self.decisoes:
            if d.get("t_aplicada", d.get("t_pedido", 0.0)) <= t:
                atual = d
        if atual is None or t - atual.get("t_aplicada", atual.get("t_pedido", 0.0)) > DECISAO_NA_TELA_S:
            return None
        return atual

    def rastro_ate(self, t: float) -> np.ndarray:
        sel = self.telemetria["t"].to_numpy(float) <= t
        return self.telemetria.loc[sel, ["x", "y"]].to_numpy(float)


def _caixa(desenho, xy: tuple[float, float, float, float], raio: int, cor=(8, 12, 20, 170)) -> None:
    desenho.rounded_rectangle(xy, radius=raio, fill=cor, outline=(255, 255, 255, 40), width=1)


def desenhar_hud(quadro, dados: DadosCorrida, t: float):
    """Devolve o quadro (PIL.Image RGB) com o HUD da corrida no instante t."""
    from PIL import Image, ImageDraw

    largura, altura = quadro.size
    u = altura / 1080.0  # escala
    camada = Image.new("RGBA", quadro.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(camada)
    e = dados.no_instante(t)
    resumo = dados.resumo
    modo = resumo.get("modo", e.get("modo", "agentes"))
    voltas = int(resumo.get("voltas", 1) or 1)

    # título (alto, à direita)
    titulo = "MOVER · gêmeo digital do caminhão autônomo"
    subtitulo = f"{'com a camada agêntica' if modo == 'agentes' else 'sem agentes (caminhão sozinho)'} · {resumo.get('simulador', 'carla').upper()}"
    f_t, f_s = fonte(int(30 * u), True), fonte(int(22 * u))
    w_t = d.textlength(titulo, font=f_t)
    w_s = d.textlength(subtitulo, font=f_s)
    w = max(w_t, w_s) + 40 * u
    _caixa(d, (largura - w - 24 * u, 24 * u, largura - 24 * u, 120 * u), int(14 * u))
    d.text((largura - w - 4 * u, 36 * u), titulo, font=f_t, fill=(240, 244, 250, 255))
    d.text((largura - w - 4 * u, 78 * u), subtitulo, font=f_s, fill=(110, 220, 170, 255) if modo == "agentes" else (240, 190, 90, 255))

    # velocímetro (alto, à esquerda)
    _caixa(d, (24 * u, 24 * u, 470 * u, 220 * u), int(16 * u))
    vel = float(e.get("speed_kmh", e.get("speed_mps", 0.0) * 3.6))
    d.text((48 * u, 30 * u), num(vel), font=fonte(int(96 * u), True), fill=(255, 255, 255, 255))
    d.text((48 * u + d.textlength(num(vel), font=fonte(int(96 * u), True)) + 12 * u, 92 * u), "km/h",
           font=fonte(int(30 * u)), fill=(200, 208, 220, 255))
    desejada, limite = float(e.get("v_desejada_kmh", 0.0)), float(e.get("v_limite_kmh", 0.0))
    d.text((48 * u, 140 * u), f"desejada {num(desejada)} · limite {num(limite)} km/h", font=fonte(int(24 * u)),
           fill=(190, 160, 255, 255))
    a = float(e.get("acc_long", 0.0))
    volta = min(int(e.get("volta", 0)) + 1, voltas)
    d.text((48 * u, 176 * u), f"{num(a, 1)} m/s²   ·   {num(t, 1)} s   ·   volta {volta}/{voltas}", font=fonte(int(22 * u)),
           fill=(200, 208, 220, 255))
    # barra de aceleração (−3 a +3 m/s²)
    x0, x1, y = 300 * u, 450 * u, 60 * u
    meio = (x0 + x1) / 2
    d.rounded_rectangle((x0, y, x1, y + 16 * u), radius=int(8 * u), fill=(40, 46, 58, 220))
    ponta = meio + max(-1.0, min(1.0, a / 3.0)) * (x1 - x0) / 2
    cor_a = (90, 200, 140, 255) if a >= 0 else (235, 90, 80, 255)
    d.rectangle((min(meio, ponta), y + 2 * u, max(meio, ponta), y + 14 * u), fill=cor_a)
    d.line((meio, y - 4 * u, meio, y + 20 * u), fill=(255, 255, 255, 160), width=max(1, int(2 * u)))

    # consumo (baixo, à esquerda)
    ml = float(e.get("combustivel_ml", 0.0))
    custo = float(e.get("custo_r", 0.0))
    _caixa(d, (24 * u, altura - 232 * u, 470 * u, altura - 24 * u), int(16 * u))
    d.text((48 * u, altura - 220 * u), "DIESEL NESTA CORRIDA", font=fonte(int(20 * u), True), fill=(150, 160, 176, 255))
    d.text((48 * u, altura - 190 * u), f"{num(ml)} mL", font=fonte(int(52 * u), True), fill=(100, 210, 230, 255))
    d.text((48 * u, altura - 122 * u), f"R$ {num(custo, 2)}", font=fonte(int(24 * u)), fill=(200, 208, 220, 255))
    base_ml = dados.base_ml(float(e.get("d", 0.0))) if modo == "agentes" else None
    if base_ml is not None and base_ml > 1:
        rotulo = "sem agentes (previsto)"
        d.text((48 * u, altura - 84 * u), f"{rotulo}: {num(base_ml)} mL ({num(100.0 * (ml - base_ml) / base_ml)} %)",
               font=fonte(int(22 * u)), fill=(120, 230, 170, 255) if ml <= base_ml else (240, 120, 100, 255))

    # decisão dos agentes (baixo, no meio)
    decisao = dados.decisao_em(t)
    if decisao is not None and modo == "agentes":
        opcao = NOMES_OPCAO.get(decisao.get("opcao"), decisao.get("opcao"))
        origem = "LLM" if decisao.get("texto_origem") == "llm" else "regra"
        cabecalho = f"Decisão dos agentes ({origem}) · {opcao}"
        linhas = textwrap.wrap(decisao.get("texto", ""), width=78)[:3]
        x_d0, x_d1 = 500 * u, largura - 420 * u
        topo = altura - (70 + 34 * len(linhas)) * u - 24 * u
        _caixa(d, (x_d0, topo, x_d1, altura - 24 * u), int(16 * u), cor=(10, 40, 28, 190))
        d.text((x_d0 + 22 * u, topo + 14 * u), cabecalho, font=fonte(int(24 * u), True), fill=(120, 230, 170, 255))
        for k, linha in enumerate(linhas):
            d.text((x_d0 + 22 * u, topo + (50 + 34 * k) * u), linha, font=fonte(int(23 * u)), fill=(232, 236, 242, 255))

    # minimapa (baixo, à direita)
    if dados.rota_xy is not None and len(dados.rota_xy):
        lado = 360 * u
        mx0, my0 = largura - lado - 24 * u, altura - lado - 24 * u
        _caixa(d, (mx0, my0, mx0 + lado, my0 + lado), int(16 * u))
        pts = dados.rota_xy
        xmin, ymin = pts.min(axis=0)
        xmax, ymax = pts.max(axis=0)
        escala = (lado - 40 * u) / max(xmax - xmin, ymax - ymin, 1.0)
        cx, cy = mx0 + lado / 2, my0 + lado / 2
        meio_x, meio_y = (xmin + xmax) / 2, (ymin + ymax) / 2

        def tela(x: float, y: float) -> tuple[float, float]:
            return cx + (x - meio_x) * escala, cy - (y - meio_y) * escala

        d.line([tela(x, y) for x, y in pts] + [tela(*pts[0])], fill=(120, 130, 150, 200), width=max(2, int(5 * u)))
        rastro = dados.rastro_ate(t)
        if len(rastro) > 1:
            d.line([tela(x, y) for x, y in rastro[::4]] + [tela(*rastro[-1])], fill=(80, 200, 230, 255), width=max(2, int(5 * u)))
        x, y = float(e.get("x", pts[0][0])), float(e.get("y", pts[0][1]))
        px, py = tela(x, y)
        r = 9 * u
        d.ellipse((px - r, py - r, px + r, py + r), fill=(235, 80, 60, 255), outline=(255, 255, 255, 255), width=max(1, int(2 * u)))
        rumo = math.radians(float(e.get("rumo_graus", 0.0)))
        d.line((px, py, px + 22 * u * math.cos(rumo), py - 22 * u * math.sin(rumo)), fill=(255, 255, 255, 255), width=max(2, int(3 * u)))

    return Image.alpha_composite(quadro.convert("RGBA"), camada).convert("RGB")


__all__ = ["DadosCorrida", "desenhar_hud", "fonte"]
