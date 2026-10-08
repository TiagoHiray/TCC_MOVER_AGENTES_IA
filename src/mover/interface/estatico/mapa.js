/* mapa.js — os dois desenhos sobre a cena 2D:
 * - VistaPerseguicao: painel 1 sem o CARLA. Visão de cima que segue o caminhão, rumo para cima.
 * - MapaVolta: dashboard. Volta inteira com o norte para cima, trecho percorrido, previsão e problemas.
 */
import {
  FONTE,
  Vista,
  cor,
  corNivel,
  desenharCaminhao,
  desenharEscala,
  desenharMarcador,
  desenharMensagem,
  desenharNorte,
  desenharRotulo,
  desenharSeta,
  difAngulo,
  fmt,
  NOME_NIVEL,
  prepararCanvas,
  tracarTrajeto,
} from "./desenho.js";

const ALTURA_VISIVEL_M = 46; // painel 1: metros de pista na altura do canvas
const RAIO_CLIQUE_PX = 12;

function pintarVias(ctx, vias, { borda = true, larguraMinima = 0 } = {}) {
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  if (borda) {
    // primeiro as bordas de todas as vias, depois o asfalto: os cruzamentos ficam limpos
    for (const via of vias) {
      ctx.strokeStyle = cor("bordaVia", 0.85);
      ctx.lineWidth = Math.max(via.largura + 0.45, larguraMinima);
      ctx.stroke(via.caminho);
    }
  }
  for (const via of vias) {
    ctx.strokeStyle = cor("asfalto");
    ctx.lineWidth = Math.max(via.largura - (borda ? 0.25 : 0), larguraMinima);
    ctx.stroke(via.caminho);
  }
}

/** Painel 1 sem o CARLA: a pista ao redor do caminhão, com o rumo sempre para cima. */
export class VistaPerseguicao {
  /** `cartao`: elemento HTML por cima do canvas (velocidade); os rótulos dos problemas desviam dele. */
  constructor(canvas, cartao = null) {
    this.canvas = canvas;
    this.cartao = cartao;
    this.angulo = null; // rumo da câmera, suavizado
    this.ultimoQuadro = 0;
  }

  /** Retângulo do cartão em px do canvas, com folga, ou null. */
  areaDoCartao() {
    if (!this.cartao || this.cartao.hidden) return null;
    const a = this.cartao.getBoundingClientRect();
    const c = this.canvas.getBoundingClientRect();
    if (!a.width || !a.height) return null;
    return { x: a.left - c.left - 4, y: a.top - c.top - 4, w: a.width + 8, h: a.height + 8 };
  }

  desenhar(d) {
    const tela = prepararCanvas(this.canvas);
    if (!tela) return;
    const { ctx, w, h } = tela;
    ctx.fillStyle = cor("chao");
    ctx.fillRect(0, 0, w, h);
    if (!d.cena || !d.pose) {
      desenharMensagem(ctx, w, h, d.mensagem?.titulo || "Aguardando a cena", d.mensagem?.detalhe || "");
      return;
    }
    // giro da câmera suavizado (~0,15 s), sem depender da taxa de quadros
    const agora = performance.now();
    const dt = Math.min(0.25, (agora - (this.ultimoQuadro || agora)) / 1000);
    this.ultimoQuadro = agora;
    const erro = this.angulo === null ? Infinity : difAngulo(d.pose.rumo, this.angulo);
    if (Math.abs(erro) > 1.2) this.angulo = d.pose.rumo;
    else this.angulo += erro * (1 - Math.exp(-dt / 0.15));

    const escala = h / ALTURA_VISIVEL_M;
    const vista = new Vista(w / 2, h * 0.64, escala, d.pose.x, d.pose.y, Math.PI / 2 - this.angulo);
    const tr = d.cena.tr;

    ctx.save();
    vista.aplicar(ctx);
    pintarVias(ctx, d.cena.vias);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    if (d.t !== null && tracarTrajeto(ctx, tr, tr.t[0], d.t)) {
      ctx.strokeStyle = cor("trajeto", 0.75);
      ctx.lineWidth = 0.55;
      ctx.stroke();
    }
    if (d.t !== null && d.horizonte > d.t && tracarTrajeto(ctx, tr, d.t, d.horizonte)) {
      ctx.setLineDash([1.6, 1.3]);
      ctx.strokeStyle = cor("trajeto", 0.9);
      ctx.lineWidth = 0.45;
      ctx.stroke();
      ctx.setLineDash([]);
    }
    desenharCaminhao(ctx, d.pose.x, d.pose.y, d.pose.rumo);
    ctx.restore();

    // problemas na pista (em px, para o texto não girar)
    const cartao = this.areaDoCartao();
    for (const p of d.problemas) {
      if (!p.pose) continue;
      const [sx, sy] = vista.paraTela(p.pose.x, p.pose.y);
      if (sx < -60 || sx > w + 60 || sy < -30 || sy > h + 30) continue;
      desenharMarcador(ctx, sx, sy, p.nivel, p.previsto, 6);
      const rotulo = `${NOME_NIVEL[p.nivel].toUpperCase()} · ${p.nome} · ${fmt(p.t)} s${p.previsto ? " · previsto" : ""}`;
      const direita = sx < w * 0.62;
      desenharRotulo(ctx, rotulo, direita ? sx + 14 : sx - 14, sy, {
        corTexto: corNivel(p.nivel),
        alinhar: direita ? "left" : "right",
        evitar: cartao,
      });
    }
    // o "N" gira em volta da rosa (raio + 8 px): longe das bordas para não encostar nelas
    desenharNorte(ctx, w - 36, 38, vista.anguloNaTela(Math.PI / 2));
    desenharEscala(ctx, 14, h - 14, escala);
  }
}

/** Dashboard: a volta inteira, norte para cima. Clique num problema -> destaque no log. */
export class MapaVolta {
  constructor(canvas, aoSelecionar) {
    this.canvas = canvas;
    this.fundo = null;
    this.chaveFundo = "";
    this.alvos = []; // posições na tela dos problemas desenhados
    this.aoSelecionar = aoSelecionar;
    canvas.addEventListener("click", (ev) => {
      const alvo = this.alvoEm(ev);
      if (alvo && this.aoSelecionar) this.aoSelecionar(alvo.id);
    });
    canvas.addEventListener("mousemove", (ev) => {
      const alvo = this.alvoEm(ev);
      canvas.style.cursor = alvo ? "pointer" : "default";
      canvas.title = alvo ? alvo.titulo : "";
    });
  }

  alvoEm(ev) {
    const r = this.canvas.getBoundingClientRect();
    const x = ev.clientX - r.left;
    const y = ev.clientY - r.top;
    let melhor = null;
    let menor = RAIO_CLIQUE_PX;
    for (const alvo of this.alvos) {
      const dist = Math.hypot(alvo.sx - x, alvo.sy - y);
      if (dist <= menor) {
        menor = dist;
        melhor = alvo;
      }
    }
    return melhor;
  }

  vista(w, h, limites) {
    const margem = 18;
    const [x0, y0, x1, y1] = limites;
    const escala = Math.min((w - 2 * margem) / (x1 - x0 || 1), (h - 2 * margem) / (y1 - y0 || 1));
    return new Vista(w / 2, h / 2, escala, (x0 + x1) / 2, (y0 + y1) / 2, 0);
  }

  desenharFundo(w, h, dpr, vista, cena) {
    const fundo = document.createElement("canvas");
    fundo.width = Math.round(w * dpr);
    fundo.height = Math.round(h * dpr);
    const ctx = fundo.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.save();
    vista.aplicar(ctx);
    pintarVias(ctx, cena.vias, { borda: false, larguraMinima: 3.2 / vista.escala });
    ctx.setLineDash([2.5 / vista.escala, 2.5 / vista.escala]);
    ctx.strokeStyle = cor("texto3", 0.75);
    ctx.lineWidth = 1.1 / vista.escala;
    ctx.stroke(cena.trajetoCompleto);
    ctx.restore();
    return fundo;
  }

  desenhar(d) {
    const tela = prepararCanvas(this.canvas);
    if (!tela) return;
    const { ctx, w, h, dpr } = tela;
    ctx.clearRect(0, 0, w, h);
    this.alvos = [];
    if (!d.cena) {
      desenharMensagem(ctx, w, h, d.mensagem?.titulo || "Mapa indisponível", d.mensagem?.detalhe || "");
      return;
    }
    const vista = this.vista(w, h, d.cena.limites);
    const chave = `${w}x${h}@${dpr}#${d.cena.versao}`;
    if (chave !== this.chaveFundo) {
      this.fundo = this.desenharFundo(w, h, dpr, vista, d.cena);
      this.chaveFundo = chave;
    }
    ctx.drawImage(this.fundo, 0, 0, w, h);

    const tr = d.cena.tr;
    ctx.save();
    vista.aplicar(ctx);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    if (d.t !== null && tracarTrajeto(ctx, tr, tr.t[0], d.t)) {
      ctx.strokeStyle = cor("trajeto");
      ctx.lineWidth = 2.6 / vista.escala;
      ctx.stroke();
    }
    if (d.t !== null && d.horizonte > d.t && tracarTrajeto(ctx, tr, d.t, d.horizonte)) {
      ctx.setLineDash([5 / vista.escala, 3.5 / vista.escala]);
      ctx.strokeStyle = cor("trajeto", 0.85);
      ctx.lineWidth = 2.2 / vista.escala;
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.restore();

    for (const p of d.problemas) {
      if (!p.pose) continue;
      const [sx, sy] = vista.paraTela(p.pose.x, p.pose.y);
      desenharMarcador(ctx, sx, sy, p.nivel, p.previsto, p.nivel === "critico" ? 5 : 4.2);
      this.alvos.push({
        id: p.id,
        sx,
        sy,
        titulo: `#${p.id} · ${NOME_NIVEL[p.nivel]} · ${p.nome} · ${fmt(p.t)} s${p.previsto ? " (previsão)" : ""}`,
      });
    }
    if (d.pose) {
      const [sx, sy] = vista.paraTela(d.pose.x, d.pose.y);
      desenharSeta(ctx, sx, sy, vista.anguloNaTela(d.pose.rumo), 8);
    }
    desenharNorte(ctx, w - 26, 34, vista.anguloNaTela(Math.PI / 2), 11);
    desenharEscala(ctx, 14, h - 12, vista.escala, 90);
    if (d.cena.origem === "config") {
      // canto superior esquerdo: embaixo fica a escala
      ctx.save();
      ctx.font = `500 10px ${FONTE}`;
      ctx.fillStyle = cor("texto3");
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText("cena do config (o replay ainda não mandou a dele)", 10, 9);
      ctx.restore();
    }
  }
}
