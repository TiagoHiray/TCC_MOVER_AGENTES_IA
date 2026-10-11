/* mapa.js — os dois desenhos sobre a cena 2D (vias, rota e plano de missão):
 * - VistaPerseguicao: painel 1 sem o CARLA. Visão de cima que segue o caminhão, rumo para cima.
 * - MapaVolta: dashboard. Volta inteira com o norte para cima; o rastro do caminhão cresce ao vivo.
 *
 * As zonas do plano (lombadas, faixa, curvas) aparecem coloridas por tipo. Uma zona com intervenção
 * decidida pelos agentes ganha um contorno verde (vazado se a opção for "sem intervenção").
 */
import {
  FONTE,
  NOME_NIVEL,
  NOME_OPCAO,
  NOME_TIPO,
  Vista,
  cor,
  corNivel,
  corTipo,
  desenharCaminhao,
  desenharEscala,
  desenharMarcador,
  desenharMensagem,
  desenharNorte,
  desenharRotulo,
  desenharSeta,
  difAngulo,
  fmt,
  indiceAte,
  prepararCanvas,
} from "./desenho.js";

const ALTURA_VISIVEL_M = 52; // painel 1: metros de pista na altura do canvas
const RAIO_CLIQUE_PX = 12;
const ROTULOS_ATE_M = 60;    // painel 1: zonas à frente com rótulo

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

/** Traça o rastro (sem pintar) até o instante t, com a ponta na pose atual. */
function tracarRastro(ctx, rastro, t, pose, desde = -Infinity) {
  const n = rastro.t.length;
  if (!n || t === null) return false;
  const fim = Math.min(n - 1, indiceAte(rastro.t, t));
  let i = Math.max(0, indiceAte(rastro.t, desde));
  if (fim < i) return false;
  ctx.beginPath();
  ctx.moveTo(rastro.x[i], rastro.y[i]);
  for (i += 1; i <= fim; i++) ctx.lineTo(rastro.x[i], rastro.y[i]);
  if (pose) ctx.lineTo(pose.x, pose.y);
  return true;
}

/** Situação da zona na volta atual: { opcao, origem } se os agentes já decidiram. */
function decisaoDaZona(d, id) {
  return d.decisoes?.get(`${id}@${d.volta ?? 0}`) || null;
}

function contornarDecisao(ctx, caminho, decisao, largura) {
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.lineWidth = largura;
  ctx.strokeStyle = decisao.opcao === "nenhuma" ? cor("texto2", 0.55) : cor("escolhido", 0.85);
  if (decisao.opcao === "nenhuma") ctx.setLineDash([largura * 0.9, largura * 0.7]);
  ctx.stroke(caminho);
  ctx.restore();
}

/** Painel 1 sem o CARLA: a pista ao redor do caminhão, com o rumo sempre para cima. */
export class VistaPerseguicao {
  /** `cartao`: elemento HTML por cima do canvas (velocidade); os rótulos desviam dele. */
  constructor(canvas, cartao = null) {
    this.canvas = canvas;
    this.cartao = cartao;
    this.angulo = null; // rumo da câmera, suavizado
    this.ultimoQuadro = 0;
  }

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
    const pose = d.pose || d.cena?.largada || null;
    if (!d.cena || !pose) {
      desenharMensagem(ctx, w, h, d.mensagem?.titulo || "Aguardando a cena", d.mensagem?.detalhe || "");
      return;
    }
    // giro da câmera suavizado (~0,15 s), sem depender da taxa de quadros
    const agora = performance.now();
    const dt = Math.min(0.25, (agora - (this.ultimoQuadro || agora)) / 1000);
    this.ultimoQuadro = agora;
    const erro = this.angulo === null ? Infinity : difAngulo(pose.rumo, this.angulo);
    if (Math.abs(erro) > 1.2) this.angulo = pose.rumo;
    else this.angulo += erro * (1 - Math.exp(-dt / 0.15));

    const escala = h / ALTURA_VISIVEL_M;
    const vista = new Vista(w / 2, h * 0.66, escala, pose.x, pose.y, Math.PI / 2 - this.angulo);
    const cena = d.cena;

    ctx.save();
    vista.aplicar(ctx);
    pintarVias(ctx, cena.vias);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    // rota à frente (tracejada) e zonas como faixas coloridas sobre o asfalto
    ctx.setLineDash([1.6, 1.4]);
    ctx.strokeStyle = cor("texto2", 0.35);
    ctx.lineWidth = 0.25;
    ctx.stroke(cena.rota);
    ctx.setLineDash([]);
    for (const z of cena.zonas) {
      ctx.strokeStyle = corTipo(z.tipo, z.tipo === "curva" ? 0.18 : 0.42);
      ctx.lineWidth = 3.0;
      ctx.stroke(z.caminho);
      const decisao = decisaoDaZona(d, z.id);
      if (decisao) contornarDecisao(ctx, z.caminho, decisao, 0.35);
    }
    if (tracarRastro(ctx, d.rastro, d.t, d.pose, (d.t ?? 0) - 40)) {
      ctx.strokeStyle = cor("trajeto", 0.8);
      ctx.lineWidth = 0.55;
      ctx.stroke();
    }
    for (const p of cena.paradas) {
      ctx.fillStyle = corTipo("portaria", 0.9);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 0.9, 0, 2 * Math.PI);
      ctx.fill();
    }
    desenharCaminhao(ctx, pose.x, pose.y, pose.rumo);
    ctx.restore();

    // rótulos das zonas à frente (em px, para o texto não girar)
    const cartao = this.areaDoCartao();
    const dAtual = d.dAtual ?? 0;
    const L = cena.comprimento || 1;
    for (const z of [...cena.zonas, ...cena.paradas.map((p) => ({ ...p, tipo: "parada", s_ini: p.s, meio: p }))]) {
      let falta = ((z.s_ini - (dAtual % L)) % L + L) % L;
      if (falta > ROTULOS_ATE_M) continue;
      const [sx, sy] = vista.paraTela(z.meio.x, z.meio.y);
      if (sx < -60 || sx > w + 60 || sy < -30 || sy > h + 30) continue;
      const decisao = decisaoDaZona(d, z.id);
      const limite = z.tipo === "parada" ? `parar ${fmt(z.duracao_s, 0)} s` : `${fmt(z.v_max_kmh, 0)} km/h`;
      const acao = decisao ? ` · ${NOME_OPCAO[decisao.opcao] || decisao.opcao}` : "";
      const direita = sx < w * 0.6;
      desenharRotulo(ctx, `${z.nome || NOME_TIPO[z.tipo] || z.tipo} · ${limite}${acao}`, direita ? sx + 14 : sx - 14, sy, {
        corTexto: decisao && decisao.opcao !== "nenhuma" ? cor("escolhido") : corTipo(z.tipo),
        alinhar: direita ? "left" : "right",
        evitar: cartao,
      });
    }
    for (const p of d.problemas) {
      if (!p.pose) continue;
      const [sx, sy] = vista.paraTela(p.pose.x, p.pose.y);
      if (sx < -60 || sx > w + 60 || sy < -30 || sy > h + 30) continue;
      desenharMarcador(ctx, sx, sy, p.nivel, false, 6);
    }
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
    this.alvos = []; // posições na tela dos marcadores clicáveis
    this.aoSelecionar = aoSelecionar;
    canvas.addEventListener("click", (ev) => {
      const alvo = this.alvoEm(ev);
      if (alvo && alvo.id !== undefined && this.aoSelecionar) this.aoSelecionar(alvo.id);
    });
    canvas.addEventListener("mousemove", (ev) => {
      const alvo = this.alvoEm(ev);
      canvas.style.cursor = alvo && alvo.id !== undefined ? "pointer" : "default";
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
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.setLineDash([2.5 / vista.escala, 2.5 / vista.escala]);
    ctx.strokeStyle = cor("texto3", 0.75);
    ctx.lineWidth = 1.1 / vista.escala;
    ctx.stroke(cena.rota);
    ctx.setLineDash([]);
    for (const z of cena.zonas) {
      ctx.strokeStyle = corTipo(z.tipo, z.tipo === "curva" ? 0.45 : 0.9);
      ctx.lineWidth = (z.tipo === "curva" ? 3.0 : 4.2) / vista.escala;
      ctx.stroke(z.caminho);
    }
    ctx.restore();
    // largada/chegada e paradas (em px)
    if (cena.largada) {
      const [sx, sy] = vista.paraTela(cena.largada.x, cena.largada.y);
      const a = vista.anguloNaTela(cena.largada.rumo) + Math.PI / 2;
      ctx.save();
      ctx.translate(sx, sy);
      ctx.rotate(a);
      for (let k = -2; k < 2; k++) {
        ctx.fillStyle = k % 2 ? "rgba(255,255,255,0.9)" : "rgba(20,20,20,0.9)";
        ctx.fillRect(k * 3, -1.5, 3, 3);
        ctx.fillStyle = k % 2 ? "rgba(20,20,20,0.9)" : "rgba(255,255,255,0.9)";
        ctx.fillRect(k * 3, 1.5, 3, 3);
      }
      ctx.restore();
    }
    for (const p of cena.paradas) {
      const [sx, sy] = vista.paraTela(p.x, p.y);
      ctx.beginPath();
      ctx.arc(sx, sy, 4.5, 0, 2 * Math.PI);
      ctx.fillStyle = corTipo("portaria");
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = "rgba(255,255,255,0.85)";
      ctx.stroke();
    }
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
    const cena = d.cena;
    const vista = this.vista(w, h, cena.limites);
    const chave = `${w}x${h}@${dpr}#${cena.versao}`;
    if (chave !== this.chaveFundo) {
      this.fundo = this.desenharFundo(w, h, dpr, vista, cena);
      this.chaveFundo = chave;
    }
    ctx.drawImage(this.fundo, 0, 0, w, h);

    ctx.save();
    vista.aplicar(ctx);
    for (const z of cena.zonas) {
      const decisao = decisaoDaZona(d, z.id);
      if (decisao) contornarDecisao(ctx, z.caminho, decisao, 7.5 / vista.escala);
    }
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    if (tracarRastro(ctx, d.rastro, d.t, d.pose)) {
      ctx.strokeStyle = cor("trajeto");
      ctx.lineWidth = 2.6 / vista.escala;
      ctx.stroke();
    }
    ctx.restore();

    for (const z of cena.zonas) {
      const [sx, sy] = vista.paraTela(z.meio.x, z.meio.y);
      const decisao = decisaoDaZona(d, z.id);
      const acao = decisao ? ` · ${NOME_OPCAO[decisao.opcao] || decisao.opcao} (${decisao.origem === "llm" ? "LLM" : "regra"})` : "";
      this.alvos.push({ sx, sy, titulo: `${z.nome} · ${fmt(z.v_max_kmh, 0)} km/h${acao}` });
    }
    for (const p of d.problemas) {
      if (!p.pose) continue;
      const [sx, sy] = vista.paraTela(p.pose.x, p.pose.y);
      desenharMarcador(ctx, sx, sy, p.nivel, false, p.nivel === "critico" ? 5 : 4.2);
      this.alvos.push({ id: p.id, sx, sy, titulo: `#${p.id} · ${NOME_NIVEL[p.nivel]} · ${p.nome} · ${fmt(p.t)} s` });
    }
    if (d.pose) {
      const [sx, sy] = vista.paraTela(d.pose.x, d.pose.y);
      desenharSeta(ctx, sx, sy, vista.anguloNaTela(d.pose.rumo), 8);
    }
    desenharNorte(ctx, w - 26, 34, vista.anguloNaTela(Math.PI / 2), 11);
    desenharEscala(ctx, 14, h - 12, vista.escala, 90);
    if (cena.origem === "config") {
      ctx.save();
      ctx.font = `500 10px ${FONTE}`;
      ctx.fillStyle = cor("texto3");
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText("cena do config (o piloto ainda não mandou a dele)", 10, 9);
      ctx.restore();
    }
  }
}

export { corNivel };
