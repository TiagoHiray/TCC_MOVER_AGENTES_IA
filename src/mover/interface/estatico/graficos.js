/* graficos.js — gráficos do dashboard: velocidade e aceleração longitudinal no tempo.
 *
 * Linha cheia = já percorrido (até o instante atual); tracejada = o resto do bloco atual e o
 * próximo bloco já analisado (previsão). Linha vertical = instante atual. Triângulos no topo =
 * problemas do log (vazados se ainda são previsão). O mouse mostra o valor nos dois gráficos.
 */
import { FONTE, FONTE_MONO, cor, corNivel, fmt, indiceAte, prepararCanvas, valorNoTempo } from "./desenho.js";

const MARGEM = { esq: 38, dir: 12, topo: 28, base: 20 };

/** Junta a telemetria publicada e a da previsão numa série só (t crescente). */
function juntarSerie(serie, chave) {
  const ts = [];
  const vs = [];
  const pub = serie.publicada;
  const n = pub.t.length;
  for (let i = 0; i < n; i++) {
    ts.push(pub.t[i]);
    vs.push(pub[chave][i] ?? null);
  }
  const ultimo = n ? pub.t[n - 1] : -Infinity;
  const prev = serie.previsao;
  if (prev && prev.t) {
    for (let i = 0; i < prev.t.length; i++) {
      if (prev.t[i] > ultimo + 1e-6) {
        ts.push(prev.t[i]);
        vs.push(prev[chave]?.[i] ?? null);
      }
    }
  }
  return { ts, vs };
}

/** Polilinha de (ts, vs) entre ta e tb, com as pontas interpoladas. Devolve os pontos em px. */
function pontosEntre(ts, vs, ta, tb, X, Y) {
  const pts = [];
  if (!(tb > ta) || !ts.length) return pts;
  const ini = Math.max(ta, ts[0]);
  const fim = Math.min(tb, ts[ts.length - 1]);
  if (!(fim > ini)) return pts;
  const va = valorNoTempo(ts, vs, ini);
  if (va !== null) pts.push([X(ini), Y(va)]);
  for (let i = indiceAte(ts, ini) + 1; i < ts.length && ts[i] < fim; i++) {
    if (vs[i] !== null) pts.push([X(ts[i]), Y(vs[i])]);
  }
  const vb = valorNoTempo(ts, vs, fim);
  if (vb !== null) pts.push([X(fim), Y(vb)]);
  return pts;
}

function tracar(ctx, pts) {
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
}

export class Grafico {
  /**
   * opcoes: { titulo, unidade, chave, faixa: [min, max], simetrica, casas, cor, sinal }
   * hover: objeto compartilhado entre os gráficos ({ t: número | null }).
   */
  constructor(canvas, opcoes, hover) {
    this.canvas = canvas;
    this.op = opcoes;
    this.hover = hover;
    this.layout = null;
    canvas.addEventListener("mousemove", (ev) => {
      if (!this.layout) return;
      const r = canvas.getBoundingClientRect();
      const x = ev.clientX - r.left;
      const { x0, x1, tMin, tMax } = this.layout;
      this.hover.t = x >= x0 && x <= x1 ? tMin + ((x - x0) / (x1 - x0)) * (tMax - tMin) : null;
    });
    canvas.addEventListener("mouseleave", () => {
      this.hover.t = null;
    });
  }

  faixa(vs) {
    let maior = -Infinity;
    let menor = Infinity;
    for (const v of vs) {
      if (v === null || !Number.isFinite(v)) continue;
      if (v > maior) maior = v;
      if (v < menor) menor = v;
    }
    const [base0, base1] = this.op.faixa;
    const temDados = Number.isFinite(maior) && Number.isFinite(menor);
    if (this.op.simetrica) {
      const extremo = temDados ? Math.max(Math.abs(menor), Math.abs(maior)) : 0;
      const limite = Math.max(base1, Math.ceil(extremo));
      return { min: -limite, max: limite, passo: limite <= 4 ? 2 : limite <= 8 ? 4 : 5 };
    }
    const max = Math.max(base1, temDados ? Math.ceil(maior / 10) * 10 : 0);
    return { min: base0, max, passo: max <= 50 ? 10 : 20 };
  }

  desenhar(d) {
    const tela = prepararCanvas(this.canvas);
    if (!tela) return;
    const { ctx, w, h } = tela;
    ctx.clearRect(0, 0, w, h);
    const x0 = MARGEM.esq;
    const x1 = w - MARGEM.dir;
    const y0 = MARGEM.topo;
    const y1 = h - MARGEM.base;
    if (x1 - x0 < 60 || y1 - y0 < 30) return;
    const tMin = d.t0;
    const tMax = d.t0 + Math.max(d.duracao, 1);
    const X = (t) => x0 + ((t - tMin) / (tMax - tMin)) * (x1 - x0);
    const { ts, vs } = juntarSerie(d.serie, this.op.chave);
    const { min, max, passo } = this.faixa(vs);
    const Y = (v) => y1 - ((v - min) / (max - min)) * (y1 - y0);
    this.layout = { x0, x1, tMin, tMax };
    const corSerie = cor(this.op.cor);

    // título, unidade e valor atual
    ctx.textBaseline = "alphabetic";
    ctx.textAlign = "left";
    ctx.font = `600 10.5px ${FONTE}`;
    if ("letterSpacing" in ctx) ctx.letterSpacing = "0.08em";
    ctx.fillStyle = cor("texto2");
    ctx.fillText(this.op.titulo.toUpperCase(), x0, 17);
    const larguraTitulo = ctx.measureText(this.op.titulo.toUpperCase()).width;
    if ("letterSpacing" in ctx) ctx.letterSpacing = "0px";
    ctx.font = `500 10.5px ${FONTE}`;
    ctx.fillStyle = cor("texto3");
    ctx.fillText(this.op.unidade, x0 + larguraTitulo + 8, 17);
    ctx.textAlign = "right";
    ctx.font = `700 13px ${FONTE}`;
    ctx.fillStyle = cor("texto");
    if (d.valorAtual !== null && d.valorAtual !== undefined) {
      ctx.fillText(`${fmt(d.valorAtual, this.op.casas)} ${this.op.unidade}`, x1, 18);
    }

    // faixas do bloco atual e do bloco em previsão
    if (d.blocoAtual) {
      ctx.fillStyle = "rgba(255, 255, 255, 0.035)";
      ctx.fillRect(X(d.blocoAtual.t_ini), y0, X(d.blocoAtual.t_fim) - X(d.blocoAtual.t_ini), y1 - y0);
    }
    if (d.blocoPrevisao) {
      const a = X(d.blocoPrevisao.t_ini);
      const b = X(d.blocoPrevisao.t_fim);
      ctx.fillStyle = cor("trajeto", 0.07);
      ctx.fillRect(a, y0, b - a, y1 - y0);
      if (b - a > 44) {
        ctx.font = `600 9.5px ${FONTE}`;
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillStyle = cor("trajeto", 0.9);
        ctx.fillText("previsão", (a + b) / 2, y0 + 3);
      }
    }

    // grade: horizontais nos valores, verticais a cada bloco (10 s)
    ctx.lineWidth = 1;
    ctx.font = `500 10px ${FONTE_MONO}`;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let i = 0, v = min; i < 60 && v <= max + 1e-9; i++, v += passo) {
      const y = Math.round(Y(v)) + 0.5;
      ctx.strokeStyle = v === 0 && min < 0 ? cor("grade", 0.17) : cor("grade", 0.07);
      ctx.beginPath();
      ctx.moveTo(x0, y);
      ctx.lineTo(x1, y);
      ctx.stroke();
      ctx.fillStyle = cor("texto3");
      ctx.fillText(fmt(v, 0), x0 - 6, y);
    }
    const bloco = d.duracaoBloco > 0 ? d.duracaoBloco : 10;
    const pxPorBloco = ((x1 - x0) * bloco) / (tMax - tMin);
    const rotuloCada = pxPorBloco >= 34 ? 1 : pxPorBloco >= 17 ? 2 : 3;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let k = 0, t = tMin; k < 60 && t <= tMax + 1e-9; k++, t += bloco) {
      const x = Math.round(X(t)) + 0.5;
      ctx.strokeStyle = cor("grade", 0.06);
      ctx.beginPath();
      ctx.moveTo(x, y0);
      ctx.lineTo(x, y1);
      ctx.stroke();
      if (k % rotuloCada === 0 && x < x1 - 16) {
        ctx.fillStyle = cor("texto3");
        ctx.fillText(fmt(t - tMin, 0), x, y1 + 5);
      }
    }
    ctx.textAlign = "right";
    ctx.fillStyle = cor("texto3");
    ctx.fillText("s", x1, y1 + 5);

    const agora = d.t;
    const passado = agora !== null ? pontosEntre(ts, vs, tMin, agora, X, Y) : [];
    const futuro = agora !== null ? pontosEntre(ts, vs, agora, tMax, X, Y) : pontosEntre(ts, vs, tMin, tMax, X, Y);

    // área sob a curva do que já passou
    if (passado.length > 1) {
      if (this.op.sinal) {
        const y = Y(0);
        for (const [acima, nome] of [[true, "positivo"], [false, "negativo"]]) {
          ctx.save();
          ctx.beginPath();
          if (acima) ctx.rect(x0, y0, x1 - x0, y - y0);
          else ctx.rect(x0, y, x1 - x0, y1 - y);
          ctx.clip();
          tracar(ctx, passado);
          ctx.lineTo(passado[passado.length - 1][0], y);
          ctx.lineTo(passado[0][0], y);
          ctx.closePath();
          ctx.fillStyle = cor(nome, 0.3);
          ctx.fill();
          ctx.restore();
        }
      } else {
        const gradiente = ctx.createLinearGradient(0, y0, 0, y1);
        gradiente.addColorStop(0, cor(this.op.cor, 0.3));
        gradiente.addColorStop(1, cor(this.op.cor, 0));
        tracar(ctx, passado);
        ctx.lineTo(passado[passado.length - 1][0], y1);
        ctx.lineTo(passado[0][0], y1);
        ctx.closePath();
        ctx.fillStyle = gradiente;
        ctx.fill();
      }
    }
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    if (futuro.length > 1) {
      tracar(ctx, futuro);
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = cor(this.op.cor, 0.6);
      ctx.lineWidth = 1.4;
      ctx.stroke();
      ctx.setLineDash([]);
    }
    if (passado.length > 1) {
      tracar(ctx, passado);
      ctx.strokeStyle = corSerie;
      ctx.lineWidth = 1.8;
      ctx.stroke();
    }

    // problemas do log
    for (const p of d.problemas) {
      if (p.t < tMin || p.t > tMax) continue;
      const x = X(p.t);
      ctx.save();
      ctx.setLineDash([2, 3]);
      ctx.strokeStyle = corNivel(p.nivel, 0.35);
      ctx.beginPath();
      ctx.moveTo(x, y0 + 8);
      ctx.lineTo(x, y1);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(x - 4.5, y0);
      ctx.lineTo(x + 4.5, y0);
      ctx.lineTo(x, y0 + 7);
      ctx.closePath();
      if (p.previsto) {
        ctx.lineWidth = 1.4;
        ctx.strokeStyle = corNivel(p.nivel);
        ctx.stroke();
      } else {
        ctx.fillStyle = corNivel(p.nivel);
        ctx.fill();
      }
      ctx.restore();
    }

    // instante atual
    if (agora !== null && agora >= tMin && agora <= tMax) {
      const x = Math.round(X(agora)) + 0.5;
      ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(x, y0);
      ctx.lineTo(x, y1);
      ctx.stroke();
      const v = valorNoTempo(ts, vs, agora);
      if (v !== null) {
        ctx.beginPath();
        ctx.arc(x, Y(v), 3.6, 0, 2 * Math.PI);
        ctx.fillStyle = corSerie;
        ctx.fill();
        ctx.lineWidth = 1.6;
        ctx.strokeStyle = cor("fundoEscuro");
        ctx.stroke();
      }
      const texto = `${fmt(agora, 1)} s`;
      ctx.font = `600 10px ${FONTE_MONO}`;
      const largura = ctx.measureText(texto).width + 10;
      const xr = Math.min(Math.max(x - largura / 2, x0), x1 - largura);
      ctx.fillStyle = "rgba(240, 244, 250, 0.95)";
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(xr, y1 + 3, largura, 15, 4);
      else ctx.rect(xr, y1 + 3, largura, 15);
      ctx.fill();
      ctx.fillStyle = cor("fundoEscuro");
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(texto, xr + 5, y1 + 11);
    }

    // leitura sob o mouse (nos dois gráficos ao mesmo tempo)
    const th = this.hover.t;
    if (th !== null && th >= tMin && th <= tMax) {
      const v = valorNoTempo(ts, vs, th);
      const x = Math.round(X(th)) + 0.5;
      ctx.strokeStyle = "rgba(255, 255, 255, 0.3)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, y0);
      ctx.lineTo(x, y1);
      ctx.stroke();
      if (v !== null) {
        const futuroNoMouse = agora !== null && th > agora;
        const texto = `${fmt(th, 1)} s · ${fmt(v, this.op.casas)} ${this.op.unidade}${futuroNoMouse ? " · previsto" : ""}`;
        ctx.font = `600 10.5px ${FONTE}`;
        const largura = ctx.measureText(texto).width + 12;
        const xr = x + 8 + largura > x1 ? x - 8 - largura : x + 8;
        ctx.fillStyle = cor("fundoEscuro", 0.9);
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(xr, y0 + 14, largura, 19, 5);
        else ctx.rect(xr, y0 + 14, largura, 19);
        ctx.fill();
        ctx.fillStyle = cor("texto");
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(texto, xr + 6, y0 + 24);
        ctx.beginPath();
        ctx.arc(x, Y(v), 3, 0, 2 * Math.PI);
        ctx.fillStyle = "white";
        ctx.fill();
      }
    }
  }
}
