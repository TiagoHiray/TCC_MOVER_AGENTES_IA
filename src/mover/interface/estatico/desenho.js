/* desenho.js — utilidades compartilhadas pelos painéis: paleta, formatação pt-BR, canvas,
 * interpolação da pose no trajeto e as figuras (caminhão, seta, norte, escala).
 *
 * Referencial do mapa (o mesmo da cena): x = leste, y = norte, em metros; rumo em radianos,
 * anti-horário a partir do leste. A Vista leva o mapa para a tela (y para baixo).
 */

export const FONTE = "Inter, 'Segoe UI', system-ui, -apple-system, Roboto, sans-serif";
export const FONTE_MONO = "'JetBrains Mono', 'Cascadia Mono', Consolas, ui-monospace, monospace";

// Paleta em HSL (matiz, saturação %, luminosidade %). A mesma do painel.css.
export const PALETA = {
  chao: [165, 9, 10],
  asfalto: [220, 7, 25],
  bordaVia: [215, 10, 52],
  trajeto: [188, 72, 52],
  texto: [220, 16, 92],
  texto2: [220, 10, 66],
  texto3: [220, 8, 46],
  grade: [220, 14, 70],
  info: [205, 70, 62],
  atencao: [40, 95, 56],
  critico: [2, 82, 60],
  positivo: [150, 55, 48],
  negativo: [2, 75, 58],
  caminhao: [4, 78, 54],
  cabine: [4, 62, 38],
  fundoEscuro: [222, 25, 6],
};

export function cor(nome, alfa = 1) {
  const [h, s, l] = PALETA[nome] || PALETA.texto;
  return `hsl(${h} ${s}% ${l}% / ${alfa})`;
}

export function corNivel(nivel, alfa = 1) {
  return cor(nivel === "critico" ? "critico" : nivel === "atencao" ? "atencao" : "info", alfa);
}

export const NOME_NIVEL = { info: "info", atencao: "atenção", critico: "crítico" };
const NOME_EVENTO = {
  frenagem_brusca: "frenagem brusca",
  corte_aceleracao: "corte brusco de aceleração",
  arrancada_brusca: "arrancada brusca",
  soltura_freio: "soltura brusca do freio",
};

/** Nome curto de um problema do log (lombada quando a causa é irregularidade na via). */
export function nomeProblema(entrada) {
  if (entrada.causa === "irregularidade_via") return "provável lombada";
  return NOME_EVENTO[entrada.evento] || String(entrada.evento || "evento").replaceAll("_", " ");
}

export const maiuscula = (texto) => (texto ? texto[0].toUpperCase() + texto.slice(1) : "");
export const doisDigitos = (n) => String(n).padStart(2, "0");

const formatadores = new Map();
/** Número em pt-BR (vírgula decimal, sinal de menos tipográfico); "—" se não houver valor. */
export function fmt(valor, casas = 1) {
  if (valor === null || valor === undefined || !Number.isFinite(Number(valor))) return "—";
  let f = formatadores.get(casas);
  if (!f) {
    f = new Intl.NumberFormat("pt-BR", { minimumFractionDigits: casas, maximumFractionDigits: casas });
    formatadores.set(casas, f);
  }
  const texto = f.format(Number(valor));
  return texto === `-${f.format(0)}` ? f.format(0) : texto.replace("-", "−");
}

export const fmtTempo = (t, casas = 1) => `${fmt(t, casas)} s`;

/** "40,0–45,0 s" para logs; "51,4 s" (pico) para problemas. */
export function tempoDaEntrada(entrada) {
  if (entrada.tipo === "problema") return fmtTempo(entrada.t_pico);
  return `${fmt(entrada.t_ini)}–${fmt(entrada.t_fim)} s`;
}

/** '2026-10-03T14:36:35.427-03:00' -> '14:36:35'. */
export function horaCurta(hora) {
  const m = /T?(\d{2}:\d{2}:\d{2})/.exec(String(hora || ""));
  return m ? m[1] : "";
}

/** Ajusta o canvas ao tamanho na tela (com devicePixelRatio) e devolve o contexto em px de CSS. */
export function prepararCanvas(canvas) {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (!w || !h) return null; // escondido
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const largura = Math.round(w * dpr);
  const altura = Math.round(h * dpr);
  if (canvas.width !== largura || canvas.height !== altura) {
    canvas.width = largura;
    canvas.height = altura;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h, dpr };
}

/** Último índice i com ts[i] <= t (-1 se nenhum). ts em ordem crescente. */
export function indiceAte(ts, t) {
  let lo = 0;
  let hi = ts.length - 1;
  let r = -1;
  while (lo <= hi) {
    const m = (lo + hi) >> 1;
    if (ts[m] <= t) {
      r = m;
      lo = m + 1;
    } else {
      hi = m - 1;
    }
  }
  return r;
}

/** a - b no intervalo (-π, π]. */
export function difAngulo(a, b) {
  let d = (a - b) % (2 * Math.PI);
  if (d > Math.PI) d -= 2 * Math.PI;
  if (d <= -Math.PI) d += 2 * Math.PI;
  return d;
}

/** Pose interpolada no trajeto {t, x, y, rumo} (rumo em radianos). */
export function poseNoTempo(tr, t) {
  const n = tr.t.length;
  if (!n || t === null || t === undefined) return null;
  if (t <= tr.t[0]) return { x: tr.x[0], y: tr.y[0], rumo: tr.rumo[0] };
  if (t >= tr.t[n - 1]) return { x: tr.x[n - 1], y: tr.y[n - 1], rumo: tr.rumo[n - 1] };
  const i = indiceAte(tr.t, t);
  const f = (t - tr.t[i]) / (tr.t[i + 1] - tr.t[i] || 1);
  return {
    x: tr.x[i] + f * (tr.x[i + 1] - tr.x[i]),
    y: tr.y[i] + f * (tr.y[i + 1] - tr.y[i]),
    rumo: tr.rumo[i] + f * difAngulo(tr.rumo[i + 1], tr.rumo[i]),
  };
}

/** Valor da série (ts, vs) no instante t, interpolado; null fora dos dados. */
export function valorNoTempo(ts, vs, t) {
  const n = ts.length;
  if (!n || t < ts[0] || t > ts[n - 1]) return null;
  const i = indiceAte(ts, t);
  if (i >= n - 1 || ts[i] === t) return vs[i];
  const a = vs[i];
  const b = vs[i + 1];
  if (a === null || b === null) return a ?? b;
  return a + ((b - a) * (t - ts[i])) / (ts[i + 1] - ts[i]);
}

/**
 * Do mapa para a tela: o ponto `centro` vai para (cx, cy), `escala` em px por metro e
 * `rotacao` (radianos, anti-horária) aplicada ao mapa antes de inverter o y.
 */
export class Vista {
  constructor(cx, cy, escala, centroX, centroY, rotacao = 0) {
    Object.assign(this, { cx, cy, escala, centroX, centroY, rotacao });
    this.c = Math.cos(rotacao);
    this.s = Math.sin(rotacao);
  }

  /** Aplica a transformação no contexto (larguras de linha passam a ser em metros). */
  aplicar(ctx) {
    ctx.translate(this.cx, this.cy);
    ctx.scale(this.escala, -this.escala);
    ctx.rotate(this.rotacao);
    ctx.translate(-this.centroX, -this.centroY);
  }

  paraTela(x, y) {
    const dx = x - this.centroX;
    const dy = y - this.centroY;
    return [this.cx + (this.c * dx - this.s * dy) * this.escala, this.cy - (this.s * dx + this.c * dy) * this.escala];
  }

  /** Ângulo na tela (y para baixo) de uma direção do mapa. */
  anguloNaTela(rumo) {
    return -(rumo + this.rotacao);
  }
}

/** Traça (sem pintar) o trajeto entre os instantes ta e tb, com as pontas interpoladas. */
export function tracarTrajeto(ctx, tr, ta, tb) {
  if (!(tb > ta) || !tr.t.length) return false;
  const a = poseNoTempo(tr, ta);
  const b = poseNoTempo(tr, tb);
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  for (let i = indiceAte(tr.t, ta) + 1; i < tr.t.length && tr.t[i] < tb; i++) ctx.lineTo(tr.x[i], tr.y[i]);
  ctx.lineTo(b.x, b.y);
  return true;
}

function retangulo(ctx, x, y, w, h, r) {
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(x, y, w, h, r);
  else ctx.rect(x, y, w, h);
}

/** Caminhão visto de cima, em metros (com a Vista já aplicada). A frente aponta para o rumo. */
export function desenharCaminhao(ctx, x, y, rumo, comprimento = 8.5, largura = 2.5) {
  const c = comprimento;
  const l = largura;
  ctx.save();
  ctx.translate(x + 0.35, y - 0.35); // sombra deslocada sempre para o mesmo lado
  ctx.rotate(rumo);
  ctx.fillStyle = "rgba(0, 0, 0, 0.38)";
  retangulo(ctx, -c / 2, -l / 2, c, l, 0.5);
  ctx.fill();
  ctx.restore();

  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(rumo);
  ctx.fillStyle = cor("caminhao");
  retangulo(ctx, -c / 2, -l / 2, c, l, 0.45);
  ctx.fill();
  ctx.fillStyle = cor("cabine"); // cabine na frente (+x)
  retangulo(ctx, c / 2 - 2.3, -l / 2, 2.3, l, 0.45);
  ctx.fill();
  ctx.fillStyle = "rgba(205, 232, 255, 0.85)"; // para-brisa
  ctx.fillRect(c / 2 - 0.8, -l / 2 + 0.28, 0.42, l - 0.56);
  ctx.strokeStyle = "rgba(255, 255, 255, 0.4)"; // escada no teto
  ctx.lineWidth = 0.1;
  ctx.beginPath();
  ctx.moveTo(-c / 2 + 0.5, -0.45);
  ctx.lineTo(c / 2 - 2.7, -0.45);
  ctx.moveTo(-c / 2 + 0.5, 0.45);
  ctx.lineTo(c / 2 - 2.7, 0.45);
  for (let s = -c / 2 + 0.9; s < c / 2 - 2.7; s += 0.7) {
    ctx.moveTo(s, -0.45);
    ctx.lineTo(s, 0.45);
  }
  ctx.stroke();
  ctx.restore();
}

/** Seta do caminhão no mapa da volta (em px), apontando para `anguloTela`. */
export function desenharSeta(ctx, sx, sy, anguloTela, tamanho = 8) {
  ctx.save();
  ctx.translate(sx, sy);
  ctx.beginPath();
  ctx.arc(0, 0, tamanho * 1.55, 0, 2 * Math.PI);
  ctx.fillStyle = cor("caminhao", 0.2);
  ctx.fill();
  ctx.rotate(anguloTela);
  ctx.beginPath();
  ctx.moveTo(tamanho, 0);
  ctx.lineTo(-tamanho * 0.75, tamanho * 0.7);
  ctx.lineTo(-tamanho * 0.35, 0);
  ctx.lineTo(-tamanho * 0.75, -tamanho * 0.7);
  ctx.closePath();
  ctx.fillStyle = cor("caminhao");
  ctx.fill();
  ctx.lineWidth = 1.4;
  ctx.strokeStyle = "rgba(255, 255, 255, 0.92)";
  ctx.stroke();
  ctx.restore();
}

/** Marcador de problema (em px): cheio se publicado, vazado se ainda é previsão. */
export function desenharMarcador(ctx, sx, sy, nivel, previsto, raio = 5) {
  ctx.save();
  ctx.beginPath();
  ctx.arc(sx, sy, raio * 2, 0, 2 * Math.PI);
  ctx.fillStyle = corNivel(nivel, previsto ? 0.1 : 0.2);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(sx, sy, raio, 0, 2 * Math.PI);
  if (previsto) {
    ctx.setLineDash([2.5, 2]);
    ctx.lineWidth = 2;
    ctx.strokeStyle = corNivel(nivel);
    ctx.fillStyle = cor("fundoEscuro", 0.7);
    ctx.fill();
    ctx.stroke();
  } else {
    ctx.fillStyle = corNivel(nivel);
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = cor("fundoEscuro", 0.9);
    ctx.stroke();
  }
  ctx.restore();
}

/** Rosa dos ventos mínima: círculo com a seta do norte apontando para `anguloTela`. */
export function desenharNorte(ctx, sx, sy, anguloTela, raio = 13) {
  ctx.save();
  ctx.translate(sx, sy);
  ctx.beginPath();
  ctx.arc(0, 0, raio, 0, 2 * Math.PI);
  ctx.fillStyle = cor("fundoEscuro", 0.62);
  ctx.fill();
  ctx.lineWidth = 1;
  ctx.strokeStyle = "rgba(255, 255, 255, 0.12)";
  ctx.stroke();
  ctx.rotate(anguloTela + Math.PI / 2); // a seta é desenhada apontando para cima
  ctx.beginPath();
  ctx.moveTo(0, -raio + 3);
  ctx.lineTo(4, 1);
  ctx.lineTo(-4, 1);
  ctx.closePath();
  ctx.fillStyle = cor("critico");
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(0, raio - 3);
  ctx.lineTo(4, 1);
  ctx.lineTo(-4, 1);
  ctx.closePath();
  ctx.fillStyle = cor("texto2", 0.7);
  ctx.fill();
  ctx.restore();
  ctx.save();
  ctx.font = `700 10px ${FONTE}`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillStyle = cor("texto");
  const r = raio + 8;
  ctx.fillText("N", sx + r * Math.cos(anguloTela), sy + r * Math.sin(anguloTela));
  ctx.restore();
}

/** Barra de escala com um comprimento redondo de até `maxPx`. (x, y) = canto inferior esquerdo. */
export function desenharEscala(ctx, x, y, escala, maxPx = 110) {
  let metros = 1;
  for (const opcao of [1, 2, 5, 10, 20, 25, 50, 100, 200, 500]) if (opcao * escala <= maxPx) metros = opcao;
  const px = metros * escala;
  ctx.save();
  ctx.strokeStyle = cor("texto", 0.75);
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(x, y - 5);
  ctx.lineTo(x, y);
  ctx.lineTo(x + px, y);
  ctx.lineTo(x + px, y - 5);
  ctx.stroke();
  ctx.font = `500 10.5px ${FONTE}`;
  ctx.fillStyle = cor("texto", 0.8);
  ctx.textAlign = "left";
  ctx.textBaseline = "bottom";
  ctx.fillText(`${metros} m`, x + px + 6, y + 1);
  ctx.restore();
}

/** Rótulo em "pílula" (px) com o fundo escuro translúcido. Devolve a largura. */
export function desenharRotulo(ctx, texto, sx, sy, { corTexto = cor("texto"), alinhar = "left", fonte = `600 11px ${FONTE}`,
                                                    evitar = null } = {}) {
  ctx.save();
  ctx.font = fonte;
  const w = ctx.measureText(texto).width + 12;
  const h = 19;
  let x = alinhar === "center" ? sx - w / 2 : alinhar === "right" ? sx - w : sx;
  let y = sy;
  // `evitar` = {x, y, w, h}: área ocupada por um elemento HTML sobre o canvas (o cartão de velocidade)
  if (evitar && x < evitar.x + evitar.w && x + w > evitar.x && y - h / 2 < evitar.y + evitar.h && y + h / 2 > evitar.y) {
    if (alinhar === "left") x = evitar.x + evitar.w + 6; // na mesma linha do marcador, logo depois do cartão
    else y = evitar.y + evitar.h + h / 2 + 6;
  }
  ctx.fillStyle = cor("fundoEscuro", 0.78);
  retangulo(ctx, x, y - h / 2, w, h, 6);
  ctx.fill();
  ctx.fillStyle = corTexto;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText(texto, x + 6, y + 0.5);
  ctx.restore();
  return w;
}

/** Mensagem centralizada num canvas vazio (sem sessão, sem cena...). */
export function desenharMensagem(ctx, w, h, titulo, detalhe = "") {
  ctx.save();
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = `600 13px ${FONTE}`;
  ctx.fillStyle = cor("texto2");
  ctx.fillText(titulo, w / 2, h / 2 - (detalhe ? 9 : 0));
  if (detalhe) {
    ctx.font = `400 11.5px ${FONTE}`;
    ctx.fillStyle = cor("texto3");
    const maximo = Math.max(20, Math.floor((w - 40) / 6.2));
    ctx.fillText(detalhe.length > maximo ? `${detalhe.slice(0, maximo - 1)}…` : detalhe, w / 2, h / 2 + 11);
  }
  ctx.restore();
}
