/* painel.js — página dos 4 painéis do MOVER (Etapa 4): simulação, dashboard, log e chat.
 *
 * De onde vêm os dados:
 * - WebSocket /ws: "historico" ao conectar e, ao vivo, "sessao", "entrada", "bloco", "previsao"
 *   e "estado" (pose e telemetria do caminhão a ~5 Hz);
 * - GET /cena: vias do mapa e trajeto alinhado (x = leste, y = norte), pedida ao abrir a página
 *   e a cada sessão nova;
 * - GET /camera/info: se há quadros recentes da câmera do CARLA, o painel 1 mostra a transmissão
 *   MJPEG; senão, a visão 2D de perseguição;
 * - POST /chat: painel 4 (chat.js).
 *
 * Entre dois estados o instante é extrapolado pelo fator de tempo do replay (no máximo 0,6 s) e a
 * pose é interpolada no trajeto da cena, para os desenhos andarem a ~30 quadros por segundo.
 */
import { NOME_NIVEL, doisDigitos, fmt, nomeProblema, poseNoTempo } from "./desenho.js";
import { Grafico } from "./graficos.js";
import { MapaVolta, VistaPerseguicao } from "./mapa.js";
import { PainelChat } from "./chat.js";
import { PainelLog } from "./log.js";

const $ = (id) => document.getElementById(id);
const EXTRAPOLACAO_MAX_S = 0.6;
const HORIZONTE_AVISO_S = 10;
const INTERVALO_CAMERA_MS = 1000;
const INTERVALO_CENA_MS = 5000;
const QUADRO_MS = 33;
const NOMES_MANOBRA = {
  cruzeiro: "velocidade constante",
  acelerar: "acelerando",
  frear: "freando",
  curva_esq: "curva à esquerda",
  curva_dir: "curva à direita",
  parar: "parado",
};

const telemetriaVazia = () => ({ t: [], speed_kmh: [], acc_long: [] });

const estado = {
  conexao: "conectando",
  sessao: null,
  entradas: [],
  porId: new Map(),
  blocos: [],
  telemetria: telemetriaVazia(),
  previsao: null,
  ultimo: null, // último estado do caminhão
  recebidoEm: 0,
  cena: null,
  cenaErro: null,
  camera: { ativa: false, transmitindo: false },
};
let versaoDados = 0; // muda quando entradas, previsão ou cena mudam

// ------------------------------------------------------------------ painéis
const log = new PainelLog({
  lista: $("lista-log"),
  previsao: $("previsao"),
  contagem: $("contagem-log"),
  botaoFim: $("ir-fim-log"),
});
const chat = new PainelChat({
  conversa: $("conversa"),
  sugestoes: $("sugestoes"),
  form: $("form-chat"),
  entrada: $("pergunta"),
  botao: $("enviar"),
  aoCitar: (id) => log.destacar(id),
});
const perseguicao = new VistaPerseguicao($("canvas-perseguicao"), $("hud"));
const mapa = new MapaVolta($("canvas-mapa"), (id) => log.destacar(id));
const hover = { t: null };
const graficoVelocidade = new Grafico($("canvas-velocidade"),
  { titulo: "Velocidade", unidade: "km/h", chave: "speed_kmh", faixa: [0, 40], casas: 0, cor: "trajeto" }, hover);
const graficoAceleracao = new Grafico($("canvas-aceleracao"),
  { titulo: "Aceleração longitudinal", unidade: "m/s²", chave: "acc_long", faixa: [-4, 4], simetrica: true, casas: 1,
    cor: "texto", sinal: true }, hover);

// ------------------------------------------------------------------ estado da volta
function ultimoBlocoPublicado() {
  return estado.blocos.length ? estado.blocos[estado.blocos.length - 1].bloco : null;
}

function fimPublicado() {
  return estado.blocos.length ? estado.blocos[estado.blocos.length - 1].t_fim : null;
}

function intervaloDoBloco(k) {
  const s = estado.sessao;
  const a = s.t0 + k * s.duracao_bloco_s;
  return [a, Math.min(a + s.duracao_bloco_s, s.t0 + s.duracao_volta_s)];
}

function anexarTelemetria(origem) {
  if (!origem || !origem.t) return;
  const destino = estado.telemetria;
  let ultimo = destino.t.length ? destino.t[destino.t.length - 1] : -Infinity;
  for (let i = 0; i < origem.t.length; i++) {
    if (origem.t[i] <= ultimo + 1e-6) continue;
    destino.t.push(origem.t[i]);
    destino.speed_kmh.push(origem.speed_kmh?.[i] ?? null);
    destino.acc_long.push(origem.acc_long?.[i] ?? null);
    ultimo = origem.t[i];
  }
}

function reiniciar(sessao) {
  estado.sessao = sessao ? { ...sessao } : null;
  estado.entradas = [];
  estado.porId.clear();
  estado.blocos = [];
  estado.telemetria = telemetriaVazia();
  estado.previsao = null;
  estado.ultimo = null;
  log.limpar();
  versaoDados++;
}

function adicionarEntrada(entrada, rolar = true) {
  if (!entrada || estado.porId.has(entrada.id)) return;
  estado.entradas.push(entrada);
  estado.porId.set(entrada.id, entrada);
  log.adicionar(entrada, rolar);
  versaoDados++;
}

function resumoLocal() {
  const problemas = estado.entradas.filter((e) => e.tipo === "problema");
  const contar = (lista, chave) => lista.reduce((acc, e) => ({ ...acc, [e[chave]]: (acc[e[chave]] || 0) + 1 }), {});
  return {
    entradas: estado.entradas.length,
    blocos_publicados: estado.blocos.length,
    problemas: problemas.length,
    problemas_por_nivel: contar(problemas, "nivel"),
    texto_origem: contar(estado.entradas, "texto_origem"),
    maior_espera_s: Math.max(0, ...estado.blocos.slice(1).map((b) => b.espera_s || 0)),
  };
}

function aplicarHistorico(m) {
  reiniciar(m.sessao);
  for (const entrada of m.entradas || []) adicionarEntrada(entrada, false);
  estado.blocos = (m.blocos || []).map((b) => ({ ...b }));
  anexarTelemetria(m.telemetria);
  estado.previsao = m.previsao || null;
  if (m.estado) aplicarEstado(m.estado);
  if (m.sessao?.encerrada) log.definirResumo(resumoLocal());
  log.rolarParaFim();
  atualizarPrevisao();
  carregarCena();
}

function aplicarSessao(m) {
  if (m.evento === "iniciada") {
    reiniciar(m.sessao);
    if (m.previsao) estado.previsao = m.previsao;
    carregarCena(); // o replay manda a cena (PUT /cena) antes de abrir a sessão
  } else if (m.evento === "encerrada") {
    if (estado.sessao && m.sessao && m.sessao.id !== estado.sessao.id) return;
    estado.sessao = { ...m.sessao };
    estado.previsao = null;
    log.definirResumo(m.resumo || resumoLocal());
    versaoDados++;
  }
  atualizarPrevisao();
}

function aplicarBloco(m) {
  const { telemetria, tipo, ...resumo } = m;
  if (estado.blocos.some((b) => b.bloco === m.bloco)) return;
  estado.blocos.push(resumo);
  anexarTelemetria(telemetria);
  if (estado.previsao && estado.previsao.bloco <= m.bloco) estado.previsao = null;
  if (estado.sessao) estado.sessao.proximo_bloco = m.bloco + 1;
  versaoDados++;
  atualizarPrevisao();
}

function aplicarPrevisao(m) {
  // a previsão do bloco 0 de uma sessão nova chega antes do "iniciada", que a traz de novo
  if (m.sessao && estado.sessao && m.sessao !== estado.sessao.id) return;
  const ultimo = ultimoBlocoPublicado();
  if (ultimo !== null && m.bloco <= ultimo) return; // já publicado
  const { tipo, ...previsao } = m;
  estado.previsao = previsao;
  versaoDados++;
  atualizarPrevisao();
}

function aplicarEstado(m) {
  const { tipo, ...dados } = m;
  estado.ultimo = dados;
  estado.recebidoEm = performance.now();
}

function tratarMensagem(m) {
  switch (m.tipo) {
    case "historico": aplicarHistorico(m); break;
    case "sessao": aplicarSessao(m); break;
    case "entrada": adicionarEntrada(m.entrada); break;
    case "bloco": aplicarBloco(m); break;
    case "previsao": aplicarPrevisao(m); break;
    case "estado": aplicarEstado(m); break;
    default: break;
  }
}

function atualizarPrevisao() {
  const s = estado.sessao;
  let info;
  if (!s) {
    info = { estado: "sem-sessao" };
  } else if (s.encerrada) {
    info = { estado: "encerrada" };
  } else {
    const ultimo = ultimoBlocoPublicado();
    const proximo = ultimo === null ? 0 : ultimo + 1;
    const p = estado.previsao;
    if (p && p.bloco >= proximo) {
      info = { estado: "pronta", ...p, problemas: (p.entradas || []).filter((e) => e.tipo === "problema") };
    } else if (proximo >= s.n_blocos) {
      info = { estado: "ultimo" };
    } else {
      const [a, b] = intervaloDoBloco(proximo);
      info = { estado: "analisando", bloco: proximo, t_ini: a, t_fim: b };
    }
  }
  log.definirPrevisao(info);
}

// ------------------------------------------------------------------ tempo e pose
function tAtual() {
  const s = estado.sessao;
  const u = estado.ultimo;
  if (!s) return u ? u.sim_time : null;
  if (!u) return estado.blocos.length ? estado.blocos[0].t_ini : null;
  let t = u.sim_time;
  if (!s.encerrada && s.tempo_real !== false) {
    const passou = ((performance.now() - estado.recebidoEm) / 1000) * (s.fator_tempo || 1);
    t += Math.min(EXTRAPOLACAO_MAX_S, Math.max(0, passou));
  }
  const fim = fimPublicado();
  return fim !== null ? Math.min(t, fim) : t;
}

function poseAtual(t) {
  const tr = estado.cena?.tr;
  if (tr) return poseNoTempo(tr, t ?? tr.t[0]);
  const u = estado.ultimo;
  if (u && Number.isFinite(u.x) && Number.isFinite(u.y)) {
    return { x: u.x, y: -u.y, rumo: (-(u.yaw || 0) * Math.PI) / 180 }; // referencial do CARLA -> mapa
  }
  return null;
}

function blocoNoTempo(t) {
  const s = estado.sessao;
  const ultimo = ultimoBlocoPublicado();
  if (!s || t === null || ultimo === null) return null;
  return Math.max(0, Math.min(Math.floor((t - s.t0) / s.duracao_bloco_s + 1e-9), ultimo, s.n_blocos - 1));
}

function horizonteConhecido() {
  if (estado.previsao) return estado.previsao.t_fim;
  return fimPublicado();
}

let memoProblemas = { versao: -1, lista: [] };
function listaProblemas() {
  if (memoProblemas.versao === versaoDados) return memoProblemas.lista;
  const tr = estado.cena?.tr;
  const lista = [];
  const adicionar = (e, previsto) => lista.push({
    id: e.id, t: e.t_pico, nivel: e.nivel, previsto, nome: nomeProblema(e), pose: tr ? poseNoTempo(tr, e.t_pico) : null,
  });
  for (const e of estado.entradas) if (e.tipo === "problema") adicionar(e, false);
  for (const e of estado.previsao?.entradas || []) {
    if (e.tipo === "problema" && !estado.porId.has(e.id)) adicionar(e, true);
  }
  memoProblemas = { versao: versaoDados, lista };
  return lista;
}

// ------------------------------------------------------------------ cena
function prepararCena(c) {
  const tr = {
    t: Float64Array.from(c.trajeto.t),
    x: Float64Array.from(c.trajeto.x),
    y: Float64Array.from(c.trajeto.y),
    rumo: Float64Array.from(c.trajeto.rumo_graus, (g) => (g * Math.PI) / 180),
  };
  const porLargura = new Map();
  for (const via of c.vias) {
    const chave = Number(via.largura_m || 3.2).toFixed(1);
    if (!porLargura.has(chave)) porLargura.set(chave, new Path2D());
    const caminho = porLargura.get(chave);
    via.pontos.forEach(([x, y], i) => (i ? caminho.lineTo(x, y) : caminho.moveTo(x, y)));
  }
  const trajetoCompleto = new Path2D();
  for (let i = 0; i < tr.t.length; i++) {
    if (i) trajetoCompleto.lineTo(tr.x[i], tr.y[i]);
    else trajetoCompleto.moveTo(tr.x[i], tr.y[i]);
  }
  return {
    origem: c.origem,
    limites: c.limites,
    tr,
    vias: [...porLargura].map(([largura, caminho]) => ({ largura: Number(largura), caminho })),
    trajetoCompleto,
    duracao: tr.t.length ? tr.t[tr.t.length - 1] - tr.t[0] : 0,
    versao: performance.now(),
  };
}

let pedidoCena = null;
let tentativaCena = null;
function carregarCena() {
  if (pedidoCena) return pedidoCena;
  clearTimeout(tentativaCena);
  pedidoCena = (async () => {
    try {
      const r = await fetch("/cena", { cache: "no-store" });
      if (r.ok) {
        estado.cena = prepararCena(await r.json());
        estado.cenaErro = null;
        versaoDados++;
        return;
      }
      const corpo = await r.json().catch(() => ({}));
      estado.cenaErro = corpo?.detail?.mensagem || `HTTP ${r.status}`;
    } catch (erro) {
      estado.cenaErro = `sem resposta do servidor (${erro.message || erro})`;
    } finally {
      pedidoCena = null;
    }
    if (!estado.cena) tentativaCena = setTimeout(carregarCena, INTERVALO_CENA_MS);
  })();
  return pedidoCena;
}

// ------------------------------------------------------------------ câmera do CARLA
const img = $("camera");
// se a transmissão cair, a próxima verificação (1 s) reabre o MJPEG enquanto houver quadros novos
img.addEventListener("error", () => { estado.camera.transmitindo = false; });

function definirCamera(ativa) {
  if (ativa && !estado.camera.transmitindo) {
    img.src = `/camera.mjpg?t=${Date.now()}`;
    estado.camera.transmitindo = true;
  } else if (!ativa && estado.camera.transmitindo) {
    img.removeAttribute("src");
    estado.camera.transmitindo = false;
  }
  if (estado.camera.ativa !== ativa) {
    estado.camera.ativa = ativa;
    img.hidden = !ativa;
    $("canvas-perseguicao").hidden = ativa;
    $("fonte-simulacao").textContent = ativa ? "CARLA · câmera de perseguição" : "mapa 2D · sem câmera do CARLA";
    $("painel-simulacao").dataset.fonte = ativa ? "carla" : "mapa";
  }
}

async function verificarCamera() {
  try {
    const r = await fetch("/camera/info", { cache: "no-store" });
    definirCamera(r.ok && Boolean((await r.json()).ativa));
  } catch {
    definirCamera(false);
  }
  setTimeout(verificarCamera, INTERVALO_CAMERA_MS);
}

// ------------------------------------------------------------------ WebSocket
let tentativasWs = 0;
function conectar() {
  const protocolo = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${protocolo}//${location.host}/ws`);
  ws.addEventListener("open", () => {
    tentativasWs = 0;
    estado.conexao = "conectado";
  });
  ws.addEventListener("message", (ev) => {
    try {
      tratarMensagem(JSON.parse(ev.data));
    } catch (erro) {
      console.error("Mensagem do WebSocket com erro:", erro);
      window.__registrarErro?.(`mensagem do WS: ${erro.message || erro}`);
    }
  });
  ws.addEventListener("close", () => {
    estado.conexao = "desconectado";
    const espera = Math.min(10000, 500 * 2 ** tentativasWs++);
    setTimeout(conectar, espera);
  });
  ws.addEventListener("error", () => ws.close());
}

// ------------------------------------------------------------------ cabeçalho e HUD
function definirTexto(elemento, texto) {
  if (elemento.textContent !== texto) elemento.textContent = texto;
}

function definirDado(elemento, chave, valor) {
  if (elemento.dataset[chave] !== valor) elemento.dataset[chave] = valor;
}

const ROTULO_CONEXAO = { conectando: "conectando…", conectado: "ao vivo", desconectado: "reconectando…" };

function atualizarProgresso(t, k) {
  const barra = $("progresso");
  const s = estado.sessao;
  if (!s) {
    if (barra.childElementCount) barra.replaceChildren();
    return;
  }
  if (barra.childElementCount !== s.n_blocos) {
    barra.replaceChildren(...Array.from({ length: s.n_blocos }, (_, i) => {
      const seg = document.createElement("span");
      seg.className = "seg";
      seg.title = `bloco ${doisDigitos(i)}`;
      seg.append(document.createElement("i"));
      return seg;
    }));
  }
  const kp = estado.previsao ? estado.previsao.bloco : null;
  [...barra.children].forEach((seg, i) => {
    const e = s.encerrada || (k !== null && i < k) ? "feito" : i === k ? "atual" : i === kp ? "previsao" : "futuro";
    definirDado(seg, "estado", e);
    if (e === "atual") {
      const [a, b] = intervaloDoBloco(i);
      seg.firstChild.style.width = `${Math.round(100 * Math.min(1, Math.max(0, (t - a) / (b - a || 1))))}%`;
    }
  });
}

function atualizarCabecalho(t, k) {
  const s = estado.sessao;
  const conexao = $("chip-conexao");
  definirDado(conexao, "estado", estado.conexao);
  definirTexto(conexao.querySelector(".rotulo"), ROTULO_CONEXAO[estado.conexao]);
  definirTexto($("chip-sessao"), s ? (s.encerrada ? `volta concluída · ${s.id}` : `sessão ${s.id}`) : "sem sessão");
  const llm = $("chip-llm");
  definirTexto(llm, s ? `LLM ${s.llm}` : "LLM —");
  llm.title = s ? `Provedor: ${s.provedor || "padrão (.env ou YAML)"} · especialista de ML ${s.especialista_ml ? "ligado" : "desligado"}` : "";
  $("chip-sintetico").hidden = !(s && s.injetar_eventos);
  definirTexto($("chip-bloco"), k !== null ? `bloco ${doisDigitos(k)}` : s ? "antes do bloco 00" : "bloco —");
  let tempo = "— s";
  if (s) {
    const fator = s.fator_tempo && s.fator_tempo !== 1 ? ` · ×${fmt(s.fator_tempo, s.fator_tempo % 1 ? 1 : 0)}` : "";
    tempo = `${fmt(t ?? s.t0, 1)} / ${fmt(s.duracao_volta_s, 0)} s${fator}`;
  }
  definirTexto($("chip-tempo"), tempo);
  atualizarProgresso(t, k);
}

let chaveAviso = "";
function atualizarHud(t, k, problemas) {
  const u = estado.ultimo;
  const s = estado.sessao;
  definirTexto($("hud-velocidade"), u ? fmt(u.speed_kmh, 0) : "—");
  definirTexto($("hud-aceleracao"), u ? `${fmt(u.acc_long, 1)} m/s²` : "— m/s²");
  definirTexto($("hud-tempo"), t !== null ? `${fmt(t, 1)} s` : "— s");
  definirTexto($("hud-bloco"), k !== null ? `bloco ${doisDigitos(k)}` : "bloco —");
  definirTexto($("hud-manobra"), u ? NOMES_MANOBRA[u.manobra] || u.manobra || "" : "");

  // mensagem central do painel 1
  let titulo = "";
  let detalhe = "";
  if (estado.conexao !== "conectado") {
    titulo = "Sem conexão com o servidor";
    detalhe = "tentando reconectar…";
  } else if (!s) {
    titulo = "Aguardando uma volta";
    detalhe = "python -m mover.simulacao.voltas_com_agentes --iniciar-servidor";
  } else if (!u && !s.encerrada) {
    titulo = "Bloco 00 analisado · partindo";
  }
  const mensagem = $("mensagem-simulacao");
  mensagem.hidden = !titulo;
  definirTexto(mensagem.querySelector("strong"), titulo);
  definirTexto(mensagem.querySelector("code"), detalhe);
  mensagem.querySelector("code").hidden = !detalhe;

  // próximo problema nos 10 s à frente (ou acontecendo agora)
  const aviso = $("aviso-proximo");
  let alvo = null;
  if (t !== null && s && !s.encerrada) {
    for (const p of problemas) {
      if (p.t >= t - 1.0 && p.t <= t + HORIZONTE_AVISO_S && (!alvo || p.t < alvo.t)) alvo = p;
    }
  }
  if (!alvo) {
    if (!aviso.hidden) aviso.hidden = true;
    chaveAviso = "";
    return;
  }
  const falta = alvo.t - t;
  const quando = falta <= 0.3 ? "agora" : `em ${fmt(falta, 1)} s`;
  const chave = `${alvo.id}|${quando}`;
  if (chave === chaveAviso) return;
  chaveAviso = chave;
  aviso.hidden = false;
  aviso.dataset.nivel = alvo.nivel;
  definirTexto(aviso.querySelector(".aviso-quando"), quando);
  const selo = aviso.querySelector(".selo-nivel");
  selo.dataset.nivel = alvo.nivel;
  definirTexto(selo, NOME_NIVEL[alvo.nivel]);
  definirTexto(aviso.querySelector(".aviso-texto"), `${alvo.nome} · pico aos ${fmt(alvo.t, 1)} s`);
  aviso.querySelector(".aviso-previsto").hidden = !alvo.previsto;
}

// ------------------------------------------------------------------ laço de desenho
function desenhar() {
  const t = tAtual();
  const k = blocoNoTempo(t);
  const pose = poseAtual(t);
  const problemas = listaProblemas();
  const s = estado.sessao;
  const mensagem = !estado.cena
    ? { titulo: "Cena indisponível", detalhe: estado.cenaErro || "carregando…" }
    : null;
  const dados = { t: s ? t : null, pose, cena: estado.cena, problemas, horizonte: horizonteConhecido(), mensagem };
  if (!estado.camera.ativa) perseguicao.desenhar(dados);
  mapa.desenhar(dados);

  const blocoAtual = k !== null ? intervaloDoBloco(k) : null;
  const blocoPrevisao = estado.previsao ? [estado.previsao.t_ini, estado.previsao.t_fim] : null;
  const comum = {
    t: s ? t : null,
    t0: s ? s.t0 : estado.cena?.tr.t[0] || 0,
    duracao: s ? s.duracao_volta_s : estado.cena?.duracao || 150,
    duracaoBloco: s ? s.duracao_bloco_s : 10,
    serie: { publicada: estado.telemetria, previsao: estado.previsao?.telemetria },
    problemas,
    blocoAtual: blocoAtual && !s?.encerrada ? { t_ini: blocoAtual[0], t_fim: blocoAtual[1] } : null,
    blocoPrevisao: blocoPrevisao ? { t_ini: blocoPrevisao[0], t_fim: blocoPrevisao[1] } : null,
  };
  graficoVelocidade.desenhar({ ...comum, valorAtual: estado.ultimo?.speed_kmh });
  graficoAceleracao.desenhar({ ...comum, valorAtual: estado.ultimo?.acc_long });
  atualizarCabecalho(t, k);
  atualizarHud(t, k, problemas);
}

let ultimoQuadro = 0;
function laco(agora) {
  requestAnimationFrame(laco);
  if (agora - ultimoQuadro < QUADRO_MS) return;
  ultimoQuadro = agora;
  try {
    desenhar();
  } catch (erro) {
    window.__registrarErro?.(`desenho: ${erro.message || erro}`);
    throw erro;
  }
}

// ------------------------------------------------------------------ início
atualizarPrevisao();
definirCamera(false);
conectar();
carregarCena();
verificarCamera();
requestAnimationFrame(laco);
window.mover = { estado, chat, log }; // para depurar no console do navegador
document.documentElement.dataset.pronto = "1";
