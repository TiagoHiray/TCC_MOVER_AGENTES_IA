/* painel.js — página dos 4 painéis do MOVER: simulação, dashboard, log e chat.
 *
 * De onde vêm os dados:
 * - WebSocket /ws: "historico" ao conectar e, ao vivo, "sessao", "entrada", "telemetria" (pontos
 *   novos da série a 5 Hz e o estado do caminhão com as métricas) e "previsao" (a decisão mais
 *   recente dos agentes, com as curvas previstas);
 * - GET /cena: vias do mapa, rota e plano de missão (zonas e paradas), pedida ao abrir a página e a
 *   cada sessão nova;
 * - GET /camera/info: se há quadros recentes da câmera do CARLA, o painel 1 mostra a transmissão
 *   MJPEG; senão, a visão 2D de perseguição;
 * - GET /benchmark/ultimo: o resultado do benchmark mais recente (cartão de comparação);
 * - POST /chat: painel 4 (chat.js).
 *
 * Os gráficos e o rastro no mapa só têm o que o caminhão já percorreu: crescem com a telemetria.
 * Entre dois estados o instante é extrapolado (no máximo 0,6 s) para os desenhos andarem suaves.
 */
import { NOME_NIVEL, NOME_OPCAO, NOME_TIPO, fmt, nomeProblema, poseNoTempo } from "./desenho.js";
import { Grafico } from "./graficos.js";
import { MapaVolta, VistaPerseguicao } from "./mapa.js";
import { PainelChat } from "./chat.js";
import { PainelLog } from "./log.js";

const $ = (id) => document.getElementById(id);
const EXTRAPOLACAO_MAX_S = 0.6;
const AVISO_ATE_M = 80;
const INTERVALO_CAMERA_MS = 1000;
const INTERVALO_CENA_MS = 5000;
const QUADRO_MS = 33;
const COLUNAS_SERIE = ["t", "speed_kmh", "acc_long", "v_desejada_kmh", "v_limite_kmh", "x", "y", "rumo_graus", "d", "combustivel_ml"];

const serieVazia = () => Object.fromEntries([...COLUNAS_SERIE, "rumo"].map((c) => [c, []]));

const estado = {
  conexao: "conectando",
  sessao: null,
  entradas: [],
  porId: new Map(),
  decisoes: new Map(), // "lombada_5@0" -> { opcao, origem, id }
  serie: serieVazia(),
  previsao: null,
  ultimo: null,
  recebidoEm: 0,
  metricas: null,
  base: null,
  cena: null,
  cenaErro: null,
  camera: { ativa: false, transmitindo: false },
  benchmark: null,
};
let versaoDados = 0;

// ------------------------------------------------------------------ painéis
const log = new PainelLog({ lista: $("lista-log"), previsao: $("previsao"), contagem: $("contagem-log"), botaoFim: $("ir-fim-log") });
const chat = new PainelChat({
  conversa: $("conversa"), sugestoes: $("sugestoes"), form: $("form-chat"), entrada: $("pergunta"), botao: $("enviar"),
  aoCitar: (id) => log.destacar(id),
});
const perseguicao = new VistaPerseguicao($("canvas-perseguicao"), $("hud"));
const mapa = new MapaVolta($("canvas-mapa"), (id) => log.destacar(id));
const hover = { t: null };
const graficoVelocidade = new Grafico($("canvas-velocidade"), {
  titulo: "Velocidade", unidade: "km/h", chave: "speed_kmh", chavePrevisao: "v_kmh", faixa: [0, 40], casas: 0, cor: "trajeto",
  extras: [{ chave: "v_limite_kmh", cor: "limite", tracejado: true, degrau: true, largura: 1.1, alfa: 0.7 },
           { chave: "v_desejada_kmh", cor: "desejada", largura: 1.1, alfa: 0.8 }],
}, hover);
const graficoAceleracao = new Grafico($("canvas-aceleracao"), {
  titulo: "Aceleração longitudinal", unidade: "m/s²", chave: "acc_long", chavePrevisao: "a", faixa: [-2, 2], simetrica: true,
  casas: 1, cor: "texto", sinal: true,
}, hover);

// ------------------------------------------------------------------ estado da corrida
function anexarSerie(pontos) {
  if (!pontos || !pontos.t) return;
  const s = estado.serie;
  let ultimo = s.t.length ? s.t[s.t.length - 1] : -Infinity;
  for (let i = 0; i < pontos.t.length; i++) {
    if (pontos.t[i] <= ultimo + 1e-6) continue;
    for (const c of COLUNAS_SERIE) s[c].push(pontos[c]?.[i] ?? null);
    s.rumo.push(((pontos.rumo_graus?.[i] ?? 0) * Math.PI) / 180);
    ultimo = pontos.t[i];
  }
}

function reiniciar(sessao) {
  estado.sessao = sessao ? { ...sessao } : null;
  estado.entradas = [];
  estado.porId.clear();
  estado.decisoes.clear();
  estado.serie = serieVazia();
  estado.previsao = null;
  estado.ultimo = null;
  estado.metricas = null;
  estado.base = null;
  log.limpar();
  versaoDados++;
}

function registrarDecisao(e) {
  for (const inst of e.instancias || []) {
    const aplicada = (e.aplicacao || []).find((a) => a.chave === inst.chave);
    const opcao = aplicada?.opcao ?? e.opcao;
    if (opcao) estado.decisoes.set(inst.chave, { opcao, origem: e.texto_origem, id: e.id });
  }
}

function adicionarEntrada(entrada, rolar = true) {
  if (!entrada || estado.porId.has(entrada.id)) return;
  estado.entradas.push(entrada);
  estado.porId.set(entrada.id, entrada);
  if (entrada.tipo === "intervencao") registrarDecisao(entrada);
  log.adicionar(entrada, rolar);
  versaoDados++;
}

/** Resumo da corrida montado com o que a página já tem (página aberta depois do fim). */
function resumoLocal() {
  const contar = (lista, chave) => lista.reduce((acc, e) => ({ ...acc, [e[chave]]: (acc[e[chave]] || 0) + 1 }), {});
  const decisoes = estado.entradas.filter((e) => e.tipo === "intervencao");
  const problemas = estado.entradas.filter((e) => e.tipo === "problema");
  return { decisoes: decisoes.length, decisoes_por_origem: contar(decisoes, "texto_origem"), problemas: problemas.length,
           problemas_por_nivel: contar(problemas, "nivel"), metricas: estado.metricas || {} };
}

function aplicarEstado(m) {
  if (!m) return;
  estado.ultimo = m;
  estado.recebidoEm = performance.now();
}

function aplicarHistorico(m) {
  reiniciar(m.sessao);
  for (const entrada of m.entradas || []) adicionarEntrada(entrada, false);
  anexarSerie(m.serie);
  estado.previsao = m.previsao || null;
  estado.metricas = m.metricas || null;
  estado.base = m.linha_de_base || null;
  atualizarRotuloBase();
  aplicarEstado(m.estado);
  if (m.sessao?.encerrada) log.definirResumo(resumoLocal());
  log.rolarParaFim();
  atualizarPrevisao();
  carregarCena();
}

function aplicarSessao(m) {
  if (m.evento === "iniciada") {
    reiniciar(m.sessao);
    estado.base = m.linha_de_base || null;
    atualizarRotuloBase();
    carregarCena(true); // o piloto manda a cena (PUT /cena) antes de abrir a sessão
  } else if (m.evento === "encerrada") {
    if (estado.sessao && m.sessao && m.sessao.id !== estado.sessao.id) return;
    estado.sessao = { ...m.sessao };
    estado.previsao = null;
    log.definirResumo(m.resumo || resumoLocal());
    carregarBenchmark();
    versaoDados++;
  }
  atualizarPrevisao();
}

function aplicarTelemetria(m) {
  anexarSerie(m.pontos);
  aplicarEstado(m.estado);
  if (m.metricas) estado.metricas = m.metricas;
}

function aplicarPrevisao(m) {
  if (m.sessao && estado.sessao && m.sessao !== estado.sessao.id) return;
  const { tipo, ...previsao } = m;
  estado.previsao = previsao;
  versaoDados++;
  atualizarPrevisao();
}

function tratarMensagem(m) {
  switch (m.tipo) {
    case "historico": aplicarHistorico(m); break;
    case "sessao": aplicarSessao(m); break;
    case "entrada": adicionarEntrada(m.entrada); break;
    case "telemetria": aplicarTelemetria(m); break;
    case "previsao": aplicarPrevisao(m); break;
    default: break;
  }
}

function atualizarPrevisao() {
  const s = estado.sessao;
  const info = !s ? { estado: "sem-sessao" }
    : s.encerrada ? { estado: "encerrada" }
      : { estado: estado.previsao ? "pronta" : "aguardando", previsao: estado.previsao, modo: s.modo };
  log.definirPrevisao(info);
}

// ------------------------------------------------------------------ tempo e pose
function tAtual() {
  const s = estado.sessao;
  const u = estado.ultimo;
  if (!u) return null;
  let t = u.t;
  if (s && !s.encerrada && s.tempo_real !== false) {
    const passou = ((performance.now() - estado.recebidoEm) / 1000) * (s.fator_tempo || 1);
    t += Math.min(EXTRAPOLACAO_MAX_S, Math.max(0, passou));
  }
  return t;
}

function poseAtual(t) {
  const u = estado.ultimo;
  if (!u || !Number.isFinite(u.x)) return null;
  const rumo = ((u.rumo_graus || 0) * Math.PI) / 180;
  const extra = Math.max(0, (t ?? u.t) - u.t);
  const passo = ((u.speed_kmh || 0) / 3.6) * extra; // um pouco à frente do último estado, no rumo atual
  return { x: u.x + passo * Math.cos(rumo), y: u.y + passo * Math.sin(rumo), rumo };
}

let memoProblemas = { versao: -1, lista: [] };
function listaProblemas() {
  if (memoProblemas.versao === versaoDados) return memoProblemas.lista;
  const s = estado.serie;
  const tr = { t: s.t, x: s.x, y: s.y, rumo: s.rumo };
  const lista = estado.entradas.filter((e) => e.tipo === "problema").map((e) => ({
    id: e.id, t: e.t_pico, nivel: e.nivel, nome: nomeProblema(e), pose: s.t.length ? poseNoTempo(tr, e.t_pico) : null,
  }));
  memoProblemas = { versao: versaoDados, lista };
  return lista;
}

function marcasDecisao() {
  return estado.entradas.filter((e) => e.tipo === "intervencao")
    .map((e) => ({ t: e.t_aplicada ?? e.t_pedido, ativa: e.opcao && e.opcao !== "nenhuma" }));
}

/** O próximo elemento do plano à frente do caminhão (zona ou parada) e a decisão dos agentes para ele. */
function proximoElemento() {
  const c = estado.cena;
  const u = estado.ultimo;
  if (!c || !u) return null;
  const L = c.comprimento || 1;
  const d = u.d || 0;
  const volta = Math.floor(d / L);
  let melhor = null;
  const candidatos = [...c.zonas.map((z) => ({ ...z, inicio: z.s_ini })), ...c.paradas.map((p) => ({ ...p, tipo: "parada", inicio: p.s }))];
  for (const z of candidatos) {
    for (const k of [volta, volta + 1]) {
      const falta = k * L + z.inicio - d;
      if (falta < -2 || falta > AVISO_ATE_M) continue;
      if (!melhor || falta < melhor.falta) melhor = { ...z, falta, chave: `${z.id}@${k}` };
    }
  }
  if (!melhor) return null;
  return { ...melhor, decisao: estado.decisoes.get(melhor.chave) || null };
}

// ------------------------------------------------------------------ cena
function caminhoDe(xs, ys, fechar = false) {
  const caminho = new Path2D();
  for (let i = 0; i < xs.length; i++) (i ? caminho.lineTo(xs[i], ys[i]) : caminho.moveTo(xs[i], ys[i]));
  if (fechar) caminho.closePath();
  return caminho;
}

function prepararCena(c) {
  const porLargura = new Map();
  for (const via of c.vias || []) {
    const chave = Number(via.largura_m || 3.2).toFixed(1);
    if (!porLargura.has(chave)) porLargura.set(chave, new Path2D());
    const caminho = porLargura.get(chave);
    via.pontos.forEach(([x, y], i) => (i ? caminho.lineTo(x, y) : caminho.moveTo(x, y)));
  }
  const zonas = (c.zonas || []).map((z) => {
    const pts = z.pontos || [];
    const meio = pts.length ? { x: pts[Math.floor(pts.length / 2)][0], y: pts[Math.floor(pts.length / 2)][1] } : { x: 0, y: 0 };
    return { ...z, caminho: caminhoDe(pts.map((p) => p[0]), pts.map((p) => p[1])), meio };
  });
  return {
    origem: c.origem,
    limites: c.limites,
    comprimento: c.rota.comprimento_m,
    rota: caminhoDe(c.rota.x, c.rota.y, true),
    vias: [...porLargura].map(([largura, caminho]) => ({ largura: Number(largura), caminho })),
    zonas,
    paradas: c.paradas || [],
    largada: c.largada ? { x: c.largada.x, y: c.largada.y, rumo: ((c.largada.rumo_graus || 0) * Math.PI) / 180 } : null,
    versao: performance.now(),
  };
}

let pedidoCena = null;
let tentativaCena = null;
function carregarCena(forcar = false) {
  if (pedidoCena) return pedidoCena;
  if (estado.cena && !forcar) return Promise.resolve();
  clearTimeout(tentativaCena);
  pedidoCena = (async () => {
    try {
      const r = await fetch("/cena", { cache: "no-store" });
      if (r.ok) {
        estado.cena = prepararCena(await r.json());
        estado.cenaErro = null;
        versaoDados++;
        atualizarProgressoMarcas();
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

async function carregarBenchmark() {
  try {
    const r = await fetch("/benchmark/ultimo", { cache: "no-store" });
    estado.benchmark = r.ok ? await r.json() : null;
  } catch {
    estado.benchmark = null;
  }
  atualizarBenchmark();
}

// ------------------------------------------------------------------ câmera do CARLA
const img = $("camera");
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

// ------------------------------------------------------------------ cabeçalho, HUD e métricas
function definirTexto(elemento, texto) {
  if (elemento && elemento.textContent !== texto) elemento.textContent = texto;
}

function definirDado(elemento, chave, valor) {
  if (elemento.dataset[chave] !== valor) elemento.dataset[chave] = valor;
}

const ROTULO_CONEXAO = { conectando: "conectando…", conectado: "ao vivo", desconectado: "reconectando…" };

let chaveMarcas = "";
function atualizarProgressoMarcas() {
  const barra = $("progresso");
  const s = estado.sessao;
  const c = estado.cena;
  const voltas = s?.voltas || 1;
  const chave = `${s?.id}|${voltas}|${c?.versao}`;
  if (chave === chaveMarcas) return;
  chaveMarcas = chave;
  barra.replaceChildren(...Array.from({ length: s ? voltas : 0 }, (_, k) => {
    const volta = document.createElement("span");
    volta.className = "volta";
    volta.title = `volta ${k + 1}`;
    const cheio = document.createElement("i");
    volta.append(cheio);
    for (const z of c?.zonas || []) {
      const marca = document.createElement("b");
      marca.className = "marca-zona";
      marca.dataset.tipo = z.tipo;
      marca.style.left = `${(100 * z.s_ini) / (c.comprimento || 1)}%`;
      marca.style.width = `${Math.max(0.6, (100 * (z.s_fim - z.s_ini)) / (c.comprimento || 1))}%`;
      volta.append(marca);
    }
    return volta;
  }));
}

function atualizarProgresso() {
  atualizarProgressoMarcas();
  const s = estado.sessao;
  const c = estado.cena;
  const u = estado.ultimo;
  if (!s || !c) return;
  const L = c.comprimento || 1;
  const d = s.encerrada ? L * s.voltas : u?.d || 0;
  [...$("progresso").children].forEach((volta, k) => {
    const fracao = Math.min(1, Math.max(0, (d - k * L) / L));
    volta.firstChild.style.width = `${(100 * fracao).toFixed(2)}%`;
  });
}

function atualizarCabecalho(t) {
  const s = estado.sessao;
  const u = estado.ultimo;
  const conexao = $("chip-conexao");
  definirDado(conexao, "estado", estado.conexao);
  definirTexto(conexao.querySelector(".rotulo"), ROTULO_CONEXAO[estado.conexao]);
  definirTexto($("chip-sessao"), s ? `${s.encerrada ? "concluída" : "sessão"} ${s.id} · ${s.modo === "bruto" ? "sem agentes" : "com agentes"}` : "sem sessão");
  const sim = $("chip-simulador");
  sim.hidden = !s;
  definirTexto(sim, s ? (s.simulador === "carla" ? "CARLA" : "simulador substituto") : "");
  definirDado(sim, "simulador", s?.simulador || "");
  const llm = $("chip-llm");
  definirTexto(llm, s ? (s.modo === "bruto" ? "sem agentes" : `decisões: ${s.llm_decisoes || "regra"}`) : "LLM —");
  llm.title = s ? `LLM dos logs e do chat: ${s.llm_logs || "texto-modelo"}` : "";
  const voltaAtual = u ? Math.min((u.volta ?? 0) + 1, s?.voltas || 1) : 1;
  definirTexto($("chip-volta"), s ? `volta ${voltaAtual}/${s.voltas}` : "volta —");
  let tempo = "— s";
  if (s) {
    const fator = s.fator_tempo && s.fator_tempo !== 1 ? ` · ×${fmt(s.fator_tempo, s.fator_tempo % 1 ? 1 : 0)}` : "";
    tempo = `${fmt(t ?? 0, 1)} s${fator}`;
  }
  definirTexto($("chip-tempo"), tempo);
  atualizarProgresso();
}

let chaveAviso = "";
function atualizarHud(t) {
  const u = estado.ultimo;
  const s = estado.sessao;
  definirTexto($("hud-velocidade"), u ? fmt(u.speed_kmh, 0) : "—");
  definirTexto($("hud-alvo"), u ? `desejada ${fmt(u.v_desejada_kmh, 0)} · limite ${fmt(u.v_limite_kmh, 0)} km/h` : "");
  definirTexto($("hud-aceleracao"), u ? `${fmt(u.acc_long, 1)} m/s²` : "— m/s²");
  definirTexto($("hud-tempo"), t !== null ? `${fmt(t, 1)} s` : "— s");
  definirTexto($("hud-volta"), u && s ? `volta ${Math.min((u.volta ?? 0) + 1, s.voltas)}/${s.voltas}` : "volta —");

  let titulo = "";
  let detalhe = "";
  if (estado.conexao !== "conectado") {
    titulo = "Sem conexão com o servidor";
    detalhe = "tentando reconectar…";
  } else if (!s) {
    titulo = "Aguardando o caminhão";
    detalhe = "python -m mover.autonomo.dirigir";
  } else if (!u && !s.encerrada) {
    titulo = "Primeiro trecho decidido · partindo";
  }
  const mensagem = $("mensagem-simulacao");
  mensagem.hidden = !titulo;
  definirTexto(mensagem.querySelector("strong"), titulo);
  definirTexto(mensagem.querySelector("code"), detalhe);
  mensagem.querySelector("code").hidden = !detalhe;

  // próximo elemento do plano e o que os agentes decidiram para ele
  const aviso = $("aviso-proximo");
  const alvo = s && !s.encerrada ? proximoElemento() : null;
  if (!alvo) {
    if (!aviso.hidden) aviso.hidden = true;
    chaveAviso = "";
    return;
  }
  const quando = alvo.falta <= 3 ? "agora" : `em ${fmt(alvo.falta, 0)} m`;
  const decisao = alvo.decisao;
  const acao = s.modo === "bruto" ? "sem agentes"
    : decisao ? `${NOME_OPCAO[decisao.opcao] || decisao.opcao} (${decisao.origem === "llm" ? "LLM" : "regra"})` : "aguardando decisão";
  const chave = `${alvo.chave}|${quando}|${acao}`;
  if (chave === chaveAviso) return;
  chaveAviso = chave;
  aviso.hidden = false;
  aviso.dataset.estado = s.modo === "bruto" ? "bruto" : decisao ? (decisao.opcao === "nenhuma" ? "nenhuma" : "decidido") : "pendente";
  definirTexto(aviso.querySelector(".aviso-quando"), quando);
  const limite = alvo.tipo === "parada" ? `parar ${fmt(alvo.duracao_s, 0)} s` : `${fmt(alvo.v_max_kmh, 0)} km/h`;
  definirTexto(aviso.querySelector(".aviso-texto"), `${alvo.nome || NOME_TIPO[alvo.tipo] || alvo.tipo} · ${limite}`);
  definirTexto(aviso.querySelector(".aviso-acao"), acao);
}

/** Rótulo da comparação: o plano do caminhão sem agentes, previsto (numa corrida só não dá para medir os dois). */
function atualizarRotuloBase() {
  const b = estado.base;
  definirTexto($("m-base-rotulo"), "sem agentes (previsto)");
  $("m-base-rotulo").title = b?.fonte
    ? `Na mesma distância percorrida: ${b.fonte}. A comparação medida é a do benchmark.`
    : "O plano do caminhão sem os agentes, previsto pelo modelo do caminhão, na mesma distância percorrida";
}

function atualizarMetricas() {
  const m = estado.metricas;
  const s = estado.sessao;
  definirTexto($("m-combustivel"), m ? `${fmt(m.combustivel_ml, 0)} mL` : "—");
  definirTexto($("m-custo"), m ? `R$ ${fmt(m.custo_r, 2)} · ${fmt((m.co2_g || 0) / 1000, 2)} kg CO₂` : "");
  $("m-custo").title = m ? `custo a R$ ${fmt(s?.preco_r_l, 2)}/L (${s?.combustivel || "diesel"})` : "";
  const temBase = m && m.base_combustivel_ml !== null && m.base_combustivel_ml !== undefined && s?.modo === "agentes";
  definirTexto($("m-base"), temBase ? `${fmt(m.base_combustivel_ml, 0)} mL` : s?.modo === "bruto" ? "esta corrida" : "—");
  const economia = $("m-economia");
  if (temBase && m.economia_medida_ml !== null) {
    const sinal = m.economia_medida_ml >= 0 ? "−" : "+";
    definirTexto(economia, `${sinal}${fmt(Math.abs(m.economia_medida_ml), 0)} mL (${sinal}${fmt(Math.abs(m.economia_medida_pct ?? 0), 0)} %)`);
    definirDado(economia, "sinal", m.economia_medida_ml >= 0 ? "ganho" : "perda");
  } else {
    definirTexto(economia, "—");
    definirDado(economia, "sinal", "");
  }
  definirTexto($("m-decisoes"), m ? `${m.decisoes} · ${m.decisoes_llm} LLM` : "—");
  $("m-decisoes").title = m?.latencia_llm_media_s ? `latência média do LLM: ${fmt(m.latencia_llm_media_s, 1)} s` : "";
  definirTexto($("m-prevista"), m ? `${fmt(m.economia_prevista_ml, 0)} mL` : "—");
  definirTexto($("m-eventos"), m ? `${m.problemas_medidos}${temBase && m.base_eventos !== null ? ` (sem: ${m.base_eventos})` : ""}` : "—");
}

function atualizarBenchmark() {
  const b = estado.benchmark;
  const caixa = $("m-benchmark");
  if (!b || !b.extrapolacao) {
    definirTexto(caixa, "Sem benchmark ainda: python -m mover.autonomo.benchmark --voltas 3");
    caixa.title = "";
    return;
  }
  const e = b.extrapolacao;
  const dif = b.diferenca_pct || {};
  const comSinal = (v) => `${v > 0 ? "+" : ""}${fmt(v, 1)}`;
  definirTexto(caixa, `Benchmark (${b.voltas}+${b.voltas} voltas, ${b.simulador}): ${comSinal(dif.combustivel_ml)} % de diesel, ` +
    `${comSinal(dif.tempo_s)} % de tempo. Em ${e.voltas} voltas: ${fmt(e.economia_l, 1)} L a menos (R$ ${fmt(e.economia_r, 2)}, ` +
    `${fmt(e.economia_co2_kg, 1)} kg CO₂), ${e.eventos_sem_agentes} → ${e.eventos_com_agentes} eventos bruscos.`);
  caixa.title = `${b.data} · ${b.pasta || ""}`;
}

// ------------------------------------------------------------------ laço de desenho
function desenhar() {
  const t = tAtual();
  const pose = poseAtual(t);
  const problemas = listaProblemas();
  const s = estado.sessao;
  const serie = estado.serie;
  const mensagem = !estado.cena ? { titulo: "Cena indisponível", detalhe: estado.cenaErro || "carregando…" } : null;
  const rastro = { t: serie.t, x: serie.x, y: serie.y };
  const dados = { t, pose, cena: estado.cena, rastro, problemas, decisoes: estado.decisoes, volta: estado.ultimo?.volta ?? 0,
                  dAtual: estado.ultimo?.d ?? 0, mensagem };
  if (!estado.camera.ativa) perseguicao.desenhar(dados);
  mapa.desenhar(dados);
  const comum = {
    t,
    duracao: s?.duracao_prevista_s || 60,
    duracaoBloco: s?.duracao_bloco_s || 10,
    serie,
    previsao: s && !s.encerrada ? estado.previsao : null,
    problemas,
    decisoes: marcasDecisao(),
  };
  graficoVelocidade.desenhar({ ...comum, valorAtual: estado.ultimo?.speed_kmh });
  graficoAceleracao.desenhar({ ...comum, valorAtual: estado.ultimo?.acc_long });
  atualizarCabecalho(t);
  atualizarHud(t);
  atualizarMetricas();
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
carregarBenchmark();
verificarCamera();
requestAnimationFrame(laco);
window.mover = { estado, chat, log, NOME_NIVEL }; // para depurar no console do navegador
document.documentElement.dataset.pronto = "1";
