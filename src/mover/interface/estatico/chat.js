/* chat.js — painel 4: perguntas do operador sobre o log (POST /chat).
 *
 * A resposta vem curta e cita as entradas como (#id, tempo); cada citação vira um botão que rola
 * o log até a entrada. O selo mostra se a resposta veio do LLM ou da busca no log (provedor
 * falso, LLM fora do ar ou resposta recusada pela guarda de números) e o motivo, se houver.
 */
import { NOME_NIVEL, fmt } from "./desenho.js";

const SUGESTOES = ["Teve algum problema?", "O que vem pela frente?", "Qual foi a velocidade máxima?", "Resuma a volta até agora"];
const MAX_TURNOS = 6; // últimas falas enviadas como histórico (3 rodadas)
const MAX_TEXTO_TURNO = 1900;
const CITACAO = /\(#(\d+)[^)]*\)/g;

function el(tag, classe, texto) {
  const elemento = document.createElement(tag);
  if (classe) elemento.className = classe;
  if (texto !== undefined && texto !== null) elemento.textContent = texto;
  return elemento;
}

export class PainelChat {
  constructor({ conversa, sugestoes, form, entrada, botao, aoCitar }) {
    Object.assign(this, { conversa, sugestoes, form, entrada, botao, aoCitar });
    this.historico = [];
    this.ocupado = false;
    form.addEventListener("submit", (ev) => {
      ev.preventDefault();
      this.perguntar(entrada.value);
    });
    entrada.addEventListener("input", () => this.atualizarBotao());
    for (const texto of SUGESTOES) {
      const b = el("button", "sugestao", texto);
      b.type = "button";
      b.addEventListener("click", () => this.perguntar(texto));
      sugestoes.append(b);
    }
    this.boasVindas();
    this.atualizarBotao();
  }

  boasVindas() {
    const li = el("li", "fala");
    li.dataset.papel = "assistente";
    li.append(el("p", "fala-texto",
      "Pergunte sobre o log: problemas, a previsão do próximo bloco, velocidade, curvas ou um trecho " +
      "(\u201centre 40 e 60 s\u201d, \u201cbloco 5\u201d). As respostas citam as entradas como (#id, tempo)."));
    this.conversa.append(li);
  }

  atualizarBotao() {
    this.botao.disabled = this.ocupado || !this.entrada.value.trim();
    this.entrada.disabled = this.ocupado;
    for (const b of this.sugestoes.querySelectorAll("button")) b.disabled = this.ocupado;
  }

  rolar() {
    this.conversa.scrollTo({ top: this.conversa.scrollHeight, behavior: "smooth" });
  }

  adicionarFala(papel, texto) {
    const li = el("li", "fala");
    li.dataset.papel = papel;
    li.append(el("p", "fala-texto", texto));
    this.conversa.append(li);
    this.rolar();
    return li;
  }

  async perguntar(texto) {
    const pergunta = String(texto || "").trim().slice(0, 500);
    if (!pergunta || this.ocupado) return;
    this.ocupado = true;
    this.entrada.value = "";
    this.atualizarBotao();
    this.adicionarFala("operador", pergunta);
    const pendente = el("li", "fala fala-pendente");
    pendente.dataset.papel = "assistente";
    pendente.setAttribute("aria-label", "Respondendo");
    pendente.append(el("span", "digitando"), el("span", "digitando"), el("span", "digitando"));
    this.conversa.append(pendente);
    this.rolar();
    try {
      const r = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pergunta, historico: this.historico.slice(-MAX_TURNOS) }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const resposta = await r.json();
      pendente.replaceWith(this.falaResposta(resposta));
      this.historico.push(
        { papel: "operador", texto: pergunta },
        { papel: "assistente", texto: String(resposta.resposta || "").slice(0, MAX_TEXTO_TURNO) },
      );
    } catch (erro) {
      const li = el("li", "fala fala-erro");
      li.dataset.papel = "assistente";
      li.append(el("p", "fala-texto", `Não consegui responder agora (${erro.message || erro}). Tente de novo.`));
      pendente.replaceWith(li);
    } finally {
      this.ocupado = false;
      this.atualizarBotao();
      this.rolar();
      this.entrada.focus();
    }
  }

  botaoCitacao(id, rotulo, info) {
    const b = el("button", "citacao", rotulo);
    b.type = "button";
    b.dataset.id = String(id);
    if (info) {
      b.dataset.nivel = info.nivel;
      if (info.previsao) b.dataset.previsao = "1";
      b.title = `#${id} · ${info.tipo === "problema" ? `problema ${NOME_NIVEL[info.nivel]}` : "log"} · bloco ${info.bloco}` +
        `${info.previsao ? " · previsão (ainda não percorrido)" : ""}. Clique para ver no log.`;
    } else {
      b.title = "Clique para ver no log";
    }
    b.addEventListener("click", () => this.aoCitar?.(id));
    return b;
  }

  falaResposta(r) {
    const li = el("li", "fala");
    li.dataset.papel = "assistente";
    const porId = new Map((r.citacoes || []).map((c) => [c.id, c]));
    const p = el("p", "fala-texto");
    const texto = String(r.resposta || "");
    const noTexto = new Set();
    let ultimo = 0;
    for (const m of texto.matchAll(CITACAO)) {
      if (m.index > ultimo) p.append(document.createTextNode(texto.slice(ultimo, m.index)));
      const id = Number(m[1]);
      noTexto.add(id);
      p.append(this.botaoCitacao(id, m[0].slice(1, -1), porId.get(id)));
      ultimo = m.index + m[0].length;
    }
    if (ultimo < texto.length) p.append(document.createTextNode(texto.slice(ultimo)));
    li.append(p);

    const fora = (r.citacoes || []).filter((c) => !noTexto.has(c.id));
    if (fora.length) {
      const linha = el("div", "fala-citacoes");
      for (const c of fora) linha.append(this.botaoCitacao(c.id, `#${c.id}, ${c.tempo}`, c));
      li.append(linha);
    }

    const meta = el("div", "fala-meta");
    const llm = r.origem === "llm";
    const selo = el("span", "selo", llm ? `LLM${r.provedor ? ` · ${r.provedor}` : ""}` : "busca no log");
    selo.dataset.origem = llm ? "llm" : "modelo";
    meta.append(selo, el("span", "fala-latencia", `${fmt(r.latencia_s, 2)} s`));
    if (r.contexto?.previsao_bloco !== null && r.contexto?.previsao_bloco !== undefined) {
      meta.append(el("span", "fala-latencia", `com previsão do bloco ${String(r.contexto.previsao_bloco).padStart(2, "0")}`));
    }
    li.append(meta);
    if (r.observacao) li.append(el("p", "fala-obs", r.observacao));
    return li;
  }
}
