/* log.js — painel 3: o log da camada agêntica.
 *
 * Cada entrada (log ou problema) vira um item com nível, tempo, bloco, origem do texto (LLM ou
 * texto-modelo) e selo de evento sintético. Embaixo fica o cartão da previsão: o próximo bloco
 * já analisado pelos agentes (ou "em análise…"). Todo texto entra com textContent.
 */
import { NOME_NIVEL, doisDigitos, fmt, horaCurta, maiuscula, nomeProblema, tempoDaEntrada } from "./desenho.js";

function el(tag, classe, texto) {
  const elemento = document.createElement(tag);
  if (classe) elemento.className = classe;
  if (texto !== undefined && texto !== null) elemento.textContent = texto;
  return elemento;
}

function seloOrigem(entrada) {
  const llm = entrada.texto_origem === "llm";
  const selo = el("span", "selo", llm ? `LLM${entrada.provedor ? ` · ${entrada.provedor}` : ""}` : "texto-modelo");
  selo.dataset.origem = llm ? "llm" : "modelo";
  selo.title = entrada.observacao
    ? `Texto-modelo: ${entrada.observacao}`
    : llm
      ? "Texto escrito pelo LLM do Supervisor (números conferidos pela guarda)"
      : "Texto-modelo montado a partir dos fatos medidos";
  return selo;
}

// Uma rolagem conta como "do usuário" se vier até este tempo depois de roda, toque ou tecla.
const JANELA_ACAO_USUARIO_MS = 1200;

export class PainelLog {
  constructor({ lista, previsao, contagem, botaoFim = null }) {
    this.lista = lista;
    this.cartao = previsao;
    this.contagem = contagem;
    this.botaoFim = botaoFim;
    this.n = 0;
    this.nProblemas = 0;
    this.resumo = null;
    this.chavePrevisao = "";
    // Acompanha o fim da lista até o usuário rolar para cima. Só a rolagem feita pelo usuário
    // muda este modo: a rolagem automática é suave e, no meio dela, a lista ainda não está no
    // fim (medir a posição nessa hora desligava o acompanhamento quando chegavam várias
    // entradas de uma vez).
    this.seguir = true;
    this.novas = 0; // entradas que chegaram enquanto o usuário lia outro trecho
    this.pendenteFim = false; // o resumo da volta chegou enquanto o usuário lia outro trecho
    this.ultimaAcaoUsuario = -Infinity;
    const marcar = (ev) => {
      // pointerdown só conta na própria lista (barra de rolagem), não nos itens
      if (ev.type !== "pointerdown" || ev.target === lista) this.ultimaAcaoUsuario = performance.now();
    };
    for (const tipo of ["wheel", "touchmove", "keydown", "pointerdown"]) {
      lista.addEventListener(tipo, marcar, { passive: true });
    }
    lista.addEventListener("scroll", () => {
      if (performance.now() - this.ultimaAcaoUsuario > JANELA_ACAO_USUARIO_MS) return;
      this.definirSeguir(this.pertoDoFim());
    }, { passive: true });
    this.botaoFim?.addEventListener("click", () => this.rolarParaFim(true));
    this.vazio();
  }

  vazio() {
    const li = el("li", "log-vazio");
    li.id = "log-vazio";
    li.append(el("strong", null, "O log aparece aqui."), el("span", null, " Cada bloco de 10 s é publicado quando o caminhão entra nele."));
    this.lista.append(li);
  }

  limpar() {
    this.lista.replaceChildren();
    this.n = 0;
    this.nProblemas = 0;
    this.resumo = null;
    this.vazio();
    this.atualizarContagem();
    this.definirSeguir(true);
  }

  atualizarContagem() {
    this.contagem.textContent = this.n
      ? `${this.n} entrada${this.n > 1 ? "s" : ""} · ${this.nProblemas} problema${this.nProblemas === 1 ? "" : "s"}`
      : "";
  }

  pertoDoFim() {
    return this.lista.scrollHeight - this.lista.scrollTop - this.lista.clientHeight < 80;
  }

  definirSeguir(seguir) {
    this.seguir = seguir;
    if (seguir) {
      this.novas = 0;
      this.pendenteFim = false;
    }
    this.atualizarBotaoFim();
  }

  atualizarBotaoFim() {
    if (!this.botaoFim) return;
    const mostrar = !this.seguir && (this.novas > 0 || this.pendenteFim);
    this.botaoFim.hidden = !mostrar;
    if (mostrar) {
      this.botaoFim.textContent = this.novas > 0
        ? `↓ ${this.novas} ${this.novas === 1 ? "entrada nova" : "entradas novas"}`
        : "↓ ir para o fim";
    }
  }

  rolarParaFim(suave = false) {
    this.definirSeguir(true);
    this.lista.scrollTo({ top: this.lista.scrollHeight, behavior: suave ? "smooth" : "auto" });
  }

  adicionar(entrada, rolar = true) {
    document.getElementById("log-vazio")?.remove();
    const item = this.item(entrada);
    if (this.resumo) this.lista.insertBefore(item, this.resumo);
    else this.lista.append(item);
    this.n += 1;
    if (entrada.tipo === "problema") this.nProblemas += 1;
    this.atualizarContagem();
    if (!rolar) return;
    if (this.seguir) {
      this.rolarParaFim(true);
    } else {
      this.novas += 1;
      this.atualizarBotaoFim();
    }
  }

  item(e) {
    const li = el("li", "entrada");
    li.id = `entrada-${e.id}`;
    li.dataset.tipo = e.tipo;
    li.dataset.nivel = e.nivel;
    li.tabIndex = -1;

    const cab = el("div", "entrada-cab");
    const nivel = el("span", "selo-nivel", e.tipo === "problema" ? NOME_NIVEL[e.nivel] : "log");
    nivel.dataset.nivel = e.nivel;
    nivel.dataset.tipo = e.tipo;
    cab.append(nivel, el("span", "entrada-tempo", tempoDaEntrada(e)), el("span", "entrada-bloco", `bloco ${doisDigitos(e.bloco)}`));
    cab.append(el("span", "entrada-id", `#${e.id}`));
    li.append(cab);

    if (e.tipo === "problema") {
      const f = e.fatos || {};
      li.append(el("p", "entrada-titulo", `${maiuscula(nomeProblema(e))} · jerk ${fmt(f.jerk_max_abs_mps3)} m/s³ a ${fmt(f.vel_kmh, 0)} km/h`));
      li.append(el("p", "entrada-texto", e.diagnostico || e.texto));
      if (e.acao) {
        const acao = el("p", "entrada-acao");
        acao.append(el("strong", null, "Ação: "), document.createTextNode(e.acao));
        li.append(acao);
      }
      if (e.ajuste) {
        const ajuste = el("p", "entrada-acao");
        ajuste.append(el("strong", null, "Ajuste no caminhão: "), document.createTextNode(e.ajuste.descricao));
        li.append(ajuste);
      }
    } else {
      li.append(el("p", "entrada-texto", e.texto));
    }

    const rodape = el("div", "entrada-rodape");
    rodape.append(seloOrigem(e));
    if (e.fonte && e.fonte !== "real") {
      const sintetico = el("span", "selo selo-sintetico", "evento sintético");
      sintetico.title = "Trecho com evento injetado por --injetar-eventos (a pose continua a real)";
      rodape.append(sintetico);
    }
    if (e.continuacao) rodape.append(el("span", "selo", "continuação"));
    if ((e.detectado_por || []).includes("especialista_ml")) {
      const ml = el("span", "selo", "ML: trecho atípico");
      ml.title = "O IsolationForest marcou quadros atípicos nesta janela";
      rodape.append(ml);
    }
    const hora = horaCurta(e.hora_local);
    if (hora) rodape.append(el("span", "entrada-hora", hora));
    li.append(rodape);
    return li;
  }

  /**
   * info: { estado: "sem-sessao" | "analisando" | "pronta" | "ultimo" | "encerrada",
   *         bloco, t_ini, t_fim, n_entradas, n_problemas, latencia_s, problemas: [entradas] }
   */
  definirPrevisao(info) {
    const chave = JSON.stringify([info.estado, info.bloco, info.n_entradas, (info.problemas || []).map((p) => p.id)]);
    if (chave === this.chavePrevisao) return;
    this.chavePrevisao = chave;
    const c = this.cartao;
    c.replaceChildren();
    c.dataset.estado = info.estado;
    c.hidden = info.estado === "encerrada";
    const trecho = info.bloco !== undefined && info.bloco !== null
      ? `bloco ${doisDigitos(info.bloco)} (${fmt(info.t_ini, 0)}–${fmt(info.t_fim, 0)} s)`
      : "";

    const cab = el("div", "previsao-cab");
    const titulo = el("span", "previsao-titulo", "Previsão");
    cab.append(titulo);
    if (info.estado === "sem-sessao") {
      cab.append(el("span", "previsao-detalhe", "sem volta em andamento"));
      c.append(cab, el("p", "previsao-texto", "Rode uma volta (python -m mover.simulacao.voltas_com_agentes) para a camada agêntica começar."));
      return;
    }
    if (info.estado === "ultimo") {
      cab.append(el("span", "previsao-detalhe", "último bloco"));
      c.append(cab, el("p", "previsao-texto", "O caminhão está no último bloco: não há trecho adiante para analisar."));
      return;
    }
    if (info.estado === "analisando") {
      cab.append(el("span", "previsao-detalhe", trecho));
      const texto = el("p", "previsao-texto previsao-analisando", "Os agentes estão analisando o próximo bloco");
      texto.append(el("span", "reticencias"));
      c.append(cab, texto, el("div", "barra-analise"));
      return;
    }
    // pronta
    cab.append(el("span", "previsao-detalhe", `${trecho} · analisado em ${fmt(info.latencia_s, 1)} s`));
    const n = info.n_entradas || 0;
    const np = info.n_problemas || 0;
    const resumo = el("p", "previsao-texto",
      `${n} ${n === 1 ? "entrada pronta" : "entradas prontas"} para quando o caminhão chegar; ` +
      (np ? `${np} problema${np === 1 ? "" : "s"} previsto${np === 1 ? "" : "s"}:` : "nenhum problema previsto."));
    c.append(cab, resumo);
    if (np) {
      const lista = el("ul", "previsao-lista");
      for (const p of info.problemas) {
        const li = el("li", "previsao-item");
        li.id = `previsao-${p.id}`;
        li.dataset.nivel = p.nivel;
        li.tabIndex = -1;
        const nivel = el("span", "selo-nivel", NOME_NIVEL[p.nivel]);
        nivel.dataset.nivel = p.nivel;
        nivel.dataset.tipo = "problema";
        const f = p.fatos || {};
        li.append(nivel, el("span", "previsao-item-texto",
          `${maiuscula(nomeProblema(p))} · pico aos ${fmt(p.t_pico)} s · jerk ${fmt(f.jerk_max_abs_mps3)} m/s³`),
          el("span", "entrada-id", `#${p.id}`));
        lista.append(li);
      }
      c.append(lista);
    }
  }

  /** Cartão de fim de volta, no fim da lista. */
  definirResumo(r) {
    this.resumo?.remove();
    const li = el("li", "resumo-volta");
    li.append(el("strong", null, "Volta concluída"));
    const niveis = r.problemas_por_nivel || {};
    const textos = r.texto_origem || {};
    const partes = [
      `${r.entradas} entradas em ${r.blocos_publicados} blocos`,
      `${r.problemas} problema${r.problemas === 1 ? "" : "s"} (${niveis.critico || 0} crítico${(niveis.critico || 0) === 1 ? "" : "s"}, ${niveis.atencao || 0} de atenção)`,
      `textos: ${textos.llm || 0} do LLM, ${textos.modelo || 0} texto-modelo`,
    ];
    if (r.maior_espera_s !== undefined) partes.push(`maior espera pela análise: ${fmt(r.maior_espera_s, 2)} s`);
    li.append(el("span", null, partes.join(" · ")));
    document.getElementById("log-vazio")?.remove();
    this.lista.append(li);
    this.resumo = li;
    if (this.seguir) {
      this.rolarParaFim(true);
    } else {
      this.pendenteFim = true;
      this.atualizarBotaoFim();
    }
  }

  /** Rola até a entrada (ou o item da previsão) e a destaca por um instante. */
  destacar(id) {
    const naLista = document.getElementById(`entrada-${id}`);
    const alvo = naLista || document.getElementById(`previsao-${id}`);
    if (!alvo) return false;
    // Quem pediu para ver uma entrada antiga não deve ser levado de volta ao fim pela próxima
    // entrada; o botão "↓ entradas novas" aparece no lugar.
    const fimAlvo = naLista ? naLista.offsetTop + naLista.offsetHeight : 0;
    if (naLista && this.lista.scrollHeight - fimAlvo > this.lista.clientHeight) this.definirSeguir(false);
    alvo.scrollIntoView({ behavior: "smooth", block: "nearest" });
    alvo.classList.remove("destaque");
    void alvo.offsetWidth; // reinicia a animação
    alvo.classList.add("destaque");
    return true;
  }
}
