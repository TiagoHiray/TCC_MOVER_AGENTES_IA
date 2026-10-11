/* log.js — painel 3: o log da camada agêntica.
 *
 * Três tipos de entrada:
 * - intervencao: decisão dos agentes para o trecho seguinte (opção escolhida, se veio do LLM ou da
 *   regra, o plano que o caminhão mandou e a economia prevista);
 * - log: o que aconteceu num trecho já percorrido (medido), em linguagem natural;
 * - problema: um evento de jerk medido (frenagem ou arrancada brusca).
 * Embaixo fica o cartão "próximo trecho" com a decisão mais recente. Todo texto entra com textContent.
 */
import { NOME_NIVEL, NOME_OPCAO, doisDigitos, fmt, horaCurta, maiuscula, nomeProblema, tempoDaEntrada } from "./desenho.js";

function el(tag, classe, texto) {
  const elemento = document.createElement(tag);
  if (classe) elemento.className = classe;
  if (texto !== undefined && texto !== null) elemento.textContent = texto;
  return elemento;
}

function seloOrigem(entrada) {
  const llm = entrada.texto_origem === "llm";
  const decisao = entrada.tipo === "intervencao";
  const rotulo = llm ? `LLM${entrada.provedor ? ` · ${entrada.provedor}` : ""}` : decisao ? "regra" : "texto-modelo";
  const selo = el("span", "selo", rotulo);
  selo.dataset.origem = llm ? "llm" : "modelo";
  selo.title = entrada.observacao
    ? `${decisao ? "Regra" : "Texto-modelo"}: ${entrada.observacao}`
    : llm
      ? decisao
        ? "Opção escolhida e texto escritos pelo Supervisor (LLM local), dentro das opções seguras e com os números conferidos"
        : "Texto escrito pelo LLM do Supervisor (números conferidos pela guarda)"
      : decisao
        ? "Decisão da regra (menor custo de combustível e tempo sem eventos bruscos)"
        : "Texto-modelo montado a partir dos fatos medidos";
  return selo;
}

function nomeElementoBruto(b) {
  const tipo = b.tipo === "frenagem_brusca" ? "frenagem brusca" : "arrancada brusca";
  return `${tipo}${b.nome_elemento ? (b.tipo === "frenagem_brusca" ? ` para a ${b.nome_elemento}` : ` depois da ${b.nome_elemento}`) : ""}`;
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
    this.nIntervencoes = 0;
    this.resumo = null;
    this.chavePrevisao = "";
    // Acompanha o fim da lista até o usuário rolar para cima (só a rolagem do usuário muda o modo).
    this.seguir = true;
    this.novas = 0;
    this.pendenteFim = false;
    this.ultimaAcaoUsuario = -Infinity;
    const marcar = (ev) => {
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
    li.append(el("strong", null, "O log aparece aqui."),
      el("span", null, " Os agentes decidem cada trecho de 10 s antes de o caminhão chegar nele e registram o que aconteceu."));
    this.lista.append(li);
  }

  limpar() {
    this.lista.replaceChildren();
    this.n = 0;
    this.nProblemas = 0;
    this.nIntervencoes = 0;
    this.resumo = null;
    this.chavePrevisao = "";
    this.vazio();
    this.atualizarContagem();
    this.definirSeguir(true);
  }

  atualizarContagem() {
    this.contagem.textContent = this.n
      ? `${this.n} entrada${this.n > 1 ? "s" : ""} · ${this.nIntervencoes} decis${this.nIntervencoes === 1 ? "ão" : "ões"} · ` +
        `${this.nProblemas} problema${this.nProblemas === 1 ? "" : "s"}`
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
    if (entrada.tipo === "intervencao") this.nIntervencoes += 1;
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
    const rotulo = e.tipo === "problema" ? NOME_NIVEL[e.nivel] : e.tipo === "intervencao" ? "decisão" : "log";
    const nivel = el("span", "selo-nivel", rotulo);
    nivel.dataset.nivel = e.nivel;
    nivel.dataset.tipo = e.tipo;
    cab.append(nivel, el("span", "entrada-tempo", tempoDaEntrada(e)));
    if (e.tipo === "intervencao") {
      cab.append(el("span", "entrada-bloco", `para ${fmt(e.t_ini, 0)}–${fmt(e.t_fim, 0)} s`));
    } else {
      cab.append(el("span", "entrada-bloco", `bloco ${doisDigitos(e.bloco)}`));
    }
    cab.append(el("span", "entrada-id", `#${e.id}`));
    li.append(cab);

    if (e.tipo === "intervencao") {
      const f = e.fatos || {};
      const ativa = e.opcao && e.opcao !== "nenhuma";
      li.dataset.ativa = ativa ? "1" : "0";
      const titulo = el("p", "entrada-titulo");
      titulo.append(document.createTextNode(ativa ? "Alteração " : ""));
      const opcao = el("span", "opcao", NOME_OPCAO[e.opcao] || e.opcao);
      opcao.dataset.opcao = e.opcao;
      titulo.append(opcao);
      if (e.brutas?.length) titulo.append(document.createTextNode(` · em vez de ${nomeElementoBruto(e.brutas[0])}`));
      li.append(titulo, el("p", "entrada-texto", e.texto));
      if (e.brutas?.length) {
        const numeros = el("div", "comparacao");
        const item = (rotuloItem, antes, depois, classe) => {
          const span = el("span", `comp ${classe || ""}`);
          span.append(el("small", null, rotuloItem), el("b", null, `${antes} → ${depois}`));
          numeros.append(span);
        };
        item("desaceleração", `${fmt(f.desacel_bruta_mps2)}`, `${fmt(f.desacel_escolhida_mps2)} m/s²`);
        item("eventos bruscos", `${f.eventos_brutos ?? "—"}`, `${f.eventos_escolhidos ?? "—"}`);
        item("diesel previsto", `${fmt(f.combustivel_bruto_ml, 0)}`, `${fmt(f.combustivel_escolhido_ml, 0)} mL`,
          (f.economia_prevista_ml ?? 0) > 0 ? "ganho" : "");
        item("tempo", "", `${(f.tempo_extra_s ?? 0) >= 0 ? "+" : ""}${fmt(f.tempo_extra_s)} s`);
        li.append(numeros);
      }
    } else if (e.tipo === "problema") {
      const f = e.fatos || {};
      li.append(el("p", "entrada-titulo", `${maiuscula(nomeProblema(e))} · jerk ${fmt(f.jerk_max_abs_mps3)} m/s³ a ${fmt(f.vel_kmh, 0)} km/h`));
      li.append(el("p", "entrada-texto", e.diagnostico || e.texto));
      if (e.acao) {
        const acao = el("p", "entrada-acao");
        acao.append(el("strong", null, "Ação: "), document.createTextNode(e.acao));
        li.append(acao);
      }
    } else {
      li.append(el("p", "entrada-texto", e.texto));
    }

    const rodape = el("div", "entrada-rodape");
    rodape.append(seloOrigem(e));
    if (e.tipo === "intervencao") {
      if (e.recomendada && e.recomendada !== e.opcao) {
        const troca = el("span", "selo", `regra: ${NOME_OPCAO[e.recomendada] || e.recomendada}`);
        troca.title = "O Supervisor (LLM) trocou a opção recomendada pela regra";
        rodape.append(troca);
      }
      const atrasadas = (e.aplicacao || []).filter((a) => !a.a_tempo).length;
      if (atrasadas) {
        const atraso = el("span", "selo selo-atraso", `${atrasadas} elemento(s) sem tempo de aplicar`);
        atraso.title = "A decisão chegou depois de o piloto começar a frenagem para o elemento";
        rodape.append(atraso);
      }
      if (e.latencia_s !== undefined) rodape.append(el("span", "entrada-hora", `${fmt(e.latencia_s, 1)} s`));
    } else {
      if (e.fonte === "substituto") {
        const selo = el("span", "selo", "simulador substituto");
        selo.title = "Medido no simulador substituto (sem o CARLA)";
        rodape.append(selo);
      }
      if (e.continuacao) rodape.append(el("span", "selo", "continuação"));
      const hora = horaCurta(e.hora_local);
      if (hora) rodape.append(el("span", "entrada-hora", hora));
    }
    li.append(rodape);
    return li;
  }

  /**
   * info: { estado: "sem-sessao" | "aguardando" | "pronta" | "encerrada", previsao, modo }
   */
  definirPrevisao(info) {
    const p = info.previsao;
    const chave = JSON.stringify([info.estado, info.modo, p?.id]);
    if (chave === this.chavePrevisao) return;
    this.chavePrevisao = chave;
    const c = this.cartao;
    c.replaceChildren();
    c.dataset.estado = info.estado;
    c.hidden = info.estado === "encerrada";
    const cab = el("div", "previsao-cab");
    cab.append(el("span", "previsao-titulo", "Próximo trecho"));
    if (info.estado === "sem-sessao") {
      cab.append(el("span", "previsao-detalhe", "sem corrida em andamento"));
      c.append(cab, el("p", "previsao-texto", "Rode o piloto (python -m mover.autonomo.dirigir) para o caminhão sair."));
      return;
    }
    if (info.modo === "bruto") {
      cab.append(el("span", "previsao-detalhe", "corrida sem agentes"));
      c.append(cab, el("p", "previsao-texto", "Corrida de comparação: o caminhão segue o próprio plano, sem alterações."));
      return;
    }
    if (info.estado === "aguardando" || !p) {
      const texto = el("p", "previsao-texto previsao-analisando", "Os agentes estão analisando o trecho seguinte");
      texto.append(el("span", "reticencias"));
      c.append(cab, texto, el("div", "barra-analise"));
      return;
    }
    const origem = p.texto_origem === "llm" ? "LLM" : "regra";
    cab.append(el("span", "previsao-detalhe", `${fmt(p.t_ini, 0)}–${fmt(p.t_fim, 0)} s · ${NOME_OPCAO[p.opcao] || p.opcao} (${origem})`));
    c.append(cab);
    const f = p.fatos || {};
    const elementos = (p.instancias || []).map((i) => i.nome).filter(Boolean);
    const linha = el("p", "previsao-texto");
    linha.textContent = (elementos.length ? `${elementos.slice(0, 4).join(", ")}: ` : "") +
      (p.brutas?.length
        ? `o caminhão planeja ${nomeElementoBruto(p.brutas[0])} (${fmt(p.brutas[0].acel_mps2)} m/s²); ` +
          `com a alteração, ` +
          `${f.eventos_escolhidos ?? 0} evento(s) e ${fmt(f.economia_prevista_ml, 0)} mL a menos.`
        : "sem decisões bruscas previstas.");
    c.append(linha);
    const legenda = el("div", "previsao-legenda");
    legenda.append(el("span", "leg-bruto", "plano do caminhão"), el("span", "leg-escolhido", "com a alteração"));
    c.append(legenda);
  }

  /** Cartão de fim de corrida, no fim da lista. */
  definirResumo(r) {
    this.resumo?.remove();
    const li = el("li", "resumo-volta");
    li.append(el("strong", null, "Corrida concluída"));
    const m = r.metricas || {};
    const niveis = r.problemas_por_nivel || {};
    const origem = r.decisoes_por_origem || {};
    const partes = [
      `${r.decisoes ?? 0} decisões (${origem.llm || 0} do LLM, ${origem.regra || 0} da regra)`,
      `${r.problemas ?? 0} problema${r.problemas === 1 ? "" : "s"} medido${r.problemas === 1 ? "" : "s"} (${niveis.critico || 0} crítico${(niveis.critico || 0) === 1 ? "" : "s"})`,
    ];
    if (m.combustivel_ml !== undefined) partes.push(`${fmt(m.combustivel_ml, 0)} mL de diesel (R$ ${fmt(m.custo_r, 2)})`);
    if (m.economia_medida_ml !== null && m.economia_medida_ml !== undefined) {
      partes.push(`${fmt(m.economia_medida_ml, 0)} mL a menos que o plano sem agentes (previsto)`);
    }
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

  /** Rola até a entrada e a destaca por um instante. */
  destacar(id) {
    const alvo = document.getElementById(`entrada-${id}`);
    if (!alvo) return false;
    const fimAlvo = alvo.offsetTop + alvo.offsetHeight;
    if (this.lista.scrollHeight - fimAlvo > this.lista.clientHeight) this.definirSeguir(false);
    alvo.scrollIntoView({ behavior: "smooth", block: "nearest" });
    alvo.classList.remove("destaque");
    void alvo.offsetWidth; // reinicia a animação
    alvo.classList.add("destaque");
    return true;
  }
}
