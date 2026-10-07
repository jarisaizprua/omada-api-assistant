// Conversation controller: sends messages, consumes the SSE stream and renders the thread.

import { streamChat } from "./api.js";
import { renderMarkdownInto } from "./markdown.js";
import { store } from "./store.js";
import { $, clear, fmtDuration, fmtNumber, h, icon, methodBadge } from "./ui.js";

function fmtUsd(value) {
  if (value < 0.01) return `US$ ${value.toFixed(4)}`;
  return `US$ ${value.toFixed(2)}`;
}

/** "hoy US$ 0.04 de 1.00" (your own usage against your daily limit), or null when there is no limit. */
function budgetText(b) {
  if (!b) return null;
  if (b.user_daily_limit_usd) return `hoy ${fmtUsd(b.user_today_usd)} de ${b.user_daily_limit_usd.toFixed(2)}`;
  if (b.requests_limit) return `hoy ${b.requests_today} de ${b.requests_limit} preguntas`;
  return null;
}


const MAX_TITLE = 52;
const RENDER_INTERVAL_MS = 60;

/** Human-readable description of a tool call, in Spanish. */
export function describeStep(step) {
  const a = step.args || {};
  const quote = (s) => `«${String(s ?? "").slice(0, 70)}»`;
  const running = step.status === "running";
  switch (step.name) {
    case "search_api":
      return running ? `Buscando ${quote(a.query)}…` : `Buscó ${quote(a.query)}`;
    case "get_operation":
      return running ? `Leyendo ${a.operation_id}…` : `Leyó el detalle de ${a.operation_id}`;
    case "list_categories":
      return running ? "Revisando categorías…" : "Revisó las categorías";
    case "search_guides":
      return running ? `Buscando en las guías ${quote(a.query)}…` : `Buscó en las guías ${quote(a.query)}`;
    default:
      return running ? `Ejecutando ${step.name}…` : step.name;
  }
}

/** Operations an answer most likely rests on: the ones read in detail, else the top search hits. */
export function chipOperations(msg) {
  const ops = msg.ops || [];
  const detailIds = new Set(
    (msg.steps || []).filter((s) => s.name === "get_operation" && s.status === "done" && s.opIds).flatMap((s) => s.opIds)
  );
  const detailed = ops.filter((o) => detailIds.has(o.id));
  return detailed.length ? detailed : ops.slice(0, 4);
}

class AssistantView {
  constructor(msg, chat) {
    this.msg = msg;
    this.chat = chat;
    this.stepsHost = h("div");
    this.textHost = h("div", { class: "md" });
    this.extraHost = h("div");
    this.el = h(
      "article",
      { class: "msg-assistant" },
      h("div", { class: "avatar" }, icon("spark")),
      h("div", { class: "msg-body" }, this.stepsHost, this.textHost, this.extraHost)
    );
    this._lastRender = 0;
    this._timer = null;
    this.renderSteps(false);
    this.renderText(false);
    this.renderExtras(false);
  }

  // --- steps (tool activity) -------------------------------------------------------
  renderSteps(live) {
    const steps = this.msg.steps || [];
    const previous = this.stepsHost.firstChild;
    const open = live ? (previous ? previous.open : true) : false; // keep the user's choice while streaming
    clear(this.stepsHost);
    if (!steps.length) return;
    const running = steps.some((s) => s.status === "running");
    const list = h(
      "ol",
      {},
      steps.map((s) =>
        h(
          "li",
          { class: `step ${s.status}` },
          h("span", { class: "state" }),
          h("span", {}, describeStep(s), s.summary && s.status !== "running" ? h("span", { class: "ms" }, `  ·  ${s.summary}`) : null),
          h("span", { class: "ms" }, s.ms != null ? fmtDuration(s.ms) : "")
        )
      )
    );
    const label = running || (live && !this.msg.content)
      ? "Consultando la base de conocimiento…"
      : `Consultó la base de conocimiento · ${steps.length} ${steps.length === 1 ? "paso" : "pasos"}`;
    const chevron = icon("chevron", 14);
    chevron.classList.add("chev");
    const details = h(
      "details",
      { class: "steps" },
      h("summary", {}, running || live ? h("span", { class: "spinner" }) : icon("spark", 14), label, chevron),
      list
    );
    details.open = open;
    this.stepsHost.append(details);
  }

  // --- answer text -----------------------------------------------------------------
  renderText(streaming) {
    renderMarkdownInto(this.textHost, this.msg.content || "");
    this.textHost.classList.toggle("cursor", Boolean(streaming));
    this._lastRender = performance.now();
  }

  scheduleText() {
    const wait = RENDER_INTERVAL_MS - (performance.now() - this._lastRender);
    if (wait <= 0) return this._flush();
    if (!this._timer) this._timer = setTimeout(() => this._flush(), wait);
  }

  _flush() {
    clearTimeout(this._timer);
    this._timer = null;
    this.renderText(true);
    this.chat.autoscroll();
  }

  // --- error / chips / usage ---------------------------------------------------------
  renderExtras(live) {
    clear(this.extraHost);
    const { msg } = this;
    if (msg.notice) this.extraHost.append(h("div", { class: "msg-note" }, msg.notice));
    if (msg.error) {
      this.extraHost.append(
        h(
          "div",
          { class: "msg-error", role: "alert" },
          icon("alert"),
          h("div", { class: "grow" }, msg.error.message),
          msg.error.code === "auth" ? null : h("button", { type: "button", onclick: () => this.chat.retry() }, "Reintentar")
        )
      );
    }
    if (live) return;
    const chips = chipOperations(msg);
    if (chips.length) {
      const total = (msg.ops || []).length;
      this.extraHost.append(
        h(
          "div",
          { class: "chips" },
          chips.slice(0, 5).map((op) =>
            h(
              "button",
              { class: "chip", type: "button", title: op.summary, onclick: () => this.chat.openOperation(op.id) },
              methodBadge(op.method),
              h("code", {}, op.path)
            )
          ),
          total > chips.length ? h("button", { class: "chip more", type: "button", onclick: () => this.chat.showRail() }, `+${total - chips.length} más`) : null
        )
      );
    }
    const u = msg.usage;
    if (u && (u.input_tokens || u.output_tokens)) {
      this.extraHost.append(
        h(
          "div",
          { class: "meta-line" },
          h("span", {}, `${fmtNumber(u.input_tokens)} tokens de entrada`),
          h("span", {}, `${fmtNumber(u.output_tokens)} de salida`),
          u.tool_calls ? h("span", {}, `${u.tool_calls} ${u.tool_calls === 1 ? "consulta" : "consultas"} al RAG`) : null,
          msg.elapsed_ms ? h("span", {}, fmtDuration(msg.elapsed_ms)) : null,
          typeof msg.cost_usd === "number" ? h("span", { title: "Costo estimado de esta respuesta" }, `≈ ${fmtUsd(msg.cost_usd)}`) : null,
          budgetText(msg.budget) ? h("span", { title: "Lo que llevas gastado hoy frente a tu límite diario" }, budgetText(msg.budget)) : null,
          msg.model ? h("span", {}, msg.model) : null
        )
      );
    }
  }

  finalize() {
    clearTimeout(this._timer);
    this._timer = null;
    this.renderText(false);
    this.renderSteps(false);
    this.renderExtras(false);
  }
}

export class Chat {
  constructor({ messagesEl, stageEl, homeEl, threadEl, titleEl, hooks }) {
    this.messagesEl = messagesEl;
    this.stageEl = stageEl;
    this.homeEl = homeEl;
    this.threadEl = threadEl;
    this.titleEl = titleEl;
    this.hooks = { onBusy() {}, onOps() {}, onChange() {}, onHome() {}, onOpenOperation() {}, onShowRail() {}, ...hooks };
    this.conversation = store.create();
    this.busy = false;
    this.abort = null;
    this.live = null;
  }

  // --- conversation lifecycle ------------------------------------------------------
  startNew() {
    this.stop();
    this.conversation = store.create();
    this.render();
    this.hooks.onChange(this.conversation);
  }

  load(id) {
    const found = store.get(id);
    if (!found) return this.startNew();
    this.stop();
    this.conversation = found;
    const last = found.messages[found.messages.length - 1];
    if (last && last.role === "user") {
      found.messages.push({ role: "assistant", content: "", steps: [], ops: [], error: { message: "La respuesta anterior no se completó.", code: "interrupted" } });
    }
    this.render();
    this.hooks.onChange(this.conversation);
  }

  allOps() {
    const seen = new Map();
    for (const m of [...this.conversation.messages].reverse()) {
      for (const op of m.ops || []) if (!seen.has(op.id)) seen.set(op.id, op);
    }
    return [...seen.values()];
  }

  // --- rendering ---------------------------------------------------------------------
  render() {
    const { messages } = this.conversation;
    clear(this.messagesEl);
    this.live = null;
    const empty = messages.length === 0;
    this.homeEl.hidden = !empty;
    this.threadEl.hidden = empty;
    this.titleEl.textContent = empty ? "Nueva conversación" : this.conversation.title;
    for (const msg of messages) {
      if (msg.role === "user") this.messagesEl.append(h("div", { class: "msg-user" }, msg.content));
      else {
        const view = new AssistantView(msg, this);
        view.finalize();
        this.messagesEl.append(view.el);
      }
    }
    this.hooks.onOps(this.allOps());
    this.hooks.onHome(empty);
    this.stageEl.scrollTop = empty ? 0 : this.stageEl.scrollHeight;
  }

  autoscroll(force = false) {
    const el = this.stageEl;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 140;
    if (force || nearBottom) el.scrollTop = el.scrollHeight;
  }

  openOperation(id) { this.hooks.onOpenOperation(id); }
  showRail() { this.hooks.onShowRail(); }

  // --- sending -----------------------------------------------------------------------
  async send(text) {
    const content = text.trim();
    if (!content || this.busy) return;
    const conv = this.conversation;
    if (!conv.messages.length) conv.title = content.length > MAX_TITLE ? `${content.slice(0, MAX_TITLE - 1)}…` : content;
    conv.messages.push({ role: "user", content });
    store.upsert(conv);
    this.hooks.onChange(conv);
    await this._run();
  }

  async retry() {
    if (this.busy) return;
    const { messages } = this.conversation;
    while (messages.length && messages[messages.length - 1].role === "assistant") messages.pop();
    if (!messages.length) return;
    await this._run();
  }

  stop() {
    this.abort?.abort();
  }

  async _run() {
    const conv = this.conversation;
    const msg = { role: "assistant", content: "", steps: [], ops: [], usage: null, error: null, notice: null };
    conv.messages.push(msg);
    this.render();
    const view = new AssistantView(msg, this);
    // replace the finalized placeholder rendered above by a live view
    this.messagesEl.lastElementChild.replaceWith(view.el);
    view.renderSteps(true);
    view.renderText(true);
    this.live = view;
    this.autoscroll(true);

    const history = conv.messages
      .slice(0, -1)
      .filter((m) => m.content && !m.error)
      .map(({ role, content }) => ({ role, content }));

    this.busy = true;
    this.hooks.onBusy(true);
    this.abort = new AbortController();
    let stoppedByUser = false;

    try {
      for await (const ev of streamChat(history, this.abort.signal)) {
        switch (ev.type) {
          case "start":
            msg.model = ev.model;
            break;
          case "token":
            msg.content += ev.text;
            view.scheduleText();
            break;
          case "tool_start":
            msg.steps.push({ id: ev.id, name: ev.name, args: ev.args, status: "running" });
            view.renderSteps(true);
            this.autoscroll();
            break;
          case "tool_result": {
            const step = msg.steps.find((s) => s.id === ev.id);
            if (step) {
              step.status = ev.ok ? "done" : "fail";
              step.summary = ev.summary;
              step.ms = ev.ms;
              step.opIds = (ev.operations || []).map((o) => o.id);
            }
            for (const op of ev.operations || []) if (!msg.ops.some((o) => o.id === op.id)) msg.ops.push(op);
            view.renderSteps(true);
            this.hooks.onOps(this.allOps());
            break;
          }
          case "notice":
            msg.notice = ev.message;
            break;
          case "error":
            msg.error = { message: ev.message, code: ev.code };
            break;
          case "done":
            msg.usage = ev.usage;
            msg.elapsed_ms = ev.elapsed_ms;
            msg.cost_usd = ev.cost_usd;
            msg.budget = ev.budget;
            break;
          default:
            break;
        }
      }
    } catch (err) {
      if (err?.name === "AbortError") {
        stoppedByUser = true;
      } else {
        msg.error = { message: `Se perdió la conexión con el servidor. ${err?.message || ""}`.trim(), code: "network" };
      }
    } finally {
      if (stoppedByUser && !msg.error) msg.notice = "Respuesta detenida.";
      if (!msg.content && !msg.error && !stoppedByUser) msg.error = { message: "El modelo no devolvió texto. Inténtalo de nuevo.", code: "empty" };
      this.busy = false;
      this.abort = null;
      this.live = null;
      view.finalize();
      store.upsert(conv);
      this.hooks.onBusy(false);
      this.hooks.onOps(this.allOps());
      this.hooks.onChange(conv);
      this.autoscroll();
    }
  }
}
