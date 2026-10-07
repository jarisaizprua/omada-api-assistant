// Entry point: wires the UI modules together.

import { getHealth, getStats, getToken, setAuthHandler, setToken } from "./api.js";
import { Chat } from "./chat.js";
import { createDrawer } from "./drawer.js";
import { createPalette } from "./palette.js";
import { prefs, store } from "./store.js";
import { FAMILIES, Topology } from "./topology.js";
import { $, clear, copyText, fmtDate, fmtNumber, h, icon, methodBadge } from "./ui.js";

const PROMPTS = [
  { topic: "Sitios", text: "¿Cómo creo un sitio nuevo y qué permisos necesito?" },
  { topic: "Automatización", text: "Dame un script en Python que liste los clientes conectados de un sitio." },
  { topic: "Redes inalámbricas", text: "¿Qué endpoints hay para crear y modificar un SSID?" },
  { topic: "Errores", text: "¿Qué significa el error -33101 y en qué operaciones aparece?" },
];

const els = {
  app: $("#app"),
  scrim: $("#scrim"),
  history: $("#history"),
  kbBody: $("#kbBody"),
  banner: $("#banner"),
  stage: $("#stage"),
  home: $("#home"),
  thread: $("#thread"),
  messages: $("#messages"),
  title: $("#chatTitle"),
  input: $("#input"),
  send: $("#sendBtn"),
  stop: $("#stopBtn"),
  composer: $("#composer"),
  railList: $("#railList"),
  railBtn: $("#railBtn"),
  railCount: $("#railCount"),
  themeBtn: $("#themeBtn"),
  heroCopy: $(".hero-copy"),
  heroSub: $("#heroSub"),
  legend: $("#legend"),
  prompts: $("#prompts"),
};

let topology = null;
let busy = false;

/* ───────────── Theme ───────────── */
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  prefs.set("theme", theme);
  const glyph = theme === "dark" ? "sun" : "moon";
  els.themeBtn.replaceChildren(icon(glyph));
  els.themeBtn.setAttribute("aria-label", theme === "dark" ? "Cambiar a tema claro" : "Cambiar a tema oscuro");
  topology?.setTheme(theme);
  topology?.resize();
}
const savedTheme = prefs.get("theme", null);
applyTheme(savedTheme || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark"));
els.themeBtn.addEventListener("click", () => applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));

/* ───────────── Drawer + palette ───────────── */
const drawer = createDrawer({
  onAsk: (op) => {
    const prefix = op.kind === "guide" ? "" : `${op.method} ${op.path}`;
    prefill(`Explícame cómo usar ${op.id} (${prefix}) con un ejemplo completo en Python.`);
  },
});
const palette = createPalette({ onOpen: (id) => drawer.open(id) });

/* ───────────── Rail (endpoints consulted) ───────────── */
function renderRail(ops) {
  clear(els.railList);
  els.railCount.hidden = ops.length === 0;
  els.railCount.textContent = String(ops.length);
  if (!ops.length) {
    els.railList.append(h("p", { class: "rail-empty" }, "Aquí aparecerán los endpoints que el asistente consulte durante la conversación."));
    return;
  }
  for (const op of ops) {
    els.railList.append(
      h(
        "button",
        { class: "op-card", type: "button", onclick: () => drawer.open(op.id) },
        h("div", { class: "top" }, methodBadge(op.method), h("span", { class: "summary" }, op.summary || op.title)),
        h("div", { class: "path" }, op.path),
        h(
          "div",
          { class: "tags" },
          h("span", { class: "tag-pill" }, op.tag),
          op.template ? h("span", { class: "tag-pill" }, "plantilla") : null,
          op.deprecated ? h("span", { class: "tag-pill warn" }, "obsoleto") : null
        )
      )
    );
  }
}

const wide = () => window.matchMedia("(min-width: 1181px)").matches;
function setRail(open, remember = true) {
  els.app.classList.toggle("rail-open", open);
  els.railBtn.setAttribute("aria-pressed", String(open));
  if (remember) prefs.set("rail", open ? "open" : "closed");
  setTimeout(() => topology?.resize(), 320);
}
els.railBtn.addEventListener("click", () => setRail(!els.app.classList.contains("rail-open")));
$("#railClose").addEventListener("click", () => setRail(false));

/* ───────────── Sidebar ───────────── */
function renderHistory() {
  clear(els.history);
  const list = store.list();
  if (!list.length) {
    els.history.append(h("p", { class: "history-empty" }, "Tus conversaciones aparecerán aquí."));
    return;
  }
  els.history.append(h("div", { class: "history-label" }, "Conversaciones"));
  for (const conv of list) {
    const del = h("button", { class: "del", type: "button", "aria-label": `Eliminar ${conv.title}` }, icon("trash", 15));
    del.addEventListener("click", (e) => {
      e.stopPropagation();
      const wasCurrent = chat.conversation.id === conv.id;
      store.remove(conv.id);
      if (wasCurrent) chat.startNew();
      renderHistory();
    });
    const open = h("button", { class: "open", type: "button", title: conv.title }, conv.title);
    open.addEventListener("click", () => { chat.load(conv.id); closeMenu(); });
    els.history.append(h("div", { class: "history-item", "aria-current": String(chat.conversation.id === conv.id) }, open, del));
  }
}

function openMenu() { els.app.classList.add("menu-open"); els.scrim.hidden = false; }
function closeMenu() { els.app.classList.remove("menu-open"); els.scrim.hidden = true; }
$("#menuBtn").addEventListener("click", openMenu);
els.scrim.addEventListener("click", closeMenu);

function renderKnowledgeBase(stats) {
  const semantic = String(stats.semantic || "").startsWith("enabled");
  const row = (label, value) => h("div", { class: "kb-row" }, h("span", {}, label), h("b", {}, value));
  els.kbBody.replaceChildren(
    row("Controller", stats.spec?.controller_version || "—"),
    row("Operaciones", fmtNumber(stats.operations)),
    row("Categorías", fmtNumber(stats.categories?.length)),
    row("Guías", fmtNumber(stats.guides)),
    row("Actualizada", fmtDate(stats.built_at) || "—"),
    h(
      "div",
      { class: "kb-status", title: stats.semantic },
      h("span", { class: `dot ${semantic ? "" : "warn"}` }),
      semantic ? "Búsqueda semántica activa" : "Solo palabras clave"
    )
  );
}

/* ───────────── Home: hero, legend, prompts ───────────── */
function renderHome(stats) {
  els.heroSub.textContent = "Las respuestas se basan en la especificación OpenAPI indexada, no en la memoria del modelo.";
  els.heroCopy.querySelector(".hero-stats")?.remove();
  els.heroCopy.append(
    h(
      "div",
      { class: "hero-stats" },
      h("div", {}, h("b", {}, fmtNumber(stats.operations)), "operaciones"),
      h("div", {}, h("b", {}, fmtNumber(stats.categories.length)), "categorías"),
      h("div", {}, h("b", {}, stats.spec?.controller_version || "—"), "controller")
    )
  );
  els.legend.replaceChildren(
    ...FAMILIES.map((f) => h("li", {}, h("i", { style: `background:${document.documentElement.dataset.theme === "light" ? f.light : f.dark}` }), f.label))
  );
}

function renderPrompts() {
  els.prompts.replaceChildren(
    ...PROMPTS.map((p) =>
      h("button", { class: "prompt", type: "button", onclick: () => chat.send(p.text) }, h("small", {}, p.topic), h("span", {}, p.text))
    )
  );
}

/* ───────────── Banner ───────────── */
function showBanner(kind, ...content) {
  els.banner.className = `banner ${kind}`;
  const close = h("button", { type: "button", "aria-label": "Cerrar aviso" }, icon("close", 16));
  close.addEventListener("click", () => (els.banner.hidden = true));
  els.banner.replaceChildren(icon("alert"), h("div", { class: "grow" }, ...content), close);
  els.banner.hidden = false;
}

function renderIssues(issues) {
  const find = (code) => issues.find((i) => i.code === code);
  if (find("index_missing")) {
    showBanner("bad", "La base de conocimiento aún no existe. Ejecuta ", h("code", {}, "python manage.py build-index"), " en la terminal y recarga esta página.");
  } else if (find("openai_key_missing")) {
    showBanner("bad", "Falta ", h("code", {}, "OPENAI_API_KEY"), " en tu archivo ", h("code", {}, ".env"), ". Añádela y reinicia el servidor.");
  } else if (find("keyword_only")) {
    showBanner("warn", find("keyword_only").message, " — las consultas funcionan, pero con menor calidad.");
  } else {
    els.banner.hidden = true;
  }
}

/* ───────────── Chat ───────────── */
const chat = new Chat({
  messagesEl: els.messages,
  stageEl: els.stage,
  homeEl: els.home,
  threadEl: els.thread,
  titleEl: els.title,
  hooks: {
    onBusy(value) {
      busy = value;
      els.stop.hidden = !value;
      els.send.hidden = value;
      syncSend();
    },
    onOps(ops) {
      renderRail(ops);
      if (ops.length && wide() && prefs.get("rail", null) !== "closed") setRail(true, false);
    },
    onChange() { renderHistory(); },
    onHome(visible) {
      if (visible) topology?.start(); else topology?.stop();
      if (visible) setTimeout(() => topology?.resize(), 0);
    },
    onOpenOperation: (id) => drawer.open(id),
    onShowRail: () => setRail(true),
  },
});

function syncSend() {
  els.send.disabled = busy || !els.input.value.trim();
}
function autosize() {
  els.input.style.height = "auto";
  els.input.style.height = `${Math.min(els.input.scrollHeight, 200)}px`;
}
function prefill(text) {
  els.input.value = text;
  autosize();
  syncSend();
  els.input.focus();
}

els.input.addEventListener("input", () => { autosize(); syncSend(); });
els.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    els.composer.requestSubmit();
  }
});
els.composer.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = els.input.value;
  if (!text.trim() || busy) return;
  els.input.value = "";
  autosize();
  syncSend();
  chat.send(text);
});
els.stop.addEventListener("click", () => chat.stop());
$("#newChat").addEventListener("click", () => { chat.startNew(); closeMenu(); els.input.focus(); });

/* ───────────── Shortcuts ───────────── */
const isMac = /Mac|iPhone|iPad/.test(navigator.platform);
$("#kbdHint").textContent = isMac ? "⌘ K" : "Ctrl K";
$("#searchBtn").addEventListener("click", () => palette.open());
document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
    e.preventDefault();
    palette.open();
  } else if (e.key === "/" && !/INPUT|TEXTAREA/.test(document.activeElement?.tagName || "")) {
    e.preventDefault();
    els.input.focus();
  }
});

/* ───────────── Access token ───────────── */
const tokenDlg = $("#tokenDlg");
setAuthHandler(() => { if (!tokenDlg.open) tokenDlg.showModal(); });
$("#tokenForm").addEventListener("submit", () => {
  setToken($("#tokenInput").value.trim());
  location.reload();
});

/* ───────────── Boot ───────────── */
async function boot() {
  renderPrompts();
  renderHistory();
  chat.render();
  renderRail([]);

  let health;
  try {
    health = await getHealth();
  } catch {
    showBanner("bad", "No se pudo contactar con el servidor. ¿Sigue en ejecución?");
    return;
  }
  renderIssues(health.issues || []);
  $("#kbBody").textContent = "";
  if (health.auth_required && !getToken()) {
    tokenDlg.showModal();
    return;
  }
  if (!health.index_ready) {
    els.kbBody.textContent = "Sin construir";
    return;
  }

  try {
    const stats = await getStats();
    renderKnowledgeBase(stats);
    renderHome(stats);
    palette.setTotal(stats.operations);
    const canvas = $("#topology");
    topology = new Topology(canvas, $("#topoTip"), {
      onPick: (tag) => chat.send(`Muéstrame los endpoints de la categoría «${tag}» y para qué sirve cada uno.`),
    });
    topology.setTheme(document.documentElement.dataset.theme);
    topology.setData(stats.categories);
    if (!chat.conversation.messages.length) topology.start();
  } catch (err) {
    if (err.message && !/Token/.test(err.message)) showBanner("bad", err.message);
  }
}

boot();

// Keep a reference for debugging in the browser console.
window.__omada = { chat, copyText };
