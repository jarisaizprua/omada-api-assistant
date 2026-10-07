// Operation detail drawer: the full documentation of one endpoint, straight from the index.

import { getOperation } from "./api.js";
import { $, copyText, clear, h, methodClass } from "./ui.js";

export function createDrawer({ onAsk }) {
  const wrap = $("#drawerWrap");
  const methodEl = $("#drawerMethod");
  const titleEl = $("#drawerTitle");
  const pathEl = $("#drawerPath");
  const metaEl = $("#drawerMeta");
  const bodyEl = $("#drawerBody");
  const tabsEl = $("#drawerTabs");
  const copyBtn = $("#drawerCopy");
  const askBtn = $("#drawerAsk");

  let detail = null;
  let tab = "header";
  let opener = null;

  function showTab(name) {
    tab = name;
    tabsEl.querySelectorAll("button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
    bodyEl.textContent = detail ? detail[name] || "—" : "";
    bodyEl.scrollTop = 0;
  }

  function close() {
    wrap.hidden = true;
    document.removeEventListener("keydown", onKey);
    opener?.focus?.();
  }

  function onKey(e) {
    if (e.key === "Escape") close();
  }

  tabsEl.addEventListener("click", (e) => {
    const button = e.target.closest("button[data-tab]");
    if (button) showTab(button.dataset.tab);
  });
  $("#drawerClose").addEventListener("click", close);
  $("#drawerBackdrop").addEventListener("click", close);
  copyBtn.addEventListener("click", async () => {
    if (!detail) return;
    const ok = await copyText(detail.path);
    copyBtn.textContent = ok ? "Copiado" : "Error";
    setTimeout(() => (copyBtn.textContent = "Copiar ruta"), 1300);
  });
  askBtn.addEventListener("click", () => {
    if (!detail) return;
    close();
    onAsk(detail);
  });

  return {
    async open(id) {
      opener = document.activeElement;
      detail = null;
      wrap.hidden = false;
      document.addEventListener("keydown", onKey);
      methodEl.className = "method";
      methodEl.textContent = "";
      titleEl.textContent = id;
      pathEl.textContent = "";
      clear(metaEl);
      bodyEl.textContent = "Cargando…";
      $("#drawerClose").focus();
      try {
        detail = await getOperation(id);
      } catch (err) {
        bodyEl.textContent = err.message || "No se pudo cargar la operación.";
        return;
      }
      methodEl.className = `method ${methodClass(detail.method)}`;
      methodEl.textContent = detail.method || "DOC";
      titleEl.textContent = detail.id;
      pathEl.textContent = detail.path;
      const pills = [
        detail.tag && h("span", { class: "tag-pill" }, detail.tag),
        detail.version && h("span", { class: "tag-pill" }, detail.version),
        detail.template && h("span", { class: "tag-pill" }, "plantilla de sitio"),
        detail.deprecated && h("span", { class: "tag-pill warn" }, "obsoleto"),
      ].filter(Boolean);
      metaEl.append(...pills);
      const isGuide = detail.kind === "guide";
      tabsEl.hidden = isGuide;
      askBtn.hidden = isGuide;
      showTab("header");
    },
    close,
  };
}
