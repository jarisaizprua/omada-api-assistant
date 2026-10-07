// Command palette ("Explorar la API"): instant search over every indexed operation.

import { search } from "./api.js";
import { $, clear, debounce, h, methodBadge } from "./ui.js";

const METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE"];

export function createPalette({ onOpen }) {
  const dialog = $("#palette");
  const input = $("#paletteInput");
  const filtersEl = $("#paletteFilters");
  const resultsEl = $("#paletteResults");
  const footEl = $("#paletteFoot");

  let method = null;
  let items = [];
  let selected = 0;
  let controller = null;
  let total = 0;

  const filterButtons = [null, ...METHODS].map((m) => {
    const button = h("button", { class: "filter", type: "button", "aria-pressed": String(m === null) }, m || "Todos");
    button.addEventListener("click", () => {
      method = m;
      filterButtons.forEach((b, i) => b.setAttribute("aria-pressed", String([null, ...METHODS][i] === m)));
      run();
      input.focus();
    });
    return button;
  });
  filtersEl.append(...filterButtons);

  function renderHint(message) {
    clear(resultsEl);
    resultsEl.append(h("li", { class: "palette-empty" }, message));
    footEl.textContent = "";
  }

  function renderResults(mode, warning) {
    clear(resultsEl);
    if (!items.length) return renderHint("Sin resultados. Prueba con otras palabras, en inglés si es posible.");
    items.forEach((op, i) => {
      const row = h(
        "li",
        { class: "result", role: "option", "aria-selected": String(i === selected), dataset: { index: i } },
        methodBadge(op.method),
        h(
          "div",
          { class: "r-main" },
          h("div", { class: "r-sum" }, op.summary || op.title, op.deprecated ? h("span", { class: "tag-pill warn", style: "margin-left:8px" }, "obsoleto") : null),
          h("div", { class: "r-path" }, op.path),
          h("div", { class: "r-tag" }, op.tag + (op.template ? " · plantilla" : ""))
        )
      );
      row.addEventListener("click", () => choose(i));
      row.addEventListener("mousemove", () => select(i, false));
      resultsEl.append(row);
    });
    footEl.textContent = `${items.length} resultados · ${mode === "hybrid" ? "búsqueda híbrida" : "solo palabras clave"} · ↑↓ navegar · Enter abrir`;
    if (warning) footEl.title = warning;
  }

  function select(i, scroll = true) {
    selected = Math.max(0, Math.min(i, items.length - 1));
    [...resultsEl.children].forEach((row, idx) => row.setAttribute("aria-selected", String(idx === selected)));
    if (scroll) resultsEl.children[selected]?.scrollIntoView({ block: "nearest" });
  }

  function choose(i) {
    const op = items[i];
    if (!op) return;
    dialog.close();
    onOpen(op.id);
  }

  async function run() {
    const query = input.value.trim();
    controller?.abort();
    if (query.length < 2) {
      items = [];
      return renderHint(total ? `Escribe para buscar entre ${total.toLocaleString("es")} operaciones.` : "Escribe para buscar.");
    }
    controller = new AbortController();
    try {
      const data = await search(query, { method, limit: 14, signal: controller.signal });
      items = data.results;
      selected = 0;
      renderResults(data.mode, data.warning);
    } catch (err) {
      if (err.name === "AbortError") return;
      items = [];
      renderHint(err.message || "No se pudo buscar.");
    }
  }
  const runDebounced = debounce(run, 200);

  input.addEventListener("input", runDebounced);
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); select(selected + 1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); select(selected - 1); }
    else if (e.key === "Enter") { e.preventDefault(); choose(selected); }
  });
  dialog.addEventListener("click", (e) => { if (e.target === dialog) dialog.close(); });
  dialog.addEventListener("close", () => controller?.abort());

  return {
    setTotal(n) { total = n; },
    open(prefill = "") {
      if (!dialog.open) dialog.showModal();
      input.value = prefill;
      input.focus();
      input.select();
      run();
    },
  };
}
