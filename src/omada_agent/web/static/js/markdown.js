// Markdown rendering for assistant answers: marked -> DOMPurify -> highlight.js,
// with a small toolbar (language + copy) on every code block.

import { copyText, h } from "./ui.js";

const hasLibs = () => typeof window.marked !== "undefined" && typeof window.DOMPurify !== "undefined";

if (hasLibs()) {
  window.marked.setOptions({ gfm: true, breaks: false });
  // Open links safely in a new tab.
  window.DOMPurify.addHook("afterSanitizeAttributes", (node) => {
    if (node.tagName === "A") {
      node.setAttribute("target", "_blank");
      node.setAttribute("rel", "noopener noreferrer");
    }
  });
}

export function renderMarkdownInto(container, text) {
  if (!hasLibs()) {
    container.textContent = text; // graceful fallback: plain text
    return;
  }
  const html = window.marked.parse(text || "");
  container.innerHTML = window.DOMPurify.sanitize(html, { USE_PROFILES: { html: true } });
  enhanceCodeBlocks(container);
}

function enhanceCodeBlocks(container) {
  for (const pre of container.querySelectorAll("pre")) {
    const code = pre.querySelector("code");
    if (!code) continue;
    const lang = (code.className.match(/language-([\w+-]+)/) || [])[1] || "texto";
    if (window.hljs && lang !== "texto") {
      try {
        if (window.hljs.getLanguage(lang)) window.hljs.highlightElement(code);
      } catch { /* leave unhighlighted */ }
    }
    const button = h("button", { type: "button", "aria-label": "Copiar código" }, "Copiar");
    button.addEventListener("click", async () => {
      const ok = await copyText(code.textContent || "");
      button.textContent = ok ? "Copiado" : "Error";
      setTimeout(() => (button.textContent = "Copiar"), 1400);
    });
    const wrapper = h("div", { class: "codeblock" }, h("header", {}, h("span", {}, lang), button));
    pre.replaceWith(wrapper);
    wrapper.append(pre);
  }
}
