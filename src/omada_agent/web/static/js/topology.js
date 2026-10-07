// Topology map: every API category is a node around the controller hub.
//   - node size  = number of operations in the category
//   - node color = API family (wired, wireless, gateway, security, management)
// It is generated from the real index, so it always reflects what is searchable.
// Pure helpers (`familyOf`, `layoutNodes`) are exported so they can be tested in Node.

export const FAMILIES = [
  { key: "wired", label: "Cableada", dark: "#5aa9ff", light: "#1f6fd1" },
  { key: "wireless", label: "Inalámbrica", dark: "#22d3bb", light: "#078a7b" },
  { key: "gateway", label: "Gateway y VPN", dark: "#b58cff", light: "#7a4cc9" },
  { key: "security", label: "Seguridad y políticas", dark: "#f5b94a", light: "#a86a00" },
  { key: "mgmt", label: "Gestión y sistema", dark: "#8fb0bb", light: "#547a88" },
];

const RULES = [
  ["security", /acl|firewall|security|portal|authenticat|radius|access control|application control|rule|policy|ips\b|url|dpi|attack|filter/],
  ["gateway", /gateway|vpn|\bwan\b|routing|route|\bnat\b|dhcp|ddns|ipsec|ospf|bgp|load balanc|isp/],
  ["wireless", /wireless|ssid|^ap\b|\bap\b|wlan|radio|mesh|bluetooth|wi-?fi|roaming|hotspot|\brf\b/],
  ["wired", /switch|wired|stack|\bport|ethernet|vlan|\blan\b|poe|\bolt\b|\bont\b|\bstp\b|\blag\b|mirror/],
];

export function familyOf(tag) {
  const name = String(tag).toLowerCase();
  for (const [key, regex] of RULES) if (regex.test(name)) return key;
  return "mgmt";
}

const GOLDEN = Math.PI * (3 - Math.sqrt(5));
export const nodeRadius = (count) => 3.2 + Math.sqrt(Math.max(count, 1)) * 1.45;

/**
 * Deterministic layout: sunflower spiral (big categories near the hub), then a few
 * rounds of collision resolution so circles never overlap.
 * Returns [{x, y, r}] in canvas pixels, same order as `categories`.
 */
export function layoutNodes(categories, width, height, hubRadius = 15) {
  const cx = width / 2;
  const cy = height / 2;
  const margin = 14;
  const ex = Math.max(width / 2 - margin, 10);
  const ey = Math.max(height / 2 - margin, 10);
  const n = categories.length;
  const scale = Math.min(1, Math.min(width, height) / 420); // shrink circles on small canvases
  const nodes = categories.map((c, i) => {
    const t = Math.sqrt((i + 0.5) / Math.max(n, 1));
    const rad = 0.16 + 0.84 * t;
    const a = i * GOLDEN;
    return { x: cx + Math.cos(a) * ex * rad, y: cy + Math.sin(a) * ey * rad, r: nodeRadius(c.count) * Math.max(scale, 0.62) };
  });

  for (let iter = 0; iter < 160; iter++) {
    for (let i = 0; i < n; i++) {
      const a = nodes[i];
      // keep clear of the hub
      let dx = a.x - cx, dy = a.y - cy;
      let d = Math.hypot(dx, dy) || 0.001;
      const minHub = hubRadius + a.r + 12;
      if (d < minHub) { a.x = cx + (dx / d) * minHub; a.y = cy + (dy / d) * minHub; }
      for (let j = i + 1; j < n; j++) {
        const b = nodes[j];
        dx = b.x - a.x; dy = b.y - a.y;
        d = Math.hypot(dx, dy) || 0.001;
        const min = a.r + b.r + 3;
        if (d < min) {
          const push = (min - d) / 2;
          const ux = dx / d, uy = dy / d;
          a.x -= ux * push; a.y -= uy * push;
          b.x += ux * push; b.y += uy * push;
        }
      }
      a.x = Math.min(Math.max(a.x, margin + a.r), width - margin - a.r);
      a.y = Math.min(Math.max(a.y, margin + a.r), height - margin - a.r);
    }
  }
  return nodes;
}

export class Topology {
  constructor(canvas, tipEl, { onPick } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext ? canvas.getContext("2d") : null;
    this.tip = tipEl;
    this.onPick = onPick || (() => {});
    this.categories = [];
    this.nodes = [];
    this.hover = -1;
    this.packets = [];
    this.running = false;
    this.w = 0;
    this.h = 0;
    this.dpr = 1;
    this.reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
    this.theme = "dark";
    this.colors = {};
    this._frame = this._frame.bind(this);

    canvas.addEventListener("pointermove", (e) => this._onMove(e));
    canvas.addEventListener("pointerleave", () => this._setHover(-1));
    canvas.addEventListener("click", (e) => this._onClick(e));
    canvas.setAttribute("role", "img");

    if (typeof ResizeObserver !== "undefined") {
      let timer;
      this.observer = new ResizeObserver(() => {
        clearTimeout(timer);
        timer = setTimeout(() => this.resize(), 100);
      });
      this.observer.observe(canvas);
    }
  }

  setData(categories) {
    this.categories = categories;
    this.resize();
  }

  setTheme(theme) {
    this.theme = theme;
    const css = getComputedStyle(document.documentElement);
    const read = (name) => css.getPropertyValue(name).trim();
    this.colors = {
      text: read("--text") || "#e8f2f4",
      dim: read("--text-dim") || "#9bb8c1",
      bg: read("--bg-raised") || "#0f2937",
      line: read("--line-strong") || "#27566a",
      accent: read("--accent") || "#22d3bb",
    };
    this.famColor = Object.fromEntries(FAMILIES.map((f) => [f.key, theme === "light" ? f.light : f.dark]));
  }

  resize() {
    if (!this.ctx) return;
    const rect = this.canvas.getBoundingClientRect();
    if (rect.width < 10 || rect.height < 10) return;
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.w = rect.width;
    this.h = rect.height;
    this.canvas.width = Math.round(this.w * this.dpr);
    this.canvas.height = Math.round(this.h * this.dpr);
    this.nodes = layoutNodes(this.categories, this.w, this.h).map((p, i) => ({
      ...p,
      phase: i * 1.7,
      tag: this.categories[i].tag,
      count: this.categories[i].count,
      family: familyOf(this.categories[i].tag),
    }));
    this._seedPackets();
    if (!this.running) this._draw(0);
  }

  start() {
    if (this.running || !this.ctx) return;
    this.running = true;
    if (!this.colors.text) this.setTheme(this.theme);
    requestAnimationFrame(this._frame);
  }

  stop() { this.running = false; }

  _seedPackets() {
    const count = this.reduced ? 0 : Math.min(7, this.nodes.length);
    this.packets = Array.from({ length: count }, (_, i) => ({ node: this._pickNode(i), p: (i * 0.37) % 1, speed: 0.0035 + (i % 3) * 0.0012 }));
  }

  _pickNode(seed) {
    if (!this.nodes.length) return 0;
    // Prefer big categories: they are the busiest "devices".
    const pool = Math.max(8, Math.floor(this.nodes.length * 0.45));
    return Math.floor((Math.sin(seed * 9301 + Date.now() * 0.001) * 0.5 + 0.5) * pool) % this.nodes.length;
  }

  _pos(node, t) {
    if (this.reduced) return { x: node.x, y: node.y };
    return { x: node.x + Math.sin(t * 0.0006 + node.phase) * 2.2, y: node.y + Math.cos(t * 0.00052 + node.phase) * 2.2 };
  }

  _frame(t) {
    if (!this.running) return;
    this._draw(t);
    requestAnimationFrame(this._frame);
  }

  _draw(t) {
    const { ctx } = this;
    if (!ctx || !this.w) return;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.w, this.h);
    const cx = this.w / 2, cy = this.h / 2;
    const c = this.colors.text ? this.colors : (this.setTheme(this.theme), this.colors);
    const hovering = this.hover >= 0;

    // links
    ctx.lineWidth = 1;
    this.nodes.forEach((n, i) => {
      const p = this._pos(n, t);
      const active = i === this.hover;
      ctx.globalAlpha = active ? 0.85 : hovering ? 0.05 : 0.14;
      ctx.strokeStyle = active ? this.famColor[n.family] : c.line;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(p.x, p.y);
      ctx.stroke();
    });

    // packets travelling along links
    ctx.globalAlpha = hovering ? 0.35 : 0.9;
    for (const pk of this.packets) {
      pk.p += pk.speed;
      if (pk.p > 1) { pk.p = 0; pk.node = this._pickNode(pk.node + t); }
      const n = this.nodes[pk.node];
      if (!n) continue;
      const p = this._pos(n, t);
      ctx.fillStyle = this.famColor[n.family];
      ctx.beginPath();
      ctx.arc(cx + (p.x - cx) * pk.p, cy + (p.y - cy) * pk.p, 2, 0, Math.PI * 2);
      ctx.fill();
    }

    // nodes
    this.nodes.forEach((n, i) => {
      const p = this._pos(n, t);
      const active = i === this.hover;
      ctx.globalAlpha = hovering && !active ? 0.35 : 0.92;
      ctx.fillStyle = this.famColor[n.family];
      ctx.beginPath();
      ctx.arc(p.x, p.y, n.r, 0, Math.PI * 2);
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = active ? c.text : c.bg;
      ctx.stroke();
    });

    // hub
    ctx.globalAlpha = 0.18;
    ctx.fillStyle = c.accent;
    ctx.beginPath(); ctx.arc(cx, cy, 24 + (this.reduced ? 0 : Math.sin(t * 0.002) * 2), 0, Math.PI * 2); ctx.fill();
    ctx.globalAlpha = 1;
    ctx.fillStyle = c.accent;
    ctx.beginPath(); ctx.arc(cx, cy, 14, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = this.theme === "light" ? "#fff" : "#04302a";
    ctx.font = "600 9px 'IBM Plex Sans', system-ui, sans-serif";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText("API", cx, cy + 0.5);

    // labels for the biggest categories (skip any that would overlap)
    ctx.font = "500 11px 'IBM Plex Sans', system-ui, sans-serif";
    ctx.textAlign = "left";
    const boxes = [];
    const labelled = new Set([...Array(Math.min(12, this.nodes.length)).keys(), this.hover].filter((i) => i >= 0));
    for (const i of labelled) {
      const n = this.nodes[i];
      const p = this._pos(n, t);
      const text = n.tag.length > 22 ? `${n.tag.slice(0, 21)}…` : n.tag;
      const width = ctx.measureText(text).width;
      let x = p.x + n.r + 5;
      if (x + width > this.w - 4) x = p.x - n.r - 5 - width;
      const box = { x, y: p.y - 7, w: width, h: 14 };
      const clash = i !== this.hover && boxes.some((b) => box.x < b.x + b.w && box.x + box.w > b.x && box.y < b.y + b.h && box.y + box.h > b.y);
      if (clash) continue;
      boxes.push(box);
      ctx.globalAlpha = hovering && i !== this.hover ? 0.35 : 1;
      ctx.lineWidth = 3;
      ctx.strokeStyle = c.bg;
      ctx.strokeText(text, x, p.y);
      ctx.fillStyle = i === this.hover ? c.text : c.dim;
      ctx.fillText(text, x, p.y);
    }
    ctx.globalAlpha = 1;
  }

  _hit(event) {
    const rect = this.canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    let best = -1, bestDist = Infinity;
    this.nodes.forEach((n, i) => {
      const d = Math.hypot(n.x - x, n.y - y);
      if (d <= n.r + 7 && d < bestDist) { best = i; bestDist = d; }
    });
    return { index: best, x, y };
  }

  _onMove(event) {
    const { index } = this._hit(event);
    this._setHover(index);
  }

  _onClick(event) {
    const { index } = this._hit(event);
    if (index >= 0) this.onPick(this.nodes[index].tag);
  }

  _setHover(index) {
    if (index === this.hover) return;
    this.hover = index;
    this.canvas.style.cursor = index >= 0 ? "pointer" : "default";
    if (!this.tip) return;
    if (index < 0) { this.tip.hidden = true; return; }
    const n = this.nodes[index];
    this.tip.replaceChildren();
    const b = document.createElement("b");
    b.textContent = n.tag;
    const s = document.createElement("span");
    s.textContent = `${n.count} ops`;
    this.tip.append(b, s);
    this.tip.style.left = `${n.x}px`;
    this.tip.style.top = `${n.y - n.r}px`;
    this.tip.hidden = false;
    if (!this.running) this._draw(0);
  }

  destroy() {
    this.stop();
    this.observer?.disconnect();
  }
}
