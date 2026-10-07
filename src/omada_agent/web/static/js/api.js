// Thin client for the backend HTTP API (including the streaming chat endpoint).

const TOKEN_KEY = "omada.token";
let onAuthRequired = () => {};

export const setAuthHandler = (fn) => { onAuthRequired = fn; };
export const getToken = () => sessionStorage.getItem(TOKEN_KEY) || "";
export const setToken = (t) => sessionStorage.setItem(TOKEN_KEY, t);

function headers(extra = {}) {
  const token = getToken();
  return { ...(token ? { Authorization: `Bearer ${token}` } : {}), ...extra };
}

async function request(path, options = {}) {
  const res = await fetch(path, { ...options, headers: headers(options.headers) });
  if (res.status === 401) {
    onAuthRequired();
    throw new Error("Token de acceso requerido.");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* not JSON */ }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json();
}

export const getHealth = () => fetch("/api/health").then((r) => r.json());
export const getStats = () => request("/api/index/stats");
export const getOperation = (id) => request(`/api/operation?id=${encodeURIComponent(id)}`);

export function search(query, { method, limit = 12, signal } = {}) {
  const params = new URLSearchParams({ q: query, limit: String(limit) });
  if (method) params.set("method", method);
  return request(`/api/search?${params}`, { signal });
}

/**
 * Parse a Server-Sent-Events byte stream into JSON events.
 * Exported separately so it can be unit-tested without a network.
 */
export async function* parseSSE(body) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let boundary;
      while ((boundary = buffer.search(/\r?\n\r?\n/)) !== -1) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary).replace(/^\r?\n\r?\n/, "");
        const data = frame
          .split(/\r?\n/)
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).replace(/^ /, ""))
          .join("\n");
        if (data) {
          try { yield JSON.parse(data); } catch { /* ignore malformed frame */ }
        }
      }
    }
  } finally {
    reader.releaseLock?.();
  }
}

/** Stream a chat turn. Yields backend events (token, tool_start, tool_result, done, error...). */
export async function* streamChat(messages, signal) {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: headers({ "Content-Type": "application/json" }),
    body: JSON.stringify({ messages }),
    signal,
  });
  if (res.status === 401) {
    onAuthRequired();
    yield { type: "error", code: "auth", message: "Este portal requiere un token de acceso." };
    return;
  }
  if (!res.ok || !res.body) {
    let message = `Error del servidor (${res.status}).`;
    try {
      const detail = (await res.json()).detail;
      if (detail) message = typeof detail === "string" ? detail : JSON.stringify(detail);
    } catch { /* ignore */ }
    yield { type: "error", code: String(res.status), message };
    return;
  }
  yield* parseSSE(res.body);
}
