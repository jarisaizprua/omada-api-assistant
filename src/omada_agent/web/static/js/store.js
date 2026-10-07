// Conversation persistence in the browser (localStorage). The server stays stateless.

const KEY = "omada.conversations.v1";
const MAX_CONVERSATIONS = 50;

function load() {
  try {
    const parsed = JSON.parse(localStorage.getItem(KEY) || "[]");
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function save(list) {
  try {
    localStorage.setItem(KEY, JSON.stringify(list.slice(0, MAX_CONVERSATIONS)));
  } catch { /* storage full or disabled: the app keeps working in memory */ }
}

const uid = () => (crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(16).slice(2)}`);

export const store = {
  list() {
    return load().sort((a, b) => b.updatedAt - a.updatedAt);
  },
  get(id) {
    return load().find((c) => c.id === id) || null;
  },
  create() {
    return { id: uid(), title: "Nueva conversación", createdAt: Date.now(), updatedAt: Date.now(), messages: [] };
  },
  upsert(conversation) {
    const list = load().filter((c) => c.id !== conversation.id);
    conversation.updatedAt = Date.now();
    list.unshift(conversation);
    save(list);
  },
  remove(id) {
    save(load().filter((c) => c.id !== id));
  },
};

export const prefs = {
  get(key, fallback) {
    try { return localStorage.getItem(`omada.${key}`) ?? fallback; } catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(`omada.${key}`, value); } catch { /* ignore */ }
  },
};
