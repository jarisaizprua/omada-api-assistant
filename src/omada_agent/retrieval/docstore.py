"""SQLite document store with FTS5 keyword search.

This is the *source of truth* for document text. It serves three jobs:

1. **Keyword search** (BM25 through SQLite FTS5): excellent for exact identifiers
   such as ``createNewSite``, a field name, or an error code.
2. **Detail lookup**: the full request/response schema of an operation, served on
   demand to the agent and to the web UI.
3. **Statistics** for the dashboard (categories, methods, counts).

SQLite ships with Python, needs no server and is rebuilt in about a second, which
keeps the whole project installable with a single ``pip install``.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from collections.abc import Iterable
from difflib import get_close_matches
from pathlib import Path

from omada_agent.ingestion.openapi_parser import split_words
from omada_agent.models import Doc, Filters

SCHEMA_VERSION = 1

_COLUMNS = (
    "doc_id, kind, title, method, path, path_norm, version, tag, tags_json, deprecated, "
    "is_template, summary, description, permissions_json, errors_json, brief_text, card_text, "
    "header_text, request_text, response_text"
)
_LIGHT_COLUMNS = (
    "doc_id, kind, title, method, path, path_norm, version, tag, tags_json, deprecated, "
    "is_template, summary, '' AS description, permissions_json, errors_json, brief_text, '' AS card_text, "
    "'' AS header_text, '' AS request_text, '' AS response_text"
)

_DDL = """
CREATE TABLE docs (
    rowid            INTEGER PRIMARY KEY,
    doc_id           TEXT NOT NULL UNIQUE,
    kind             TEXT NOT NULL,
    title            TEXT NOT NULL,
    method           TEXT NOT NULL DEFAULT '',
    path             TEXT NOT NULL DEFAULT '',
    path_norm        TEXT NOT NULL DEFAULT '',
    version          TEXT NOT NULL DEFAULT '',
    tag              TEXT NOT NULL DEFAULT '',
    tags_json        TEXT NOT NULL DEFAULT '[]',
    deprecated       INTEGER NOT NULL DEFAULT 0,
    is_template      INTEGER NOT NULL DEFAULT 0,
    summary          TEXT NOT NULL DEFAULT '',
    description      TEXT NOT NULL DEFAULT '',
    permissions_json TEXT NOT NULL DEFAULT '[]',
    errors_json      TEXT NOT NULL DEFAULT '[]',
    brief_text       TEXT NOT NULL DEFAULT '',
    card_text        TEXT NOT NULL DEFAULT '',
    header_text      TEXT NOT NULL DEFAULT '',
    request_text     TEXT NOT NULL DEFAULT '',
    response_text    TEXT NOT NULL DEFAULT ''
);
CREATE INDEX idx_docs_kind ON docs(kind);
CREATE INDEX idx_docs_tag ON docs(tag);
CREATE INDEX idx_docs_method ON docs(method);
CREATE INDEX idx_docs_pathnorm ON docs(path_norm);
CREATE VIRTUAL TABLE docs_fts USING fts5(
    title, keywords, description, fields,
    tokenize = 'unicode61 remove_diacritics 2'
);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

# BM25 column weights: title > keywords > description > field names.
_BM25_WEIGHTS = "6.0, 4.0, 2.0, 1.5"

_STOPWORDS = frozenset(
    """a an and are as at be by can do does for from get how i in is it of on or that the this to
    use using what when where which with you your via about api endpoint endpoints me my
    el la los las un una unos unas de del y o en con por para que como cual cuales se es son al
    lo mi mis""".split()
)

_METHOD_PATH_RE = re.compile(r"\b(GET|POST|PUT|PATCH|DELETE)\s+(/\S+)", re.IGNORECASE)


def tokenize_query(text: str) -> list[str]:
    tokens: list[str] = []
    for word in split_words(text):
        w = word.lower()
        if w in _STOPWORDS or (len(w) < 2 and not w.isdigit()):
            continue
        if w not in tokens:
            tokens.append(w)
    return tokens[:24]


def build_fts_query(text: str) -> str | None:
    tokens = tokenize_query(text)
    if not tokens:
        return None
    return " OR ".join(f'"{t}"*' if len(t) >= 3 else f'"{t}"' for t in tokens)


def _normalize_path(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "{}", path).rstrip("/").lower()


def _row_to_doc(row: sqlite3.Row) -> Doc:
    return Doc(
        doc_id=row["doc_id"],
        kind=row["kind"],
        title=row["title"],
        method=row["method"],
        path=row["path"],
        path_norm=row["path_norm"],
        version=row["version"],
        tag=row["tag"],
        tags=json.loads(row["tags_json"]),
        deprecated=bool(row["deprecated"]),
        is_template=bool(row["is_template"]),
        summary=row["summary"],
        description=row["description"],
        permissions=json.loads(row["permissions_json"]),
        errors=[tuple(e) for e in json.loads(row["errors_json"])],
        brief_text=row["brief_text"],
        card_text=row["card_text"],
        header_text=row["header_text"],
        request_text=row["request_text"],
        response_text=row["response_text"],
    )


def _where(filters: Filters | None, alias: str = "d") -> tuple[str, list]:
    clauses: list[str] = []
    params: list = []
    if filters:
        if filters.kind:
            clauses.append(f"{alias}.kind = ?")
            params.append(filters.kind)
        if filters.method:
            clauses.append(f"{alias}.method = ?")
            params.append(filters.method.upper())
        if filters.tag:
            clauses.append(f"{alias}.tag = ? COLLATE NOCASE")
            params.append(filters.tag)
        if not filters.include_deprecated:
            clauses.append(f"{alias}.deprecated = 0")
        if not filters.include_templates:
            clauses.append(f"{alias}.is_template = 0")
    return (" AND " + " AND ".join(clauses)) if clauses else "", params


class DocStore:
    """Read-only access to the SQLite index (one connection per thread)."""

    def __init__(self, path: Path):
        self.path = path
        self._local = threading.local()
        if not path.exists():
            raise FileNotFoundError(path)

    # ------------------------------------------------------------------ build
    @staticmethod
    def build(path: Path, docs: Iterable[Doc], meta: dict[str, str]) -> int:
        """(Re)create the database atomically. Returns the number of rows written."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        if tmp.exists():
            tmp.unlink()
        conn = sqlite3.connect(tmp)
        try:
            conn.executescript(_DDL)
            count = 0
            for doc in docs:
                cur = conn.execute(
                    f"INSERT INTO docs ({_COLUMNS}) VALUES ({','.join('?' * 20)})",
                    (
                        doc.doc_id,
                        doc.kind,
                        doc.title,
                        doc.method,
                        doc.path,
                        doc.path_norm,
                        doc.version,
                        doc.tag,
                        json.dumps(doc.tags),
                        int(doc.deprecated),
                        int(doc.is_template),
                        doc.summary,
                        doc.description,
                        json.dumps(doc.permissions),
                        json.dumps(doc.errors),
                        doc.brief_text,
                        doc.card_text,
                        doc.header_text,
                        doc.request_text,
                        doc.response_text,
                    ),
                )
                conn.execute(
                    "INSERT INTO docs_fts (rowid, title, keywords, description, fields) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        cur.lastrowid,
                        doc.fts_title,
                        doc.fts_keywords,
                        doc.fts_description,
                        doc.fts_fields,
                    ),
                )
                count += 1
            meta = {**meta, "schema_version": str(SCHEMA_VERSION)}
            conn.executemany("INSERT INTO meta VALUES (?, ?)", meta.items())
            conn.commit()
        finally:
            conn.close()
        os.replace(tmp, path)
        return count

    # ------------------------------------------------------------------ plumbing
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    # ------------------------------------------------------------------ lookups
    def get(self, doc_id: str) -> Doc | None:
        row = self._conn().execute(
            f"SELECT {_COLUMNS} FROM docs WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return _row_to_doc(row) if row else None

    def get_many(self, ids: list[str], heavy: bool = False) -> dict[str, Doc]:
        if not ids:
            return {}
        cols = _COLUMNS if heavy else _LIGHT_COLUMNS
        marks = ",".join("?" * len(ids))
        rows = self._conn().execute(
            f"SELECT {cols} FROM docs WHERE doc_id IN ({marks})", ids
        ).fetchall()
        return {r["doc_id"]: _row_to_doc(r) for r in rows}

    def lookup_exact(self, query: str, filters: Filters | None = None) -> list[str]:
        """Find documents whose operationId or ``METHOD /path`` is spelled in the query."""
        found: list[str] = []
        conn = self._conn()
        where, params = _where(filters)

        for method, path in _METHOD_PATH_RE.findall(query):
            rows = conn.execute(
                f"SELECT d.doc_id FROM docs d WHERE d.method = ? AND d.path_norm = ?{where}",
                [method.upper(), _normalize_path(path.rstrip(".,;:)")), *params],
            ).fetchall()
            found += [r[0] for r in rows]

        for token in re.findall(r"[A-Za-z0-9_\-:#./]{4,}", query):
            token = token.strip(".,;:)(")
            if token.startswith("/"):
                rows = conn.execute(
                    f"SELECT d.doc_id FROM docs d WHERE d.path_norm = ?{where}",
                    [_normalize_path(token), *params],
                ).fetchall()
            else:
                rows = conn.execute(
                    f"SELECT d.doc_id FROM docs d WHERE lower(d.doc_id) = lower(?){where}",
                    [token, *params],
                ).fetchall()
            found += [r[0] for r in rows]
        # An operation-specific error code spelled in the query (e.g. "-33101").
        for code in re.findall(r"(?<![\w])-\d{4,6}\b", query):
            rows = conn.execute(
                f"SELECT d.doc_id FROM docs d WHERE d.errors_json LIKE ?{where}",
                [f'%"{code}"%', *params],
            ).fetchall()
            found += [r[0] for r in rows]

        # The whole query is exactly an operation summary ("Create new site").
        rows = conn.execute(
            f"SELECT d.doc_id FROM docs d WHERE d.kind = 'operation' "
            f"AND lower(d.summary) = lower(?){where}",
            [query.strip().rstrip("?.!"), *params],
        ).fetchall()
        found += [r[0] for r in rows]
        return list(dict.fromkeys(found))

    # ------------------------------------------------------------------ keyword search
    def keyword_search(
        self, query: str, limit: int, filters: Filters | None = None
    ) -> list[tuple[str, float]]:
        fts = build_fts_query(query)
        if not fts:
            return []
        where, params = _where(filters)
        sql = (
            f"SELECT d.doc_id, bm25(docs_fts, {_BM25_WEIGHTS}) AS score "
            "FROM docs_fts JOIN docs d ON d.rowid = docs_fts.rowid "
            f"WHERE docs_fts MATCH ?{where} ORDER BY score LIMIT ?"
        )
        rows = self._conn().execute(sql, [fts, *params, limit]).fetchall()
        return [(r["doc_id"], r["score"]) for r in rows]

    # ------------------------------------------------------------------ catalog / stats
    def tag_counts(self, kind: str = "operation") -> list[tuple[str, int]]:
        rows = self._conn().execute(
            "SELECT tag, COUNT(*) AS n FROM docs WHERE kind = ? GROUP BY tag ORDER BY n DESC, tag",
            (kind,),
        ).fetchall()
        return [(r["tag"], r["n"]) for r in rows]

    def resolve_tag(self, name: str) -> tuple[str | None, list[str]]:
        """Map a user/LLM-provided category name to a real one (case-insensitive)."""
        tags = [t for t, _ in self.tag_counts()]
        lowered = {t.lower(): t for t in tags}
        if name.lower() in lowered:
            return lowered[name.lower()], []
        contains = [t for t in tags if name.lower() in t.lower()]
        if len(contains) == 1:
            return contains[0], []
        suggestions = contains[:6] or get_close_matches(name, tags, n=6, cutoff=0.4)
        return None, suggestions

    def stats(self) -> dict:
        conn = self._conn()

        def scalar(sql: str, *args) -> int:
            return conn.execute(sql, args).fetchone()[0]

        return {
            "operations": scalar("SELECT COUNT(*) FROM docs WHERE kind='operation'"),
            "guides": scalar("SELECT COUNT(*) FROM docs WHERE kind='guide'"),
            "templates": scalar("SELECT COUNT(*) FROM docs WHERE kind='operation' AND is_template=1"),
            "deprecated": scalar("SELECT COUNT(*) FROM docs WHERE kind='operation' AND deprecated=1"),
            "methods": {
                r[0]: r[1]
                for r in conn.execute(
                    "SELECT method, COUNT(*) FROM docs WHERE kind='operation' GROUP BY method"
                )
            },
            "categories": [
                {"tag": t, "count": n} for t, n in self.tag_counts()
            ],
        }

    def meta(self) -> dict[str, str]:
        return {r[0]: r[1] for r in self._conn().execute("SELECT key, value FROM meta")}

    # ------------------------------------------------------------------ bulk helpers (evaluations)
    def iter_operations(self) -> list[dict]:
        """Light view of every operation (no long texts)."""
        rows = self._conn().execute(
            "SELECT doc_id, method, path, path_norm, summary, tag, deprecated, is_template "
            "FROM docs WHERE kind = 'operation' ORDER BY rowid"
        ).fetchall()
        return [dict(r) for r in rows]

    def guide_corpus(self) -> str:
        """All guide text joined (used to verify that paths mentioned by the model exist)."""
        rows = self._conn().execute("SELECT description FROM docs WHERE kind = 'guide'").fetchall()
        return "\n".join(r[0] for r in rows)

    def field_names(self) -> set[str]:
        """Lower-cased request/response field names of every operation."""
        rows = self._conn().execute(
            "SELECT f.fields FROM docs_fts f JOIN docs d ON d.rowid = f.rowid WHERE d.kind = 'operation'"
        ).fetchall()
        names: set[str] = set()
        for (fields,) in rows:
            names.update(w.lower() for w in (fields or "").split())
        return names

