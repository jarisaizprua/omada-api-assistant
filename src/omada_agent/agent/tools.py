"""Tools the LLM can call to query the knowledge base.

Progressive disclosure keeps prompts small and answers accurate:

* ``search_api``      -> short cards (summary, category, params, required body fields)
* ``get_operation``   -> the full schema of ONE operation, only when needed
* ``list_categories`` -> map of the API surface
* ``search_guides``   -> prose guides (authentication, access modes, tables...)

Every tool returns two things: ``text`` (what the model reads) and ``operations``
(small JSON cards the web UI shows in the "Endpoints consulted" panel).
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from omada_agent.models import Doc, Filters
from omada_agent.retrieval.hybrid import Retriever

log = logging.getLogger(__name__)

MAX_SEARCH_LIMIT = 15
DETAIL_CHARS_ALL = 16_000
DETAIL_CHARS_SECTION = 32_000
GUIDE_SNIPPET_CHARS = 1800


@dataclass(slots=True)
class ToolResult:
    text: str
    summary: str
    operations: list[dict] = field(default_factory=list)
    ok: bool = True


def _cap(text: str, limit: int, hint: str) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n[... truncated ({len(text) - limit} more characters). {hint}]"


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "search_api",
        "description": (
            "Search the Omada Open API specification for operations (endpoints). Hybrid semantic + "
            "keyword search. Returns short cards. Write the query in ENGLISH with Omada vocabulary "
            "(e.g. 'create site', 'modify ssid password', 'list connected clients', 'error -33101'). "
            "An exact operationId or 'METHOD /path' in the query returns that operation first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "English search query."},
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
                    "description": "Optional HTTP method filter.",
                },
                "category": {
                    "type": "string",
                    "description": "Optional category (tag) to restrict to. See list_categories.",
                },
                "include_templates": {
                    "type": "boolean",
                    "description": "Include site-template operations. Default false.",
                },
                "include_deprecated": {
                    "type": "boolean",
                    "description": "Include deprecated operations. Default false.",
                },
                "limit": {
                    "type": "integer",
                    "description": f"Number of results (1-{MAX_SEARCH_LIMIT}). Default 8.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "get_operation",
        "description": (
            "Get the complete documentation of ONE operation: description, required permissions, "
            "operation-specific error codes, parameters, and the full request/response schemas. "
            "Call this before writing code that uses an endpoint."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "operation_id": {
                    "type": "string",
                    "description": "The operationId (e.g. 'createNewSite') or 'METHOD /path'.",
                },
                "section": {
                    "type": "string",
                    "enum": ["all", "request", "response"],
                    "description": "Use 'response' or 'request' to get one section in more detail.",
                },
            },
            "required": ["operation_id"],
            "additionalProperties": False,
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "list_categories",
        "description": (
            "List API categories (tags) with their number of operations. Use it to discover how the "
            "API is organised or to find the exact category name for the search_api filter."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "contains": {
                    "type": "string",
                    "description": "Optional substring to filter category names.",
                }
            },
            "additionalProperties": False,
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "search_guides",
        "description": (
            "Search the prose guides (authentication, access modes, token handling, value tables...). "
            "These are NOT part of the OpenAPI spec. Write the query in English."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "English search query."},
                "limit": {"type": "integer", "description": "Number of chunks (1-6). Default 4."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": False,
    },
]


class ToolBox:
    """Executes tool calls against a :class:`Retriever`."""

    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.store = retriever.store

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return TOOL_SCHEMAS

    # ------------------------------------------------------------------ dispatch
    async def run(self, name: str, raw_arguments: str | dict | None) -> ToolResult:
        try:
            args = (
                raw_arguments
                if isinstance(raw_arguments, dict)
                else json.loads(raw_arguments or "{}")
            )
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            return ToolResult(f"Invalid tool arguments: {exc}", "Argumentos inválidos", ok=False)

        handler = {
            "search_api": self._search_api,
            "get_operation": self._get_operation,
            "list_categories": self._list_categories,
            "search_guides": self._search_guides,
        }.get(name)
        if handler is None:
            return ToolResult(f"Unknown tool: {name}", "Herramienta desconocida", ok=False)
        try:
            # SQLite, Chroma and the embeddings client are synchronous: keep the event loop free.
            return await asyncio.to_thread(handler, args)
        except Exception as exc:  # never crash the conversation because of a tool
            log.exception("Tool %s failed", name)
            return ToolResult(
                f"Tool '{name}' failed: {type(exc).__name__}: {exc}", "Error en la herramienta", ok=False
            )

    # ------------------------------------------------------------------ tools
    def _search_api(self, args: dict) -> ToolResult:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolResult("Missing required argument: query", "Consulta vacía", ok=False)

        filters = Filters(
            kind="operation",
            method=(args.get("method") or None),
            include_deprecated=bool(args.get("include_deprecated", False)),
            include_templates=bool(args.get("include_templates", False)),
        )
        category = args.get("category")
        if category:
            resolved, suggestions = self.store.resolve_tag(str(category))
            if resolved is None:
                hint = ", ".join(suggestions) if suggestions else "use list_categories"
                return ToolResult(
                    f"Unknown category '{category}'. Did you mean: {hint}?",
                    "Categoría desconocida",
                    ok=False,
                )
            filters.tag = resolved

        limit = max(1, min(int(args.get("limit") or 8), MAX_SEARCH_LIMIT))
        result = self.retriever.search(query, filters=filters, limit=limit)
        if not result.hits:
            return ToolResult(
                "No operations matched. Try different English keywords, remove filters "
                "(include_templates / include_deprecated), or use list_categories.",
                "Sin resultados",
            )

        lines = [
            f'{len(result.hits)} results for "{query}" ({result.mode} search). '
            "Call get_operation(operation_id) for full schemas.",
            "",
        ]
        for i, hit in enumerate(result.hits, start=1):
            lines.append(f"{i}. {hit.doc.brief_text}")
        if result.warning:
            lines += ["", f"Note: {result.warning}"]
        return ToolResult(
            "\n".join(lines),
            f"{len(result.hits)} resultados",
            [h.doc.to_card(h.score) for h in result.hits],
        )

    def _find_operation(self, ref: str) -> Doc | None:
        ref = ref.strip()
        doc = self.store.get(ref)
        if doc:
            return doc
        exact = self.store.lookup_exact(ref, Filters(kind="operation", include_deprecated=True))
        return self.store.get(exact[0]) if exact else None

    def _get_operation(self, args: dict) -> ToolResult:
        ref = str(args.get("operation_id", "")).strip()
        if not ref:
            return ToolResult("Missing required argument: operation_id", "Falta operation_id", ok=False)

        doc = self._find_operation(ref)
        if doc is None:
            near = self.retriever.search(
                ref, Filters(kind="operation", include_deprecated=True), limit=5, mode="keyword"
            )
            hint = "; ".join(f"{h.doc.doc_id} ({h.doc.title})" for h in near.hits) or "none"
            return ToolResult(
                f"No operation found for '{ref}'. Closest matches: {hint}.",
                "Operación no encontrada",
                ok=False,
            )

        section = args.get("section") or "all"
        if section == "request":
            body, limit = f"{doc.header_text}\n\n{doc.request_text}", DETAIL_CHARS_SECTION
        elif section == "response":
            body, limit = f"{doc.header_text}\n\n{doc.response_text}", DETAIL_CHARS_SECTION
        else:
            body, limit = (
                f"{doc.header_text}\n\n{doc.request_text}\n\n{doc.response_text}",
                DETAIL_CHARS_ALL,
            )
        hint = "Call get_operation again with section='request' or 'response' for that part only."
        return ToolResult(
            _cap(body, limit, hint),
            f"{doc.method} {doc.path}",
            [doc.to_card()],
        )

    def _list_categories(self, args: dict) -> ToolResult:
        needle = str(args.get("contains") or "").lower()
        rows = [(t, n) for t, n in self.store.tag_counts() if needle in t.lower()]
        if not rows:
            return ToolResult("No category matches that text.", "Sin categorías")
        text = "\n".join(f"- {tag} ({n} operations)" for tag, n in rows)
        return ToolResult(text, f"{len(rows)} categorías")

    def _search_guides(self, args: dict) -> ToolResult:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolResult("Missing required argument: query", "Consulta vacía", ok=False)
        limit = max(1, min(int(args.get("limit") or 4), 6))
        if self.store.stats()["guides"] == 0:
            return ToolResult(
                "No guides are indexed in this knowledge base, so authentication and other "
                "prose documentation are not available here.",
                "Sin guías indexadas",
            )
        result = self.retriever.search(query, Filters(kind="guide"), limit=limit)
        if not result.hits:
            return ToolResult("No guide section matched.", "Sin resultados")
        docs = self.store.get_many([h.doc.doc_id for h in result.hits], heavy=True)
        parts = []
        for hit in result.hits:
            doc = docs[hit.doc.doc_id]
            parts.append(_cap(doc.header_text, GUIDE_SNIPPET_CHARS, "Guide section truncated."))
        return ToolResult("\n\n---\n\n".join(parts), f"{len(parts)} secciones de guía")
