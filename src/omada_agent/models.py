"""Plain data containers shared by ingestion, retrieval and the agent."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass(slots=True)
class Doc:
    """One searchable unit of knowledge.

    ``kind == "operation"`` -> one API operation (method + path) from the OpenAPI spec.
    ``kind == "guide"``     -> one chunk of a hand-written Markdown guide.
    """

    doc_id: str
    kind: str
    title: str
    method: str = ""
    path: str = ""
    path_norm: str = ""
    version: str = ""
    tag: str = ""
    tags: list[str] = field(default_factory=list)
    deprecated: bool = False
    is_template: bool = False
    summary: str = ""
    description: str = ""
    permissions: list[str] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)

    # Compact view shown in search results (what the LLM sees first).
    brief_text: str = ""
    # Text that is embedded (vector search).
    card_text: str = ""
    # Full-text-search columns (keyword search).
    fts_title: str = ""
    fts_keywords: str = ""
    fts_description: str = ""
    fts_fields: str = ""
    # Detail sections served on demand by the ``get_operation`` tool.
    header_text: str = ""
    request_text: str = ""
    response_text: str = ""

    @property
    def embed_hash(self) -> str:
        """Hash of exactly what gets embedded; drives incremental re-indexing."""
        return hashlib.sha256(self.card_text.encode("utf-8")).hexdigest()[:20]

    def to_card(self, score: float | None = None) -> dict:
        """Lightweight JSON-safe view used by the web UI and tool outputs."""
        card = {
            "id": self.doc_id,
            "kind": self.kind,
            "method": self.method,
            "path": self.path,
            "title": self.title,
            "summary": self.summary,
            "tag": self.tag,
            "version": self.version,
            "deprecated": self.deprecated,
            "template": self.is_template,
        }
        if score is not None:
            card["score"] = round(score, 4)
        return card


@dataclass(slots=True)
class Filters:
    """Optional constraints applied to both vector and keyword search."""

    kind: str | None = None  # "operation" | "guide"
    method: str | None = None
    tag: str | None = None
    include_deprecated: bool = False
    include_templates: bool = True


@dataclass(slots=True)
class Hit:
    doc: Doc
    score: float
    vector_rank: int | None = None
    keyword_rank: int | None = None
    exact: bool = False
