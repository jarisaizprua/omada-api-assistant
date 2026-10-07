"""Build (or incrementally update) the knowledge base.

Pipeline::

    spec (.json/.zip) ─┐
                       ├─> Doc objects ─┬─> SQLite + FTS5   (keyword search, full detail)
    guides (*.md)  ────┘                └─> ChromaDB vectors (semantic search, OpenAI embeddings)

Re-running is cheap: the SQLite file is rebuilt in about a second, and only
documents whose embedded text changed are sent to the embeddings API again.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from omada_agent.config import Settings
from omada_agent.ingestion.guides import load_guides
from omada_agent.ingestion.loader import load_spec
from omada_agent.ingestion.openapi_parser import parse_operations
from omada_agent.models import Doc
from omada_agent.retrieval.docstore import DocStore
from omada_agent.retrieval.embeddings import Embedder, OpenAIEmbedder
from omada_agent.retrieval.manifest import (
    CHROMA_DIRNAME,
    DOCSTORE_NAME,
    IndexNotBuiltError,
    read_manifest,
    write_manifest,
)
from omada_agent.retrieval.vectorstore import VectorStore

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, int, int], None]


@dataclass(slots=True)
class BuildReport:
    operations: int
    guides: int
    embedded: int
    reused: int
    removed: int
    keyword_only: bool
    seconds: float
    spec_name: str
    controller_version: str
    embedding_model: str


def _unique(docs: list[Doc]) -> list[Doc]:
    seen: set[str] = set()
    out: list[Doc] = []
    for doc in docs:
        if doc.doc_id in seen:
            log.warning("Duplicate document id skipped: %s", doc.doc_id)
            continue
        seen.add(doc.doc_id)
        out.append(doc)
    return out


def build_index(
    settings: Settings,
    *,
    spec_location: Path | None = None,
    full: bool = False,
    keyword_only: bool = False,
    embedder: Embedder | None = None,
    progress: ProgressFn | None = None,
) -> BuildReport:
    """Build the knowledge base. See the module docstring for the pipeline."""
    started = time.perf_counter()
    report = progress or (lambda stage, done, total: None)
    index_dir = settings.index_location
    index_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load + parse -----------------------------------------------------------------
    report("Reading specification", 0, 1)
    spec = load_spec(spec_location or settings.spec_location)
    operations = parse_operations(spec.data)
    guides = load_guides(settings.guides_location)
    docs = _unique([*operations, *guides])
    report("Reading specification", 1, 1)
    log.info("Parsed %d operations and %d guide chunks", len(operations), len(guides))

    # 2) Keyword index (SQLite) -------------------------------------------------------
    report("Building keyword index", 0, len(docs))
    meta = {"spec_name": spec.source_name, "spec_sha256": spec.sha256}
    DocStore.build(index_dir / DOCSTORE_NAME, docs, meta)
    report("Building keyword index", len(docs), len(docs))

    embedded = reused = removed = 0
    model = settings.openai_embedding_model

    # 3) Vector index (ChromaDB) ------------------------------------------------------
    if not keyword_only:
        embedder = embedder or OpenAIEmbedder(settings)
        model = embedder.model
        store = VectorStore(index_dir / CHROMA_DIRNAME)

        previous = None
        try:
            previous = read_manifest(index_dir)
        except IndexNotBuiltError:
            pass
        prev_emb = (previous or {}).get("embedding", {})
        model_changed = bool(
            previous
            and prev_emb.get("vectors")
            and (
                prev_emb.get("model") != model
                or prev_emb.get("dimensions") != settings.openai_embedding_dimensions
            )
        )
        if full or model_changed:
            if model_changed:
                log.info("Embedding model changed; rebuilding all vectors.")
            store.reset()

        existing = store.existing_hashes()
        wanted = {d.doc_id: d for d in docs}
        stale = [i for i in existing if i not in wanted]
        if stale:
            store.delete(stale)
            removed = len(stale)

        todo = [d for d in docs if existing.get(d.doc_id) != d.embed_hash]
        reused = len(docs) - len(todo)
        if todo:
            report("Creating embeddings", 0, len(todo))
            vectors = embedder.embed_documents(
                [d.card_text for d in todo],
                on_batch=lambda done, total: report("Creating embeddings", done, total),
            )
            store.upsert(todo, vectors)
            embedded = len(todo)

    # 4) Manifest ---------------------------------------------------------------------
    write_manifest(
        index_dir,
        {
            "schema_version": 1,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "spec": {
                "name": spec.source_name,
                "sha256": spec.sha256,
                "controller_version": spec.controller_version,
            },
            "embedding": {
                "vectors": not keyword_only,
                "model": model if not keyword_only else None,
                "dimensions": settings.openai_embedding_dimensions,
            },
            "counts": {"operations": len(operations), "guides": len(guides)},
        },
    )

    return BuildReport(
        operations=len(operations),
        guides=len(guides),
        embedded=embedded,
        reused=reused,
        removed=removed,
        keyword_only=keyword_only,
        seconds=time.perf_counter() - started,
        spec_name=spec.source_name,
        controller_version=spec.controller_version,
        embedding_model=model if not keyword_only else "-",
    )
