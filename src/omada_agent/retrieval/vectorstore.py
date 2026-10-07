"""Persistent vector index backed by ChromaDB (HNSW, cosine distance).

ChromaDB runs in-process and stores everything in a folder, so there is no
server to install or operate. Only vectors and light metadata live here; the
document text itself lives in SQLite (see ``docstore.py``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from omada_agent.models import Doc, Filters

log = logging.getLogger(__name__)

COLLECTION = "omada_docs"
UPSERT_BATCH = 500


class VectorStore:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(directory),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    # ------------------------------------------------------------------ maintenance
    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(COLLECTION)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    def existing_hashes(self) -> dict[str, str]:
        """Map ``doc_id -> embed_hash`` for everything currently indexed."""
        out: dict[str, str] = {}
        total = self._collection.count()
        for offset in range(0, total, 1000):
            page = self._collection.get(limit=1000, offset=offset, include=["metadatas"])
            for doc_id, meta in zip(page["ids"], page["metadatas"]):
                out[doc_id] = (meta or {}).get("embed_hash", "")
        return out

    def upsert(self, docs: list[Doc], vectors: list[list[float]]) -> None:
        for start in range(0, len(docs), UPSERT_BATCH):
            chunk = docs[start : start + UPSERT_BATCH]
            self._collection.upsert(
                ids=[d.doc_id for d in chunk],
                embeddings=vectors[start : start + UPSERT_BATCH],
                metadatas=[
                    {
                        "kind": d.kind,
                        "method": d.method,
                        "tag": d.tag,
                        "deprecated": d.deprecated,
                        "template": d.is_template,
                        "embed_hash": d.embed_hash,
                    }
                    for d in chunk
                ],
            )

    def delete(self, ids: list[str]) -> None:
        for start in range(0, len(ids), UPSERT_BATCH):
            self._collection.delete(ids=ids[start : start + UPSERT_BATCH])

    # ------------------------------------------------------------------ search
    @staticmethod
    def _where(filters: Filters | None) -> dict | None:
        clauses: list[dict] = []
        if filters:
            if filters.kind:
                clauses.append({"kind": filters.kind})
            if filters.method:
                clauses.append({"method": filters.method.upper()})
            if filters.tag:
                clauses.append({"tag": filters.tag})
            if not filters.include_deprecated:
                clauses.append({"deprecated": False})
            if not filters.include_templates:
                clauses.append({"template": False})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def query(
        self, vector: list[float], limit: int, filters: Filters | None = None
    ) -> list[tuple[str, float]]:
        """Nearest neighbours as ``(doc_id, cosine_distance)``, best first."""
        if self._collection.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[vector],
            n_results=min(limit, self._collection.count()),
            where=self._where(filters),
            include=["distances"],
        )
        return list(zip(result["ids"][0], result["distances"][0]))
