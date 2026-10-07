"""Hybrid retrieval: vector search + keyword search fused with Reciprocal Rank Fusion.

Why hybrid?
-----------
* **Vectors** understand intent ("how do I add a Wi-Fi network?") even when the
  words differ from the documentation.
* **Keywords (BM25)** nail exact tokens: ``createNewSite``, ``-33101``, ``siteId``.
* **Reciprocal Rank Fusion** merges both rankings using only ranks, so the two
  incomparable score scales never need to be normalised.
* Exact ``operationId`` / ``METHOD /path`` mentions get a decisive boost.

If embeddings are unavailable (no API key, model mismatch, API outage) the
retriever transparently degrades to keyword-only search instead of failing.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, replace

from omada_agent.config import Settings
from omada_agent.models import Filters, Hit
from omada_agent.retrieval.docstore import DocStore
from omada_agent.retrieval.embeddings import Embedder, OpenAIEmbedder
from omada_agent.retrieval.queryexpand import expand_query, expansion_terms, keyword_variant
from omada_agent.retrieval.manifest import (
    CHROMA_DIRNAME,
    DOCSTORE_NAME,
    read_manifest,
)
from omada_agent.retrieval.vectorstore import VectorStore

log = logging.getLogger(__name__)

EXACT_MATCH_BOOST = 0.5
TEMPLATE_PENALTY = 0.93
DEPRECATED_PENALTY = 0.85
ACTION_MATCH_BOOST = 1.12
ACTION_CONFLICT_PENALTY = 0.93
# Weight of the search that uses only the API's own words ("reboot device") when a vocabulary rule fired.
VOCAB_LEG_WEIGHT = 2.0

# The first word of an Omada summary is its action ("Modify port forwarding"). Words that mean
# the same thing are grouped, so a question about *modifying* something prefers "Modify ..."
# over "Delete ..." even when both share every other word.
_ACTION_GROUPS = {
    "read": {"get", "list", "show", "view", "fetch", "query", "read", "retrieve", "find", "check"},
    "create": {"create", "add", "new", "make", "insert"},
    "change": {"modify", "update", "edit", "change", "set", "configure", "enable", "disable"},
    "remove": {"delete", "remove", "forget", "clear", "drop"},
}
_ACTION_OF = {word: group for group, words in _ACTION_GROUPS.items() for word in words}


def _action_of_summary(summary: str) -> str | None:
    first = summary.strip().split(" ", 1)[0].lower() if summary.strip() else ""
    return _ACTION_OF.get(first)


def _actions_in_query(query: str) -> set[str]:
    return {_ACTION_OF[w] for w in re.findall(r"[a-z]+", query.lower()) if w in _ACTION_OF}


@dataclass(slots=True)
class SearchResult:
    hits: list[Hit]
    mode: str  # "hybrid" | "keyword"
    warning: str | None = None
    queries: list[str] = field(default_factory=list)


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = 60, weights: list[float] | None = None
) -> dict[str, float]:
    """Combine several ranked id lists. ``score(d) = sum_i w_i / (k + rank_i(d))``."""
    weights = weights or [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
    return scores


class Retriever:
    def __init__(self, settings: Settings, embedder: Embedder | None = None):
        self.settings = settings
        index_dir = settings.index_location
        self.manifest = read_manifest(index_dir)
        self.store = DocStore(index_dir / DOCSTORE_NAME)

        self.vectors: VectorStore | None = None
        self.embedder: Embedder | None = None
        self.semantic_status = "disabled"

        if not self.manifest.get("embedding", {}).get("vectors"):
            self.semantic_status = "index built without vectors (keyword-only)"
            return

        built_with = self.manifest["embedding"]["model"]
        if embedder is None and not settings.has_openai_key:
            self.semantic_status = "OPENAI_API_KEY missing (keyword-only)"
            return
        if embedder is None and built_with != settings.openai_embedding_model:
            self.semantic_status = (
                f"embedding model mismatch: index={built_with}, "
                f".env={settings.openai_embedding_model}. Rebuild with: "
                "python manage.py build-index --full"
            )
            log.warning(self.semantic_status)
            return

        self.embedder = embedder or OpenAIEmbedder(settings)
        self.vectors = VectorStore(index_dir / CHROMA_DIRNAME)
        self.semantic_status = f"enabled ({built_with})"

    # ------------------------------------------------------------------ properties
    @property
    def semantic_available(self) -> bool:
        return self.vectors is not None and self.embedder is not None

    def info(self) -> dict:
        """Everything the UI wants to know about the knowledge base."""
        stats = self.store.stats()
        return {
            **stats,
            "spec": self.manifest.get("spec", {}),
            "built_at": self.manifest.get("built_at"),
            "embedding_model": self.manifest.get("embedding", {}).get("model"),
            "semantic": self.semantic_status,
        }

    # ------------------------------------------------------------------ search
    def search(
        self,
        query: str,
        filters: Filters | None = None,
        limit: int | None = None,
        mode: str = "hybrid",
    ) -> SearchResult:
        limit = limit or self.settings.rag_top_k
        candidates = max(self.settings.rag_candidates, limit * 3)
        query = query.strip()
        if not query:
            return SearchResult([], "keyword")

        if filters and filters.tag:
            # Vector metadata filters are case-sensitive; SQL ones are not. Use the canonical name.
            canonical, _ = self.store.resolve_tag(filters.tag)
            if canonical is None:
                return SearchResult([], "keyword", warning=f"Unknown category '{filters.tag}'.")
            filters = replace(filters, tag=canonical)

        # Add the API's own vocabulary ("restart" -> "reboot") as EXTRA search legs. The user's own
        # wording is always searched too, so expansion can only add chances, never remove them.
        # Exact lookups and action detection keep using what the user actually wrote.
        expand = self.settings.rag_query_expansion
        extra = expansion_terms(query) if expand else []
        expanded_text = expand_query(query) if extra else query
        keyword_text = keyword_variant(query) if extra else query
        vocab_text = " ".join(extra) if len(extra) >= 2 else ""

        exact_ids = self.store.lookup_exact(query, filters)
        keyword = self.store.keyword_search(query, candidates, filters)
        extra_keyword: list[list[tuple[str, float]]] = []
        if keyword_text != query:
            extra_keyword.append(self.store.keyword_search(keyword_text, candidates, filters))
        vocab: list[tuple[str, float]] = (
            self.store.keyword_search(vocab_text, candidates, filters) if vocab_text else []
        )

        vector: list[tuple[str, float]] = []
        vector_expanded: list[tuple[str, float]] = []
        used_mode, warning = "keyword", None
        if mode != "keyword":
            if self.semantic_available:
                try:
                    vec = self.embedder.embed_query(query)  # type: ignore[union-attr]
                    vector = self.vectors.query(vec, candidates, filters)  # type: ignore[union-attr]
                    if expanded_text != query:
                        vec2 = self.embedder.embed_query(expanded_text)  # type: ignore[union-attr]
                        vector_expanded = self.vectors.query(vec2, candidates, filters)  # type: ignore[union-attr]
                    used_mode = "hybrid"
                except Exception as exc:  # network, quota, bad key...
                    log.warning("Semantic search failed, using keyword only: %s", exc)
                    warning = f"Semantic search unavailable ({type(exc).__name__}); used keyword search."
            else:
                warning = f"Semantic search disabled: {self.semantic_status}"

        rankings = [[i for i, _ in vector], [i for i, _ in keyword]]
        weights = [1.0, 1.0]
        if vector_expanded:
            rankings.append([i for i, _ in vector_expanded])
            weights.append(1.0)
        for leg in extra_keyword:
            rankings.append([i for i, _ in leg])
            weights.append(1.0)
        if vocab:
            rankings.append([i for i, _ in vocab])
            weights.append(VOCAB_LEG_WEIGHT)
        scores = reciprocal_rank_fusion(rankings, k=self.settings.rag_rrf_k, weights=weights)
        for doc_id in exact_ids:
            scores[doc_id] = scores.get(doc_id, 0.0) + EXACT_MATCH_BOOST

        # Re-rank a slightly larger pool: unless the user asked for them, site-template variants and
        # deprecated operations should not outrank the regular, current operation.
        pool = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: limit * 2]
        docs = self.store.get_many([i for i, _ in pool])
        wants_templates = "template" in query.lower()
        query_actions = _actions_in_query(query)
        adjusted: dict[str, float] = {}
        for doc_id, score in pool:
            doc = docs.get(doc_id)
            if doc is not None:
                if doc.is_template and not wants_templates:
                    score *= TEMPLATE_PENALTY
                if doc.deprecated:
                    score *= DEPRECATED_PENALTY
                if doc.kind == "operation" and query_actions:
                    action = _action_of_summary(doc.summary)
                    if action in query_actions:
                        score *= ACTION_MATCH_BOOST
                    elif action is not None and len(query_actions) == 1:
                        score *= ACTION_CONFLICT_PENALTY
            adjusted[doc_id] = score
        ranked = sorted(adjusted.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        vec_rank = {i: r for r, (i, _) in enumerate(vector, 1)}
        kw_rank = {i: r for r, (i, _) in enumerate(keyword, 1)}
        exact_set = set(exact_ids)

        hits = [
            Hit(
                doc=docs[doc_id],
                score=score,
                vector_rank=vec_rank.get(doc_id),
                keyword_rank=kw_rank.get(doc_id),
                exact=doc_id in exact_set,
            )
            for doc_id, score in ranked
            if doc_id in docs
        ]
        return SearchResult(hits=hits, mode=used_mode, warning=warning, queries=[query, expanded_text] if expanded_text != query else [query])
