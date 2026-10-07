"""OpenAI embeddings client with batching and a small query cache."""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Protocol

from openai import OpenAI

from omada_agent.config import Settings

log = logging.getLogger(__name__)

MAX_INPUT_CHARS = 24_000  # ~6k tokens, safely below the 8k-token input limit
BATCH_SIZE = 96
QUERY_CACHE_SIZE = 512


class Embedder(Protocol):
    """Anything that can turn text into vectors. Tests plug in a fake one."""

    model: str

    def embed_documents(
        self, texts: Sequence[str], on_batch: Callable[[int, int], None] | None = None
    ) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class OpenAIEmbedder:
    def __init__(self, settings: Settings):
        self.model = settings.openai_embedding_model
        self.dimensions = settings.openai_embedding_dimensions
        self._client = OpenAI(
            api_key=settings.require_openai_key(),
            base_url=settings.openai_base_url,
            timeout=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
        )
        self._cache: OrderedDict[str, list[float]] = OrderedDict()

    def _create(self, inputs: list[str]) -> list[list[float]]:
        kwargs: dict = {"model": self.model, "input": inputs}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions
        response = self._client.embeddings.create(**kwargs)
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]

    def embed_documents(
        self, texts: Sequence[str], on_batch: Callable[[int, int], None] | None = None
    ) -> list[list[float]]:
        vectors: list[list[float]] = []
        total = len(texts)
        for start in range(0, total, BATCH_SIZE):
            batch = [t[:MAX_INPUT_CHARS] or " " for t in texts[start : start + BATCH_SIZE]]
            vectors.extend(self._create(batch))
            if on_batch:
                on_batch(min(start + BATCH_SIZE, total), total)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        key = text.strip()
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        vector = self._create([key[:MAX_INPUT_CHARS] or " "])[0]
        self._cache[key] = vector
        if len(self._cache) > QUERY_CACHE_SIZE:
            self._cache.popitem(last=False)
        return vector
