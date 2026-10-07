"""Server-Sent Events helpers."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any


def sse(event: dict[str, Any]) -> str:
    """Format one event as an SSE ``data:`` frame."""
    return f"data: {json.dumps(event, ensure_ascii=False, separators=(',', ':'))}\n\n"


async def with_heartbeat(
    source: AsyncIterator[dict[str, Any]], interval: float = 15.0
) -> AsyncIterator[dict[str, Any] | None]:
    """Yield items from ``source``; yield ``None`` whenever ``interval`` seconds pass quietly.

    The ``None`` values become SSE comments (``: ping``) that keep proxies and
    browsers from closing a connection while the model is thinking or a tool runs.
    """
    iterator = source.__aiter__()
    task: asyncio.Future | None = asyncio.ensure_future(iterator.__anext__())
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=interval)
            if not done:
                yield None
                continue
            try:
                item = task.result()
            except StopAsyncIteration:
                task = None
                return
            task = asyncio.ensure_future(iterator.__anext__())
            yield item
    finally:
        if task is not None and not task.done():
            task.cancel()
        aclose = getattr(source, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
