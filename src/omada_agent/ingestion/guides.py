"""Load hand-written guides (Markdown / text) so they are searchable too.

The OpenAPI specification describes *endpoints*, but not everything a developer
needs: authentication flows, access modes, time-zone tables, etc. live in the
prose guides of the Omada documentation. Drop those pages as ``.md`` or ``.txt``
files into ``data/guides/`` and they become part of the knowledge base.

Files whose name starts with ``_`` or ``README`` are ignored.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from omada_agent.ingestion.openapi_parser import split_words
from omada_agent.models import Doc

log = logging.getLogger(__name__)

GUIDE_SUFFIXES = {".md", ".markdown", ".txt"}
CHUNK_TARGET_CHARS = 1600
CHUNK_MAX_CHARS = 2400

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


def _iter_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    files = [
        p
        for p in sorted(directory.rglob("*"))
        if p.is_file()
        and p.suffix.lower() in GUIDE_SUFFIXES
        and not p.name.startswith("_")
        and not p.name.lower().startswith("readme")
    ]
    return files


def chunk_markdown(text: str) -> list[tuple[str, str]]:
    """Split Markdown into ``(heading path, body)`` chunks of roughly 1.6k characters.

    * Splits at headings and blank lines (never inside a fenced code block).
    * Each chunk remembers its heading trail, e.g. ``"Authorization > Client credentials"``.
    """
    chunks: list[tuple[str, str]] = []
    trail: list[tuple[int, str]] = []
    buffer: list[str] = []
    in_fence = False

    def trail_text() -> str:
        return " > ".join(title for _, title in trail)

    def flush() -> None:
        body = "\n".join(buffer).strip()
        buffer.clear()
        if body:
            chunks.append((trail_text(), body))

    for line in text.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        heading = None if in_fence else _HEADING_RE.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            while trail and trail[-1][0] >= level:
                trail.pop()
            trail.append((level, heading.group(2)))
            continue
        buffer.append(line)
        size = sum(len(x) + 1 for x in buffer)
        at_break = not in_fence and not line.strip()
        if (size >= CHUNK_TARGET_CHARS and at_break) or size >= CHUNK_MAX_CHARS:
            flush()
    flush()
    return chunks


def load_guides(directory: Path) -> list[Doc]:
    docs: list[Doc] = []
    for file in _iter_files(directory):
        rel = file.relative_to(directory).as_posix()
        try:
            text = file.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:  # pragma: no cover - defensive
            log.warning("Skipping unreadable guide %s: %s", rel, exc)
            continue
        for i, (trail, body) in enumerate(chunk_markdown(text), start=1):
            title = f"{file.stem}: {trail}" if trail else file.stem
            first_line = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
            docs.append(
                Doc(
                    doc_id=f"guide:{rel}#{i}",
                    kind="guide",
                    title=title,
                    path=rel,
                    tag="Guide",
                    tags=["Guide"],
                    summary=first_line[:160],
                    description=body,
                    brief_text=f"[GUIDE] {title}\n  {body[:300].replace(chr(10), ' ')}…",
                    card_text=f"{title}\n{body}",
                    fts_title=title,
                    fts_keywords=" ".join(split_words(file.stem)),
                    fts_description=body,
                    header_text=f"## {title}\n(source file: {rel})\n\n{body}",
                )
            )
    if docs:
        log.info("Loaded %d guide chunks from %s", len(docs), directory)
    return docs
