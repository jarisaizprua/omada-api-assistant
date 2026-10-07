"""The index manifest: a small JSON file describing what the index was built from.

It lets the server verify at start-up that query embeddings use the same model
as the stored vectors (mixing models silently ruins retrieval quality) and feeds
the "knowledge base" card in the web UI.
"""

from __future__ import annotations

import json
from pathlib import Path

MANIFEST_NAME = "manifest.json"
DOCSTORE_NAME = "docs.sqlite3"
CHROMA_DIRNAME = "chroma"


class IndexNotBuiltError(RuntimeError):
    """The knowledge base has not been built yet. The message says how to build it."""


def manifest_path(index_dir: Path) -> Path:
    return index_dir / MANIFEST_NAME


def read_manifest(index_dir: Path) -> dict:
    path = manifest_path(index_dir)
    if not path.exists() or not (index_dir / DOCSTORE_NAME).exists():
        raise IndexNotBuiltError(
            "The knowledge base has not been built yet. Run:  python manage.py build-index"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(index_dir: Path, manifest: dict) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    manifest_path(index_dir).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
