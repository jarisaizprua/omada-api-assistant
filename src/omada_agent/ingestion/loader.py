"""Locate and load the OpenAPI specification from disk.

The spec can be dropped into ``data/spec/`` exactly as downloaded from the
Omada controller: either the raw ``.json`` file or the ``.zip`` that contains it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_VERSION_RE = re.compile(r"(\d+\.\d+(?:\.\d+)*)")


class SpecNotFoundError(FileNotFoundError):
    """No usable specification was found. The message tells the user what to do."""


@dataclass(slots=True)
class LoadedSpec:
    data: dict
    source_name: str  # e.g. "Omada_CBC_Standard_6.3.0.json"
    sha256: str
    controller_version: str  # e.g. "6.3.0" ("unknown" if it cannot be inferred)


def find_spec_file(location: Path) -> Path:
    """Return the spec file to use.

    ``location`` may be a file or a directory. In a directory, the most recently
    modified ``.json``/``.zip`` file wins.
    """
    if location.is_file():
        return location
    if not location.exists():
        raise SpecNotFoundError(
            f"Spec location not found: {location}\n"
            "Place the OpenAPI file (.json or .zip) inside data/spec/ and try again."
        )
    candidates = sorted(
        (p for p in location.iterdir() if p.suffix.lower() in {".json", ".zip"}),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise SpecNotFoundError(
            f"No .json or .zip file found in {location}\n"
            "Download the OpenAPI specification from your Omada controller "
            "(Settings > Platform Integration > Open API > Online API Document) "
            "and place it in that folder."
        )
    if len(candidates) > 1:
        log.warning(
            "Several spec files found in %s; using the newest one: %s",
            location,
            candidates[0].name,
        )
    return candidates[0]


def _read_json_bytes(path: Path) -> tuple[bytes, str]:
    """Return (raw bytes, inner file name) for a .json or a .zip containing one."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            json_members = [i for i in zf.infolist() if i.filename.lower().endswith(".json")]
            if not json_members:
                raise SpecNotFoundError(f"{path.name} does not contain any .json file.")
            member = max(json_members, key=lambda i: i.file_size)
            return zf.read(member), Path(member.filename).name
    return path.read_bytes(), path.name


def load_spec(location: Path) -> LoadedSpec:
    path = find_spec_file(location)
    raw, inner_name = _read_json_bytes(path)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{inner_name} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or "paths" not in data:
        raise ValueError(f"{inner_name} does not look like an OpenAPI document (no 'paths').")

    match = _VERSION_RE.search(inner_name)
    version = match.group(1) if match else "unknown"
    return LoadedSpec(
        data=data,
        source_name=inner_name,
        sha256=hashlib.sha256(raw).hexdigest(),
        controller_version=version,
    )
