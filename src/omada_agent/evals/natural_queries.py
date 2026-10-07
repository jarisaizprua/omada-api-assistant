"""Natural-language search questions for the search evaluation.

The default search test asks "How do I <summary of the operation>?", which repeats the words of the
documentation and therefore flatters the search. Real people say "restart an AP" where the API says
"Reboot device". This module asks an LLM once to rewrite each sampled operation as the question a
developer would type *without* reusing the API's wording, and saves the result in
``evals/natural_queries.json`` so later runs are free and reproducible.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from omada_agent.evals.judge import extract_json
from omada_agent.evals.retrieval_eval import RetrievalCase, build_cases
from omada_agent.retrieval.hybrid import Retriever

log = logging.getLogger(__name__)

BATCH = 12

INSTRUCTIONS = """\
You help test a search engine for the TP-Link Omada Open API documentation.
For each operation below, write ONE short question (6-14 words, in English) that a network engineer or \
developer might type when they need that operation but do not know its API name.

Rules:
- Do NOT reuse the distinctive words of the summary. Use everyday wording and synonyms \
(for example "restart" instead of "reboot", "Wi-Fi network" instead of "SSID", "remove" instead of "delete").
- The question must still identify that operation specifically (mention the object and the intent).
- No operation names, no URLs, no code.

Reply with ONLY a JSON object that maps each id to its question: {"<id>": "<question>", ...}
"""


def _describe(op: dict) -> dict:
    return {"id": op["doc_id"], "summary": op["summary"], "method": op["method"], "category": op["tag"]}


async def _ask(client, model: str, ops: list[dict]) -> dict[str, str]:
    payload = json.dumps([_describe(o) for o in ops], ensure_ascii=False)
    try:
        response = await client.responses.create(
            model=model, instructions=INSTRUCTIONS, input=[{"role": "user", "content": payload}]
        )
        data = extract_json(getattr(response, "output_text", "") or "")
        return {k: str(v).strip() for k, v in (data or {}).items() if isinstance(v, str) and v.strip()}
    except Exception as exc:  # noqa: BLE001 - a failed batch just yields fewer cases
        log.warning("Natural-query batch failed: %s", exc)
        return {}


def load_cached(path: Path, controller_version: str) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if data.get("controller_version") not in (controller_version, None):
        return {}  # a different spec version: operations may have changed
    return {k: v for k, v in data.get("queries", {}).items() if isinstance(v, str)}


async def ensure_natural_queries(
    retriever: Retriever,
    path: Path,
    *,
    samples: int,
    client=None,
    model: str | None = None,
) -> list[RetrievalCase]:
    """Cases whose queries are natural phrasings. Generates missing ones when a client is given."""
    version = (retriever.manifest.get("spec") or {}).get("controller_version") or ""
    cases = build_cases(retriever, samples)
    cached = load_cached(path, version)
    ops = {o["doc_id"]: o for o in retriever.store.iter_operations()}

    missing = [ops[c.op_id] for c in cases if c.op_id not in cached and c.op_id in ops]
    if missing and client is not None and model:
        gate = asyncio.Semaphore(4)

        async def run(batch: list[dict]) -> dict[str, str]:
            async with gate:
                return await _ask(client, model, batch)

        batches = [missing[i : i + BATCH] for i in range(0, len(missing), BATCH)]
        for result in await asyncio.gather(*(run(b) for b in batches)):
            cached.update(result)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"controller_version": version, "queries": cached}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

    out = []
    for c in cases:
        if c.op_id in cached:
            out.append(RetrievalCase(op_id=c.op_id, query=cached[c.op_id], tag=c.tag, equivalent=c.equivalent))
    return out
