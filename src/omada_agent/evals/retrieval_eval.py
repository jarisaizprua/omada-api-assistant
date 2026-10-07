"""Evaluation 1 - can the search find the right operation?

No LLM is involved, so it is fast and almost free. For a representative sample of
operations we write the kind of question a developer would ask ("How do I get site
list?") and check at which position the right operation shows up.

Operations that share the same name and HTTP method (for example the regular and the
"v2" variants of a call) are treated as equivalent, so the metric is not punished for
a tie the documentation itself creates.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from omada_agent.models import Filters
from omada_agent.retrieval.hybrid import Retriever

TOP_K = 8


@dataclass
class RetrievalCase:
    op_id: str
    query: str
    tag: str
    equivalent: list[str]


def _question_for(summary: str) -> str:
    text = summary.strip().rstrip(".")
    return f"How do I {text[:1].lower()}{text[1:]}?"


def build_cases(retriever: Retriever, samples: int, seed: int = 7) -> list[RetrievalCase]:
    ops = [o for o in retriever.store.iter_operations() if not o["deprecated"] and not o["is_template"]]
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for o in ops:
        groups[(o["summary"].lower(), o["method"])].append(o["doc_id"])

    by_tag: dict[str, list[dict]] = defaultdict(list)
    for o in ops:
        by_tag[o["tag"]].append(o)

    rng = random.Random(seed)
    total = len(ops)
    chosen: list[dict] = []
    for tag in sorted(by_tag):  # proportional per category, at least one each
        items = by_tag[tag][:]
        rng.shuffle(items)
        quota = max(1, round(samples * len(items) / total)) if total else 0
        chosen += items[:quota]
    rng.shuffle(chosen)
    chosen = chosen[:samples] if samples < len(chosen) else chosen

    return [
        RetrievalCase(
            op_id=o["doc_id"],
            query=_question_for(o["summary"]),
            tag=o["tag"],
            equivalent=groups[(o["summary"].lower(), o["method"])],
        )
        for o in chosen
    ]


def _score(rows: list[dict]) -> dict:
    n = len(rows)
    ranks = [r["rank"] for r in rows]

    def pct(k: int) -> float:
        return sum(1 for x in ranks if x is not None and x <= k) / n if n else 0.0

    return {
        "n": n,
        "top1": pct(1),
        "top3": pct(3),
        "top8": pct(8),
        "mrr": sum(1 / x for x in ranks if x) / n if n else 0.0,
    }


def _run_cases(retriever: Retriever, cases: list[RetrievalCase], mode: str, workers: int, progress=None) -> list[dict]:
    filters = Filters(kind="operation", include_templates=False)
    done = 0

    def one(case: RetrievalCase) -> dict:
        nonlocal done
        result = retriever.search(case.query, filters, TOP_K, mode)
        rank = next((i for i, h in enumerate(result.hits, 1) if h.doc.doc_id in case.equivalent), None)
        done += 1
        if progress:
            progress(done, len(cases))
        return {
            **asdict(case),
            "rank": rank,
            "top1": result.hits[0].doc.doc_id if result.hits else None,
            "mode": result.mode,
        }

    with ThreadPoolExecutor(max_workers=workers if mode == "hybrid" else 1) as pool:
        return list(pool.map(one, cases))


def _misses(rows: list[dict], limit: int = 40) -> list[dict]:
    return [
        {"query": r["query"], "expected": r["op_id"], "got": r["top1"], "rank": r["rank"]}
        for r in rows
        if r["rank"] is None or r["rank"] > 3
    ][:limit]


def run_retrieval_eval(
    retriever: Retriever,
    samples: int,
    *,
    keyword_only: bool = False,
    workers: int = 8,
    progress: Callable[[int, int], None] | None = None,
    natural: list[RetrievalCase] | None = None,
    curated: list[RetrievalCase] | None = None,
) -> dict:
    """Three views of the same question - "does search find the right operation?":

    * literal  - "How do I <summary>?"  (easy: repeats the documentation's own words)
    * natural  - the same operations asked in everyday words (closer to real use)
    * curated  - the hand-written questions of the answer evaluation (Spanish, as users write them)
    """
    cases = build_cases(retriever, samples)
    mode = "keyword" if keyword_only or not retriever.semantic_available else "hybrid"
    rows = _run_cases(retriever, cases, mode, workers, progress)
    n = len(rows)

    by_tag: dict[str, list[int | None]] = defaultdict(list)
    for r in rows:
        by_tag[r["tag"]].append(r["rank"])
    categories = [
        {
            "tag": tag,
            "n": len(rs),
            "top3": sum(1 for x in rs if x is not None and x <= 3) / len(rs),
            "top8": sum(1 for x in rs if x is not None and x <= 8) / len(rs),
        }
        for tag, rs in by_tag.items()
    ]
    result = {
        "mode": rows[0]["mode"] if rows else mode,
        **_score(rows),
        "categories": sorted(categories, key=lambda c: (c["top3"], -c["n"])),
        "misses": _misses(rows),
        "natural": None,
        "curated": None,
    }
    for key, extra in (("natural", natural), ("curated", curated)):
        if extra:
            extra_rows = _run_cases(retriever, extra, mode, workers)
            result[key] = {**_score(extra_rows), "misses": _misses(extra_rows, 25)}
    return result


def curated_cases(questions, catalog_ids: set[str]) -> list[RetrievalCase]:
    """Search cases built from the curated questions that expect an operation."""
    cases = []
    for q in questions:
        expected = [i for i in q.all_expected_ids if i in catalog_ids]
        if q.expect_operations or q.expect_all_operations:
            if expected:
                cases.append(RetrievalCase(op_id=expected[0], query=q.question, tag=q.topic, equivalent=expected))
    return cases
