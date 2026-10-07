"""The curated question set (``evals/questions.json``) and its consistency checks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
import re
from pathlib import Path

from omada_agent.evals.scoring import Catalog, fold, term_present

TYPES = {"buscar", "configurar", "errores", "varios_pasos", "autenticacion", "referencia", "fuera_de_alcance"}
DIFFICULTIES = ("facil", "media", "dificil")


@dataclass
class Question:
    id: str
    topic: str
    difficulty: str
    type: str
    question: str
    reference: str = ""
    expect_operations: list[str] = field(default_factory=list)  # naming ANY of these is enough
    expect_all_operations: list[str] = field(default_factory=list)  # ALL of these must be named
    expect_terms: list[str] = field(default_factory=list)  # all must appear; verified against the docs by `lint`
    derived_terms: list[str] = field(default_factory=list)  # must appear but are not literally in the docs (e.g. a unit conversion)
    needs_guide: bool = False  # only answerable once the access guide is imported

    @property
    def out_of_scope(self) -> bool:
        return self.type == "fuera_de_alcance"

    @property
    def all_terms(self) -> list[str]:
        return [*self.expect_terms, *self.derived_terms]

    @property
    def all_expected_ids(self) -> list[str]:
        return [*self.expect_operations, *self.expect_all_operations]


def load_questions(path: Path) -> list[Question]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    questions = [Question(**item) for item in raw["questions"]]
    seen: set[str] = set()
    for q in questions:
        if q.id in seen:
            raise ValueError(f"Duplicate question id: {q.id}")
        seen.add(q.id)
        if q.type not in TYPES:
            raise ValueError(f"{q.id}: unknown type '{q.type}' (use one of {sorted(TYPES)})")
        if q.difficulty not in DIFFICULTIES:
            raise ValueError(f"{q.id}: unknown difficulty '{q.difficulty}'")
    return questions


_REQUIRED_TOP_RE = re.compile(r"^- ([A-Za-z0-9_.\-]+)\*", re.MULTILINE)
SPEC_EXTRACT_CHARS = 2500


def spec_facts(catalog: Catalog, op_id: str) -> str:
    """What the specification itself says about an operation (always right; no human in the loop).

    Hand-written reference notes can be incomplete, so the judge also receives the real permissions,
    error codes, required fields and a slice of the request schema.
    """
    doc = catalog.store.get(op_id) if catalog.store else None
    if doc is None:
        return ""
    lines = [f"[{op_id}] {doc.method} {doc.path} - {doc.summary}"]
    if doc.permissions:
        lines.append("Permissions (any one): " + "; ".join(doc.permissions))
    if doc.errors:
        lines.append("Error codes: " + "; ".join(f"{c} = {m}" for c, m in doc.errors))
    required = list(dict.fromkeys(_REQUIRED_TOP_RE.findall(doc.request_text or "")))
    if required:
        lines.append("Required fields (marked * in the request body, nested ones included): " + ", ".join(required))
    lines.append("Request schema excerpt:\n" + (doc.request_text or "")[:SPEC_EXTRACT_CHARS])
    return "\n".join(lines)


def full_reference(q: Question, catalog: Catalog) -> str:
    """Hand-written notes plus the exact operations, and the specification's own facts about them."""
    lines: list[str] = []
    facts = [spec_facts(catalog, op_id) for op_id in q.all_expected_ids if op_id in catalog.operations]
    facts = [f for f in facts if f]
    if facts:
        lines.append("AUTHORITATIVE FACTS FROM THE SPECIFICATION (always correct and complete):")
        lines += facts
    if q.reference:
        lines.append("EXPERT NOTES (may be incomplete; they never make a correct extra detail wrong):")
        lines.append(q.reference.strip())
    for op_id in q.all_expected_ids:
        op = catalog.operations.get(op_id)
        if op:
            lines.append(f"Expected operation: {op_id} = {op['method']} {op['path']} ({op['summary']})")
    if q.expect_all_operations:
        lines.append("All of the operations above are needed (it is a multi-step task).")
    elif len(q.expect_operations) > 1:
        lines.append("Naming any one of the operations above is acceptable.")
    return "\n".join(lines)


def lint(questions: list[Question], catalog: Catalog) -> tuple[list[str], list[str]]:
    """Check the dataset against the index. Returns (problems, notes)."""
    problems: list[str] = []
    notes: list[str] = []
    for q in questions:
        for op_id in q.all_expected_ids:
            if op_id not in catalog.operations:
                problems.append(f"{q.id}: expected operation '{op_id}' does not exist in this spec")
        if q.needs_guide and not catalog.guides:
            notes.append(f"{q.id}: needs the access guide (not imported) - it will be skipped")
            continue
        # every expected term should be verifiable in the documentation
        corpus = ""
        for op_id in q.all_expected_ids:
            doc = catalog.store.get(op_id) if op_id in catalog.operations else None
            if doc:
                corpus += f"\n{doc.header_text}\n{doc.request_text}\n{doc.response_text}\n{doc.path}"
        if q.needs_guide:
            corpus += "\n" + catalog.store.guide_corpus()
        if q.expect_terms and (corpus.strip() or q.needs_guide):
            low = fold(corpus)
            for term in q.expect_terms:
                if not term_present(low, term):
                    problems.append(f"{q.id}: term '{term}' is not found in the documentation of its operations")
        if q.out_of_scope and q.all_expected_ids:
            problems.append(f"{q.id}: out-of-scope questions must not expect operations")
        if not q.out_of_scope and not (q.all_expected_ids or q.all_terms or q.reference):
            problems.append(f"{q.id}: nothing to check (no operations, terms or reference)")
    return problems, notes
