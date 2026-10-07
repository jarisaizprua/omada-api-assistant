"""LLM-as-judge: a second model call that reads an answer and grades it.

The deterministic checks (see ``scoring.py``) catch wrong endpoints, invented paths and
missing key terms. The judge covers what strings cannot: is the explanation correct and
complete, does it contradict the documentation, did it refuse when it should have?
It receives a short reference written by a person plus the exact expected operations.
"""

from __future__ import annotations

import json
import re

JUDGE_INSTRUCTIONS = """\
You grade answers given by an assistant that documents the TP-Link Omada Open API.
You receive the user's QUESTION, REFERENCE facts (ground truth, written by an expert) and the ANSWER.

Rules:
- The REFERENCE has two parts. "AUTHORITATIVE FACTS FROM THE SPECIFICATION" are always right and complete: \
a field, permission or error code listed there is real, and a required field listed there is truly required. \
"EXPERT NOTES" are a short summary written by a person and may be incomplete.
- Never mark something wrong only because it is absent from the expert notes. Check it against the \
authoritative facts first. Extra correct detail is fine; what is not acceptable is contradicting the \
facts, inventing endpoints, fields or values, or naming a different operation for the task.
- An answer may name an operation under its operationId or as METHOD /path; both are fine.
- Code examples build URLs with variables (f-strings, os.getenv, placeholders such as {omadacId}); that is \
normal and not an invented path.
- If the reference says the capability does not exist, a good answer clearly says it is not available \
and does not invent an endpoint.
- Be strict but fair. Do not reward length.

Scores: 5 = correct and complete; 4 = correct, minor omission; 3 = partly correct or misses a key point; \
2 = mostly wrong or misleading; 1 = wrong or invented.
Verdict: "correct" (score 4-5), "partial" (3), "incorrect" (1-2).

Reply with ONLY a JSON object (no markdown):
{"score": <1-5>, "verdict": "correct|partial|incorrect", "missing": [<short strings>], "wrong": [<short strings>], \
"comment": "<one or two plain sentences in Spanish explaining the grade>"}
"""


def extract_json(text: str) -> dict | None:
    """First JSON object found in a model reply, or ``None``."""
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def parse_judgement(text: str) -> dict:
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return {"error": "El juez no devolvió un JSON válido."}
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return {"error": "El juez devolvió un JSON inválido."}
    try:
        score = max(1, min(5, int(data.get("score", 0))))
    except (TypeError, ValueError):
        return {"error": "El juez no dio una nota válida."}
    verdict = data.get("verdict")
    if verdict not in {"correct", "partial", "incorrect"}:
        verdict = "correct" if score >= 4 else "partial" if score == 3 else "incorrect"
    return {
        "score": score,
        "verdict": verdict,
        "missing": [str(x) for x in data.get("missing", [])][:6],
        "wrong": [str(x) for x in data.get("wrong", [])][:6],
        "comment": str(data.get("comment", ""))[:400],
    }


async def judge_answer(client, model: str, question: str, reference: str, answer: str) -> dict:
    payload = f"QUESTION:\n{question}\n\nREFERENCE:\n{reference or '(none)'}\n\nANSWER:\n{answer[:6000]}"
    try:
        response = await client.responses.create(
            model=model, instructions=JUDGE_INSTRUCTIONS, input=[{"role": "user", "content": payload}]
        )
        return parse_judgement(getattr(response, "output_text", "") or "")
    except Exception as exc:  # a failed judgement must never abort the whole evaluation
        return {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
