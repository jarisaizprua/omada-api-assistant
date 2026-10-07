"""Evaluation 2 - does the assistant answer correctly?

Every curated question is sent to the real agent (same prompt, tools and model as the
portal). The answer is then graded with:

1. deterministic checks - right endpoint named, key terms present, no invented paths;
2. an LLM judge (optional) - is the explanation correct and complete?

Each answer ends in one of four states: ok / partial / bad / skipped.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from omada_agent.evals.dataset import Question, full_reference
from omada_agent.evals.judge import judge_answer
from omada_agent.evals.scoring import (
    Catalog,
    estimate_cost,
    find_invented_paths,
    find_suspicious_names,
    looks_like_refusal,
    mentions_operation,
    missing_terms,
)


async def _ask(agent, question: str) -> dict[str, Any]:
    text: list[str] = []
    tools: list[dict] = []
    retrieved: list[str] = []
    out: dict[str, Any] = {"error": None, "usage": {}, "elapsed_ms": 0}
    started = time.perf_counter()
    async for ev in agent.run([{"role": "user", "content": question}]):
        kind = ev["type"]
        if kind == "token":
            text.append(ev["text"])
        elif kind == "tool_start":
            tools.append({"name": ev["name"], "args": ev["args"]})
        elif kind == "tool_result":
            retrieved += [op["id"] for op in ev.get("operations", [])]
        elif kind == "error":
            out["error"] = ev["message"]
        elif kind == "done":
            out["usage"] = ev["usage"]
            out["elapsed_ms"] = ev["elapsed_ms"]
    if not out["elapsed_ms"]:
        out["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
    out.update(answer="".join(text), tools=tools, retrieved=list(dict.fromkeys(retrieved)))
    return out


def grade(
    q: Question,
    run: dict[str, Any],
    catalog: Catalog,
    judgement: dict | None,
    judge_independent: bool = True,
) -> dict[str, Any]:
    """Turn one raw run into checks, a status and plain-language reasons."""
    answer: str = run["answer"]
    reasons: list[str] = []
    ops = [catalog.operations[i] for i in q.expect_operations if i in catalog.operations]
    ops_all = [catalog.operations[i] for i in q.expect_all_operations if i in catalog.operations]

    named_any = [o["doc_id"] for o in ops if mentions_operation(answer, o)]
    named_all = [o["doc_id"] for o in ops_all if mentions_operation(answer, o)]
    has_expectation = bool(ops or ops_all)
    endpoint_ok = (not ops or bool(named_any)) and len(named_all) == len(ops_all)
    retrieved = set(run["retrieved"])
    retrieval_ok = (not ops or any(o["doc_id"] in retrieved for o in ops)) and all(
        o["doc_id"] in retrieved for o in ops_all
    )

    lacking = missing_terms(answer, q.all_terms)
    invented = find_invented_paths(answer, catalog)
    suspicious = find_suspicious_names(answer, catalog)
    tool_names = [t["name"] for t in run["tools"]]
    looked_up = any(n in ("search_api", "search_guides") for n in tool_names)
    read_detail = "get_operation" in tool_names

    judge_ok = judgement is not None and "error" not in judgement
    verdict = judgement.get("verdict") if judge_ok else None
    # A judge that is the very same model as the assistant tends to mark correct answers wrong (it
    # does not know the spec better than the assistant does). Its "incorrect" is then only a warning
    # unless a deterministic check agrees; an independent, stronger judge keeps full weight.
    judge_blocks = verdict == "incorrect" and judge_independent

    if run["error"]:
        status = "bad"
        reasons.append(f"El asistente devolvió un error: {run['error']}")
    elif q.out_of_scope:
        refused = (verdict != "incorrect") if (judge_ok and judge_independent) else looks_like_refusal(answer)
        status = "ok" if refused and not invented else "bad"
        if not refused:
            reasons.append("Debía decir que esa función no existe en la API, pero no lo reconoció.")
        if invented:
            reasons.append("Inventó rutas que no existen: " + ", ".join(invented[:3]))
    else:
        # For "what does error X mean?" the explanation matters more than naming the endpoint.
        endpoint_required = has_expectation and q.type != "errores"
        if has_expectation and not endpoint_ok and endpoint_required:
            wanted = [o["doc_id"] for o in ops + ops_all if o["doc_id"] not in named_any + named_all]
            reasons.append(f"No mencionó el endpoint esperado ({', '.join(wanted)}).")
            reasons.append(
                "La búsqueda sí lo había encontrado, pero el modelo no lo usó."
                if retrieval_ok
                else "La búsqueda tampoco lo encontró."
            )
        if invented:
            reasons.append("Inventó rutas que no existen: " + ", ".join(invented[:3]))
        if lacking:
            reasons.append("Faltó mencionar: " + ", ".join(lacking) + ".")
        if has_expectation and not endpoint_ok and not endpoint_required:
            reasons.append("No nombró la operación donde aparece el error (conviene, pero no es indispensable).")
        if verdict in ("partial", "incorrect") and judgement:
            prefix = "Juez" if judge_independent or verdict == "partial" else "Juez (mismo modelo, solo aviso)"
            reasons.append(f"{prefix}: " + judgement.get("comment", ""))
        all_terms_lost = bool(q.all_terms) and len(lacking) == len(q.all_terms)
        hard_bad = (
            (endpoint_required and not endpoint_ok) or bool(invented) or judge_blocks or all_terms_lost
        )
        soft = (
            bool(lacking)
            or verdict == "partial"
            or (verdict == "incorrect" and not judge_independent)
            or (has_expectation and not endpoint_ok and not endpoint_required)
        )
        status = "bad" if hard_bad else "partial" if soft else "ok"

    if suspicious and status != "bad":
        reasons.append("Nombres a revisar (parecen operaciones pero no existen): " + ", ".join(suspicious[:3]))
    if not looked_up and not q.out_of_scope:
        reasons.append("Respondió sin consultar la documentación.")

    return {
        "status": status,
        "reasons": reasons,
        "checks": {
            "endpoint_ok": endpoint_ok if has_expectation else None,
            "retrieval_ok": retrieval_ok if has_expectation else None,
            "terms_missing": lacking,
            "invented_paths": invented,
            "suspicious_names": suspicious,
            "looked_up": looked_up,
            "read_detail": read_detail,
        },
    }


async def run_answer_eval(
    agent,
    catalog: Catalog,
    questions: list[Question],
    *,
    model: str,
    judge_client=None,
    judge_model: str | None = None,
    concurrency: int = 3,
    price_override: tuple[float | None, float | None] = (None, None),
    progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    gate = asyncio.Semaphore(max(1, concurrency))
    done = 0

    async def one(q: Question) -> dict[str, Any]:
        nonlocal done
        base = {"id": q.id, "topic": q.topic, "difficulty": q.difficulty, "type": q.type, "question": q.question}
        if q.needs_guide and not catalog.guides:
            done += 1
            if progress:
                progress(done, len(questions))
            return {**base, "status": "skipped", "reasons": ["Requiere la guía de acceso (aún no importada)."]}
        async with gate:
            run = await _ask(agent, q.question)
            judgement = None
            if judge_client and run["answer"] and not run["error"]:
                judgement = await judge_answer(
                    judge_client, judge_model or model, q.question, full_reference(q, catalog), run["answer"]
                )
        graded = grade(q, run, catalog, judgement, judge_independent=bool(judge_model) and judge_model != model)
        usage = run["usage"]
        cost = estimate_cost(model, usage.get("input_tokens", 0), usage.get("output_tokens", 0), price_override)
        done += 1
        if progress:
            progress(done, len(questions))
        return {
            **base,
            **graded,
            "answer": run["answer"],
            "judge": judgement,
            "tools": run["tools"],
            "usage": usage,
            "elapsed_ms": run["elapsed_ms"],
            "cost": cost,
        }

    return list(await asyncio.gather(*(one(q) for q in questions)))
