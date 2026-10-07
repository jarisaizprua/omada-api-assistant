"""Orchestrates an evaluation run and saves its results.

``evals/results/<timestamp>/`` receives:
  report.md    the readable report
  results.json every number and every answer (for comparing runs or digging deeper)
and ``evals/results/latest.json`` always points to the most recent run.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from omada_agent import __version__
from omada_agent.agent.agent import OmadaAgent
from omada_agent.agent.tools import ToolBox
from omada_agent.config import Settings
from omada_agent.evals.answer_eval import run_answer_eval
from omada_agent.evals.dataset import Question, lint, load_questions
from omada_agent.evals.report import overall_score, recommendations, render_markdown, summarize_answers
from omada_agent.evals.natural_queries import ensure_natural_queries
from omada_agent.evals.retrieval_eval import curated_cases, run_retrieval_eval
from omada_agent.evals.scoring import Catalog
from omada_agent.retrieval.hybrid import Retriever

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, int, int], None]


def default_questions_path(settings: Settings) -> Path:
    return settings.evals_location / "questions.json"


def select_questions(
    questions: list[Question], *, ids: list[str] | None = None, topic: str | None = None, limit: int | None = None
) -> list[Question]:
    if ids:
        wanted = {i.strip().upper() for i in ids}
        questions = [q for q in questions if q.id.upper() in wanted]
    if topic:
        questions = [q for q in questions if topic.lower() in q.topic.lower()]
    if limit:
        questions = questions[:limit]
    return questions


def check_dataset(settings: Settings, questions_path: Path | None = None) -> tuple[int, list[str], list[str]]:
    """Validate the question set against the current index. Returns (count, problems, notes)."""
    retriever = Retriever(settings)
    questions = load_questions(questions_path or default_questions_path(settings))
    problems, notes = lint(questions, Catalog.from_store(retriever.store))
    return len(questions), problems, notes


def load_baseline(settings: Settings, ref: str | None) -> dict | None:
    if not ref:
        return None
    path = settings.evals_location / "results" / "latest.json" if ref == "latest" else Path(ref).expanduser()
    if not path.exists():
        log.warning("Baseline not found: %s", path)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


async def run_evaluation(
    settings: Settings,
    *,
    only: str | None = None,  # "retrieval" | "answers" | None (both)
    ids: list[str] | None = None,
    topic: str | None = None,
    limit: int | None = None,
    judge: bool = True,
    keyword_only: bool = False,
    samples: int | None = None,
    concurrency: int | None = None,
    questions_path: Path | None = None,
    baseline_ref: str | None = None,
    retriever: Retriever | None = None,
    agent=None,
    judge_client=None,
    progress: ProgressFn | None = None,
    save: bool = True,
    make_natural: bool = False,
) -> dict:
    started = datetime.now()
    clock = time.perf_counter()
    note = progress or (lambda stage, done, total: None)

    baseline = load_baseline(settings, baseline_ref)
    retriever = retriever or Retriever(settings)
    catalog = Catalog.from_store(retriever.store)
    info = retriever.info()

    run: dict = {"retrieval": None, "answers": None}

    judge_model = settings.openai_judge_model or settings.openai_model

    if only in (None, "retrieval"):
        note("Búsqueda", 0, 1)
        sample_n = samples or settings.eval_retrieval_samples
        gen_client = None
        if make_natural and not keyword_only and settings.has_openai_key:
            from openai import AsyncOpenAI

            gen_client = AsyncOpenAI(
                api_key=settings.require_openai_key(),
                base_url=settings.openai_base_url,
                timeout=settings.openai_timeout_seconds,
                max_retries=settings.openai_max_retries,
            )
        natural = await ensure_natural_queries(
            retriever,
            settings.evals_location / "natural_queries.json",
            samples=sample_n,
            client=gen_client,
            model=judge_model,
        )
        try:
            curated = curated_cases(
                load_questions(questions_path or default_questions_path(settings)), set(catalog.operations)
            )
        except (OSError, ValueError, KeyError):
            curated = []
        run["retrieval"] = run_retrieval_eval(
            retriever,
            sample_n,
            keyword_only=keyword_only,
            progress=lambda d, t: note("Búsqueda", d, t),
            natural=natural,
            curated=curated,
        )

    judge_model = settings.openai_judge_model or settings.openai_model
    judging = False
    if only in (None, "answers"):
        questions = select_questions(
            load_questions(questions_path or default_questions_path(settings)), ids=ids, topic=topic, limit=limit
        )
        if not questions:
            raise ValueError("No questions match the selected filters.")
        agent = agent or OmadaAgent(settings, ToolBox(retriever))
        if judge and judge_client is None:
            from openai import AsyncOpenAI

            judge_client = AsyncOpenAI(
                api_key=settings.require_openai_key(),
                base_url=settings.openai_base_url,
                timeout=settings.openai_timeout_seconds,
                max_retries=settings.openai_max_retries,
            )
        judging = bool(judge and judge_client)
        records = await run_answer_eval(
            agent,
            catalog,
            questions,
            model=settings.openai_model,
            judge_client=judge_client if judging else None,
            judge_model=judge_model,
            concurrency=concurrency or settings.eval_concurrency,
            price_override=(settings.eval_price_input_per_mtok, settings.eval_price_output_per_mtok),
            progress=lambda d, t: note("Respuestas", d, t),
        )
        run["answers"] = {"records": records, "summary": summarize_answers(records)}

    score, label = overall_score(run["retrieval"], (run["answers"] or {}).get("summary"))
    run.update(
        overall=score,
        verdict=label,
        meta={
            "version": __version__,
            "started_at": started.isoformat(timespec="seconds"),
            "seconds": time.perf_counter() - clock,
            "model": settings.openai_model,
            "judge_enabled": judging,
            "judge_model": judge_model if judging else None,
            "embedding_model": info.get("embedding_model"),
            "semantic": info.get("semantic"),
            "controller_version": (info.get("spec") or {}).get("controller_version"),
            "operations": info.get("operations"),
        },
    )
    run["recommendations"] = recommendations(run)
    run["report_markdown"] = render_markdown(run, baseline)

    if save:
        out_dir = settings.evals_location / "results" / started.strftime("%Y%m%d-%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.md").write_text(run["report_markdown"], encoding="utf-8")
        serializable = {k: v for k, v in run.items() if k != "report_markdown"}
        (out_dir / "results.json").write_text(json.dumps(serializable, ensure_ascii=False, indent=1), encoding="utf-8")
        shutil.copyfile(out_dir / "results.json", out_dir.parent / "latest.json")
        run["report_path"] = str(out_dir / "report.md")
    return run
