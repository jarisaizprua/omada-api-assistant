"""Command-line interface.

    python manage.py build-index     build the knowledge base (RAG)
    python manage.py import-guide F  add a guide (auth, tables...) to the RAG
    python manage.py eval            measure search and answer quality
    python manage.py serve           start the web portal
    python manage.py search "query"  try the retriever from the terminal
    python manage.py stats           show what is indexed
    python manage.py usage           spending and usage report (who asked, how much it cost)
    python manage.py doctor          diagnose your setup
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from omada_agent import __version__
from omada_agent.config import ConfigError, Settings, get_settings
from omada_agent.logging_setup import setup_logging

console = Console()


# --------------------------------------------------------------------------- commands
def cmd_build_index(args: argparse.Namespace, settings: Settings) -> int:
    from omada_agent.ingestion.indexer import build_index
    from omada_agent.ingestion.loader import SpecNotFoundError

    if not args.keyword_only and not settings.has_openai_key:
        console.print(
            "[red]OPENAI_API_KEY is not set.[/red] Add it to your .env file, or build a "
            "keyword-only index with [bold]--keyword-only[/bold] (no API key needed, lower quality)."
        )
        return 2

    spec = Path(args.spec).expanduser() if args.spec else None
    with Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as bar:
        tasks: dict[str, int] = {}

        def on_progress(stage: str, done: int, total: int) -> None:
            if stage not in tasks:
                tasks[stage] = bar.add_task(stage, total=max(total, 1))
            bar.update(tasks[stage], completed=done, total=max(total, 1))

        try:
            report = build_index(
                settings,
                spec_location=spec,
                full=args.full,
                keyword_only=args.keyword_only,
                progress=on_progress,
            )
        except SpecNotFoundError as exc:
            bar.stop()
            console.print(f"[red]{exc}[/red]")
            return 2

    table = Table(box=box.SIMPLE_HEAD, show_header=False)
    table.add_row("Specification", f"{report.spec_name} (controller {report.controller_version})")
    table.add_row("Operations indexed", f"{report.operations:,}")
    table.add_row("Guide sections indexed", f"{report.guides:,}")
    if report.keyword_only:
        table.add_row("Vectors", "skipped (keyword-only index)")
    else:
        table.add_row("Embedding model", report.embedding_model)
        table.add_row("Embedded now / reused / removed", f"{report.embedded:,} / {report.reused:,} / {report.removed:,}")
    table.add_row("Time", f"{report.seconds:.1f} s")
    console.print(Panel(table, title="[green]Knowledge base ready", border_style="green"))
    console.print("Next step: [bold]python manage.py serve[/bold]")
    return 0


def cmd_import_guide(args: argparse.Namespace, settings: Settings) -> int:
    import re

    from omada_agent.ingestion.guide_import import normalize_guide_text
    from omada_agent.ingestion.guides import chunk_markdown

    source = Path(args.file).expanduser()
    if not source.is_file():
        console.print(f"[red]File not found:[/red] {source}")
        return 2
    text = normalize_guide_text(source.read_text(encoding="utf-8", errors="replace"))
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", args.name or source.stem).strip("_") or "guide"
    target = settings.guides_location / f"{name}.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    chunks = chunk_markdown(text)
    headings = sum(1 for line in text.splitlines() if line.startswith("#"))
    console.print(
        f"[green]Guide saved:[/green] {target}\n"
        f"  {len(text):,} characters · {headings} headings · {len(chunks)} searchable sections"
    )
    console.print("Next step: [bold]python manage.py build-index[/bold]")
    return 0


def cmd_eval(args: argparse.Namespace, settings: Settings) -> int:
    import asyncio

    from omada_agent.evals.report import render_console
    from omada_agent.evals.runner import check_dataset, default_questions_path, run_evaluation
    from omada_agent.retrieval.manifest import IndexNotBuiltError

    questions_path = Path(args.questions).expanduser() if args.questions else default_questions_path(settings)
    try:
        if args.check:
            total, problems, notes = check_dataset(settings, questions_path)
            console.print(f"{total} preguntas revisadas contra el índice actual.")
            for note in notes:
                console.print(f" [yellow]·[/yellow] {note}")
            for problem in problems:
                console.print(f" [red]✘[/red] {problem}")
            console.print("[green]El conjunto de preguntas es consistente.[/green]" if not problems else f"[red]{len(problems)} problema(s).[/red]")
            return 1 if problems else 0
    except IndexNotBuiltError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2

    needs_key = args.only != "retrieval" or not args.keyword_only
    if needs_key and not settings.has_openai_key:
        console.print("[red]OPENAI_API_KEY is not set.[/red] Usa [bold]--only retrieval --keyword-only[/bold] para una prueba gratuita de la búsqueda.")
        return 2

    with Progress(
        TextColumn("[bold cyan]{task.description}"), BarColumn(), TaskProgressColumn(), TimeElapsedColumn(), console=console
    ) as bar:
        tasks: dict[str, int] = {}

        def on_progress(stage: str, done: int, total: int) -> None:
            if stage not in tasks:
                tasks[stage] = bar.add_task(stage, total=max(total, 1))
            bar.update(tasks[stage], completed=done, total=max(total, 1))

        try:
            run = asyncio.run(
                run_evaluation(
                    settings,
                    only=args.only,
                    ids=args.ids.split(",") if args.ids else None,
                    topic=args.topic,
                    limit=args.limit,
                    judge=not args.no_judge,
                    keyword_only=args.keyword_only,
                    samples=args.samples,
                    concurrency=args.concurrency,
                    questions_path=questions_path,
                    baseline_ref=args.baseline,
                    progress=on_progress,
                    make_natural=args.natural,
                )
            )
        except (IndexNotBuiltError, ValueError) as exc:
            bar.stop()
            console.print(f"[red]{exc}[/red]")
            return 2
    render_console(run, console, run.get("report_path", ""))
    return 0


def cmd_serve(args: argparse.Namespace, settings: Settings) -> int:
    import uvicorn

    host = args.host or settings.host
    port = args.port or settings.port
    console.print(
        Panel.fit(
            f"[bold]Omada API Assistant[/bold] v{__version__}\n"
            f"Open  [cyan]http://{'localhost' if host in ('0.0.0.0', '127.0.0.1') else host}:{port}[/cyan]",
            border_style="cyan",
        )
    )
    uvicorn.run(
        "omada_agent.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=args.reload,
        reload_dirs=[str(Path(__file__).resolve().parent)] if args.reload else None,
        log_level=settings.log_level.lower(),
        access_log=False,
    )
    return 0


def cmd_search(args: argparse.Namespace, settings: Settings) -> int:
    from omada_agent.models import Filters
    from omada_agent.retrieval.hybrid import Retriever
    from omada_agent.retrieval.manifest import IndexNotBuiltError

    try:
        retriever = Retriever(settings)
    except IndexNotBuiltError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    filters = Filters(
        kind="guide" if args.guides else "operation",
        method=args.method,
        tag=args.tag,
        include_deprecated=args.deprecated,
        include_templates=not args.no_templates,
    )
    result = retriever.search(
        " ".join(args.query), filters, args.limit, "keyword" if args.keyword_only else "hybrid"
    )
    console.print(f"[dim]mode: {result.mode}" + (f" · {result.warning}" if result.warning else "") + "[/dim]")
    table = Table(box=box.SIMPLE_HEAD)
    for col in ("#", "Method", "Path", "Summary", "Category", "Score"):
        table.add_column(col, overflow="fold")
    for i, hit in enumerate(result.hits, start=1):
        d = hit.doc
        table.add_row(
            str(i),
            d.method or "-",
            d.path,
            d.summary[:60] + (" [red](deprecated)[/red]" if d.deprecated else ""),
            d.tag,
            f"{hit.score:.3f}",
        )
    console.print(table)
    return 0


def cmd_stats(args: argparse.Namespace, settings: Settings) -> int:
    from omada_agent.retrieval.hybrid import Retriever
    from omada_agent.retrieval.manifest import IndexNotBuiltError

    try:
        info = Retriever(settings).info()
    except IndexNotBuiltError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    spec = info.get("spec", {})
    console.print(
        Panel.fit(
            f"Specification: {spec.get('name')} (controller {spec.get('controller_version')})\n"
            f"Built: {info.get('built_at')}\n"
            f"Operations: {info['operations']:,}   Guides: {info['guides']:,}   "
            f"Templates: {info['templates']:,}   Deprecated: {info['deprecated']:,}\n"
            f"Embedding model: {info.get('embedding_model')}   Semantic search: {info['semantic']}",
            title="Knowledge base",
        )
    )
    table = Table(box=box.SIMPLE_HEAD, title=f"Top categories (of {len(info['categories'])})")
    table.add_column("Category")
    table.add_column("Operations", justify="right")
    for cat in info["categories"][: args.top]:
        table.add_row(cat["tag"], str(cat["count"]))
    console.print(table)
    return 0


def cmd_usage(args: argparse.Namespace, settings: Settings) -> int:
    from omada_agent.usage import BudgetGuard, UsageStore, price_for

    path = settings.usage_db_location
    if not path.exists():
        console.print(f"Todavía no hay registro de uso ({path}). Se crea con la primera pregunta en el portal.")
        return 0
    store = UsageStore(path, settings.usage_timezone)
    days = max(1, args.days)
    today, month = store.day_key(), store.month_key()

    def usd(v: float) -> str:
        return f"US$ {v:,.4f}" if v < 1 else f"US$ {v:,.2f}"

    console.print(f"[bold]Uso del asistente[/bold]  ·  zona horaria {settings.usage_timezone}  ·  modelo {settings.openai_model}\n")
    if price_for(settings) is None:
        console.print("[yellow]No conozco el precio de este modelo: los costos salen en 0. Define PRICE_INPUT_PER_MTOK y PRICE_OUTPUT_PER_MTOK.[/yellow]\n")

    limits = Table(box=box.SIMPLE_HEAD, title="Límites")
    for col in ("Límite", "Usado", "Tope", "Porcentaje"):
        limits.add_column(col, justify="left" if col == "Límite" else "right")
    t, m = store.spent(day=today), store.spent(month=month)
    rows = [
        ("Hoy (todos)", t["cost"], settings.budget_daily_usd, True),
        ("Este mes (todos)", m["cost"], settings.budget_monthly_usd, True),
    ]
    for label, used, cap, is_usd in rows:
        limits.add_row(label, usd(used), usd(cap) if cap else "sin tope", f"{used / cap:.0%}" if cap else "—")
    console.print(limits)

    by_day = store.by_day(days)
    table = Table(box=box.SIMPLE_HEAD, title=f"Últimos {days} días")
    for col in ("Día", "Preguntas", "Tokens", "Costo"):
        table.add_column(col, justify="left" if col == "Día" else "right")
    for r in by_day[-14:]:
        table.add_row(r["day"], str(r["requests"]), f"{(r['input_tokens'] or 0) + (r['output_tokens'] or 0):,}", usd(r["cost"] or 0))
    console.print(table if by_day else "Sin actividad en ese periodo.")

    users = store.top_users(days, limit=10)
    if users:
        ut = Table(box=box.SIMPLE_HEAD, title="Quién usa más")
        for col in ("Usuario", "Preguntas", "Costo", "Última vez"):
            ut.add_column(col, justify="left" if col in ("Usuario", "Última vez") else "right")
        for u in users:
            ut.add_row(u["user"], str(u["requests"]), usd(u["cost"] or 0), (u["last_seen"] or "")[:16].replace("T", " "))
        console.print(ut)
    avg = store.averages(days)
    if avg["n"]:
        console.print(
            f"Respuesta típica: {usd(avg['cost'])} · {avg['ms'] / 1000:.1f} s · {avg['tools']:.1f} consultas al RAG "
            f"({avg['n']} respuestas correctas)."
        )
    states = {r["status"]: r["requests"] for r in store.by_status(days)}
    if states.get("error") or states.get("aborted"):
        console.print(f"Con problemas: {states.get('error', 0)} con error, {states.get('aborted', 0)} detenidas por el usuario.")
    if args.user:
        mine = store.spent(day=today, user=args.user)
        snap = BudgetGuard(settings, store).snapshot(args.user)
        console.print(f"\n{args.user}: hoy {mine['requests']} preguntas, {usd(mine['cost'])} (tope {usd(snap['limits']['user_daily_usd']) if snap['limits']['user_daily_usd'] else 'sin tope'}).")
    return 0


def cmd_doctor(args: argparse.Namespace, settings: Settings) -> int:
    from omada_agent.ingestion.loader import SpecNotFoundError, find_spec_file
    from omada_agent.retrieval.manifest import IndexNotBuiltError, read_manifest
    from omada_agent.usage import price_for

    ok = True

    def check(label: str, passed: bool, detail: str = "", warn: bool = False) -> None:
        nonlocal ok
        icon = "[green]✔[/green]" if passed else ("[yellow]![/yellow]" if warn else "[red]✘[/red]")
        if not passed and not warn:
            ok = False
        console.print(f" {icon} {label}" + (f"  [dim]{detail}[/dim]" if detail else ""))

    console.print(f"[bold]Omada API Assistant {__version__}[/bold] — setup check\n")
    check("Python >= 3.10", sys.version_info >= (3, 10), sys.version.split()[0])
    env_file = Settings.model_config["env_file"]
    check(".env file", Path(str(env_file)).exists(), str(env_file))
    check("OPENAI_API_KEY", settings.has_openai_key, "set" if settings.has_openai_key else "missing")
    console.print(f"   model: {settings.openai_model} · embeddings: {settings.openai_embedding_model}")

    try:
        spec = find_spec_file(settings.spec_location)
        check("OpenAPI spec file", True, spec.name)
    except SpecNotFoundError:
        check("OpenAPI spec file", False, f"nothing in {settings.spec_location}")

    try:
        manifest = read_manifest(settings.index_location)
        emb = manifest.get("embedding", {})
        check("Knowledge base built", True, f"{manifest['counts']['operations']} operations")
        if emb.get("vectors"):
            same = emb.get("model") == settings.openai_embedding_model
            check(
                "Embedding model matches the index",
                same,
                f"index={emb.get('model')} · .env={settings.openai_embedding_model}",
            )
        else:
            check("Vector index", False, "keyword-only index", warn=True)
    except IndexNotBuiltError:
        check("Knowledge base built", False, "run: python manage.py build-index")

    price = price_for(settings)
    check(
        "Model price known (for spending limits)",
        price is not None,
        f"${price[0]}/${price[1]} per 1M tokens" if price else "set PRICE_INPUT_PER_MTOK / PRICE_OUTPUT_PER_MTOK",
        warn=True,
    )
    console.print(
        f"   limits: ${settings.budget_daily_usd}/day · ${settings.budget_monthly_usd}/month · "
        f"${settings.budget_user_daily_usd}/person/day · {settings.user_daily_requests} questions/person/day  (0 = off)"
    )
    check("Login header configured", bool(settings.portal_user_header) or not settings.portal_behind_proxy,
          settings.portal_user_header or "users are told apart by IP", warn=True)

    console.print("\n[green]All good.[/green]" if ok else "\n[red]Fix the items marked ✘ and run again.[/red]")
    return 0 if ok else 1


# --------------------------------------------------------------------------- entry point
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="manage.py", description="Omada API Assistant")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-index", help="Build or update the knowledge base (RAG)")
    p.add_argument("--spec", help="Path to the spec (.json/.zip) or a folder (default: SPEC_PATH)")
    p.add_argument("--full", action="store_true", help="Discard existing vectors and re-embed everything")
    p.add_argument("--keyword-only", action="store_true", help="Skip embeddings (no API key needed)")
    p.set_defaults(func=cmd_build_index)

    p = sub.add_parser("import-guide", help="Add a guide (plain text/Markdown) to the knowledge base")
    p.add_argument("file", help="Text or Markdown file with the guide")
    p.add_argument("--name", help="Name of the saved guide (default: the file name)")
    p.set_defaults(func=cmd_import_guide)

    p = sub.add_parser("eval", help="Evaluate search and answer quality (report in evals/results/)")
    p.add_argument("--only", choices=["retrieval", "answers"], help="Run just one of the two evaluations")
    p.add_argument("--check", action="store_true", help="Only validate the question set against the index")
    p.add_argument("--limit", type=int, help="Use only the first N questions")
    p.add_argument("--ids", help="Comma-separated question ids, e.g. Q02,Q03")
    p.add_argument("--topic", help="Only questions whose topic contains this text")
    p.add_argument("--no-judge", action="store_true", help="Skip the LLM judge (cheaper, rule-based checks only)")
    p.add_argument("--keyword-only", action="store_true", help="Search evaluation without embeddings (free)")
    p.add_argument("--samples", type=int, help="Operations sampled for the search evaluation (default 300)")
    p.add_argument(
        "--natural",
        action="store_true",
        help="Write (once, cached) everyday-wording questions for the search evaluation using the judge model",
    )
    p.add_argument("--concurrency", type=int, help="Questions answered in parallel (default 3)")
    p.add_argument("--questions", help="Alternative questions file")
    p.add_argument("--baseline", help="Compare with a previous results.json, or 'latest'")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("usage", help="Spending and usage report")
    p.add_argument("--days", type=int, default=30, help="Period to summarise (default 30)")
    p.add_argument("--user", help="Also show today's usage and limits of one user")
    p.set_defaults(func=cmd_usage)

    p = sub.add_parser("serve", help="Start the web portal")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--reload", action="store_true", help="Auto-reload on code changes (development)")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("search", help="Query the knowledge base from the terminal")
    p.add_argument("query", nargs="+")
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--method", choices=["GET", "POST", "PUT", "PATCH", "DELETE"])
    p.add_argument("--tag", help="Restrict to one category")
    p.add_argument("--guides", action="store_true", help="Search guides instead of operations")
    p.add_argument("--keyword-only", action="store_true")
    p.add_argument("--deprecated", action="store_true", help="Include deprecated operations")
    p.add_argument("--no-templates", action="store_true", help="Hide site-template operations")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("stats", help="Show what is indexed")
    p.add_argument("--top", type=int, default=15)
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("doctor", help="Check your setup")
    p.set_defaults(func=cmd_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = get_settings()
    except Exception as exc:  # invalid .env values
        console.print(f"[red]Invalid configuration in .env:[/red] {exc}")
        return 2
    setup_logging(settings.log_level, settings.log_format, settings.log_file)
    try:
        return int(args.func(args, settings) or 0)
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2
    except KeyboardInterrupt:
        return 130
