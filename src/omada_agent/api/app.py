"""FastAPI application factory.

``create_app`` accepts an optional pre-built retriever/agent so tests can inject
fakes; in production everything is built from the ``.env`` settings.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from omada_agent import __version__
from omada_agent.agent.agent import OmadaAgent
from omada_agent.agent.tools import ToolBox
from omada_agent.api.identity import identify, is_admin
from omada_agent.api.ratelimit import RateLimiter
from omada_agent.api.schemas import ChatRequest
from omada_agent.api.streaming import sse, with_heartbeat
from omada_agent.config import ConfigError, Settings, get_settings
from omada_agent.logging_setup import log_event, request_id_var
from omada_agent.models import Filters
from omada_agent.retrieval.hybrid import Retriever
from omada_agent.retrieval.manifest import IndexNotBuiltError
from omada_agent.usage import BudgetGuard, UsageStore, cost_of, price_for

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
STATIC_DIR = WEB_DIR / "static"

CSP = (
    "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
    "script-src 'self'; font-src 'self'; connect-src 'self'; object-src 'none'; "
    "base-uri 'self'; frame-ancestors 'none'"
)


@dataclass
class AppContext:
    settings: Settings
    retriever: Retriever | None = None
    agent: OmadaAgent | None = None
    issues: list[dict] = field(default_factory=list)
    chat_limiter: RateLimiter | None = None
    read_limiter: RateLimiter | None = None
    usage: UsageStore | None = None
    budget: BudgetGuard | None = None


def _build_context(
    settings: Settings, retriever: Retriever | None, agent: OmadaAgent | None
) -> AppContext:
    ctx = AppContext(
        settings=settings,
        chat_limiter=RateLimiter(settings.rate_limit_per_minute),
        read_limiter=RateLimiter(settings.rate_limit_per_minute * 6 if settings.rate_limit_per_minute else 0),
    )
    try:
        ctx.usage = UsageStore(settings.usage_db_location, settings.usage_timezone)
        ctx.budget = BudgetGuard(settings, ctx.usage)
    except Exception as exc:  # unwritable folder, corrupt file...
        log.exception("Usage ledger unavailable")
        ctx.issues.append(
            {
                "code": "usage_unavailable",
                "message": f"No se pudo abrir el registro de uso ({type(exc).__name__}); los límites de gasto NO se aplican.",
            }
        )
    if ctx.budget is not None and price_for(settings) is None:
        ctx.issues.append(
            {
                "code": "model_unpriced",
                "message": (
                    f"No conozco el precio del modelo '{settings.openai_model}', así que los límites en dólares no pueden "
                    "aplicarse. Define PRICE_INPUT_PER_MTOK y PRICE_OUTPUT_PER_MTOK en el .env (el límite de preguntas por "
                    "día sí funciona)."
                ),
            }
        )
    try:
        ctx.retriever = retriever or Retriever(settings)
    except IndexNotBuiltError as exc:
        ctx.issues.append({"code": "index_missing", "message": str(exc)})
    if ctx.retriever is not None:
        if agent is not None:
            ctx.agent = agent
        else:
            try:
                ctx.agent = OmadaAgent(settings, ToolBox(ctx.retriever))
            except ConfigError as exc:
                ctx.issues.append({"code": "openai_key_missing", "message": str(exc)})
    if ctx.retriever is not None and not ctx.retriever.semantic_available:
        ctx.issues.append(
            {"code": "keyword_only", "message": f"Búsqueda semántica: {ctx.retriever.semantic_status}"}
        )
    return ctx


def _budget_summary(ctx: AppContext, user: str, extra_cost: float = 0.0) -> dict | None:
    """Small summary shown under each answer. ``extra_cost`` is this answer, not yet in the ledger."""
    if ctx.budget is None:
        return None
    snap = ctx.budget.snapshot(user)
    lim, used = snap["limits"], snap["used"]
    return {
        "user_today_usd": used["user_daily_usd"] + extra_cost,
        "user_daily_limit_usd": lim["user_daily_usd"] or None,
        "today_usd": used["daily_usd"] + extra_cost,
        "daily_limit_usd": lim["daily_usd"] or None,
        "requests_today": used["user_daily_requests"] + 1,
        "requests_limit": lim["user_daily_requests"] or None,
    }


def _record_usage(ctx: AppContext, settings: Settings, *, user, rid, status, error_code, usage, elapsed_ms, question, streamed_chars):
    """Write the ledger row and the structured log line. Never raises: accounting must not break answers."""
    in_tok, out_tok = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    if status == "aborted" and not out_tok:
        out_tok = streamed_chars // 4  # the stream stopped early: estimate what was already generated
    cost = cost_of(settings, in_tok, out_tok)
    try:
        if ctx.usage is not None:
            ctx.usage.record(
                user=user, model=settings.openai_model, input_tokens=in_tok, output_tokens=out_tok, cost_usd=cost,
                tool_calls=usage.get("tool_calls", 0), rounds=usage.get("rounds", 0), elapsed_ms=elapsed_ms,
                status=status, error_code=error_code, request_id=rid, question_chars=len(question),
            )
    except Exception:
        log.exception("Could not write the usage ledger")
    fields = dict(
        request_id=rid, user=user, model=settings.openai_model, status=status, error_code=error_code, input_tokens=in_tok,
        output_tokens=out_tok, cost_usd=round(cost, 6) if cost is not None else None,
        tool_calls=usage.get("tool_calls", 0), rounds=usage.get("rounds", 0), ms=elapsed_ms, question_chars=len(question),
    )
    if settings.log_questions:
        fields["question"] = question[:500]
    cost_text = f"${cost:.4f}" if cost is not None else "n/a"
    log_event(log, "chat", f"chat {status}: {in_tok}+{out_tok} tokens, {cost_text}, {elapsed_ms} ms", **fields)


def create_app(
    settings: Settings | None = None,
    *,
    retriever: Retriever | None = None,
    agent: OmadaAgent | None = None,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ctx = _build_context(settings, retriever, agent)
        for issue in app.state.ctx.issues:
            log.warning("Setup issue [%s]: %s", issue["code"], issue["message"])
        yield

    app = FastAPI(
        title="Omada API Assistant",
        version=__version__,
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )

    if settings.cors_origin_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origin_list,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type"],
        )

    _RID_OK = re.compile(r"^[A-Za-z0-9._\-]{6,64}$")

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        incoming = request.headers.get("x-request-id", "")
        rid = incoming if _RID_OK.match(incoming) else uuid.uuid4().hex[:12]
        token = request_id_var.set(rid)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("Unhandled error on %s %s", request.method, request.url.path)
            raise
        finally:
            request_id_var.reset(token)
        ms = int((time.perf_counter() - started) * 1000)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers["X-Request-ID"] = rid
        if not request.url.path.startswith("/api/docs") and request.url.path != "/api/openapi.json":
            response.headers.setdefault("Content-Security-Policy", CSP)
        path = request.url.path
        if path.startswith("/api/") and path != "/api/health":
            request_id_var.set(rid)
            log_event(
                log,
                "request",
                f"{request.method} {path} -> {response.status_code} ({ms} ms)",
                method=request.method,
                path=path,
                status=response.status_code,
                ms=ms,
                user=identify(request, settings),
            )
        return response

    # ------------------------------------------------------------------ dependencies
    def ctx_of(request: Request) -> AppContext:
        return request.app.state.ctx

    def require_auth(request: Request) -> None:
        token = settings.access_token
        if not token:
            return
        header = request.headers.get("authorization", "")
        supplied = header[7:] if header.lower().startswith("bearer ") else ""
        if not hmac.compare_digest(supplied.encode(), token.encode()):
            raise HTTPException(status_code=401, detail="Token de acceso inválido o ausente.")

    def client_key(request: Request) -> str:
        return identify(request, settings)

    def limit_chat(request: Request) -> None:
        if not ctx_of(request).chat_limiter.allow(client_key(request)):  # type: ignore[union-attr]
            raise HTTPException(429, "Demasiadas solicitudes. Espera un momento e inténtalo de nuevo.")

    def limit_read(request: Request) -> None:
        if not ctx_of(request).read_limiter.allow(client_key(request)):  # type: ignore[union-attr]
            raise HTTPException(429, "Demasiadas solicitudes. Espera un momento e inténtalo de nuevo.")

    def need_retriever(request: Request) -> Retriever:
        retr = ctx_of(request).retriever
        if retr is None:
            raise HTTPException(
                503, "La base de conocimiento no está construida. Ejecuta: python manage.py build-index"
            )
        return retr

    api = APIRouter(prefix="/api")

    # ------------------------------------------------------------------ routes
    @api.get("/health")
    async def health(request: Request):
        ctx = ctx_of(request)
        return {
            "status": "ok" if not any(i["code"] != "keyword_only" for i in ctx.issues) else "degraded",
            "version": __version__,
            "index_ready": ctx.retriever is not None,
            "agent_ready": ctx.agent is not None,
            "model": settings.openai_model,
            "auth_required": settings.access_token is not None,
            "budget_enabled": ctx.budget is not None
            and any(
                v > 0
                for v in (
                    settings.budget_daily_usd,
                    settings.budget_monthly_usd,
                    settings.budget_user_daily_usd,
                    settings.user_daily_requests,
                )
            ),
            "issues": ctx.issues,
        }

    @api.get("/index/stats", dependencies=[Depends(require_auth)])
    async def index_stats(request: Request):
        retr = need_retriever(request)
        return await run_in_threadpool(retr.info)

    @api.get("/search", dependencies=[Depends(require_auth), Depends(limit_read)])
    async def search(
        request: Request,
        q: str = Query(min_length=1, max_length=300),
        method: str | None = Query(default=None, pattern="^(GET|POST|PUT|PATCH|DELETE)$"),
        tag: str | None = Query(default=None, max_length=100),
        kind: str | None = Query(default="operation", pattern="^(operation|guide)$"),
        limit: int = Query(default=12, ge=1, le=30),
        include_deprecated: bool = False,
        include_templates: bool = True,
        mode: str = Query(default="hybrid", pattern="^(hybrid|keyword)$"),
    ):
        retr = need_retriever(request)
        filters = Filters(
            kind=kind,
            method=method,
            tag=tag,
            include_deprecated=include_deprecated,
            include_templates=include_templates,
        )
        result = await run_in_threadpool(retr.search, q, filters, limit, mode)
        return {
            "mode": result.mode,
            "warning": result.warning,
            "results": [h.doc.to_card(h.score) for h in result.hits],
        }

    @api.get("/operation", dependencies=[Depends(require_auth), Depends(limit_read)])
    async def operation(request: Request, id: str = Query(min_length=1, max_length=300)):
        retr = need_retriever(request)
        doc = await run_in_threadpool(retr.store.get, id)
        if doc is None:
            raise HTTPException(404, f"No existe la operación '{id}'.")
        return {
            **doc.to_card(),
            "description": doc.description,
            "permissions": doc.permissions,
            "errors": [{"code": c, "message": m} for c, m in doc.errors],
            "header": doc.header_text,
            "request": doc.request_text,
            "response": doc.response_text,
        }

    @api.get("/usage", dependencies=[Depends(require_auth), Depends(limit_read)])
    async def usage(request: Request, days: int = Query(default=30, ge=1, le=366)):
        """Your own usage and limits; administrators also get the whole picture."""
        ctx = ctx_of(request)
        if ctx.usage is None or ctx.budget is None:
            raise HTTPException(503, "El registro de uso no está disponible.")
        user = identify(request, settings)
        store = ctx.usage
        out = {
            "user": user,
            "budget": ctx.budget.snapshot(user),
            "price_per_mtok": price_for(settings),
            "me": {
                "today": store.spent(day=store.day_key(), user=user),
                "by_day": store.by_day(days, user=user),
            },
            "admin": None,
        }
        if is_admin(user, settings):
            out["admin"] = {
                "today": store.spent(day=store.day_key()),
                "month": store.spent(month=store.month_key()),
                "by_day": store.by_day(days),
                "top_users": store.top_users(days),
                "by_status": store.by_status(days),
                "averages": store.averages(days),
                "unpriced_requests_this_month": store.unpriced_requests(store.month_key()),
            }
        return out

    @api.post("/chat", dependencies=[Depends(require_auth), Depends(limit_chat)])
    async def chat(body: ChatRequest, request: Request):
        ctx = ctx_of(request)
        history = [m.model_dump() for m in body.messages]
        user = identify(request, settings)
        rid = request_id_var.get()
        question = history[-1]["content"] if history else ""

        warning = None
        if ctx.budget is not None:
            decision = await run_in_threadpool(ctx.budget.check, user)
            if not decision.allowed:
                log_event(
                    log, "budget_blocked", f"Blocked {user}: {decision.code}", level=logging.WARNING,
                    user=user, code=decision.code,
                )
                raise HTTPException(429, decision.message)
            warning = ctx.budget.warning(ctx.budget.snapshot(user))

        async def stream():
            if ctx.agent is None:
                issue = next(
                    (i for i in ctx.issues if i["code"] in ("index_missing", "openai_key_missing")),
                    {"code": "not_ready", "message": "El asistente no está listo."},
                )
                yield sse({"type": "error", "message": issue["message"], "code": issue["code"]})
                return

            started = time.perf_counter()
            status, error_code, usage_info, streamed_chars = "ok", None, {}, 0
            if warning:
                yield sse({"type": "notice", "message": warning})
            try:
                async for event in with_heartbeat(ctx.agent.run(history)):
                    if event is None:
                        yield ": ping\n\n"
                        continue
                    kind = event["type"]
                    if kind == "token":
                        streamed_chars += len(event["text"])
                    elif kind in ("done", "error"):
                        usage_info = event.get("usage") or {}
                        if kind == "error":
                            status, error_code = "error", event.get("code")
                        else:
                            cost = cost_of(
                                settings, usage_info.get("input_tokens", 0), usage_info.get("output_tokens", 0)
                            )
                            event = {
                                **event,
                                "cost_usd": cost,
                                "budget": _budget_summary(ctx, user, extra_cost=cost or 0.0),
                            }
                    yield sse(event)
            except asyncio.CancelledError:
                status = "aborted"  # the user pressed Stop or closed the tab
                raise
            finally:
                _record_usage(
                    ctx, settings, user=user, rid=rid, status=status, error_code=error_code, usage=usage_info,
                    elapsed_ms=int((time.perf_counter() - started) * 1000), question=question,
                    streamed_chars=streamed_chars,
                )

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
        )

    app.include_router(api)

    # ------------------------------------------------------------------ web UI
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        icon = STATIC_DIR / "img" / "favicon.svg"
        if icon.exists():
            return FileResponse(icon, media_type="image/svg+xml")
        return Response(status_code=204)

    return app
