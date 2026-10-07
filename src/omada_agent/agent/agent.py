"""The agent loop.

Uses OpenAI's **Responses API** (the recommended interface for new projects) with
function calling and streaming:

1. Send the conversation + tool definitions.
2. Stream text tokens to the caller as they arrive.
3. If the model requests tools, run them (in parallel) against the RAG and send the
   results back using ``previous_response_id``; repeat (bounded by
   ``AGENT_MAX_TOOL_ROUNDS``).
4. When the model answers without tools, we are done.

The agent is *stateless between requests*: the web client sends the history on
every call, so the server scales horizontally with no session storage.

``AgentEvent`` dictionaries yielded by :meth:`OmadaAgent.run`:

========== ===============================================================
type       payload
========== ===============================================================
start      model
tool_start id, name, args
tool_result id, name, ok, summary, operations[], ms
token      text
notice     message
error      message, code
done       usage{input_tokens, output_tokens, tool_calls, rounds}, elapsed_ms
========== ===============================================================
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import openai
from openai import AsyncOpenAI

from omada_agent.agent.guide_hints import guide_query_for
from omada_agent.agent.prompts import build_system_prompt
from omada_agent.agent.tools import ToolBox, ToolResult
from omada_agent.config import Settings

log = logging.getLogger(__name__)

AgentEvent = dict[str, Any]


class AgentError(RuntimeError):
    def __init__(self, message: str, code: str = "agent_error"):
        super().__init__(message)
        self.code = code


def friendly_openai_error(exc: Exception) -> tuple[str, str]:
    """Translate OpenAI SDK exceptions into (message, code) the user can act on."""
    if isinstance(exc, openai.AuthenticationError):
        return ("OpenAI rechazó la API key. Revisa OPENAI_API_KEY en tu archivo .env.", "auth")
    if isinstance(exc, openai.PermissionDeniedError):
        return (
            "Tu cuenta de OpenAI no tiene acceso a este modelo. Revisa OPENAI_MODEL en el .env.",
            "permission",
        )
    if isinstance(exc, openai.NotFoundError):
        return (
            "El modelo configurado no existe o no está disponible para tu cuenta. "
            "Revisa OPENAI_MODEL en el .env.",
            "model_not_found",
        )
    if isinstance(exc, openai.RateLimitError):
        return (
            "OpenAI indica límite de uso o falta de saldo (rate limit / quota). "
            "Espera un momento o revisa tu plan de facturación.",
            "rate_limit",
        )
    if isinstance(exc, openai.APIConnectionError):
        return ("No se pudo conectar con la API de OpenAI. Revisa tu conexión a internet.", "connection")
    if isinstance(exc, openai.BadRequestError):
        return (
            f"OpenAI rechazó la solicitud: {getattr(exc, 'message', str(exc))} "
            "(si menciona 'temperature' o 'reasoning', ajusta esas variables en el .env).",
            "bad_request",
        )
    if isinstance(exc, openai.APIStatusError):
        return (f"Error de la API de OpenAI ({exc.status_code}): {getattr(exc, 'message', exc)}", "api_status")
    return (f"Error inesperado: {type(exc).__name__}: {exc}", "unexpected")


class OmadaAgent:
    def __init__(
        self,
        settings: Settings,
        toolbox: ToolBox,
        *,
        client: AsyncOpenAI | None = None,
    ):
        self.settings = settings
        self.toolbox = toolbox
        self._unsupported: set[str] = set()  # optional params this model rejected
        self.client = client or AsyncOpenAI(
            api_key=settings.require_openai_key(),
            base_url=settings.openai_base_url,
            timeout=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
        )
        info = toolbox.retriever.info()
        self.instructions = build_system_prompt(
            controller_version=info.get("spec", {}).get("controller_version", ""),
            operations=info.get("operations", 0),
            categories=len(info.get("categories", [])),
            guides=info.get("guides", 0),
        )

    # ------------------------------------------------------------------ helpers
    def _prepare_history(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        s = self.settings
        trimmed = messages[-s.chat_max_history_messages :]
        while trimmed and trimmed[0]["role"] != "user":  # always start on a user turn
            trimmed = trimmed[1:]
        return [
            {"role": m["role"], "content": m["content"][: s.chat_max_message_chars]} for m in trimmed
        ]

    def _request_kwargs(
        self,
        input_items: list[dict[str, Any]],
        previous_response_id: str | None,
        allow_tools: bool,
        instructions: str | None = None,
    ) -> dict[str, Any]:
        s = self.settings
        kwargs: dict[str, Any] = {
            "model": s.openai_model,
            "instructions": instructions or self.instructions,  # not carried over by previous_response_id
            "input": input_items,
            "tools": self.toolbox.schemas,
            "stream": True,
        }
        if allow_tools:
            kwargs["parallel_tool_calls"] = True
        else:
            kwargs["tool_choice"] = "none"
        if previous_response_id:
            kwargs["previous_response_id"] = previous_response_id
        if s.openai_reasoning_effort:
            kwargs["reasoning"] = {"effort": s.openai_reasoning_effort}
        if s.openai_temperature is not None:
            kwargs["temperature"] = s.openai_temperature
        if s.openai_max_output_tokens:
            kwargs["max_output_tokens"] = s.openai_max_output_tokens
        for key in self._unsupported:
            kwargs.pop(key, None)
        return kwargs

    async def _create_stream(self, input_items, previous_id, allow_tools, instructions=None):
        """Open the response stream; if the model rejects an optional parameter
        (e.g. ``reasoning`` on gpt-4o-mini, ``temperature`` on a reasoning model),
        remember that, drop it and retry instead of failing the conversation."""
        for _ in range(4):
            kwargs = self._request_kwargs(input_items, previous_id, allow_tools, instructions)
            try:
                return await self.client.responses.create(**kwargs)
            except openai.BadRequestError as exc:
                text = str(exc).lower()
                culprit = next(
                    (k for k in ("reasoning", "temperature", "max_output_tokens") if k in text and k in kwargs),
                    None,
                )
                if culprit is None:
                    raise
                log.warning("Model %s rejected '%s'; continuing without it.", self.settings.openai_model, culprit)
                self._unsupported.add(culprit)
        return await self.client.responses.create(
            **self._request_kwargs(input_items, previous_id, allow_tools, instructions)
        )

    async def _prefetch_guides(self, history: list[dict[str, str]]) -> tuple[str, dict | None]:
        """Search the guides ourselves when the question needs them. Returns (extra instructions, event info)."""
        if not self.settings.agent_guide_prefetch or not history:
            return "", None
        query = guide_query_for(history[-1]["content"])
        if query is None:
            return "", None
        started = time.perf_counter()
        result = await self.toolbox.run("search_guides", {"query": query, "limit": 3})
        if not result.ok or result.text.startswith(("No guide", "No guides")):
            return "", None
        note = (
            "\n\n## Guide excerpts retrieved automatically for this question\n"
            "These come from the official guides and are authoritative. Use them for authentication, "
            "header formats, limits and value tables; do not contradict them.\n\n"
            + result.text[:6000]
        )
        info = {"query": query, "summary": result.summary, "ms": int((time.perf_counter() - started) * 1000)}
        return note, info

    async def _execute(self, call: Any) -> tuple[Any, ToolResult, int]:
        started = time.perf_counter()
        result = await self.toolbox.run(call.name, call.arguments)
        return call, result, int((time.perf_counter() - started) * 1000)

    @staticmethod
    def _parse_args(raw: str | None) -> dict:
        try:
            parsed = json.loads(raw or "{}")
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    # ------------------------------------------------------------------ main loop
    async def run(self, messages: list[dict[str, str]]) -> AsyncIterator[AgentEvent]:
        started = time.perf_counter()
        usage = {"input_tokens": 0, "output_tokens": 0, "tool_calls": 0, "rounds": 0}
        yield {"type": "start", "model": self.settings.openai_model}

        input_items: list[dict[str, Any]] = list(self._prepare_history(messages))  # type: ignore[arg-type]
        previous_id: str | None = None
        max_rounds = self.settings.agent_max_tool_rounds
        instructions = self.instructions
        try:
            extra, info = await self._prefetch_guides(input_items)  # type: ignore[arg-type]
        except Exception:  # a helper must never break the answer
            log.exception("Guide prefetch failed")
            extra, info = "", None
        if info:
            instructions += extra
            yield {"type": "tool_start", "id": "auto-guides", "name": "search_guides", "args": {"query": info["query"], "auto": True}}
            yield {"type": "tool_result", "id": "auto-guides", "name": "search_guides", "ok": True, "summary": info["summary"], "operations": [], "ms": info["ms"]}

        try:
            for round_no in range(max_rounds + 1):
                allow_tools = round_no < max_rounds
                usage["rounds"] += 1
                final = None
                stream = await self._create_stream(input_items, previous_id, allow_tools, instructions)
                try:
                    async for event in stream:
                        etype = event.type
                        if etype == "response.output_text.delta":
                            yield {"type": "token", "text": event.delta}
                        elif etype in ("response.completed", "response.incomplete"):
                            final = event.response
                        elif etype == "response.failed":
                            err = getattr(event.response, "error", None)
                            raise AgentError(
                                getattr(err, "message", None) or "La respuesta del modelo falló.",
                                "response_failed",
                            )
                        elif etype == "error":
                            raise AgentError(getattr(event, "message", "Error en el stream."), "stream")
                finally:
                    close = getattr(stream, "close", None)
                    if close:
                        await close()

                if final is None:
                    raise AgentError("El stream terminó sin una respuesta final.", "no_final")

                if getattr(final, "usage", None):
                    usage["input_tokens"] += final.usage.input_tokens or 0
                    usage["output_tokens"] += final.usage.output_tokens or 0

                if getattr(final, "status", "") == "incomplete":
                    reason = getattr(getattr(final, "incomplete_details", None), "reason", "")
                    yield {
                        "type": "notice",
                        "message": "La respuesta se cortó"
                        + (" por el límite de tokens de salida (OPENAI_MAX_OUTPUT_TOKENS)." if reason == "max_output_tokens" else f" ({reason})."),
                    }

                calls = [item for item in final.output if getattr(item, "type", "") == "function_call"]
                if not calls:
                    break

                usage["tool_calls"] += len(calls)
                for call in calls:
                    yield {
                        "type": "tool_start",
                        "id": call.call_id,
                        "name": call.name,
                        "args": self._parse_args(call.arguments),
                    }

                outputs: dict[str, str] = {}
                pending = [asyncio.ensure_future(self._execute(c)) for c in calls]
                try:
                    for future in asyncio.as_completed(pending):
                        call, result, ms = await future
                        outputs[call.call_id] = result.text
                        yield {
                            "type": "tool_result",
                            "id": call.call_id,
                            "name": call.name,
                            "ok": result.ok,
                            "summary": result.summary,
                            "operations": result.operations,
                            "ms": ms,
                        }
                finally:
                    for task in pending:
                        task.cancel()

                input_items = [
                    {
                        "type": "function_call_output",
                        "call_id": c.call_id,
                        "output": outputs.get(c.call_id, "Tool did not return a result."),
                    }
                    for c in calls
                ]
                previous_id = final.id

        except AgentError as exc:
            yield {"type": "error", "message": str(exc), "code": exc.code, "usage": usage}
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("Agent run failed")
            message, code = friendly_openai_error(exc)
            yield {"type": "error", "message": message, "code": code, "usage": usage}
            return

        yield {
            "type": "done",
            "usage": usage,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }
