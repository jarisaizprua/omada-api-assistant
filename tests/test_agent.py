import httpx
import openai
import pytest

from omada_agent.agent.agent import OmadaAgent
from omada_agent.agent.tools import ToolBox
from tests.conftest import FakeOpenAI, completed, function_call, text_delta


def make_agent(retriever, settings, turns):
    client = FakeOpenAI(turns)
    return OmadaAgent(settings, ToolBox(retriever), client=client), client


async def collect(agent, messages=None):
    return [e async for e in agent.run(messages or [{"role": "user", "content": "How do I create a site?"}])]


async def test_tool_round_then_streamed_answer(retriever, settings):
    agent, client = make_agent(
        retriever,
        settings,
        [
            [
                completed(
                    "resp_1",
                    output=[
                        function_call("search_api", {"query": "create site"}, "c1"),
                        function_call("get_operation", {"operation_id": "createNewSite"}, "c2"),
                    ],
                    in_tokens=100,
                    out_tokens=20,
                )
            ],
            [text_delta("Use "), text_delta("`createNewSite`."), completed("resp_2", in_tokens=300, out_tokens=40)],
        ],
    )
    events = await collect(agent)
    types = [e["type"] for e in events]

    assert types[0] == "start" and types[-1] == "done"
    assert types.count("tool_start") == 2 and types.count("tool_result") == 2
    assert "".join(e["text"] for e in events if e["type"] == "token") == "Use `createNewSite`."
    # tools are announced before any result and before the answer streams
    assert types.index("tool_start") < types.index("tool_result") < types.index("token")

    results = {e["name"]: e for e in events if e["type"] == "tool_result"}
    assert results["search_api"]["ok"] and results["get_operation"]["operations"][0]["id"] == "createNewSite"

    done = events[-1]
    assert done["usage"] == {"input_tokens": 400, "output_tokens": 60, "tool_calls": 2, "rounds": 2}

    first, second = client.calls
    assert first["stream"] is True and first["parallel_tool_calls"] is True
    assert "1.2.3" in first["instructions"] and first["tools"]
    assert first["input"] == [{"role": "user", "content": "How do I create a site?"}]
    assert "previous_response_id" not in first
    # round two continues the stored response and feeds tool outputs back
    assert second["previous_response_id"] == "resp_1"
    assert {i["call_id"] for i in second["input"]} == {"c1", "c2"}
    assert all(i["type"] == "function_call_output" and i["output"] for i in second["input"])
    assert second["instructions"] == first["instructions"]  # not carried over by the API: must be resent


async def test_answer_without_tools(retriever, settings):
    agent, client = make_agent(retriever, settings, [[text_delta("Hi"), completed("r1")]])
    events = await collect(agent)
    assert [e["type"] for e in events] == ["start", "token", "done"]
    assert len(client.calls) == 1


async def test_tool_round_limit_forces_a_final_answer(make_settings, built):
    settings = make_settings(agent_max_tool_rounds=1)
    from omada_agent.retrieval.hybrid import Retriever

    retriever = Retriever(settings, embedder=built[1])
    agent, client = make_agent(
        retriever,
        settings,
        [
            [completed("r1", output=[function_call("list_categories", {}, "c1")])],
            [text_delta("done"), completed("r2")],
        ],
    )
    events = await collect(agent)
    assert events[-1]["type"] == "done"
    assert client.calls[1]["tool_choice"] == "none" and "parallel_tool_calls" not in client.calls[1]


async def test_tool_failures_are_reported_not_raised(retriever, settings):
    agent, _ = make_agent(
        retriever,
        settings,
        [
            [completed("r1", output=[function_call("get_operation", {"operation_id": "doesNotExist"}, "c1")])],
            [text_delta("Not found."), completed("r2")],
        ],
    )
    events = await collect(agent)
    result = next(e for e in events if e["type"] == "tool_result")
    assert result["ok"] is False
    assert events[-1]["type"] == "done"


async def test_optional_model_parameters_are_passed_only_when_set(retriever, make_settings, built):
    from omada_agent.retrieval.hybrid import Retriever

    settings = make_settings(openai_reasoning_effort="low", openai_temperature=0.3, openai_max_output_tokens=900)
    agent, client = make_agent(Retriever(settings, embedder=built[1]), settings, [[text_delta("x"), completed("r")]])
    await collect(agent)
    call = client.calls[0]
    assert call["reasoning"] == {"effort": "low"} and call["temperature"] == 0.3 and call["max_output_tokens"] == 900

    plain, plain_client = make_agent(retriever, settings.model_copy(update={
        "openai_reasoning_effort": None, "openai_temperature": None, "openai_max_output_tokens": None}),
        [[text_delta("x"), completed("r")]])
    await collect(plain)
    call = plain_client.calls[0]
    assert not {"reasoning", "temperature", "max_output_tokens"} & set(call)


async def test_history_is_trimmed_and_starts_on_a_user_turn(retriever, make_settings, built):
    from omada_agent.retrieval.hybrid import Retriever

    settings = make_settings(chat_max_history_messages=3, chat_max_message_chars=10)
    agent, client = make_agent(Retriever(settings, embedder=built[1]), settings, [[text_delta("ok"), completed("r")]])
    history = []
    for i in range(6):
        history += [{"role": "user", "content": f"question number {i}"}, {"role": "assistant", "content": f"answer number {i}"}]
    history.append({"role": "user", "content": "the final question is long"})
    await collect(agent, history)
    sent = client.calls[0]["input"]
    assert sent[0]["role"] == "user" and sent[-1]["content"] == "the final "  # 10 chars
    assert len(sent) <= 3


async def test_incomplete_response_emits_notice(retriever, settings):
    agent, _ = make_agent(retriever, settings, [[text_delta("cut"), completed("r", status="incomplete")]])
    events = await collect(agent)
    notice = next(e for e in events if e["type"] == "notice")
    assert "OPENAI_MAX_OUTPUT_TOKENS" in notice["message"]


@pytest.mark.parametrize(
    "status, exc_class, code",
    [
        (401, openai.AuthenticationError, "auth"),
        (404, openai.NotFoundError, "model_not_found"),
        (429, openai.RateLimitError, "rate_limit"),
    ],
)
async def test_openai_errors_become_friendly_events(retriever, settings, status, exc_class, code):
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    exc = exc_class("boom", response=httpx.Response(status, request=request), body=None)
    agent, _ = make_agent(retriever, settings, [exc])
    events = await collect(agent)
    assert [e["type"] for e in events] == ["start", "error"]
    assert events[-1]["code"] == code and events[-1]["message"]


async def test_failed_stream_event_is_reported(retriever, settings):
    from types import SimpleNamespace

    failed = SimpleNamespace(type="response.failed", response=SimpleNamespace(error=SimpleNamespace(message="model overloaded")))
    agent, _ = make_agent(retriever, settings, [[failed]])
    events = await collect(agent)
    assert events[-1]["type"] == "error" and events[-1]["code"] == "response_failed"
    assert events[-1]["message"] == "model overloaded"
    assert set(events[-1]["usage"]) == {"input_tokens", "output_tokens", "tool_calls", "rounds"}


async def test_rejected_optional_parameters_are_dropped_and_retried(make_settings, built):
    """gpt-4o-mini rejects `reasoning`; the agent must recover instead of failing the chat."""
    from omada_agent.retrieval.hybrid import Retriever

    settings = make_settings(openai_reasoning_effort="low", openai_temperature=0.2)
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    bad = openai.BadRequestError(
        "Unsupported parameter: 'reasoning.effort' is not supported with this model.",
        response=httpx.Response(400, request=request),
        body=None,
    )
    agent, client = make_agent(
        Retriever(settings, embedder=built[1]),
        settings,
        [bad, [text_delta("ok"), completed("r1")], [text_delta("again"), completed("r2")]],
    )
    events = await collect(agent)
    assert events[-1]["type"] == "done" and not any(e["type"] == "error" for e in events)
    assert "reasoning" in client.calls[0] and "reasoning" not in client.calls[1]
    assert client.calls[1]["temperature"] == 0.2  # only the rejected parameter is dropped
    await collect(agent)  # remembered for later requests
    assert "reasoning" not in client.calls[2]


async def test_unrelated_bad_requests_still_surface(retriever, settings):
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    bad = openai.BadRequestError("Invalid value for input", response=httpx.Response(400, request=request), body=None)
    agent, _ = make_agent(retriever, settings, [bad])
    events = await collect(agent)
    assert events[-1]["type"] == "error" and events[-1]["code"] == "bad_request"
