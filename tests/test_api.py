import json

import pytest
from fastapi.testclient import TestClient

from omada_agent.agent.agent import OmadaAgent
from omada_agent.agent.tools import ToolBox
from omada_agent.api.app import create_app
from omada_agent.api.streaming import with_heartbeat
from tests.conftest import FakeOpenAI, completed, function_call, text_delta

CHAT = {"messages": [{"role": "user", "content": "How do I create a site?"}]}


def sse_events(response) -> list[dict]:
    frames = [f for f in response.text.split("\n\n") if f.startswith("data: ")]
    return [json.loads(f[6:]) for f in frames]


def scripted_turns():
    return [
        [completed("r1", output=[function_call("search_api", {"query": "create site"}, "c1")])],
        [text_delta("Use createNewSite."), completed("r2")],
    ]


@pytest.fixture
def make_client(retriever, make_settings):
    def _make(turns=None, **setting_overrides):
        cfg = make_settings(**setting_overrides)
        agent = OmadaAgent(cfg, ToolBox(retriever), client=FakeOpenAI(turns or scripted_turns()))
        return TestClient(create_app(cfg, retriever=retriever, agent=agent))

    return _make


def test_health_and_stats(make_client):
    with make_client() as client:
        health = client.get("/api/health").json()
        assert health["status"] == "ok" and health["index_ready"] and health["agent_ready"]
        stats = client.get("/api/index/stats").json()
        assert stats["operations"] == 7 and stats["spec"]["controller_version"] == "1.2.3"
        assert {"tag": "Site", "count": 3} in stats["categories"]


def test_search_endpoint(make_client):
    with make_client() as client:
        data = client.get("/api/search", params={"q": "create site"}).json()
        assert data["mode"] == "hybrid" and data["results"][0]["id"] == "createNewSite"
        only_get = client.get("/api/search", params={"q": "site", "method": "GET"}).json()
        assert only_get["results"] and all(r["method"] == "GET" for r in only_get["results"])
        assert client.get("/api/search", params={"q": "site", "method": "TRACE"}).status_code == 422
        assert client.get("/api/search", params={"q": ""}).status_code == 422


def test_operation_detail_and_404(make_client):
    with make_client() as client:
        detail = client.get("/api/operation", params={"id": "createNewSite"}).json()
        assert detail["method"] == "POST" and detail["permissions"] == ["Global Dashboard Manager Modify"]
        assert detail["errors"][0]["code"] == "-33101"
        assert "Request body" in detail["request"] and "Response 200" in detail["response"]
        assert client.get("/api/operation", params={"id": "nope"}).status_code == 404


def test_chat_streams_server_sent_events(make_client):
    with make_client() as client:
        res = client.post("/api/chat", json=CHAT)
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/event-stream")
        assert res.headers["x-accel-buffering"] == "no"
        events = sse_events(res)
        types = [e["type"] for e in events]
        assert types[0] == "start" and types[-1] == "done"
        assert "tool_start" in types and "tool_result" in types
        assert "".join(e["text"] for e in events if e["type"] == "token") == "Use createNewSite."


def test_chat_validation(make_client):
    with make_client() as client:
        assert client.post("/api/chat", json={"messages": []}).status_code == 422
        last_is_assistant = {"messages": [{"role": "assistant", "content": "hi"}]}
        assert client.post("/api/chat", json=last_is_assistant).status_code == 422
        assert client.post("/api/chat", json={"messages": [{"role": "system", "content": "x"}]}).status_code == 422


def test_access_token_protects_the_api_but_not_health_or_ui(make_client):
    with make_client(portal_access_token="s3cret") as client:
        assert client.get("/api/health").json()["auth_required"] is True
        assert client.get("/").status_code == 200
        assert client.get("/api/search", params={"q": "site"}).status_code == 401
        assert client.post("/api/chat", json=CHAT).status_code == 401
        bad = {"Authorization": "Bearer nope"}
        assert client.get("/api/search", params={"q": "site"}, headers=bad).status_code == 401
        good = {"Authorization": "Bearer s3cret"}
        assert client.get("/api/search", params={"q": "site"}, headers=good).status_code == 200
        assert client.post("/api/chat", json=CHAT, headers=good).status_code == 200


def test_chat_rate_limit(make_client):
    turns = [[text_delta("a"), completed("r1")], [text_delta("b"), completed("r2")], [text_delta("c"), completed("r3")]]
    with make_client(turns, rate_limit_per_minute=2) as client:
        assert client.post("/api/chat", json=CHAT).status_code == 200
        assert client.post("/api/chat", json=CHAT).status_code == 200
        limited = client.post("/api/chat", json=CHAT)
        assert limited.status_code == 429 and "Demasiadas" in limited.json()["detail"]


def test_static_ui_and_security_headers(make_client):
    with make_client() as client:
        page = client.get("/")
        assert page.status_code == 200 and "Omada API Assistant" in page.text
        assert "default-src 'self'" in page.headers["content-security-policy"]
        assert page.headers["x-content-type-options"] == "nosniff"
        for asset in ("/static/css/app.css", "/static/js/main.js", "/static/vendor/marked.umd.js", "/static/img/favicon.svg"):
            assert client.get(asset).status_code == 200, asset


def test_missing_index_is_reported_not_crashed(make_settings):
    app = create_app(make_settings())
    with TestClient(app) as client:
        health = client.get("/api/health").json()
        assert health["index_ready"] is False and health["status"] == "degraded"
        assert health["issues"][0]["code"] == "index_missing"
        assert client.get("/api/search", params={"q": "x"}).status_code == 503
        events = sse_events(client.post("/api/chat", json=CHAT))
        assert events == [{"type": "error", "message": health["issues"][0]["message"], "code": "index_missing"}]


def test_missing_api_key_keeps_search_working(built, make_settings):
    settings = make_settings(openai_api_key=None)
    with TestClient(create_app(settings)) as client:
        health = client.get("/api/health").json()
        codes = {i["code"] for i in health["issues"]}
        assert "openai_key_missing" in codes and health["index_ready"] and not health["agent_ready"]
        data = client.get("/api/search", params={"q": "create site"}).json()
        assert data["mode"] == "keyword" and data["results"]
        events = sse_events(client.post("/api/chat", json=CHAT))
        assert events[0]["type"] == "error" and events[0]["code"] == "openai_key_missing"


async def test_heartbeat_emits_none_while_idle_and_forwards_items():
    import asyncio

    async def slow():
        yield {"n": 1}
        await asyncio.sleep(0.12)
        yield {"n": 2}

    seen = [item async for item in with_heartbeat(slow(), interval=0.03)]
    assert [i for i in seen if i is not None] == [{"n": 1}, {"n": 2}]
    assert None in seen
