import json
import logging

import pytest
from fastapi.testclient import TestClient

from omada_agent.agent.agent import OmadaAgent
from omada_agent.agent.tools import ToolBox
from omada_agent.api.app import create_app
from omada_agent.logging_setup import JsonFormatter, request_id_var
from omada_agent.usage import BudgetGuard, UsageStore, cost_of, price_for
from omada_agent.usage.pricing import reference_price
from tests.conftest import FakeOpenAI, completed, function_call, text_delta
from tests.test_api import CHAT, scripted_turns, sse_events


# ------------------------------------------------------------------ pricing
def test_pricing_known_override_and_unknown(make_settings):
    cfg = make_settings()  # gpt-4o-mini: $0.15 in / $0.60 out per 1M tokens
    assert cost_of(cfg, 1_000_000, 1_000_000) == pytest.approx(0.75)
    assert reference_price("gpt-4o-mini-2024-07-18") == (0.15, 0.60)  # dated snapshot
    assert cost_of(make_settings(openai_model="mystery"), 10, 10) is None
    custom = make_settings(openai_model="mystery", price_input_per_mtok=2.0, price_output_per_mtok=8.0)
    assert price_for(custom) == (2.0, 8.0) and cost_of(custom, 1_000_000, 0) == pytest.approx(2.0)


# ------------------------------------------------------------------ ledger
def test_store_totals_by_day_user_and_month(tmp_path):
    store = UsageStore(tmp_path / "u.db", "UTC")
    store.record(user="ana", model="m", input_tokens=100, output_tokens=10, cost_usd=0.5)
    store.record(user="ana", model="m", input_tokens=50, output_tokens=5, cost_usd=0.25, status="error", error_code="rate_limit")
    store.record(user="luis", model="m", cost_usd=None)  # unknown price
    day, month = store.day_key(), store.month_key()
    assert store.spent(day=day)["requests"] == 3
    assert store.spent(day=day, user="ana") == {"cost": 0.75, "requests": 2, "input_tokens": 150, "output_tokens": 15}
    assert store.spent(month=month, user="luis")["cost"] == 0
    assert store.unpriced_requests(month) == 1
    assert store.top_users(7)[0]["user"] == "ana"
    assert {r["status"] for r in store.by_status(7)} == {"ok", "error"}
    assert store.by_day(7)[0]["requests"] == 3


def test_store_survives_reopen_and_bad_timezone(tmp_path):
    path = tmp_path / "u.db"
    UsageStore(path, "UTC").record(user="a", model="m", cost_usd=1.0)
    again = UsageStore(path, "Not/AZone")  # falls back to UTC instead of crashing
    assert again.spent(day=again.day_key())["cost"] == 1.0


# ------------------------------------------------------------------ budgets
def guard(make_settings, tmp_path, **kw):
    cfg = make_settings(**kw)
    store = UsageStore(tmp_path / "b.db", "UTC")
    return BudgetGuard(cfg, store), store


def test_user_daily_cost_limit(make_settings, tmp_path):
    g, store = guard(make_settings, tmp_path, budget_user_daily_usd=1.0, budget_daily_usd=0, budget_monthly_usd=0, user_daily_requests=0)
    assert g.check("ana").allowed
    store.record(user="ana", model="m", cost_usd=1.0)
    blocked = g.check("ana")
    assert not blocked.allowed and blocked.code == "budget_user_daily" and "US$ 1.00" in blocked.message
    assert g.check("luis").allowed  # other people are not affected


def test_global_daily_and_monthly_limits(make_settings, tmp_path):
    g, store = guard(make_settings, tmp_path, budget_user_daily_usd=0, budget_daily_usd=2.0, budget_monthly_usd=3.0, user_daily_requests=0)
    store.record(user="a", model="m", cost_usd=1.0)
    store.record(user="b", model="m", cost_usd=1.0)
    assert g.check("c").code == "budget_daily"
    g2, store2 = guard(make_settings, tmp_path / "x", budget_user_daily_usd=0, budget_daily_usd=0, budget_monthly_usd=1.0, user_daily_requests=0)
    store2.record(user="a", model="m", cost_usd=1.5)
    assert g2.check("z").code == "budget_monthly"


def test_request_cap_works_even_without_prices(make_settings, tmp_path):
    g, store = guard(make_settings, tmp_path, openai_model="mystery", budget_user_daily_usd=0, budget_daily_usd=0, budget_monthly_usd=0, user_daily_requests=2)
    for _ in range(2):
        store.record(user="ana", model="mystery", cost_usd=None)
    assert g.check("ana").code == "requests_user_daily"


def test_zero_switches_every_limit_off(make_settings, tmp_path):
    g, store = guard(make_settings, tmp_path, budget_user_daily_usd=0, budget_daily_usd=0, budget_monthly_usd=0, user_daily_requests=0)
    store.record(user="ana", model="m", cost_usd=999)
    assert g.check("ana").allowed and g.warning(g.snapshot("ana")) is None


def test_warning_at_eighty_percent(make_settings, tmp_path):
    g, store = guard(make_settings, tmp_path, budget_user_daily_usd=1.0, budget_daily_usd=0, budget_monthly_usd=0, user_daily_requests=0)
    store.record(user="ana", model="m", cost_usd=0.85)
    assert "85 %" in g.warning(g.snapshot("ana"))


# ------------------------------------------------------------------ API
@pytest.fixture
def make_client(retriever, make_settings):
    def _make(turns=None, **overrides):
        cfg = make_settings(**overrides)
        agent = OmadaAgent(cfg, ToolBox(retriever), client=FakeOpenAI(turns or scripted_turns()))
        return TestClient(create_app(cfg, retriever=retriever, agent=agent)), cfg

    return _make


def test_chat_is_recorded_with_cost_and_budget(make_client):
    client, cfg = make_client(portal_user_header="X-Forwarded-Email")
    with client:
        res = client.post("/api/chat", json=CHAT, headers={"X-Forwarded-Email": "Ana@Example.com"})
        done = sse_events(res)[-1]
        assert done["type"] == "done" and done["cost_usd"] is not None
        assert done["budget"]["requests_today"] == 1 and done["budget"]["user_daily_limit_usd"] == 1.0
        usage = client.get("/api/usage", headers={"X-Forwarded-Email": "ana@example.com"}).json()
        assert usage["user"] == "ana@example.com" and usage["me"]["today"]["requests"] == 1
        assert usage["admin"]["top_users"][0]["user"] == "ana@example.com"  # no admin list = everyone sees it


def test_chat_is_refused_when_the_limit_is_reached(make_client):
    client, cfg = make_client(user_daily_requests=1, portal_user_header="X-Forwarded-Email")
    with client:
        headers = {"X-Forwarded-Email": "ana@example.com"}
        assert client.post("/api/chat", json=CHAT, headers=headers).status_code == 200
        blocked = client.post("/api/chat", json=CHAT, headers=headers)
        assert blocked.status_code == 429 and "límite" in blocked.json()["detail"]
        other = client.post("/api/chat", json=CHAT, headers={"X-Forwarded-Email": "luis@example.com"})
        assert other.status_code == 200  # the limit is per person


def test_header_is_ignored_unless_configured(make_client):
    client, cfg = make_client()
    with client:
        client.post("/api/chat", json=CHAT, headers={"X-Forwarded-Email": "spoof@example.com"})
        assert client.get("/api/usage").json()["user"].startswith("ip:")


def test_only_admins_see_everybody(make_client):
    client, cfg = make_client(portal_user_header="X-User", portal_admins="boss@example.com")
    with client:
        client.post("/api/chat", json=CHAT, headers={"X-User": "ana@example.com"})
        ana = client.get("/api/usage", headers={"X-User": "ana@example.com"}).json()
        boss = client.get("/api/usage", headers={"X-User": "boss@example.com"}).json()
        assert ana["admin"] is None and boss["admin"]["today"]["requests"] == 1


def test_error_and_stop_are_recorded_too(make_client):
    # the stream ends without a final response -> an error event with the partial usage
    client, cfg = make_client(turns=[[text_delta("Hel")]], portal_user_header="X-User")
    with client:
        events = sse_events(client.post("/api/chat", json=CHAT, headers={"X-User": "ana"}))
        assert events[-1]["type"] == "error"
        usage = client.get("/api/usage", headers={"X-User": "ana"}).json()
        assert usage["admin"]["by_status"] == [{"status": "error", "requests": 1}]


def test_unpriced_model_is_reported_in_health(make_client):
    client, cfg = make_client(openai_model="mystery-model")
    with client:
        health = client.get("/api/health").json()
        assert any(i["code"] == "model_unpriced" for i in health["issues"])
        assert health["budget_enabled"] is True  # the per-day question cap still protects you


def test_request_id_header_and_json_log_line(make_client, caplog):
    client, cfg = make_client()
    with client, caplog.at_level(logging.INFO, logger="omada_agent"):
        res = client.get("/api/index/stats", headers={"X-Request-ID": "abc-123-xyz"})
        assert res.headers["x-request-id"] == "abc-123-xyz"
        assert client.get("/api/health", headers={"X-Request-ID": "bad id with spaces"}).headers["x-request-id"] != "bad id with spaces"
    record = next(r for r in caplog.records if getattr(r, "event", "") == "request")
    line = json.loads(JsonFormatter().format(record))
    assert line["path"] == "/api/index/stats" and line["status"] == 200 and line["event"] == "request"


def test_question_text_is_not_logged_by_default(make_client, caplog):
    client, cfg = make_client()
    with client, caplog.at_level(logging.INFO, logger="omada_agent"):
        client.post("/api/chat", json=CHAT)
    chat = next(r for r in caplog.records if getattr(r, "event", "") == "chat")
    assert not hasattr(chat, "question") and chat.question_chars == len(CHAT["messages"][0]["content"])
    assert chat.input_tokens >= 0 and chat.status == "ok"
    assert chat.request_id not in ("", "-")  # the line can be matched with the HTTP request that started it
    assert request_id_var.get() == "-"  # the context variable is reset after the request
