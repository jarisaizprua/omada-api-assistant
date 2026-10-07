"""Tests for the improvements made after the first real evaluation:
vocabulary expansion, automatic guide lookup, and the stricter prompt."""

import pytest

from omada_agent.agent.agent import OmadaAgent
from omada_agent.agent.guide_hints import guide_query_for
from omada_agent.agent.prompts import build_system_prompt
from omada_agent.agent.tools import ToolBox
from omada_agent.ingestion.guide_import import normalize_guide_text
from omada_agent.ingestion.indexer import build_index
from omada_agent.retrieval.hybrid import Retriever
from omada_agent.retrieval.queryexpand import expand_query, expansion_terms, keyword_variant
from tests.conftest import FakeOpenAI, completed, text_delta
from tests.test_guide_import import RAW


# ------------------------------------------------------------------ vocabulary
@pytest.mark.parametrize(
    "query, expected",
    [
        ("restart access point", {"reboot", "device"}),
        ("make the AP blink so I can find it", {"locate"}),
        ("cpu and memory of a switch every 5 minutes", {"statistic", "5min", "device"}),
        ("change channel and tx power on 5ghz", {"radio", "config"}),
        ("adopt a new device", {"start"}),  # "adopt" and "device" are already in the question
        ("several calls at once", {"batch"}),
    ],
)
def test_everyday_words_gain_the_apis_words(query, expected):
    assert expected <= set(expansion_terms(query))


def test_rules_stay_quiet_on_real_api_names():
    # "vlan" and "channel" are real words of many operations; they must not be rewritten
    assert expansion_terms("How do I modify AP vlan config?") == ["device"]
    assert "radio" not in expansion_terms("get channel utilization results of interference detection")


def test_original_words_are_never_lost_and_nothing_is_added_twice():
    assert expand_query("restart the AP").startswith("restart the AP")
    assert expansion_terms("reboot device") == []  # already speaks the API's language
    assert expand_query("list sites") == "list sites"


def test_keyword_variant_swaps_pure_synonyms_only():
    assert "restart" not in keyword_variant("restart the access point").split()
    assert "reboot" in keyword_variant("restart the access point").split()
    assert "switch" in keyword_variant("get switch stack").split()  # real nouns stay


def test_expansion_can_be_switched_off(make_settings, built, embedder):
    off = Retriever(make_settings(rag_query_expansion=False), embedder=embedder)
    on = Retriever(make_settings(), embedder=embedder)
    assert off.search("restart access point").queries == ["restart access point"]
    assert len(on.search("restart access point").queries) == 2


# ------------------------------------------------------------------ guides
@pytest.mark.parametrize(
    "message",
    [
        "Dame un script que use un token de acceso",
        "¿Con qué cabecera envío el token?",
        "¿Hay un límite de llamadas en un lote?",
        "¿Qué valor uso en timeZone para Madrid?",
        "¿Qué código entero tiene el protocolo UDP?",
        "What does error -44112 mean?",
        "How do I adopt a device and with which credentials?",
    ],
)
def test_guide_topics_trigger_a_lookup(message):
    assert guide_query_for(message)


def test_plain_api_questions_do_not_trigger_it():
    assert guide_query_for("¿Cómo reinicio un access point?") is None
    assert guide_query_for("How do I list the sites?") is None


@pytest.fixture
def guided(make_settings, embedder, tmp_path):
    guides = tmp_path / "guides"
    guides.mkdir()
    (guides / "access_guide.md").write_text(normalize_guide_text(RAW), encoding="utf-8")
    cfg = make_settings()
    build_index(cfg, embedder=embedder)
    return cfg, Retriever(cfg, embedder=embedder)


async def test_agent_reads_the_guide_before_answering_auth_questions(guided):
    cfg, retriever = guided
    client = FakeOpenAI([[text_delta("Usa AccessToken."), completed("r1")]])
    agent = OmadaAgent(cfg, ToolBox(retriever), client=client)
    events = [e async for e in agent.run([{"role": "user", "content": "¿Cómo obtengo el token de acceso con client credentials?"}])]

    kinds = [e["type"] for e in events]
    assert kinds[:3] == ["start", "tool_start", "tool_result"]
    assert events[1]["name"] == "search_guides" and events[1]["args"]["auto"] is True
    sent = client.calls[0]["instructions"]
    assert "Guide excerpts retrieved automatically" in sent and "client_credentials" in sent
    assert "Guide excerpts" not in agent.instructions  # the base prompt is untouched for later questions


async def test_no_guides_no_extra_events(make_settings, built, retriever):
    cfg, _, _ = built
    client = FakeOpenAI([[text_delta("ok"), completed("r1")]])
    agent = OmadaAgent(cfg, ToolBox(retriever), client=client)
    events = [e async for e in agent.run([{"role": "user", "content": "Dame un token de acceso"}])]
    assert [e["type"] for e in events] == ["start", "token", "done"]


async def test_prefetch_can_be_disabled(make_settings, guided):
    _, retriever = guided
    cfg = make_settings(agent_guide_prefetch=False)
    client = FakeOpenAI([[text_delta("ok"), completed("r1")]])
    agent = OmadaAgent(cfg, ToolBox(retriever), client=client)
    events = [e async for e in agent.run([{"role": "user", "content": "token de acceso"}])]
    assert "tool_start" not in [e["type"] for e in events]


# ------------------------------------------------------------------ prompt
def test_prompt_demands_the_guide_and_forbids_guessing():
    text = build_system_prompt(controller_version="6.3.0", operations=10, categories=2, guides=5)
    assert "search_guides" in text and "Never answer them from memory" in text
    assert "Bearer" in text  # named as the thing NOT to assume
    assert "reboot" in text and "locate" in text  # vocabulary hints
    assert "this does not exist in the API" in text
