import json

import pytest

from omada_agent.agent.tools import DETAIL_CHARS_ALL, TOOL_SCHEMAS, ToolBox


@pytest.fixture
def toolbox(retriever):
    return ToolBox(retriever)


def test_tool_schemas_are_well_formed():
    names = {t["name"] for t in TOOL_SCHEMAS}
    assert names == {"search_api", "get_operation", "list_categories", "search_guides"}
    for tool in TOOL_SCHEMAS:
        assert tool["type"] == "function" and tool["strict"] is False
        assert tool["parameters"]["type"] == "object"
        json.dumps(tool)  # must be serialisable


async def test_search_api_returns_cards_for_llm_and_ui(toolbox):
    result = await toolbox.run("search_api", json.dumps({"query": "create site"}))
    assert result.ok and "createNewSite" in result.text
    assert "get_operation" in result.text  # nudges the model to fetch details
    assert any(op["id"] == "createNewSite" for op in result.operations)
    assert result.summary.endswith("resultados")


async def test_search_api_hides_templates_and_deprecated_by_default(toolbox):
    default = await toolbox.run("search_api", {"query": "site template legacy"})
    ids = {op["id"] for op in default.operations}
    assert "createSiteTemplate" not in ids and "getLegacyThing" not in ids
    opted_in = await toolbox.run(
        "search_api", {"query": "site template legacy", "include_templates": True, "include_deprecated": True}
    )
    ids = {op["id"] for op in opted_in.operations}
    assert {"createSiteTemplate", "getLegacyThing"} <= ids


async def test_search_api_unknown_category_suggests_alternatives(toolbox):
    result = await toolbox.run("search_api", {"query": "x", "category": "Wirelss Network"})
    assert not result.ok and "Wireless Network" in result.text


async def test_get_operation_by_id_and_by_method_path(toolbox):
    by_id = await toolbox.run("get_operation", {"operation_id": "createNewSite"})
    assert by_id.ok
    for expected in ("Required permissions", "-33101", "Request body", "- name* (string)", "Response 200"):
        assert expected in by_id.text
    by_path = await toolbox.run("get_operation", {"operation_id": "POST /openapi/v1/{omadacId}/sites"})
    assert by_path.ok and by_path.operations[0]["id"] == "createNewSite"


async def test_get_operation_sections(toolbox):
    request = await toolbox.run("get_operation", {"operation_id": "createNewSite", "section": "request"})
    assert "Request body" in request.text and "Response 200" not in request.text
    response = await toolbox.run("get_operation", {"operation_id": "createNewSite", "section": "response"})
    assert "Response 200" in response.text and "Request body" not in response.text


async def test_get_operation_not_found_offers_closest(toolbox):
    result = await toolbox.run("get_operation", {"operation_id": "createSite"})
    assert not result.ok and "No operation found" in result.text


async def test_long_details_are_truncated_with_hint(toolbox, monkeypatch):
    doc = toolbox.store.get("createNewSite")
    doc.request_text = "x" * (DETAIL_CHARS_ALL + 500)
    monkeypatch.setattr(toolbox, "_find_operation", lambda ref: doc)
    result = await toolbox.run("get_operation", {"operation_id": "createNewSite"})
    assert "truncated" in result.text and len(result.text) < DETAIL_CHARS_ALL + 400


async def test_list_categories(toolbox):
    result = await toolbox.run("list_categories", {})
    assert "- Site (" in result.text and "Wireless Network" in result.text
    filtered = await toolbox.run("list_categories", {"contains": "wire"})
    assert "Wireless Network" in filtered.text and "Site (" not in filtered.text


async def test_search_guides_without_guides_is_explicit(toolbox):
    result = await toolbox.run("search_guides", {"query": "authentication"})
    assert "No guides are indexed" in result.text


async def test_bad_arguments_never_raise(toolbox):
    assert not (await toolbox.run("search_api", "{not json")).ok
    assert not (await toolbox.run("search_api", "[]")).ok
    assert not (await toolbox.run("search_api", {})).ok
    assert not (await toolbox.run("nope", {})).ok
