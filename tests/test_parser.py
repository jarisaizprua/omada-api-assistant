from omada_agent.ingestion.guides import chunk_markdown
from omada_agent.ingestion.openapi_parser import (
    SchemaRenderer,
    parse_description,
    parse_operations,
    split_words,
)
from tests.conftest import MINI_SPEC


def test_split_words_handles_camel_case_and_paths():
    assert split_words("createNewSite") == ["create", "New", "Site"]
    assert split_words("/openapi/v1/{omadacId}/sites") == ["openapi", "v1", "omadac", "Id", "sites"]


def test_parse_description_extracts_permissions_and_errors():
    raw = MINI_SPEC["paths"]["/openapi/v1/{omadacId}/sites"]["post"]["description"]
    text, perms, errors = parse_description(raw)
    assert text == "Create new site"
    assert perms == ["Global Dashboard Manager Modify"]
    assert errors[0] == ("-33101", "This name already exists. Please enter another name for this site.")
    assert [code for code, _ in errors] == ["-33101", "-33104"]


def test_parse_description_without_extras_is_untouched():
    text, perms, errors = parse_description("Just a plain description")
    assert (text, perms, errors) == ("Just a plain description", [], [])


def test_parse_operations_counts_and_flags():
    docs = {d.doc_id: d for d in parse_operations(MINI_SPEC)}
    assert len(docs) == 7
    assert docs["getLegacyThing"].deprecated is True
    assert docs["createSiteTemplate"].is_template is True
    assert docs["createNewSite"].is_template is False
    assert docs["createNewSite"].version == "v1"
    assert docs["createNewSite"].method == "POST"


def test_request_schema_resolves_refs_and_marks_required():
    doc = {d.doc_id: d for d in parse_operations(MINI_SPEC)}["createNewSite"]
    req = doc.request_text
    assert "- name* (string)" in req
    assert "- region* (string)" in req
    assert "- type (integer; enum: 0, 1)" in req
    assert "- deviceAccountSetting (object <DeviceAccount>)" in req
    assert "  - username* (string)" in req  # nested, indented
    assert "- tagIds (array<string>)" in req


def test_card_contains_search_relevant_text():
    doc = {d.doc_id: d for d in parse_operations(MINI_SPEC)}["createNewSite"]
    assert "operationId: createNewSite" in doc.card_text
    assert "Global Dashboard Manager Modify" in doc.card_text
    assert "-33101" in doc.card_text
    assert "[POST] /openapi/v1/{omadacId}/sites" in doc.brief_text
    assert "name*" in doc.brief_text


def test_array_response_and_params():
    docs = {d.doc_id: d for d in parse_operations(MINI_SPEC)}
    assert "Array of object <ClientInfo>:" in docs["getClients"].response_text
    params = docs["getSiteList"].header_text
    assert "- page (query, integer, required)" in params
    assert "max=1000" in params


def test_circular_references_terminate():
    doc = {d.doc_id: d for d in parse_operations(MINI_SPEC)}["getTree"]
    assert "recursive reference to TreeNode" in doc.response_text


def test_depth_limit_omits_deep_fields():
    schemas = {
        "A": {"type": "object", "properties": {"b": {"$ref": "#/components/schemas/B"}}},
        "B": {"type": "object", "properties": {"c": {"$ref": "#/components/schemas/C"}}},
        "C": {"type": "object", "properties": {"d": {"type": "string"}}},
    }
    lines = SchemaRenderer(schemas, max_depth=2).render({"$ref": "#/components/schemas/A"})
    assert any("nested fields omitted" in line for line in lines)
    assert not any("- d " in line for line in lines)


def test_line_budget_adds_truncation_marker():
    props = {f"f{i}": {"type": "string"} for i in range(50)}
    lines = SchemaRenderer({}, max_depth=3, max_lines=10).render({"type": "object", "properties": props})
    assert len(lines) == 11
    assert lines[-1].startswith("[... 40 more lines omitted]")


def test_chunk_markdown_keeps_headings_and_code_fences_together():
    text = "# Auth\nIntro text.\n\n## Client credentials\nSome steps.\n\n```python\n# not a heading\nprint(1)\n```\n"
    chunks = chunk_markdown(text)
    trails = [t for t, _ in chunks]
    assert "Auth" in trails
    assert "Auth > Client credentials" in trails
    code_chunk = next(body for trail, body in chunks if trail == "Auth > Client credentials")
    assert "# not a heading" in code_chunk and "print(1)" in code_chunk
