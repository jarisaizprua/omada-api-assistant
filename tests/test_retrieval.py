import json

import pytest

from omada_agent.ingestion.indexer import build_index
from omada_agent.models import Filters
from omada_agent.retrieval.hybrid import Retriever, reciprocal_rank_fusion
from omada_agent.retrieval.manifest import IndexNotBuiltError, read_manifest
from tests.conftest import MINI_SPEC


def test_rrf_rewards_agreement_between_rankings():
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c"] < scores["a"] and scores["d"] < scores["a"]


def test_build_reports_counts_and_writes_manifest(built):
    settings, _, report = built
    assert report.operations == 7 and report.embedded == 7 and report.reused == 0
    assert report.controller_version == "1.2.3"
    manifest = read_manifest(settings.index_location)
    assert manifest["spec"]["controller_version"] == "1.2.3"
    assert manifest["embedding"] == {"vectors": True, "model": "fake-embedder", "dimensions": None}


def test_incremental_rebuild_only_embeds_changes(built, spec_file, embedder):
    settings, _, _ = built
    before = embedder.calls
    again = build_index(settings, embedder=embedder)
    assert again.embedded == 0 and again.reused == 7 and embedder.calls == before

    changed = json.loads(json.dumps(MINI_SPEC))
    changed["paths"]["/openapi/v1/{omadacId}/sites"]["get"]["summary"] = "List every site"
    del changed["paths"]["/openapi/v1/{omadacId}/sites/{siteId}/legacy"]
    spec_file.write_text(json.dumps(changed), encoding="utf-8")
    third = build_index(settings, embedder=embedder)
    assert third.embedded == 1 and third.removed == 1 and third.operations == 6


def test_full_rebuild_reembeds_everything(built, embedder):
    settings, _, _ = built
    report = build_index(settings, embedder=embedder, full=True)
    assert report.embedded == 7


def test_keyword_only_index_works_without_embedder(make_settings):
    settings = make_settings(openai_api_key=None)
    report = build_index(settings, keyword_only=True)
    assert report.keyword_only and report.embedded == 0
    retriever = Retriever(settings)
    assert not retriever.semantic_available
    result = retriever.search("ssid password")
    assert result.mode == "keyword" and result.hits[0].doc.doc_id == "updateSsid"
    assert "keyword" in result.warning.lower()


def test_exact_operation_id_is_ranked_first(retriever):
    result = retriever.search("how do I call createNewSite?")
    assert result.hits[0].doc.doc_id == "createNewSite" and result.hits[0].exact


def test_method_and_path_lookup_ignores_parameter_names(retriever):
    result = retriever.search("PATCH /openapi/v1/{omadacId}/sites/{x}/wireless-network/ssids/{y}")
    assert result.hits[0].doc.doc_id == "updateSsid"


def test_error_code_lookup(retriever):
    result = retriever.search("what is error -33101")
    assert result.hits[0].doc.doc_id == "createNewSite"


def test_hybrid_finds_by_meaning_words(retriever):
    result = retriever.search("list connected clients")
    assert result.mode == "hybrid"
    assert result.hits[0].doc.doc_id == "getClients"
    assert result.hits[0].vector_rank is not None and result.hits[0].keyword_rank is not None


def test_filters(retriever):
    posts = retriever.search("site", Filters(method="POST", include_deprecated=True), limit=20)
    assert posts.hits and all(h.doc.method == "POST" for h in posts.hits)

    no_templates = retriever.search("site template", Filters(include_templates=False), limit=20)
    assert all(not h.doc.is_template for h in no_templates.hits)
    with_templates = retriever.search("site template", Filters(include_templates=True), limit=20)
    assert any(h.doc.doc_id == "createSiteTemplate" for h in with_templates.hits)

    hidden = retriever.search("legacy thing", Filters(include_deprecated=False), limit=20)
    assert all(h.doc.doc_id != "getLegacyThing" for h in hidden.hits)
    shown = retriever.search("legacy thing", Filters(include_deprecated=True), limit=20)
    assert shown.hits[0].doc.doc_id == "getLegacyThing"

    by_tag = retriever.search("clients", Filters(tag="client"), limit=20)  # case-insensitive
    assert [h.doc.doc_id for h in by_tag.hits] == ["getClients"]


def test_search_degrades_gracefully_when_embeddings_fail(retriever, built):
    _, embedder, _ = built
    embedder.fail = True
    result = retriever.search("create site")
    assert result.mode == "keyword" and result.hits
    assert "unavailable" in result.warning


def test_model_mismatch_disables_semantic_search(built, make_settings):
    settings, _, _ = built
    other = make_settings(openai_embedding_model="text-embedding-3-large")
    retriever = Retriever(other)  # no embedder injected -> compares manifest with .env
    assert not retriever.semantic_available and "mismatch" in retriever.semantic_status


def test_missing_index_raises_helpful_error(make_settings):
    with pytest.raises(IndexNotBuiltError, match="build-index"):
        Retriever(make_settings())


def test_guides_are_indexed_and_searchable(make_settings, embedder, tmp_path):
    guides = tmp_path / "guides"
    guides.mkdir()
    (guides / "auth.md").write_text(
        "# Authorization\n\n## Client credentials mode\nPOST the client id and secret to get an access token.\n",
        encoding="utf-8",
    )
    (guides / "README.md").write_text("ignored", encoding="utf-8")
    settings = make_settings()
    report = build_index(settings, embedder=embedder)
    assert report.guides == 1
    retriever = Retriever(settings, embedder=embedder)
    result = retriever.search("access token client credentials", Filters(kind="guide"))
    assert result.hits and result.hits[0].doc.kind == "guide"
    assert "Client credentials" in result.hits[0].doc.title
