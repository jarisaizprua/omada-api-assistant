import json
from types import SimpleNamespace

import pytest

from omada_agent.evals.answer_eval import grade
from omada_agent.evals.dataset import Question, lint, load_questions
from omada_agent.evals.judge import parse_judgement
from omada_agent.evals.report import comparison, overall_score, recommendations, summarize_answers
from omada_agent.evals.retrieval_eval import build_cases, run_retrieval_eval
from omada_agent.evals.runner import run_evaluation, select_questions
from omada_agent.evals.scoring import (
    Catalog,
    estimate_cost,
    find_invented_paths,
    find_suspicious_names,
    looks_like_refusal,
    mentions_operation,
    normalize_path,
)
from omada_agent.retrieval.hybrid import _action_of_summary, _actions_in_query


@pytest.fixture
def catalog(retriever):
    return Catalog.from_store(retriever.store)


# ---------------------------------------------------------------- scoring primitives
def test_normalize_path_handles_placeholders_and_queries():
    base = "/openapi/v1/{}/sites/{}/clients"
    assert normalize_path("/openapi/v1/{omadacId}/sites/{siteId}/clients") == base
    assert normalize_path("/openapi/v1/${OMADAC_ID}/sites/<site>/clients?page=1&pageSize=10") == base
    assert normalize_path("/openapi/v1/:omadacId/sites/{siteId}/clients).") == base


def test_invented_paths_are_flagged_but_prefixes_and_guide_paths_are_not(catalog):
    good = "Call `GET /openapi/v1/{omadacId}/sites` or the base /openapi/v1/{omadacId}."
    assert find_invented_paths(good, catalog) == []
    bad = "Use POST /openapi/v1/{omadacId}/sites/{siteId}/teleport then /openapi/v1/{omadacId}/sitez"
    assert find_invented_paths(bad, catalog) == [
        "/openapi/v1/{omadacId}/sites/{siteId}/teleport",
        "/openapi/v1/{omadacId}/sitez",
    ]


def test_mentions_operation_by_id_or_by_method_and_path(catalog):
    op = catalog.operations["createNewSite"]
    assert mentions_operation("el operationId es createNewSite", op)
    assert mentions_operation("requests.post(f\"{BASE}/openapi/v1/{omadac_id}/sites\", json=...)", op)
    assert not mentions_operation("requests.get(f\"{BASE}/openapi/v1/{omadac_id}/sites\")", op)  # wrong method
    assert not mentions_operation("createNewSiteTemplateX", op)  # not a whole word


def test_suspicious_names_ignore_real_operations_fields_and_code_blocks(catalog):
    text = "Usa `createNewSite` con `deviceAccountSetting`, no `createSiteMagic`.\n```python\ncreateFakeThing()\n```"
    assert find_suspicious_names(text, catalog) == ["createSiteMagic"]


def test_refusal_detection_in_both_languages():
    assert looks_like_refusal("No existe un endpoint para eso en la API.")
    assert looks_like_refusal("That is not available in the Omada API.")
    assert not looks_like_refusal("Usa POST /openapi/v1/{omadacId}/weather para obtenerlo.")


def test_cost_estimates():
    assert estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000) == pytest.approx(0.75)
    assert estimate_cost("some-unknown-model", 10, 10) is None
    assert estimate_cost("some-unknown-model", 1_000_000, 0, (2.0, 8.0)) == pytest.approx(2.0)


# ---------------------------------------------------------------- dataset
def test_shipped_question_file_is_well_formed():
    from pathlib import Path

    questions = load_questions(Path(__file__).resolve().parents[1] / "evals" / "questions.json")
    assert len(questions) >= 50
    assert len({q.id for q in questions}) == len(questions)
    assert {q.difficulty for q in questions} == {"facil", "media", "dificil"}
    assert sum(q.out_of_scope for q in questions) >= 3
    assert all(q.reference or q.all_terms or q.all_expected_ids or q.out_of_scope for q in questions)


def test_lint_detects_unknown_operations_and_unverifiable_terms(catalog):
    good = Question("A", "t", "facil", "buscar", "q", expect_operations=["createNewSite"], expect_terms=["region", "-33101"])
    bad_op = Question("B", "t", "facil", "buscar", "q", expect_operations=["nope"])
    bad_term = Question("C", "t", "facil", "buscar", "q", expect_operations=["getSiteList"], expect_terms=["zzz"])
    derived = Question("D", "t", "facil", "buscar", "q", expect_operations=["getSiteList"], derived_terms=["zzz"])
    problems, notes = lint([good, bad_op, bad_term, derived], catalog)
    assert any(p.startswith("B:") for p in problems) and any(p.startswith("C:") and "zzz" in p for p in problems)
    assert not any(p.startswith(("A:", "D:")) for p in problems)


def test_load_questions_rejects_bad_files(tmp_path):
    def write(items):
        path = tmp_path / "q.json"
        path.write_text(json.dumps({"questions": items}), encoding="utf-8")
        return path

    base = dict(id="Q1", topic="t", difficulty="facil", type="buscar", question="x")
    with pytest.raises(ValueError, match="Duplicate"):
        load_questions(write([base, base]))
    with pytest.raises(ValueError, match="unknown type"):
        load_questions(write([{**base, "type": "otro"}]))
    with pytest.raises(ValueError, match="difficulty"):
        load_questions(write([{**base, "difficulty": "imposible"}]))


# ---------------------------------------------------------------- grading one answer
def _run(answer, tools=("search_api",), retrieved=(), error=None):
    return {
        "answer": answer, "error": error, "retrieved": list(retrieved),
        "tools": [{"name": t, "args": {}} for t in tools], "usage": {}, "elapsed_ms": 1,
    }


def test_grade_statuses(catalog):
    q = Question("Q", "t", "media", "configurar", "x", expect_operations=["createNewSite"], expect_terms=["name", "region"])
    ok = grade(q, _run("POST /openapi/v1/{omadacId}/sites (createNewSite): name, region", retrieved=["createNewSite"]), catalog, None)
    assert ok["status"] == "ok" and ok["checks"]["endpoint_ok"] is True

    partial = grade(q, _run("createNewSite acepta name", retrieved=["createNewSite"]), catalog, None)
    assert partial["status"] == "partial" and "region" in " ".join(partial["reasons"])

    wrong = grade(q, _run("Usa getSiteList", retrieved=[]), catalog, None)
    assert wrong["status"] == "bad" and "tampoco lo encontró" in " ".join(wrong["reasons"])

    ignored = grade(q, _run("Usa getSiteList", retrieved=["createNewSite"]), catalog, None)
    assert "no lo usó" in " ".join(ignored["reasons"])

    invented = grade(q, _run("createNewSite name region en /openapi/v1/{omadacId}/sitez"), catalog, None)
    assert invented["status"] == "bad" and invented["checks"]["invented_paths"]

    errored = grade(q, _run("", error="boom"), catalog, None)
    assert errored["status"] == "bad" and "boom" in errored["reasons"][0]


def test_judge_can_downgrade_but_not_rescue(catalog):
    q = Question("Q", "t", "media", "configurar", "x", expect_operations=["createNewSite"])
    answer = "createNewSite"
    assert grade(q, _run(answer), catalog, {"score": 5, "verdict": "correct"})["status"] == "ok"
    assert grade(q, _run(answer), catalog, {"score": 3, "verdict": "partial", "comment": "falta"})["status"] == "partial"
    assert grade(q, _run(answer), catalog, {"score": 1, "verdict": "incorrect", "comment": "mal"})["status"] == "bad"
    assert grade(q, _run("Usa getSiteList"), catalog, {"score": 5, "verdict": "correct"})["status"] == "bad"
    # a failed judgement is ignored, not fatal
    assert grade(q, _run(answer), catalog, {"error": "timeout"})["status"] == "ok"


def test_out_of_scope_questions(catalog):
    q = Question("Q", "t", "media", "fuera_de_alcance", "x")
    assert grade(q, _run("No existe ningún endpoint para eso."), catalog, None)["status"] == "ok"
    assert grade(q, _run("Usa POST /openapi/v1/{omadacId}/weather"), catalog, None)["status"] == "bad"
    assert grade(q, _run("No existe, pero prueba /openapi/v1/{omadacId}/weather"), catalog, None)["status"] == "bad"


def test_parse_judgement_is_tolerant():
    assert parse_judgement('Claro: {"score": 4, "verdict": "correct", "comment": "bien"}')["verdict"] == "correct"
    assert parse_judgement('{"score": 3}')["verdict"] == "partial"  # verdict derived from the score
    assert parse_judgement('{"score": 9, "verdict": "???"}')["score"] == 5  # clamped
    assert "error" in parse_judgement("no json here")
    assert "error" in parse_judgement('{"score": "x"}')


# ---------------------------------------------------------------- search evaluation
def test_retrieval_eval_on_the_mini_spec(retriever):
    cases = build_cases(retriever, samples=50)
    assert cases and all(c.query.startswith("How do I ") for c in cases)
    assert all(c.op_id not in ("getLegacyThing", "createSiteTemplate") for c in cases)  # deprecated / template excluded
    result = run_retrieval_eval(retriever, samples=50)
    assert result["mode"] == "hybrid" and result["n"] == len(cases)
    assert 0 <= result["top1"] <= result["top3"] <= result["top8"] <= 1
    assert result["top8"] == 1.0  # a 7-operation spec: everything is within the first 8
    keyword = run_retrieval_eval(retriever, samples=50, keyword_only=True)
    assert keyword["mode"] == "keyword"


def test_action_words_steer_ranking(retriever):
    assert _action_of_summary("Modify port forwarding") == "change"
    assert _action_of_summary("Delete port forwarding") == "remove"
    assert _action_of_summary("Strange thing") is None
    assert _actions_in_query("How do I update or remove it?") == {"change", "remove"}
    assert retriever.search("how do I create a site").hits[0].doc.doc_id == "createNewSite"
    assert retriever.search("get the site list").hits[0].doc.doc_id == "getSiteList"


# ---------------------------------------------------------------- full run with stubs
class StubAgent:
    """Answers each curated question with a canned reply (no network)."""

    REPLIES = {
        "T1": "Usa `POST /openapi/v1/{omadacId}/sites` (createNewSite) con name y region.",
        "T2": "Usa GET /openapi/v1/{omadacId}/sitez para listarlos.",
        "T3": "getClients devuelve mac de cada cliente.",
        "T4": "No existe un endpoint para eso en la API.",
        "T5": "Usa POST /openapi/v1/{omadacId}/weather",
        "T7": "createNewSite con name y region.",
    }

    async def run(self, messages):
        text = messages[-1]["content"]
        qid = text.split(":")[0]
        yield {"type": "start", "model": "stub"}
        yield {"type": "tool_start", "id": "1", "name": "search_api", "args": {"query": text}}
        yield {"type": "tool_result", "id": "1", "name": "search_api", "ok": True, "summary": "1",
               "operations": [{"id": "createNewSite"}, {"id": "getClients"}], "ms": 5}
        yield {"type": "token", "text": self.REPLIES.get(qid, "")}
        yield {"type": "done", "usage": {"input_tokens": 1000, "output_tokens": 200, "tool_calls": 1, "rounds": 2}, "elapsed_ms": 2500}


class StubJudge:
    def __init__(self):
        self.responses = SimpleNamespace(create=self._create)
        self.calls = 0

    async def _create(self, **kwargs):
        self.calls += 1
        verdict = "incorrect" if "QUESTION:\nT7:" in kwargs["input"][0]["content"] else "correct"
        score = 1 if verdict == "incorrect" else 5
        return SimpleNamespace(output_text=json.dumps({"score": score, "verdict": verdict, "comment": "ok", "missing": [], "wrong": []}))


@pytest.fixture
def question_file(tmp_path):
    def q(id, question, **kw):
        return dict(id=id, topic="Pruebas", difficulty="facil", type=kw.pop("type", "configurar"), question=f"{id}: {question}",
                    reference=kw.pop("reference", "ref"), **kw)

    items = [
        q("T1", "crear sitio", expect_operations=["createNewSite"], expect_terms=["name", "region"]),
        q("T2", "listar sitios", expect_operations=["getSiteList"]),
        q("T3", "clientes", expect_operations=["getClients"], derived_terms=["mac", "vlan"]),
        q("T4", "tarjeta de crédito", type="fuera_de_alcance"),
        q("T5", "clima", type="fuera_de_alcance"),
        q("T6", "token", type="autenticacion", expect_terms=["client_credentials"], needs_guide=True),
        q("T7", "otra", expect_operations=["createNewSite"]),
    ]
    path = tmp_path / "questions.json"
    path.write_text(json.dumps({"questions": items}), encoding="utf-8")
    return path


async def test_judge_same_model_is_only_a_warning(settings, retriever, question_file):
    """A judge that is the very same model as the assistant cannot, alone, fail an answer."""
    run = await run_evaluation(
        settings, retriever=retriever, agent=StubAgent(), judge_client=StubJudge(), questions_path=question_file,
        baseline_ref=None, save=False,
    )
    status = {r["id"]: r["status"] for r in run["answers"]["records"]}
    assert status["T7"] == "partial"  # judge said "incorrect", but it is the same model as the assistant
    assert any("mismo modelo" in reason for reason in next(r for r in run["answers"]["records"] if r["id"] == "T7")["reasons"])


async def test_full_evaluation_run(settings, retriever, question_file, tmp_path):
    judge = StubJudge()
    settings = settings.model_copy(update={"openai_judge_model": "stronger-judge"})
    run = await run_evaluation(
        settings, retriever=retriever, agent=StubAgent(), judge_client=judge, questions_path=question_file,
        baseline_ref=None, save=False,
    )
    status = {r["id"]: r["status"] for r in run["answers"]["records"]}
    assert status == {"T1": "ok", "T2": "bad", "T3": "partial", "T4": "ok", "T5": "bad", "T6": "skipped", "T7": "bad"}
    assert judge.calls == 6  # skipped questions are not judged

    s = run["answers"]["summary"]
    assert (s["n"], s["ok"], s["partial"], s["bad"], s["skipped"]) == (6, 2, 1, 3, 1)
    assert s["invented_responses"] == 2 and s["looked_up"] == s["answerable"] == 4
    assert s["avg_ms"] == 2500 and s["judge_avg"] == pytest.approx((5 + 5 + 5 + 5 + 5 + 1) / 6)
    assert run["retrieval"]["n"] > 0 and 0 <= run["overall"] <= 100

    md = run["report_markdown"]
    for expected in ("# Informe de evaluación", "Nota global", "1. ¿Encuentra los documentos correctos?",
                     "2. ¿Responde bien?", "3. ¿Inventa cosas?", "4. Velocidad y costo", "5. Dónde falló",
                     "6. Qué mejorar", "Anexo", "T2", "sitez"):
        assert expected in md, expected
    json.dumps({k: v for k, v in run.items() if k != "report_markdown"})  # fully serialisable
    assert any("inventaron rutas" in tip for tip in run["recommendations"])
    assert any("guía de acceso" in tip for tip in run["recommendations"])


async def test_run_saves_files_and_supports_a_baseline(settings, retriever, question_file, tmp_path):
    settings = settings.model_copy(update={"evals_dir": str(tmp_path / "evals")})
    first = await run_evaluation(settings, retriever=retriever, agent=StubAgent(), judge_client=None,
                                 questions_path=question_file, judge=False)
    results = tmp_path / "evals" / "results"
    assert (results / "latest.json").exists()
    assert (next(results.glob("*/report.md"))).read_text(encoding="utf-8").startswith("# Informe")
    assert first["meta"]["judge_enabled"] is False

    second = await run_evaluation(settings, retriever=retriever, agent=StubAgent(), judge_client=None,
                                  questions_path=question_file, judge=False, baseline_ref="latest", only="answers")
    assert "Comparación con la evaluación anterior" in second["report_markdown"]
    assert second["retrieval"] is None


def test_select_questions_filters(question_file):
    qs = load_questions(question_file)
    assert [q.id for q in select_questions(qs, ids=["t1", "T2"])] == ["T1", "T2"]
    assert len(select_questions(qs, topic="prueb", limit=3)) == 3
    assert select_questions(qs, topic="nada") == []


def test_overall_score_bands_and_comparison():
    ans = {"n": 10, "ok": 9, "partial": 1, "invented_responses": 0}
    value, label = overall_score({"top3": 1.0}, ans)
    assert (value, label) == (97, "Excelente")  # 0.6*0.95 + 0.2*1 + 0.2*1
    assert overall_score({"top3": 0.5}, {"n": 10, "ok": 2, "partial": 0, "invented_responses": 5})[1] == "Necesita mejoras"
    assert overall_score(None, None) == (None, "Sin datos")
    rows = comparison({"overall": 90, "retrieval": {"top3": 0.9}}, {"overall": 80, "retrieval": {"top3": 0.95}})
    assert rows[0][3].startswith("🟢") and rows[1][3].startswith("🔴")
