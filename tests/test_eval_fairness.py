"""Regression tests for the evaluation mistakes found in the first real run."""

from omada_agent.evals.scoring import (
    fold,
    missing_terms,
    normalize_path,
    strip_interpolation,
)
from omada_agent.evals.scoring import PATH_RE


def paths(text):
    return [normalize_path(p) for p in PATH_RE.findall(strip_interpolation(text))]


def test_python_fstring_urls_are_not_cut_in_the_middle():
    code = "url = f\"{BASE}/openapi/v1/{os.environ['OMADAC_ID']}/sites/{os.getenv('SITE')}/ssids\""
    assert paths(code) == ["/openapi/v1/{}/sites/{}/ssids"]


def test_js_template_literals_and_plain_placeholders():
    assert paths("`${process.env.BASE}/openapi/v2/${omadacId}/lan-networks`") == ["/openapi/v2/{}/lan-networks"]
    assert paths("GET /openapi/v1/{omadacId}/sites/{siteId}") == ["/openapi/v1/{}/sites/{}"]


def test_terms_ignore_accents_underscores_and_case():
    answer = "En el modo código de autorización (Authorization Code) usa client credentials."
    assert missing_terms(answer, ["authorization_code", "client_credentials"]) == []


def test_terms_accept_alternatives():
    assert missing_terms("usa el código de autorización", ["authorization_code|código de autorización"]) == []
    assert missing_terms("nada relevante", ["authorization_code|código de autorización"]) != []


def test_fold_is_symmetric():
    assert fold("Código_de-Autorización") == fold("codigo de autorizacion")
