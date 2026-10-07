from omada_agent.ingestion.guide_import import normalize_guide_text
from omada_agent.ingestion.guides import chunk_markdown
from omada_agent.ingestion.indexer import build_index
from omada_agent.models import Filters
from omada_agent.retrieval.hybrid import Retriever

# Plain text exactly as copied from the Omada documentation site (tabs and all).
RAW = """Organization Level Open API Access Guide

Omada's Open API supports OAuth 2.0.
1 Overview
1.2 Authentication Mode Overview
Mode \tCredential \tBest For
Client credentials \tclient ID + client secret \tServer-to-server calls
3.3 Client Credentials Mode Access Process
3.3.1 Obtain Access Token

The client credentials mode can directly obtain the access token.

    In the Query, fill in the grant type(the field name is grant_type).

An example of the curl command of the obtaining access token interface is as follows:

Request:

curl -X POST "https://localhost:8043/openapi/authorize/token?grant_type=client_credentials" \\
  -H "Content-Type: application/json" \\
  -d '{
    "omadacId": "de382a0e",
    "client_id": "1855",
    "client_secret": "7673"
  }' \\
  -k

Response:

{
  "errorCode": 0,
  "result": {
    "accessToken": "AT-bjaJk",
    "expiresIn": 7200
  }
}

3.3.2 Access API Interface

In the Header, fill in the access token. Note prefix of the access token in the Authorization header is "AccessToken=" .
4.1 Status Code
Status Code \tExplain
400 \tBad Request
401 \tUnauthorized access
"""


def test_headings_become_markdown_levels():
    md = normalize_guide_text(RAW)
    lines = md.splitlines()
    assert lines[0] == "# Organization Level Open API Access Guide"
    assert "## 1 Overview" in lines
    assert "### 1.2 Authentication Mode Overview" in lines
    assert "#### 3.3.1 Obtain Access Token" in lines


def test_table_rows_are_not_mistaken_for_headings():
    md = normalize_guide_text(RAW)
    assert "400 | Bad Request" in md
    assert not any(line.startswith("#") and "Bad Request" in line for line in md.splitlines())
    assert "Client credentials | client ID + client secret | Server-to-server calls" in md


def test_examples_are_fenced():
    md = normalize_guide_text(RAW)
    assert "```bash\ncurl -X POST" in md and "-k\n```" in md
    assert '```json\n{\n  "errorCode": 0' in md and '}\n```' in md


def test_normalizing_twice_changes_nothing():
    once = normalize_guide_text(RAW)
    assert normalize_guide_text(once) == once


def test_chunks_keep_heading_trail_and_whole_examples():
    chunks = chunk_markdown(normalize_guide_text(RAW))
    token_chunk = next(body for trail, body in chunks if trail.endswith("3.3.1 Obtain Access Token"))
    assert "grant_type=client_credentials" in token_chunk
    assert "client_secret" in token_chunk and "```" in token_chunk  # the example is not split in two
    assert any("3.3 Client Credentials Mode Access Process" in trail for trail, _ in chunks)


def test_guide_is_found_by_the_retriever(make_settings, embedder, tmp_path):
    guides = tmp_path / "guides"
    guides.mkdir()
    (guides / "access_guide.md").write_text(normalize_guide_text(RAW), encoding="utf-8")
    settings = make_settings()
    report = build_index(settings, embedder=embedder)
    assert report.guides >= 3
    result = Retriever(settings, embedder=embedder).search(
        "obtain access token client credentials", Filters(kind="guide")
    )
    assert "3.3.1 Obtain Access Token" in result.hits[0].doc.title
