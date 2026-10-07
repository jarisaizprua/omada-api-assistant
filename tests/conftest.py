"""Shared fixtures: a tiny OpenAPI spec, a deterministic fake embedder and a fake OpenAI client.

None of the tests touch the network or need an API key.
"""

from __future__ import annotations

import hashlib
import json
import math
from types import SimpleNamespace

import pytest

from omada_agent.config import Settings
from omada_agent.ingestion.indexer import build_index
from omada_agent.ingestion.openapi_parser import split_words
from omada_agent.retrieval.hybrid import Retriever

ENVELOPE = {
    "type": "object",
    "properties": {
        "errorCode": {"type": "integer"},
        "msg": {"type": "string"},
        "result": {"type": "object"},
    },
}


def _op(op_id, summary, tag, description=None, **extra):
    base = {
        "tags": [tag],
        "summary": summary,
        "description": description or summary,
        "operationId": op_id,
        "parameters": [
            {"name": "omadacId", "in": "path", "required": True, "description": "Omada ID", "schema": {"type": "string"}}
        ],
        "responses": {
            "200": {"description": "OK", "content": {"*/*": {"schema": {"$ref": "#/components/schemas/OperationResponse"}}}}
        },
    }
    base.update(extra)
    return base


MINI_SPEC = {
    "openapi": "3.0.1",
    "info": {"title": "Omada Open API", "version": "v0.1"},
    "paths": {
        "/openapi/v1/{omadacId}/sites": {
            "get": _op(
                "getSiteList",
                "Get site list",
                "Site",
                parameters=[
                    {"name": "omadacId", "in": "path", "required": True, "schema": {"type": "string"}},
                    {"name": "page", "in": "query", "required": True, "description": "Page number", "schema": {"type": "integer"}},
                    {"name": "pageSize", "in": "query", "description": "Items per page", "schema": {"type": "integer", "maximum": 1000}},
                ],
            ),
            "post": _op(
                "createNewSite",
                "Create new site",
                "Site",
                "Create new site<br/><br/>The interface requires one of the permissions: <br/>"
                "Global Dashboard Manager Modify<br/><br/>The possible error code for the interface in the "
                "returned body is one of the following error codes (non generic error codes): <br/>"
                "-33101  -  This name already exists. Please enter another name for this site.<br/>"
                "-33104  -  The number of sites has reached the limit.",
                requestBody={
                    "required": True,
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/CreateSiteEntity"}}},
                },
            ),
        },
        "/openapi/v1/{omadacId}/sites/{siteId}/wireless-network/ssids/{ssidId}": {
            "patch": _op(
                "updateSsid",
                "Modify SSID password",
                "Wireless Network",
                requestBody={"content": {"application/json": {"schema": {"$ref": "#/components/schemas/SsidUpdate"}}}},
            )
        },
        "/openapi/v1/{omadacId}/sites/{siteId}/clients": {
            "get": _op(
                "getClients",
                "Get connected clients list",
                "Client",
                responses={
                    "200": {
                        "description": "OK",
                        "content": {
                            "application/json": {
                                "schema": {"type": "array", "items": {"$ref": "#/components/schemas/ClientInfo"}}
                            }
                        },
                    }
                },
            )
        },
        "/openapi/v1/{omadacId}/sitetemplates": {
            "post": _op("createSiteTemplate", "Create new site template", "Site Template")
        },
        "/openapi/v1/{omadacId}/sites/{siteId}/legacy": {
            "get": _op("getLegacyThing", "Get legacy thing", "Site", deprecated=True)
        },
        "/openapi/v1/{omadacId}/sites/{siteId}/tree": {
            "get": _op(
                "getTree",
                "Get device tree",
                "Device",
                responses={
                    "200": {
                        "description": "OK",
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/TreeNode"}}},
                    }
                },
            )
        },
    },
    "components": {
        "schemas": {
            "OperationResponse": ENVELOPE,
            "CreateSiteEntity": {
                "type": "object",
                "required": ["name", "region"],
                "properties": {
                    "name": {"type": "string", "description": "Name of the site, 1 to 64 characters."},
                    "region": {"type": "string", "description": "ISO country code"},
                    "type": {"type": "integer", "enum": [0, 1], "description": "0 basic, 1 pro"},
                    "deviceAccountSetting": {"$ref": "#/components/schemas/DeviceAccount"},
                    "tagIds": {"type": "array", "items": {"type": "string"}},
                },
            },
            "DeviceAccount": {
                "type": "object",
                "required": ["username"],
                "properties": {"username": {"type": "string"}, "password": {"type": "string"}},
            },
            "SsidUpdate": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "pskSetting": {"type": "object", "properties": {"securityKey": {"type": "string"}}}},
            },
            "ClientInfo": {
                "type": "object",
                "properties": {"mac": {"type": "string"}, "ip": {"type": "string"}, "ssid": {"type": "string"}},
            },
            "TreeNode": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "children": {"type": "array", "items": {"$ref": "#/components/schemas/TreeNode"}},
                },
            },
        }
    },
}


class FakeEmbedder:
    """Hashing bag-of-words embedder: similar words -> similar vectors. Deterministic, offline."""

    model = "fake-embedder"

    def __init__(self):
        self.calls = 0
        self.fail = False

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * 128
        for w in split_words(text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 128] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts, on_batch=None):
        self.calls += len(texts)
        out = [self._vec(t) for t in texts]
        if on_batch:
            on_batch(len(texts), len(texts))
        return out

    def embed_query(self, text):
        if self.fail:
            raise RuntimeError("embeddings are down")
        return self._vec(text)


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def spec_file(tmp_path):
    path = tmp_path / "spec" / "Omada_Test_1.2.3.json"
    path.parent.mkdir()
    path.write_text(json.dumps(MINI_SPEC), encoding="utf-8")
    return path


@pytest.fixture
def make_settings(tmp_path, spec_file):
    def _make(**overrides) -> Settings:
        values = dict(
            _env_file=None,
            openai_api_key="test-key",
            spec_path=str(spec_file),
            guides_dir=str(tmp_path / "guides"),
            index_dir=str(tmp_path / "index"),
            rate_limit_per_minute=0,
            usage_db_path=str(tmp_path / "usage" / "usage.db"),
            # a model the price table knows, so budgets in dollars are enforceable in tests
            openai_model="gpt-4o-mini",
        )
        values.update(overrides)
        return Settings(**values)

    return _make


@pytest.fixture
def settings(make_settings):
    return make_settings()


@pytest.fixture
def built(settings, embedder):
    report = build_index(settings, embedder=embedder)
    return settings, embedder, report


@pytest.fixture
def retriever(built):
    settings, embedder, _ = built
    return Retriever(settings, embedder=embedder)


# ---------------------------------------------------------------- fake OpenAI client
def text_delta(text: str):
    return SimpleNamespace(type="response.output_text.delta", delta=text)


def function_call(name: str, arguments: dict | str, call_id: str):
    return SimpleNamespace(
        type="function_call",
        name=name,
        arguments=arguments if isinstance(arguments, str) else json.dumps(arguments),
        call_id=call_id,
    )


def completed(resp_id: str, output=None, in_tokens=10, out_tokens=5, status="completed"):
    return SimpleNamespace(
        type="response.completed" if status == "completed" else "response.incomplete",
        response=SimpleNamespace(
            id=resp_id,
            status=status,
            output=output or [],
            usage=SimpleNamespace(input_tokens=in_tokens, output_tokens=out_tokens),
            incomplete_details=SimpleNamespace(reason="max_output_tokens") if status == "incomplete" else None,
        ),
    )


class _FakeStream:
    def __init__(self, events):
        self._events = events
        self.closed = False

    def __aiter__(self):
        async def gen():
            for e in self._events:
                yield e

        return gen()

    async def close(self):
        self.closed = True


class FakeOpenAI:
    """Replays scripted turns; records every ``responses.create`` call."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.calls: list[dict] = []
        self.responses = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.turns:
            raise AssertionError("FakeOpenAI ran out of scripted turns")
        turn = self.turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return _FakeStream(turn)
