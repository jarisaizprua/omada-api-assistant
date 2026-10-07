"""Deterministic checks used to grade answers (no LLM involved).

Everything here is plain string matching against the knowledge base, so results are
reproducible and free:

* did the answer name the expected endpoint?
* does every ``/openapi/...`` path in the answer really exist? (anti-hallucination)
* are the key terms (field names, error codes, headers) present?
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from omada_agent.retrieval.docstore import DocStore
from omada_agent.usage.pricing import REFERENCE_PRICES, reference_price

PATH_RE = re.compile(r"/openapi/[A-Za-z0-9_\-{}$.:<>%/]*")

# Code builds URLs with interpolation: f"{BASE}/openapi/v1/{os.environ['OMADAC_ID']}/sites" or
# `${process.env.SITE}`. Those brackets contain code, not path text, so each one is collapsed to a plain
# "{}" placeholder before paths are extracted. Without this a perfectly valid script looks like it
# calls a path such as "/openapi/v1/{os.environ" that does not exist.
_INTERPOLATION_RE = re.compile(r"\$?\{[^{}\n]*[().'\"\[\]$,+][^{}\n]*\}")


def strip_interpolation(text: str) -> str:
    return _INTERPOLATION_RE.sub("{}", text)

# Verbs that begin Omada operationIds (getSiteList, createNewSite, ...). Used to spot made-up names.
_OPERATION_VERBS = (
    "get", "create", "modify", "update", "delete", "add", "edit", "set", "start", "batch", "reboot",
    "forget", "adopt", "block", "unblock", "locate", "export", "import", "query", "check", "clear",
    "reset", "upload", "download", "disconnect", "reconnect", "lock", "unlock", "enable", "disable",
    "trigger", "confirm", "remove", "list", "send", "stop", "cancel", "apply", "sync",
)
_CAMEL_RE = re.compile(r"^(?:" + "|".join(_OPERATION_VERBS) + r")[A-Z][A-Za-z0-9_]*$")
_INLINE_CODE_RE = re.compile(r"(?<!`)`([^`\n]{3,80})`(?!`)")
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)

_REFUSAL_RE = re.compile(
    r"\b(no (existe|hay|encontr|dispon|ofrece|cuenta con|tiene|aparece|figura|est[aá] (incluid|document|en))"
    r"|no es posible|no se puede|no he (encontrado|podido)|no pude|no ofrece|sin endpoint"
    r"|not (available|found|exist|supported|documented|provided|part of)|no (such )?(endpoint|operation)"
    r"|does(n't| not) (exist|provide|have|offer|appear)|isn't (available|exposed|in)|cannot be)",
    re.IGNORECASE,
)

def normalize_path(path: str) -> str:
    path = path.split("?")[0].rstrip(".,;:)'\"`")
    path = re.sub(r"\$?\{[^}]*\}", "{}", path)
    path = re.sub(r"<[^>]*>", "{}", path)
    path = re.sub(r"/:[A-Za-z_][A-Za-z0-9_]*", "/{}", path)
    return path.rstrip("/").lower()


@dataclass
class Catalog:
    """What really exists in the knowledge base (built once per run)."""

    operations: dict[str, dict]  # operationId -> light row
    paths: set[str]
    guide_paths: set[str]
    field_names: set[str]
    guides: int
    store: DocStore = field(repr=False, default=None)  # type: ignore[assignment]

    @classmethod
    def from_store(cls, store: DocStore) -> "Catalog":
        rows = store.iter_operations()
        guide_text = store.guide_corpus()
        return cls(
            operations={r["doc_id"]: r for r in rows},
            paths={r["path_norm"] for r in rows},
            guide_paths={normalize_path(p) for p in PATH_RE.findall(guide_text)},
            field_names=store.field_names(),
            guides=store.stats()["guides"],
            store=store,
        )

    def path_exists(self, normalized: str) -> bool:
        known = self.paths | self.guide_paths
        if normalized in known:
            return True
        # A prefix such as "/openapi/v1/{}" (base URL) is fine.
        return any(k.startswith(normalized + "/") for k in known)


# ---------------------------------------------------------------------------- individual checks
def find_invented_paths(answer: str, catalog: Catalog) -> list[str]:
    invented: list[str] = []
    for raw in PATH_RE.findall(strip_interpolation(answer)):
        normalized = normalize_path(raw)
        if len(normalized) <= len("/openapi"):
            continue
        if not catalog.path_exists(normalized) and raw.rstrip(".,;:)'\"`") not in invented:
            invented.append(raw.rstrip(".,;:)'\"`"))
    return invented


def find_suspicious_names(answer: str, catalog: Catalog) -> list[str]:
    """Inline `camelCaseNames` that look like operationIds but are not in the spec (soft signal)."""
    text = _FENCE_RE.sub("", answer)
    known = {i.lower() for i in catalog.operations} | catalog.field_names
    out: list[str] = []
    for token in _INLINE_CODE_RE.findall(text):
        token = token.strip()
        if _CAMEL_RE.match(token) and token.lower() not in known and token not in out:
            out.append(token)
    return out


def mentions_operation(answer: str, op: dict) -> bool:
    if re.search(rf"(?<![A-Za-z0-9_]){re.escape(op['doc_id'])}(?![A-Za-z0-9_])", answer, re.IGNORECASE):
        return True
    answer = strip_interpolation(answer)
    for match in PATH_RE.finditer(answer):
        if normalize_path(match.group()) == op["path_norm"]:
            window = answer[max(0, match.start() - 90) : match.start()].lower()
            if op["method"].lower() in window:
                return True
    return False


def fold(text: str) -> str:
    """Lower-case, drop accents and treat ``_``, ``-`` and spaces alike, so "authorization_code",
    "Authorization Code" and "código de autorización" compare on equal terms."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[\s_\-]+", " ", text)


def term_present(folded_answer: str, term: str) -> bool:
    """``term`` may list alternatives separated by ``|`` (any one is enough)."""
    return any(fold(alt.strip()) in folded_answer for alt in term.split("|") if alt.strip())


def missing_terms(answer: str, terms: list[str]) -> list[str]:
    folded = fold(answer)
    return [t for t in terms if not term_present(folded, t)]


def looks_like_refusal(answer: str) -> bool:
    return bool(_REFUSAL_RE.search(answer))


def estimate_cost(
    model: str, input_tokens: int, output_tokens: int, override: tuple[float | None, float | None] = (None, None)
) -> float | None:
    price_in, price_out = override
    if price_in is None or price_out is None:
        ref = reference_price(model)
        if ref is None:
            return None
        price_in, price_out = ref
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000
