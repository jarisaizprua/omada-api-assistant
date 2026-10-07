"""Turn an OpenAPI 3 document into :class:`~omada_agent.models.Doc` objects.

Design notes
------------
* **One document per operation** (HTTP method + path). Splitting by character
  count would separate an endpoint from its parameters; splitting by operation
  keeps every answer self-contained.
* **Two layers of text.** A short *card* (summary, category, permissions, field
  names) is what gets embedded and keyword-indexed. The long *detail* (the full
  request/response schema, with ``$ref`` resolved) is stored next to it and served
  on demand by the agent. This keeps retrieval precise and prompts small, even for
  the few operations whose response schema is huge.
* ``$ref`` pointers are resolved with a depth limit and cycle protection, so the
  output is always finite.
"""

from __future__ import annotations

import hashlib
import html
import re
from typing import Any

from omada_agent.models import Doc

HTTP_METHODS = ("get", "post", "put", "patch", "delete")

REQUEST_MAX_DEPTH = 5
RESPONSE_MAX_DEPTH = 4
MAX_SCHEMA_LINES = 400
FIELD_DESC_CHARS = 400
CARD_MAX_CHARS = 3500

_TRIVIAL_FORMATS = {"int32", "int64", "float", "double"}


# --------------------------------------------------------------------------- text helpers
def split_words(text: str) -> list[str]:
    """``createNewSite`` -> ``['create', 'New', 'Site']`` (also splits snake/kebab/path)."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    return re.findall(r"[A-Za-z0-9]+", text)


def html_to_text(raw: str) -> str:
    """The spec's descriptions contain ``<br/>`` tags and entities; make them plain text."""
    text = re.sub(r"<br\s*/?>", "\n", raw or "", flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def clean_inline(raw: str | None, limit: int = FIELD_DESC_CHARS) -> str:
    text = re.sub(r"\s+", " ", html_to_text(raw or "")).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


_ERROR_LINE_RE = re.compile(r"^(-?\d+)\s*[-–]\s*(.+)$")


def parse_description(raw: str) -> tuple[str, list[str], list[tuple[str, str]]]:
    """Split a description into (main text, required permissions, error codes).

    The Omada spec appends these two blocks to every description::

        <text><br/><br/>The interface requires one of the permissions: <br/>A<br/>B
        <br/><br/>The possible error code ... : <br/>-33101  -  message
    """
    text = html_to_text(raw)
    main: list[str] = []
    perms: list[str] = []
    errors: list[tuple[str, str]] = []
    mode = "main"
    for line in text.split("\n"):
        stripped = line.strip()
        low = stripped.lower()
        if "requires one of the permissions" in low or low.startswith("the interface requires"):
            mode = "perms"
            continue
        if "possible error code" in low:
            mode = "errors"
            continue
        if not stripped:
            mode = "main" if mode != "main" else mode
            main.append("")
            continue
        if mode == "perms":
            perms.append(stripped)
        elif mode == "errors":
            m = _ERROR_LINE_RE.match(stripped)
            if m:
                errors.append((m.group(1), m.group(2).strip()))
            else:
                main.append(stripped)
        else:
            main.append(stripped)
    description = re.sub(r"\n{3,}", "\n\n", "\n".join(main)).strip()
    return description, perms, errors


# --------------------------------------------------------------------------- schema renderer
class SchemaRenderer:
    """Render JSON schemas as compact, indented, human/LLM-friendly field lists."""

    def __init__(self, schemas: dict[str, Any], max_depth: int, max_lines: int = MAX_SCHEMA_LINES):
        self.schemas = schemas
        self.max_depth = max_depth
        self.max_lines = max_lines
        self._lines: list[str] = []
        self._omitted = 0

    # -- public ---------------------------------------------------------------------
    def render(self, schema: dict[str, Any] | None) -> list[str]:
        self._lines, self._omitted = [], 0
        node, _ = self._resolve(schema or {})
        node = self._merge(node)
        if node.get("type") == "array":
            items, iref = self._resolve(node.get("items", {}))
            items = self._merge(items)
            if "properties" in items:
                self._emit(f"Array of object{self._ref_tag(iref)}:")
                self._props(items, 1, frozenset({iref} if iref else ()), 1)
            else:
                self._emit(f"Array of {self._type_label(items)}")
        elif "properties" in node:
            self._props(node, 1, frozenset(), 0)
        else:
            self._emit(self._type_label(node) + self._constraints(node))
        if self._omitted:
            self._lines.append(f"[... {self._omitted} more lines omitted]")
        return self._lines

    def top_level_fields(self, schema: dict[str, Any] | None) -> list[tuple[str, bool]]:
        node, _ = self._resolve(schema or {})
        node = self._merge(node)
        if node.get("type") == "array":
            node, _ = self._resolve(node.get("items", {}))
            node = self._merge(node)
        required = set(node.get("required") or [])
        return [(name, name in required) for name in (node.get("properties") or {})]

    @staticmethod
    def field_names(lines: list[str], limit: int = 80) -> list[str]:
        """Unique field names found in rendered lines (used for search text)."""
        seen: dict[str, None] = {}
        for line in lines:
            m = re.match(r"^\s*- ([A-Za-z0-9_.$\-]+)\*? \(", line)
            if m:
                seen.setdefault(m.group(1), None)
            if len(seen) >= limit:
                break
        return list(seen)

    # -- internals --------------------------------------------------------------------
    def _resolve(self, node: Any) -> tuple[dict[str, Any], str | None]:
        """Follow ``$ref`` chains. Returns (resolved node, last ref name)."""
        name = None
        hops = 0
        while isinstance(node, dict) and "$ref" in node and hops < 10:
            name = node["$ref"].rsplit("/", 1)[-1]
            node = self.schemas.get(name, {})
            hops += 1
        return (node if isinstance(node, dict) else {}), name

    def _merge(self, node: dict[str, Any]) -> dict[str, Any]:
        """Flatten ``allOf`` into a single schema."""
        if "allOf" not in node:
            return node
        merged = {k: v for k, v in node.items() if k != "allOf"}
        props = dict(merged.get("properties") or {})
        required = list(merged.get("required") or [])
        for part in node["allOf"]:
            resolved, _ = self._resolve(part)
            resolved = self._merge(resolved)
            props.update(resolved.get("properties") or {})
            required += resolved.get("required") or []
            for key in ("type", "description", "enum", "items"):
                if key in resolved:
                    merged.setdefault(key, resolved[key])
        merged["properties"] = props
        merged["required"] = required
        return merged

    def _emit(self, line: str) -> None:
        if len(self._lines) >= self.max_lines:
            self._omitted += 1
        else:
            self._lines.append(line)

    @staticmethod
    def _ref_tag(ref: str | None) -> str:
        return f" <{ref}>" if ref else ""

    def _type_label(self, node: dict[str, Any]) -> str:
        t = node.get("type")
        fmt = node.get("format")
        if t == "array":
            items, _ = self._resolve(node.get("items", {}))
            return f"array<{self._type_label(self._merge(items))}>"
        if t == "object" or (t is None and "properties" in node):
            extra = node.get("additionalProperties")
            if isinstance(extra, dict) and extra:
                value, _ = self._resolve(extra)
                return f"map<string, {self._type_label(self._merge(value))}>"
            return "object"
        if t is None:
            return "enum" if "enum" in node else "any"
        if fmt and fmt not in _TRIVIAL_FORMATS:
            return f"{t}({fmt})"
        return t

    @staticmethod
    def _constraints(node: dict[str, Any]) -> str:
        parts: list[str] = []
        if "enum" in node:
            values = [str(v) for v in node["enum"][:25]]
            more = "…" if len(node["enum"]) > 25 else ""
            parts.append("enum: " + ", ".join(values) + more)
        for key, label in (
            ("minimum", "min"),
            ("maximum", "max"),
            ("minLength", "minLength"),
            ("maxLength", "maxLength"),
            ("minItems", "minItems"),
            ("maxItems", "maxItems"),
        ):
            if key in node:
                parts.append(f"{label}={node[key]}")
        if node.get("default") is not None:
            parts.append(f"default={node['default']}")
        if node.get("example") is not None:
            parts.append(f"example={node['example']}")
        if node.get("nullable"):
            parts.append("nullable")
        return ("; " + "; ".join(parts)) if parts else ""

    def _props(self, node: dict[str, Any], depth: int, seen: frozenset[str], indent: int) -> None:
        required = set(node.get("required") or [])
        pad = "  " * indent
        for name, raw in (node.get("properties") or {}).items():
            if len(self._lines) >= self.max_lines:
                self._omitted += 1
                continue
            res, ref = self._resolve(raw)
            res = self._merge(res)
            desc = clean_inline((raw or {}).get("description") or res.get("description"))
            star = "*" if name in required else ""
            suffix = f": {desc}" if desc else ""

            if res.get("type") == "array":
                items, iref = self._resolve(res.get("items", {}))
                items = self._merge(items)
                if "properties" in items:
                    self._emit(f"{pad}- {name}{star} (array of object{self._ref_tag(iref)}){suffix}")
                    self._descend(items, iref, depth, seen, indent + 1)
                    continue
            elif "properties" in res:
                self._emit(f"{pad}- {name}{star} (object{self._ref_tag(ref)}){suffix}")
                self._descend(res, ref, depth, seen, indent + 1)
                continue

            label = self._type_label(res)
            self._emit(f"{pad}- {name}{star} ({label}{self._constraints(res)}){suffix}")

    def _descend(
        self, node: dict[str, Any], ref: str | None, depth: int, seen: frozenset[str], indent: int
    ) -> None:
        pad = "  " * indent
        if ref and ref in seen:
            self._emit(f"{pad}(recursive reference to {ref})")
        elif depth >= self.max_depth:
            n = len(node.get("properties") or {})
            self._emit(f"{pad}(… {n} nested fields omitted; schema {ref or 'inline'})")
        else:
            self._props(node, depth + 1, seen | ({ref} if ref else frozenset()), indent)


# --------------------------------------------------------------------------- operation parsing
def _param_line(param: dict[str, Any], renderer: SchemaRenderer) -> str:
    schema, _ = renderer._resolve(param.get("schema", {}))
    label = renderer._type_label(renderer._merge(schema))
    where = param.get("in", "?")
    req = ", required" if param.get("required") else ""
    cons = renderer._constraints(schema)
    desc = clean_inline(param.get("description"))
    tail = f": {desc}" if desc else ""
    return f"- {param.get('name')} ({where}, {label}{req}{cons}){tail}"


def _pick_content(content: dict[str, Any] | None) -> tuple[str, dict[str, Any]] | None:
    if not content:
        return None
    for ctype in ("application/json", "*/*"):
        if ctype in content:
            return ctype, content[ctype].get("schema", {})
    ctype, body = next(iter(content.items()))
    return ctype, body.get("schema", {})


def _success_response(responses: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    for code in sorted(responses):
        if code.startswith("2"):
            return code, responses[code]
    return None


def _resolve_param(spec: dict[str, Any], param: dict[str, Any]) -> dict[str, Any]:
    if "$ref" in param:
        name = param["$ref"].rsplit("/", 1)[-1]
        return spec.get("components", {}).get("parameters", {}).get(name, {})
    return param


def _normalize_path(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "{}", path).rstrip("/").lower()


def parse_operations(spec: dict[str, Any]) -> list[Doc]:
    """Convert every operation of the spec into a :class:`Doc`."""
    schemas = (spec.get("components") or {}).get("schemas") or {}
    req_renderer = SchemaRenderer(schemas, REQUEST_MAX_DEPTH)
    res_renderer = SchemaRenderer(schemas, RESPONSE_MAX_DEPTH)

    docs: list[Doc] = []
    used_ids: set[str] = set()

    for path, item in (spec.get("paths") or {}).items():
        shared_params = item.get("parameters", [])
        for method in HTTP_METHODS:
            op = item.get(method)
            if not op:
                continue

            tags = op.get("tags") or ["Untagged"]
            tag = tags[0]
            op_id = op.get("operationId") or f"{method}_{hashlib.md5(path.encode()).hexdigest()[:8]}"
            if op_id in used_ids:
                op_id = f"{op_id}__{method}"
            used_ids.add(op_id)

            description, permissions, errors = parse_description(op.get("description", ""))
            summary = clean_inline(op.get("summary"), 200) or (description.split("\n")[0][:200])
            version_match = re.match(r"/openapi/(v\d+)/", path)
            version = version_match.group(1) if version_match else ""
            deprecated = bool(op.get("deprecated"))
            is_template = "template" in tag.lower() or "/sitetemplates" in path.lower()

            # parameters (operation-level overrides path-level)
            params: dict[tuple[str, str], dict[str, Any]] = {}
            for raw_param in [*shared_params, *op.get("parameters", [])]:
                p = _resolve_param(spec, raw_param)
                if p.get("name"):
                    params[(p["name"], p.get("in", ""))] = p
            param_lines = [_param_line(p, req_renderer) for p in params.values()]

            # request body
            request_lines: list[str] = []
            body_fields: list[tuple[str, bool]] = []
            req_picked = _pick_content((op.get("requestBody") or {}).get("content"))
            if req_picked:
                ctype, schema = req_picked
                body_fields = req_renderer.top_level_fields(schema)
                request_lines = req_renderer.render(schema)
                request_text = f"Request body ({ctype}):\n" + "\n".join(request_lines)
            else:
                request_text = "Request body: none."
            request_names = SchemaRenderer.field_names(request_lines)

            # response
            response_lines: list[str] = []
            resp = _success_response(op.get("responses") or {})
            response_text = "Response: no documented body."
            if resp:
                code, body = resp
                picked = _pick_content(body.get("content"))
                if picked:
                    ctype, schema = picked
                    response_lines = res_renderer.render(schema)
                    response_text = f"Response {code} ({ctype}):\n" + "\n".join(response_lines)
                else:
                    response_text = f"Response {code}: {clean_inline(body.get('description')) or 'OK'}"
            response_names = SchemaRenderer.field_names(response_lines, limit=60)

            # ---- assemble texts -----------------------------------------------------
            status = " (DEPRECATED)" if deprecated else ""
            header = [
                f"# {method.upper()} {path}{status}",
                f"operationId: {op_id}",
                f"category: {tag}" + (" [site-template operation]" if is_template else ""),
                f"summary: {summary}",
            ]
            if version:
                header.append(f"path version: {version}")
            if description and description != summary:
                header += ["", "Description:", description]
            if permissions:
                header += ["", "Required permissions (any one of): " + "; ".join(permissions)]
            if errors:
                header += ["", "Operation-specific error codes:"]
                header += [f"- {code}: {msg}" for code, msg in errors]
            if param_lines:
                header += ["", "Parameters:", *param_lines]
            header_text = "\n".join(header)

            path_words = [w for w in split_words(path) if w.lower() not in {"openapi"}]
            id_words = split_words(op_id)
            tag_words = split_words(tag)
            keywords = " ".join(dict.fromkeys([*path_words, *tag_words, method]))

            body_brief = ", ".join(f"{n}{'*' if r else ''}" for n, r in body_fields[:14])
            if len(body_fields) > 14:
                body_brief += f", … (+{len(body_fields) - 14})"
            param_brief = ", ".join(
                f"{p['name']}({p.get('in', '?')}){'*' if p.get('required') else ''}"
                for p in list(params.values())[:10]
            )
            brief = [
                f"[{method.upper()}] {path} — {summary} (operationId: {op_id})",
                f"  category: {tag}"
                + (" | TEMPLATE operation" if is_template else "")
                + (" | DEPRECATED" if deprecated else "")
                + (f" | permissions: {'; '.join(permissions)}" if permissions else ""),
            ]
            if param_brief:
                brief.append(f"  params: {param_brief}")
            if body_brief:
                brief.append(f"  body: {body_brief}")

            card = [
                f"{method.upper()} {path} — {summary}",
                f"operationId: {op_id}",
                f"Category: {tag}",
                f"Keywords: {' '.join(id_words)}; {keywords}",
            ]
            if description and description != summary:
                card.append(f"Description: {description[:700]}")
            if permissions:
                card.append("Permissions: " + "; ".join(permissions))
            if param_brief:
                card.append("Parameters: " + param_brief)
            if request_names:
                card.append("Request fields: " + ", ".join(request_names))
            if response_names:
                card.append("Response fields: " + ", ".join(response_names))
            if errors:
                card.append("Errors: " + "; ".join(f"{c} {m}" for c, m in errors[:8]))
            if deprecated:
                card.append("Status: deprecated")
            card_text = "\n".join(card)[:CARD_MAX_CHARS]

            docs.append(
                Doc(
                    doc_id=op_id,
                    kind="operation",
                    title=f"{method.upper()} {path}",
                    method=method.upper(),
                    path=path,
                    path_norm=_normalize_path(path),
                    version=version,
                    tag=tag,
                    tags=list(tags),
                    deprecated=deprecated,
                    is_template=is_template,
                    summary=summary,
                    description=description,
                    permissions=permissions,
                    errors=errors,
                    brief_text="\n".join(brief),
                    card_text=card_text,
                    fts_title=f"{summary} {' '.join(id_words)}",
                    fts_keywords=keywords,
                    fts_description=description,
                    fts_fields=" ".join(
                        [
                            *request_names,
                            *response_names,
                            *[f"{c} {m}" for c, m in errors],
                            *permissions,
                        ]
                    ),
                    header_text=header_text,
                    request_text=request_text,
                    response_text=response_text,
                )
            )
    return docs
