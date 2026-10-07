"""Who is making this request? (for rate limits, budgets and the usage ledger)"""

from __future__ import annotations

import re

from fastapi import Request

from omada_agent.config import Settings

_UNSAFE = re.compile(r"[^\w@.+\-:]")


def clean(value: str, limit: int = 120) -> str:
    return _UNSAFE.sub("_", value.strip().lower())[:limit]


def client_ip(request: Request, settings: Settings) -> str:
    if settings.portal_behind_proxy:
        forwarded = request.headers.get("x-forwarded-for", "")
        first = forwarded.split(",")[0].strip()
        if first:
            return clean(first, 64)
    return clean(request.client.host, 64) if request.client else "unknown"


def identify(request: Request, settings: Settings) -> str:
    """The signed-in user (from the proxy's header, when configured) or else the client IP."""
    if settings.portal_user_header:
        value = request.headers.get(settings.portal_user_header, "")
        if value.strip():
            return clean(value)
    return f"ip:{client_ip(request, settings)}"


def is_admin(user: str, settings: Settings) -> bool:
    admins = settings.admin_list
    return not admins or user in admins
