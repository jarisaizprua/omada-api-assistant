"""Decide when a question needs the prose guides, and what to look for in them.

Small models often answer from memory when they should have opened the guide ("Bearer" instead of
"AccessToken=", a made-up batch limit, a time zone that is not in the table). To remove that failure
mode, the agent searches the guides *itself* whenever the user's message touches these topics, and
hands the excerpts to the model as context. The model can still search more.
"""

from __future__ import annotations

import re

# (pattern on the user's message, English words that help find the right guide section)
TRIGGERS: list[tuple[str, str]] = [
    (r"token|autentic|authentic|authoriz|autoriz|credencial|credential|client[_ ]?(id|secret)|login|iniciar sesi|sesi[oó]n", "access token authorization client credentials"),
    (r"cabecera|header|bearer|accesstoken", "Authorization header AccessToken"),
    (r"refresh|expir|renov|caduc", "refresh token expiration"),
    (r"l[ií]mite|limit|m[aá]xim|maximum|cu[aá]ntas? (llamadas|operaciones|peticiones)|batch|lote|varias llamadas|una sola vez|several calls|multiple calls|bulk", "batch API limit maximum operations"),
    (r"zona horaria|time ?zone|huso", "time zone table timeZone"),
    (r"protocolo|protocol|c[oó]digo entero|integer code|enum|valores? (posibles|v[aá]lidos)", "protocol number table values"),
    (r"-44\d{3}", "error code access token"),
    (r"omadacid|openapi/authorize", "omadacId authorize"),
    (r"adopt", "device adoption credentials"),
]
_COMPILED = [(re.compile(p, re.IGNORECASE), hint) for p, hint in TRIGGERS]


def guide_query_for(message: str) -> str | None:
    """Search text for the guides, or ``None`` when the message does not need them."""
    hints: list[str] = []
    for pattern, hint in _COMPILED:
        if pattern.search(message):
            hints.append(hint)
    if not hints:
        return None
    words = list(dict.fromkeys(" ".join(hints).split()))
    return f"{message.strip()[:200]} {' '.join(words)}"
