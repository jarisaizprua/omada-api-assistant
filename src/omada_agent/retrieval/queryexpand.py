"""Query expansion with Omada vocabulary.

Developers describe things in everyday words ("restart an AP", "make it blink"), while the
Omada API names them differently ("Reboot device", "Locate device"). Neither BM25 nor a small
embedding model reliably bridges that gap, so we *add* the API's own words to the query.

Rules are additive: the original query is always kept, we only append terms. Each rule is
plain data, so extending the vocabulary means adding one line to ``RULES``.
"""

from __future__ import annotations

import re

# (pattern on the lower-cased query, terms to append). Terms are the words the API itself uses
# in operation summaries, paths and field names.
RULES: list[tuple[str, str]] = [
    (r"\b(restart|power[- ]?cycle|bounce|reload|reboot)\w*", "reboot"),
    (r"\b(blink\w*|flash(ing)?|identify|beacon|physically)\b", "locate"),
    (r"\b(cpu|memory|ram|utili[sz]ation|performance|historical|history)\b", "statistic"),
    (r"\b5[- ]?min(ute)?s?\b|\bevery five minutes\b", "5min"),
    (r"\b(hourly|per hour|by hour)\b", "hourly"),
    (r"\b(daily|per day|by day)\b", "daily"),
    (r"\btx ?power\b|\btransmi\w+ power\b|\bchannel (and|or|&)\b|\bchannel\b.*\bpower\b|\bpower\b.*\bchannel\b|\bradio (settings?|config\w*)\b", "radio config"),
    (r"\badopt\w*|\bonboard\w*", "start adopt device"),
    (r"\b(wi-?fi|wireless network|wlan)\b", "ssid"),
    (r"\bfirmware\b|\bupgrade\b", "upgrade firmware"),
    (r"\b(connected|active|online) (clients?|stations?|users?)\b", "active clients"),
    (r"\btime ?zones?\b", "timeZone"),
    (r"\bforget\w*|\bremove (a |the )?device\b", "forget device"),
    (r"\bbatch\w*|\bbulk\b|\bseveral calls\b|\bmultiple calls\b", "batch"),
    (r"\b(bind|associate|assign|attach)\b.*\b(ssid|wlan)\b|\b(ssid|wlan)\b.*\b(bind|associate|assign)\b", "ap groups bind"),
    (r"\b(access points?|aps?|switch(es)?|gateways?|routers?)\b", "device"),
    (r"\bport[- ]?forward\w*", "port forwarding"),
    (r"\bspeed ?test\b", "speed test"),
]

_COMPILED = [(re.compile(pattern, re.IGNORECASE), terms) for pattern, terms in RULES]

# Rules whose trigger words are pure synonyms of the API term. For keyword (BM25) search the trigger
# is replaced, because a rare everyday word ("restart") would otherwise outweigh the API word
# ("reboot") and pull unrelated operations that merely mention it in their description.
_SYNONYM_TERMS = {"reboot", "locate"}


def expansion_terms(query: str) -> list[str]:
    """Extra words to append, without duplicates and without repeating what is already there."""
    present = set(re.findall(r"\w+", query.lower()))
    extra: list[str] = []
    for pattern, terms in _COMPILED:
        if pattern.search(query):
            for term in terms.split():
                if term.lower() not in present and term not in extra:
                    extra.append(term)
    return extra


def expand_query(query: str) -> str:
    """Return ``query`` plus the Omada vocabulary it suggests (unchanged when nothing applies)."""
    extra = expansion_terms(query)
    return f"{query} {' '.join(extra)}" if extra else query


def keyword_variant(query: str) -> str:
    """Query text for BM25: synonym triggers are swapped for the API word, other terms appended."""
    text = query
    for pattern, terms in _COMPILED:
        if terms in _SYNONYM_TERMS and pattern.search(text):
            text = pattern.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    extra = expansion_terms(query)
    return f"{text} {' '.join(extra)}".strip() if extra else text
