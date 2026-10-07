"""Prepare a pasted/plain-text guide so it indexes well.

Guides copied from the Omada documentation website arrive as plain text:
numbered headings ("3.2.2 Obtain Access Token"), tab-separated tables and
un-fenced ``curl``/JSON examples. :func:`normalize_guide_text` converts that into
Markdown so the chunker can

* keep the heading trail with every chunk ("... > 3.2 Authorization Code Mode > 3.2.2 Obtain Access Token"),
* keep each request/response example in one piece, next to its explanation.
"""

from __future__ import annotations

import re

# "1 Overview", "3.2.2 Obtain Access Token", "5.9.1.1 Interfaces"
# - single literal space after the number (table rows use tabs),
# - at most 2 digits per component (so "400 Bad Request" or "01001001 x" are not headings).
_HEADING_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,5}) ([^\t]{2,90})$")
_MAX_HEADING_WORDS = 10


def _is_heading(text: str) -> re.Match[str] | None:
    match = _HEADING_RE.match(text)
    if not match:
        return None
    title = match.group(2)
    if len(title.split()) > _MAX_HEADING_WORDS or title.endswith((".", ":", ",", ";")):
        return None
    return match


def normalize_guide_text(text: str) -> str:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[str] = []
    seen_content = False
    in_fence = False
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        if stripped.startswith(("```", "~~~")):  # already Markdown code: pass through
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue
        if in_fence:
            out.append(line)
            i += 1
            continue

        if not seen_content and stripped:
            seen_content = True
            if not stripped.startswith("#") and not _is_heading(stripped):
                out.append(f"# {stripped}")  # the first line is the document title
                i += 1
                continue

        # curl example: runs until the next blank line
        if stripped.startswith("curl "):
            block = []
            while i < len(lines) and lines[i].strip():
                block.append(lines[i].rstrip())
                i += 1
            out += ["```bash", *block, "```"]
            continue

        # JSON example: from a "{" at column 0 to the "}" at column 0
        if line == "{":
            block = [line]
            i += 1
            while i < len(lines):
                block.append(lines[i].rstrip())
                closed = lines[i].rstrip() == "}"
                i += 1
                if closed:
                    break
            out += ["```json", *block, "```"]
            continue

        heading = _is_heading(stripped)
        if heading:
            level = min(heading.group(1).count(".") + 2, 6)
            out += ["", f"{'#' * level} {stripped}", ""]
        elif "\t" in line:  # table row
            out.append(re.sub(r"\s*\t\s*", " | ", stripped))
        else:
            out.append(line)
        i += 1

    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip() + "\n"
