"""Text-cleanup helpers shared by every LLM call that returns code or markdown
(agents/codegen.py's skill/pipeline generation, orchestration/peer_review.py's
skill-update rewrite)."""
from __future__ import annotations

import re


def extract_code(text: str) -> str:
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # No closing fence -- e.g. the response got cut off mid-code. Still drop a
    # leading opening-fence line so we don't hand back a guaranteed line-1
    # SyntaxError; the (still likely incomplete) rest goes to the debug loop.
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
    return text.strip()


def strip_optional_fence(text: str) -> str:
    """Markdown-writing calls aren't asked to fence their output, but models
    sometimes wrap the whole document in ``` anyway -- strip that if both ends
    are present."""
    text = text.strip()
    if text.startswith("```") and text.rstrip().endswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            return text[first_newline + 1 : text.rstrip().rfind("```")].strip()
    return text
