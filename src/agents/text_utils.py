"""Text helpers shared by every LLM call that returns code or markdown (the swarm agents'
pipeline.py + notebook answers, the observer's report, the checker's findings), and by the
context the agents are shown."""
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


# Paper ids are Scopus EIDs, e.g. "2-s2.0-105000159569" -- the same id scheme used as both the
# corpus JSON filenames and the "[paper_id]" citation style the prompts ask for.
_CITATION_RE = re.compile(r"2-s2\.0-\d+")
_NOTEBOOK_RE = re.compile(r"(?is)#+\s*NOTEBOOK\s*\n(.*?)(?=```|\Z)")


def extract_notebook(text: str, cap: int) -> str:
    """The `## NOTEBOOK` section of a pipeline-writing answer (everything up to the code
    fence), capped at `cap` characters; "" if the agent wrote none."""
    match = _NOTEBOOK_RE.search(text)
    if not match:
        return ""
    return cap_text(match.group(1).strip(), cap)


def extract_cited_papers(text: str) -> list[str]:
    """Paper ids actually cited in a piece of generated text (e.g. a pipeline.py's comments),
    in first-seen order, deduplicated."""
    seen: list[str] = []
    for match in _CITATION_RE.findall(text):
        if match not in seen:
            seen.append(match)
    return seen


def cap_text(text: str, cap: int, marker: str = "\n...[truncated]") -> str:
    return text if len(text) <= cap else text[:cap] + marker


def changed_hparams(old: dict, new: dict) -> list[str]:
    """Keys of HPARAMS that were added, removed, or changed value."""
    return sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
