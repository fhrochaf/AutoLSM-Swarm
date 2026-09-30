"""Loader for src/prompts.yaml, the single home of every LLM prompt in the project.

`render("velocity.system")` / `render("reflect.user", agent_idx=3, ...)` return the prompt
text with its <<placeholders>> filled in. The module also computes the few dataset-derived
strings the prompts embed, so switching `settings.dataset_name` re-points every prompt.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from agents.pipeline_contract import build_required_funcs_doc
from config import settings
from data.registry import get_dataset_spec

PROMPTS_PATH = Path(__file__).resolve().parent.parent / "prompts.yaml"
_PLACEHOLDER = re.compile(r"<<(\w+)>>")
_PROMPTS: dict = yaml.safe_load(PROMPTS_PATH.read_text(encoding="utf-8"))


def _lookup(key: str) -> str:
    node = _PROMPTS
    for part in key.split("."):
        node = node[part]
    if not isinstance(node, str):
        raise KeyError(f"prompt key {key!r} is a group, not a prompt")
    return node


def prompt_keys() -> list[str]:
    """Every renderable prompt key (dotted), e.g. for tests."""
    keys: list[str] = []

    def walk(node: dict, prefix: str) -> None:
        for name, value in node.items():
            path = f"{prefix}{name}"
            if isinstance(value, dict):
                walk(value, f"{path}.")
            else:
                keys.append(path)

    walk(_PROMPTS, "")
    return keys


def placeholders(key: str) -> set[str]:
    return set(_PLACEHOLDER.findall(_lookup(key)))


def render(key: str, **variables: object) -> str:
    """Fill <<name>> placeholders. A placeholder without a value, or a value the prompt
    never uses, is an error -- a typo in either place would otherwise silently ship a
    broken prompt."""
    text = _lookup(key)
    wanted = set(_PLACEHOLDER.findall(text))
    missing, unused = wanted - variables.keys(), variables.keys() - wanted
    if missing or unused:
        raise KeyError(f"prompt {key!r}: missing variables {sorted(missing)}, unused variables {sorted(unused)}")
    return _PLACEHOLDER.sub(lambda m: str(variables[m.group(1)]), text).strip()


# The single source of truth for "what dataset is this run about" -- swap
# settings.dataset_name to point every prompt below at a different DatasetSpec
# (data/registry.py) without touching the YAML.
_DATASET_SPEC = get_dataset_spec(settings.dataset_name)

DATASET_DESCRIPTION = _DATASET_SPEC.description
REQUIRED_FUNCS_DOC = build_required_funcs_doc(_DATASET_SPEC)

# The corpus-retrieval query for round 0: derived from the dataset itself rather than an
# assigned niche, so every agent's search is grounded in "how do I map this kind of
# target given this kind of dataset" rather than a predetermined method family.
RETRIEVAL_QUERY = render("retrieval_query", dataset_description=DATASET_DESCRIPTION)

PIPELINE_SYSTEM_PROMPT = render(
    "pipeline.system", required_funcs_doc=REQUIRED_FUNCS_DOC, default_seed=settings.seed
)
