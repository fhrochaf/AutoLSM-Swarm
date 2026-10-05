"""Loader for the prompts YAML file -- the single home of every LLM prompt in a run.

Which file is read is `settings.prompts_file` (config.py, or the PROMPTS_FILE environment
variable / .env): a name resolved under src/ (default `prompts.yaml`), or a path. Keep one
YAML per prompt variant you want to compare; each run stores a copy of the one it used.

`render("velocity.system")` / `render("reflect.user", agent_idx=3, ...)` return the prompt
text with its <<placeholders>> filled in. The module also computes the few dataset-derived
strings the prompts embed, so switching `settings.dataset_name` re-points every prompt.

The file is checked when this module is imported, i.e. before any LLM call or training run
is paid for. The only thing that stops a run is a prompt the enabled features need but the
file does not have at all (there is no text to send). Everything about placeholders is
forgiving: a prompt may leave placeholders out (e.g. drop the trajectory from the velocity
prompt) -- they are just not shown; a placeholder the code does not supply (a typo, or one
from another version of the prompts) is reported as a warning and left as written in the
text, and render() never raises because of one.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import yaml

from agents.pipeline_contract import build_required_funcs_doc
from config import settings
from data.registry import get_dataset_spec

SRC_DIR = Path(__file__).resolve().parent.parent
_PLACEHOLDER = re.compile(r"<<(\w+)>>")

# Every prompt the code renders, with the variables it supplies for that prompt.
PROMPT_VARIABLES: dict[str, set[str]] = {
    "retrieval_query": {"dataset_description"},
    "pipeline.system": {"required_funcs_doc", "default_seed"},
    "pipeline.initial_user": {"dataset_description", "retrieved_papers"},
    "pipeline.initial_user_no_rag": {"dataset_description"},
    "pipeline.apply_velocity_user": {"dataset_description", "current_code", "velocity", "velocity_reasoning"},
    "pipeline.fix_user": {"previous_code", "error_message"},
    "peer_review.system": set(),
    "peer_review.user": {"dataset_description", "tables"},
    "reflect.system": set(),
    "reflect.grounding_suffix": set(),
    "reflect.user": {"dataset_description", "agent_idx", "pipeline_code", "peer_review"},
    "reflect.enrich_system": set(),
    "reflect.enrich_user": {"reflection", "grounding"},
    "velocity.system": set(),
    "velocity.user": {
        "prev_velocity", "trajectory", "current_dice", "best_this_round", "reflection",
        "pipeline_code", "p_best_score", "p_best_block", "g_best_score", "g_best_block",
    },
    "fidelity.system": {"required_funcs_doc", "dataset_description"},
    "fidelity.user": {"velocity", "diff", "new_code"},
    "fidelity.revise_user": {
        "dataset_description", "velocity", "velocity_reasoning", "previous_code", "flawed_code", "feedback",
        "search_hint",
    },
}
_FIDELITY_KEYS = {k for k in PROMPT_VARIABLES if k.startswith("fidelity.")}
_ENRICH_KEYS = {"reflect.grounding_suffix", "reflect.enrich_system", "reflect.enrich_user"}
_RAG_KEYS = {"retrieval_query", "pipeline.initial_user"}
_NO_RAG_KEYS = {"pipeline.initial_user_no_rag"}


def resolve_prompts_path(name: str) -> Path:
    """`name` is absolute, or relative to src/, the repo root, or the working directory."""
    p = Path(name).expanduser()
    candidates = [p] if p.is_absolute() else [SRC_DIR / p, SRC_DIR.parent / p, Path.cwd() / p]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(
        f"prompts file {name!r} (settings.prompts_file / PROMPTS_FILE) not found; looked in:\n"
        + "\n".join(f"  {c}" for c in candidates)
    )


PROMPTS_PATH = resolve_prompts_path(settings.prompts_file)
PROMPTS_SHA = hashlib.sha256(PROMPTS_PATH.read_bytes()).hexdigest()[:12]
_PROMPTS: dict = yaml.safe_load(PROMPTS_PATH.read_text(encoding="utf-8")) or {}


def _lookup(key: str) -> str:
    node = _PROMPTS
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"prompt {key!r} not found in {PROMPTS_PATH.name}")
        node = node[part]
    if not isinstance(node, str):
        raise KeyError(f"prompt key {key!r} in {PROMPTS_PATH.name} is a group, not a prompt")
    return node


def prompt_keys() -> list[str]:
    """Every renderable prompt key (dotted) present in the loaded file, e.g. for tests."""
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


def required_keys() -> set[str]:
    """Prompts the current config can actually reach: optional features only when enabled."""
    keys = set(PROMPT_VARIABLES) - _FIDELITY_KEYS - _ENRICH_KEYS - _NO_RAG_KEYS
    if settings.fidelity_check:
        keys |= _FIDELITY_KEYS
    if not settings.use_rag:
        keys = (keys - _RAG_KEYS) | _NO_RAG_KEYS
    elif settings.enriched_reflection_all or settings.random_enriched_reflection > 0 or settings.enriched_leader:
        keys |= _ENRICH_KEYS
    return keys


def validate_prompts() -> None:
    """Check the file now, with its name, rather than mid-run after round 0 has been paid for.
    A missing prompt that an enabled feature needs is an error (there is nothing to send).
    A placeholder the code does not supply only produces a warning."""
    needed = required_keys()
    missing_keys = []
    for key in sorted(PROMPT_VARIABLES):
        try:
            found = placeholders(key)
        except KeyError:
            if key in needed:
                missing_keys.append(key)
            continue
        unknown = found - PROMPT_VARIABLES[key]
        if unknown:
            allowed = ", ".join(sorted(PROMPT_VARIABLES[key])) or "none"
            print(
                f"[prompts] WARNING {PROMPTS_PATH.name}: {key} uses placeholder(s) {sorted(unknown)} that the "
                f"code does not supply (it supplies: {allowed}); they are left as written in the prompt",
                file=sys.stderr,
            )
    if missing_keys:
        raise ValueError(
            f"{PROMPTS_PATH} lacks prompt(s) the enabled features need: {missing_keys}. Add them, or switch the "
            "feature off in config.py (fidelity_check; enriched_reflection_all / random_enriched_reflection / "
            "enriched_leader)."
        )


_warned: set[tuple[str, str]] = set()


def render(key: str, **variables: object) -> str:
    """Fill <<name>> placeholders. Never raises because of a placeholder: a value the prompt does
    not use is simply left out of the text, and a placeholder the caller gave no value for is left
    as written (with a one-time warning) instead of stopping the run."""
    text = _lookup(key)

    def fill(m: re.Match) -> str:
        name = m.group(1)
        if name in variables:
            return str(variables[name])
        if (key, name) not in _warned:
            _warned.add((key, name))
            print(f"[prompts] WARNING {key}: no value for <<{name}>>; left as written", file=sys.stderr)
        return m.group(0)

    return _PLACEHOLDER.sub(fill, text).strip()


validate_prompts()

# The single source of truth for "what dataset is this run about" -- swap
# settings.dataset_name to point every prompt below at a different DatasetSpec
# (data/registry.py) without touching the YAML.
_DATASET_SPEC = get_dataset_spec(settings.dataset_name)

DATASET_DESCRIPTION = _DATASET_SPEC.description
REQUIRED_FUNCS_DOC = build_required_funcs_doc(_DATASET_SPEC)

# The corpus-retrieval query for round 0: derived from the dataset itself rather than an
# assigned niche, so every agent's search is grounded in "how do I map this kind of
# target given this kind of dataset" rather than a predetermined method family.
RETRIEVAL_QUERY = render("retrieval_query", dataset_description=DATASET_DESCRIPTION) if settings.use_rag else ""

PIPELINE_SYSTEM_PROMPT = render(
    "pipeline.system", required_funcs_doc=REQUIRED_FUNCS_DOC, default_seed=settings.seed
)
