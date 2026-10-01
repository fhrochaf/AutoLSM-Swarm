"""JSON-shaped LLM answers, independent of how a prompt is worded.

Two calls in the loop produce an answer the code must read a specific part of: the velocity
update (the directive to apply, apart from the reasoning behind it) and the fidelity judge
(a verdict, apart from its findings). Instead of asking each prompt to format its answer a
certain way and parsing text, the CODE fixes the shape: a pydantic schema is sent with the
request, so any prompt in any prompts YAML -- including older ones that say nothing about
JSON -- gets answers it can use. The prompts only have to explain what belongs in each field.

invoke_structured() tries, in order:
  1. the provider's native structured output (llm.with_structured_output(schema));
  2. a plain call with the schema's fields spelled out, parsing the JSON in the reply;
and returns (parsed_or_None, raw_text) so the caller can decide what to do if both fail.
"""
from __future__ import annotations

import json
import re
import sys
from typing import Literal, TypeVar

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field, ValidationError

from config import settings

T = TypeVar("T", bound=BaseModel)


class VelocityAnswer(BaseModel):
    "The velocity update's answer: the directive that is applied, and the reasoning behind it."

    reasoning: str = Field(
        default="",
        description=(
            "Your step-by-step reasoning that leads to the decision. It is only recorded for the "
            "researcher: the agent that edits pipeline.py never sees it."
        ),
    )
    final_velocity: str = Field(
        description=(
            "The directive only, complete and self-contained: the change(s) to make, written as the "
            "instructions require (for each parametric change the exact `CHANGE <key>: <current value> -> "
            "<new value>` line, otherwise a precise structural description). This field alone is passed "
            "on to the agent that edits pipeline.py; do not repeat the reasoning in it."
        )
    )


class FidelityVerdict(BaseModel):
    "The fidelity judge's answer."

    violations: list[str] = Field(
        default_factory=list,
        description="The concrete violations found, one per item. Empty if the update is faithful.",
    )
    verdict: Literal["FAITHFUL", "NOT_FAITHFUL"] = Field(
        description="FAITHFUL if the updated pipeline.py applies the directive faithfully, else NOT_FAITHFUL."
    )


def _coerce(out: object, schema: type[T]) -> T | None:
    if isinstance(out, schema):
        return out
    if isinstance(out, dict):
        try:
            return schema.model_validate(out)
        except ValidationError:
            return None
    return None


def json_instructions(schema: type[BaseModel]) -> str:
    "Plain-text description of the schema, for models/providers without structured output."
    props = schema.model_json_schema().get("properties", {})
    fields = "\n".join(f'- "{name}": {spec.get("description", "")}' for name, spec in props.items())
    return (
        "Respond with ONLY a single JSON object -- no prose before or after it, no markdown fence -- "
        f"with these fields:\n{fields}"
    )


def extract_json(text: str) -> dict | None:
    "The first JSON object in `text`: a whole-text object, a ```json fence, or the first balanced {...}."
    text = text.strip()
    candidates = [text]
    fence = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if fence:
        candidates.append(fence.group(1).strip())
    start = text.find("{")
    if start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    break
    for cand in candidates:
        try:
            data = json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict):
            return data
    return None


def invoke_structured(
    llm: BaseChatModel, schema: type[T], messages: list[tuple[str, str]]
) -> tuple[T | None, str]:
    """Ask `llm` for `schema`-shaped output. Returns (parsed, raw_text); parsed is None only if
    neither the native structured call nor the JSON-in-text fallback produced a valid object
    (raw_text is then the plain reply, for the caller's last-resort handling)."""
    attempts = settings.llm_retry_attempts
    try:
        out = llm.with_structured_output(schema).with_retry(stop_after_attempt=attempts).invoke(messages)
        parsed = _coerce(out, schema)
        if parsed is not None:
            return parsed, parsed.model_dump_json()
    except Exception as exc:  # noqa: BLE001 - unsupported provider / unparsable reply: use the fallback
        print(f"[structured] native structured output failed for {schema.__name__} ({type(exc).__name__}); "
              "asking for JSON in plain text", file=sys.stderr)

    role, content = messages[-1]
    plain = [*messages[:-1], (role, f"{content}\n\n{json_instructions(schema)}")]
    text = llm.with_retry(stop_after_attempt=attempts).invoke(plain).text.strip()
    data = extract_json(text)
    parsed = _coerce(data, schema) if data is not None else None
    return parsed, text
