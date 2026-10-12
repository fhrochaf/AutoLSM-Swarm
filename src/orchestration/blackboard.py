"""The blackboard: the shared list of messages the agents post to during a round's
conversation (orchestration/graph.py). Every agent reads all of it; it holds agents' posts,
the check requests they hand to the HYPOTHESIS CHECKER, and the checker's results.

The board lives in SwarmState.blackboard (one round at a time) and is saved after every post
to round_<t>/blackboard.{jsonl,md}.
"""
from __future__ import annotations

from pathlib import Path

from orchestration.state import Post


def format_post(post: Post) -> str:
    if post.kind == "check_request":
        header = f"{post.author} -> HYPOTHESIS CHECKER (check request)"
    elif post.kind == "check_result":
        header = "HYPOTHESIS CHECKER"
    else:
        header = post.author
    return f"[cycle {post.cycle + 1}] {header}:\n{post.text.strip()}"


def format_board(posts: list[Post]) -> str:
    if not posts:
        return "(no posts yet)"
    return "\n\n---\n\n".join(format_post(p) for p in posts)


def save_board(round_dir: Path, posts: list[Post]) -> None:
    (round_dir / "blackboard.jsonl").write_text(
        "".join(p.model_dump_json() + "\n" for p in posts), encoding="utf-8"
    )
    (round_dir / "blackboard.md").write_text(
        f"# Blackboard\n\n{format_board(posts)}\n", encoding="utf-8"
    )

