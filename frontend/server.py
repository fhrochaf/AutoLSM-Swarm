"""Run viewer: a small local web UI for browsing AutoLSM-Swarm runs.

    uv run python frontend/server.py [RUNS_DIR ...] [--port 8765]

With no RUNS_DIR it looks in settings.runs_dir (config.py) and every runs*/ folder at the repo
root. Read-only: it only lists and serves files that are inside a run directory. Standard library
only; the page (frontend/index.html) loads marked + highlight.js from a CDN.
"""
from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

FRONTEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = FRONTEND_DIR.parent
MAX_FILE_BYTES = 3_000_000
TEXT_SUFFIXES = {".md", ".py", ".json", ".jsonl", ".yaml", ".yml", ".txt", ".csv", ".log"}
RUN_MARKERS = ("run_config.json", "ledger.jsonl", "state_snapshot.json", "summary.json")


def default_runs_dirs() -> list[Path]:
    dirs: list[Path] = []
    try:
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from config import Settings  # noqa: PLC0415 - optional: only for the configured runs_dir

        dirs.append(Path(Settings().runs_dir))
    except Exception:  # noqa: BLE001 - the viewer must start even if config cannot load
        pass
    dirs += sorted(p for p in REPO_ROOT.glob("runs*") if p.is_dir())
    seen, unique = set(), []
    for d in dirs:
        key = str(d.resolve())
        if key not in seen and d.exists():
            seen.add(key)
            unique.append(d.resolve())
    return unique


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def _is_run(d: Path) -> bool:
    return d.is_dir() and any((d / m).exists() for m in RUN_MARKERS)


def list_runs(runs_dirs: list[Path]) -> list[dict]:
    runs = []
    for i, base in enumerate(runs_dirs):
        for d in base.iterdir():
            if not _is_run(d):
                continue
            summary = _read_json(d / "summary.json") or {}
            config = _read_json(d / "run_config.json") or {}
            ledger = _read_jsonl(d / "ledger.jsonl")
            dices = [e["dice"] for e in ledger if isinstance(e.get("dice"), (int, float))]
            rounds = sorted(int(p.name.split("_")[1]) for p in d.glob("round_*") if p.name.split("_")[1].isdigit())
            runs.append({
                "dir": i,
                "runs_dir": str(base),
                "run_id": d.name,
                "finished": (d / "summary.json").exists(),
                "n_agents": config.get("n_agents") or len({e.get("agent_idx") for e in ledger}) or None,
                "n_rounds": config.get("n_rounds"),
                "rounds_seen": (rounds[-1] + 1) if rounds else 0,
                "best_dice": summary.get("g_best_score") if summary else (max(dices) if dices else None),
                "kind": "conversational" if (d / "run_config.json").exists() else "legacy",
                "mtime": d.stat().st_mtime,
            })
    return sorted(runs, key=lambda r: r["mtime"], reverse=True)


def _files(d: Path, run_dir: Path) -> list[str]:
    return sorted(
        str(p.relative_to(run_dir)).replace("\\", "/")
        for p in d.iterdir()
        if p.is_file() and p.suffix in TEXT_SUFFIXES
    )


def run_details(run_dir: Path) -> dict:
    rounds = []
    for rd in sorted(run_dir.glob("round_*"), key=lambda p: int(p.name.split("_")[1]) if p.name.split("_")[1].isdigit() else 0):
        if not rd.is_dir():
            continue
        agents = {
            ad.name.split("_")[1]: _files(ad, run_dir)
            for ad in sorted(rd.glob("agent_*"), key=lambda p: int(p.name.split("_")[1]))
            if ad.is_dir()
        }
        checker = _files(rd / "checker", run_dir) if (rd / "checker").is_dir() else []
        rounds.append({
            "round": int(rd.name.split("_")[1]),
            "files": _files(rd, run_dir),
            "agents": agents,
            "checker": checker,
            "blackboard": _read_jsonl(rd / "blackboard.jsonl"),
        })
    return {
        "run_id": run_dir.name,
        "run_config": _read_json(run_dir / "run_config.json"),
        "summary": _read_json(run_dir / "summary.json"),
        "ledger": _read_jsonl(run_dir / "ledger.jsonl"),
        "conversation": _read_jsonl(run_dir / "conversation_ledger.jsonl"),
        "top_files": _files(run_dir, run_dir),
        "rounds": rounds,
    }


class Handler(BaseHTTPRequestHandler):
    runs_dirs: list[Path] = []

    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status: int = 200) -> None:
        self._send(json.dumps(obj, default=str).encode("utf-8"), "application/json; charset=utf-8", status)

    def _run_dir(self, q: dict) -> Path | None:
        try:
            base = self.runs_dirs[int(q["dir"][0])]
        except (KeyError, IndexError, ValueError):
            return None
        run_id = q.get("id", [""])[0]
        run_dir = (base / run_id).resolve()
        if run_dir.parent != base or not _is_run(run_dir):
            return None
        return run_dir

    def do_GET(self):  # noqa: N802
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path in ("/", "/index.html"):
            return self._send((FRONTEND_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
        if url.path == "/api/runs":
            return self._json({"runs_dirs": [str(d) for d in self.runs_dirs], "runs": list_runs(self.runs_dirs)})
        run_dir = self._run_dir(q)
        if run_dir is None:
            return self._json({"error": "unknown run"}, 404)
        if url.path == "/api/run":
            return self._json(run_details(run_dir))
        if url.path == "/api/file":
            path = (run_dir / q.get("path", [""])[0]).resolve()
            if run_dir not in path.parents or not path.is_file() or path.suffix not in TEXT_SUFFIXES:
                return self._json({"error": "file not found"}, 404)
            data = path.read_bytes()[:MAX_FILE_BYTES]
            return self._send(data, "text/plain; charset=utf-8")
        return self._json({"error": "not found"}, 404)


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoLSM-Swarm run viewer")
    parser.add_argument("runs_dirs", nargs="*", type=Path, help="folders holding run directories")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    Handler.runs_dirs = [d.resolve() for d in args.runs_dirs] or default_runs_dirs()
    if not Handler.runs_dirs:
        raise SystemExit("No runs folder found; pass one, e.g. python frontend/server.py runs_landslide4sense")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Run viewer on {url}  (runs folders: {', '.join(map(str, Handler.runs_dirs))})")
    if not args.no_browser:
        webbrowser.open(url)
    server.serve_forever()


if __name__ == "__main__":
    main()
