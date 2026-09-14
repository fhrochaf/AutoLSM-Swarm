$${\color{red}Setup config file}$$

# AutoLSM-Swarm

A swarm of LLM agents that search the landslide-mapping literature and evolve their own
landslide detection/segmentation pipelines, adapting the particle-swarm skill-evolution
framework **AgentPSO** (Hwang et al., 2026) from general LLM reasoning to a concrete
Earth-observation task: mapping landslides from satellite imagery.

Hundreds of landslide-mapping methods have been published, no single one generalizes
across regions/sensors, and reproducing them one at a time to find the best combination
for a given dataset does not scale — only ~20% of papers on the topic even release code
(see the corpus survey below). Instead of a human replicating papers sequentially, a
population of agents searches the published methodology space in parallel, each
converging on its own pipeline while learning from its peers' results.

## Methodology

Each agent is a "particle" whose **state is not a numeric vector but a natural-language
`skill.md`** describing an accumulated landslide-mapping strategy (grounded in specific
cited papers), plus the `pipeline.py` that strategy currently compiles to. Every round
follows a four-step loop
([`src/orchestration/graph.py`](src/orchestration/graph.py),
[`src/orchestration/peer_review.py`](src/orchestration/peer_review.py)):

1. **Independent solving.** Every agent runs its current `pipeline.py`
   (preprocess → build/train a model → predict) against the fixed train/val split and
   is scored by Dice/IoU against the ground-truth masks.
2. **Peer observation.** Each agent is shown its peers' *code and scores* for the round
   — never their `skill.md` text — so it must infer *why* something worked rather than
   copy another agent's wording.
3. **Self-reflective update.** Four chained LLM calls replace the numeric velocity
   update: `Reflect` → `GroundReflection` → `VelocityUpdate` → `SkillUpdate`
   (rewrite `skill.md` accordingly, then regenerate `pipeline.py` from it). This is also
   grounded in the literature corpus every round, not just at initialization — but
   grounding happens in two passes rather than folding retrieval
   into the same call that forms the hypothesis. `Reflect` first runs without any
   corpus access, purely on behavior (its own skill vs. peer code/scores), and closes
   with one explicit `GROUNDING QUERY: ...` line naming the mechanism it's least sure
   about. That line is parsed out of the response and used, unmodified, as the
   vector-search query, so retrieval is anchored on the agent's actual open question
   rather than a raw skill/peer-code dump. `GroundReflection` then reads that draft
   reflection back together with the retrieved papers and revises it — confirming,
   sharpening, or contradicting its own hypotheses and citing a specific published
   technique (`[paper_id]`) where one actually bears on it. Only this finished,
   grounded reflection reaches `VelocityUpdate`, which merges it with the previous
   velocity, the agent's own personal-best skill, and the swarm's global-best skill
   into one set of revision directives.
4. **Validation-based best tracking.** A new skill only replaces an agent's
   personal-best (or the swarm's global-best) if its Dice score improves on the
   previous best by more than a margin `p_best_epsilon`, damping noisy fluctuations.

This project uses a **global-best (fully-connected) topology**: every agent is steered
toward one swarm-wide best skill each round. Training stops after `n_rounds`, or earlier
if the global-best Dice has not improved for `plateau_patience` consecutive rounds.

Generated pipelines must satisfy a fixed contract
([`src/agents/pipeline_contract.py`](src/agents/pipeline_contract.py)) —
`preprocess(X) -> X`, `build_model(config) -> model`, `train(model, X, y) -> model`,
and a `.predict(X)` returning binary masks — which is smoke-tested on two tiles before
a pipeline is trusted to run on the full split. If a pipeline crashes or violates the
contract, its traceback is fed back to the LLM in a bounded generate/debug loop
(`max_debug_iters`) before the round gives up on it.

## Dataset

Pipelines are trained and scored on the **Landslide4Sense** benchmark
(Ghorbanzadeh et al., 2022): 128×128 multispectral + DEM-derived tiles (14 channels)
with binary landslide masks, expected under `archive/` in the same layout as the
original dataset release (`TrainData/{img,mask}/*.h5`, etc. —
[`src/data/dataset.py`](src/data/dataset.py)). A fixed, seeded 70/15/15 train/val/test
split is carved out of the labeled tiles once per machine and reused across every run.

## Literature corpus

Agents ground every skill/pipeline revision in retrieved papers rather than generic
knowledge — both when a skill is first initialized (round 0) and again every later
round, where retrieval also feeds the `Reflect` step of peer review (see Methodology
above). The corpus is a set of ~300 landslide-mapping papers, each pre-summarized
into a structured JSON (methods, datasets, reported novelty, reproducibility status)
and indexed into a local Chroma vector store for retrieval
([`src/corpus/ingest.py`](src/corpus/ingest.py),
[`src/corpus/retrieve.py`](src/corpus/retrieve.py)). This corpus — including the
reproducibility survey mentioned above (only ~20% of 2020–2025 automated
landslide-mapping papers publish code) — is browsable at
**[fhrochaf.github.io/LSM_ReproViewer](https://fhrochaf.github.io/LSM_ReproViewer/)**.
Retrieved summaries are hydrated with an excerpt of each paper's actual full text on
demand (extracted once via `pypdf`, then disk-cached —
[`src/corpus/pdf_extract.py`](src/corpus/pdf_extract.py)) before being stuffed into an
agent's prompt.

## Module overview

- [`main.py`](main.py) — CLI entry point; wires config, the RAG index, the cached
  dataset, and the round-loop graph together, then runs (or resumes) a swarm.
- [`src/config.py`](src/config.py) — the single `Settings` object (pydantic-settings,
  loaded from `.env`) every other module imports: LLM provider/model, swarm size,
  round/plateau/epsilon hyperparameters, dataset/corpus paths, retrieval settings.
- **`src/agents/`** — everything about one agent's own generate/run/debug cycle:
  - [`codegen.py`](src/agents/codegen.py) — LLM calls that produce a `skill.md`
    (from retrieved papers) or compile a `skill.md` into `pipeline.py`.
  - [`driver.py`](src/agents/driver.py) — standalone subprocess entry point that
    actually executes one `pipeline.py` end-to-end (preprocess → train → evaluate) in
    isolation from the orchestrator, auto-installing any library the agent imports.
  - [`runner.py`](src/agents/runner.py) — the generate → run → debug loop: keeps
    regenerating `pipeline.py` from the driver's traceback until it passes, or gives
    up after `max_debug_iters`.
  - [`pipeline_contract.py`](src/agents/pipeline_contract.py) — the fixed interface
    every generated pipeline must implement, plus the smoke test that checks it cheaply
    before a full run.
  - [`prompts.py`](src/agents/prompts.py) — system/user prompt templates for skill and
    pipeline generation.
  - [`skill.py`](src/agents/skill.py) — per-run/round/agent file I/O
    (`runs/<run_id>/round_<t>/agent_<i>/{skill.md,pipeline.py}`).
  - [`text_utils.py`](src/agents/text_utils.py) — small helpers for cleaning LLM output
    (stripping code fences, extracting code blocks).
- **`src/corpus/`** — the literature RAG layer:
  - [`ingest.py`](src/corpus/ingest.py) — builds the persisted Chroma index from the
    per-paper JSON summaries.
  - [`retrieve.py`](src/corpus/retrieve.py) — similarity search over that index, plus
    `retrieve_diverse`, which shards one shared candidate pool round-robin across agents
    so round-0 agents don't all ground themselves in the same top-k papers.
  - [`pdf_extract.py`](src/corpus/pdf_extract.py) — on-demand full-text PDF extraction
    and disk caching, keyed by paper id.
- **`src/data/`**:
  - [`dataset.py`](src/data/dataset.py) — the fixed, seeded train/val/test split over
    Landslide4Sense tiles, and the per-run `.npz` cache so the dataset is read from disk
    only once per run.
- **`src/eval/`** — the fixed scoring contract agents cannot override:
  - [`metrics.py`](src/eval/metrics.py) — pure Dice/IoU implementations.
  - [`infer.py`](src/eval/infer.py) — runs an agent's `preprocess`/`predict` and scores
    the result against ground truth.
- **`src/orchestration/`** — the swarm loop itself:
  - [`state.py`](src/orchestration/state.py) — `AgentState`/`SwarmState`: particle-style
    bookkeeping (skill, velocity, personal-best, global-best).
  - [`peer_review.py`](src/orchestration/peer_review.py) — the
    `Reflect → GroundReflection → VelocityUpdate → SkillUpdate` chain described above,
    including the per-round corpus retrieval (queried on `Reflect`'s own
    `GROUNDING QUERY` line) that `GroundReflection` uses to revise the reflection.
  - [`graph.py`](src/orchestration/graph.py) — the round loop as a LangGraph
    `StateGraph`: retrieval + initial skills at round 0, peer review + pipeline
    regeneration in later rounds, personal-/global-best tracking, plateau-based early
    stopping, and per-agent checkpointing (`state_snapshot.json`, updated as each agent
    finishes so an interrupted round can resume without rerunning agents already done)
    for `--resume`.

## Setup

```bash
uv sync
```

Add the required API key(s) to a `.env` file at the repo root (see `llm_provider` /
`model_name` in [`src/config.py`](src/config.py) for which key name is expected, e.g.
`ANTHROPIC_API_KEY` or `GOOGLE_GENAI_API_KEY`). `corpus_json_dir` and `publications_dir`
in the same file point at the literature corpus's JSON summaries and source PDFs on
disk — update them to wherever your copy of the corpus lives.

## Usage

```bash
python main.py [--agents N] [--rounds T] [--max-train-tiles N] [--max-val-tiles N]
                [--plateau-patience N] [--rebuild-index] [--resume RUN_ID]
```

| Flag | Description |
|---|---|
| `--agents N` | Number of agents in the swarm (default from config: 3). Ignored with `--resume`. |
| `--rounds T` | Number of rounds to train for (default from config: 5). Required alongside `--resume`, where it is the target round to train up to. |
| `--max-train-tiles N` | Cap on labeled tiles used for training (a cheap-proxy subset). Ignored with `--resume` — data is reused from the original run. |
| `--max-val-tiles N` | Cap on labeled tiles used for validation scoring. Ignored with `--resume`. |
| `--plateau-patience N` | Stop early after this many consecutive rounds with no global-best Dice improvement (default from config: 3). |
| `--rebuild-index` | Force a rebuild of the RAG literature index even if one already exists. |
| `--resume RUN_ID` | Keep training an existing run (`runs/<RUN_ID>`) instead of starting a new one, reusing its cached dataset split and current swarm state. |

Each run writes to `runs/<run_id>/`: per-round, per-agent `skill.md`/`pipeline.py`/
`driver_stdout.json`, an append-only `ledger.jsonl` of every round's scores, a
`state_snapshot.json` checkpoint (used by `--resume`), and a final `summary.json`.

## License

GPL-3.0 — see [LICENSE.txt](LICENSE.txt).

## References

Hwang, H., Kim, J., Kim, C., Chang, H., & Ye, J. C. (2026). *AgentPSO: Evolving Agent
Reasoning Skill via Multi-agent Particle Swarm Optimization*. arXiv:2605.08704.
https://arxiv.org/abs/2605.08704v2
