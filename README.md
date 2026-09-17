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
and a `.predict(X)` returning binary masks — which is smoke-tested on a handful of
fixed tiles before a pipeline is trusted to run on the full split. If a pipeline
crashes or violates the contract, its traceback is fed back to the LLM in a bounded
generate/debug loop (`max_debug_iters`) before the round gives up on it.

Before a pipeline is even run, a separate **fidelity guardrail**
([`src/agents/fidelity.py`](src/agents/fidelity.py)) — an LLM judge — checks whether it
actually implements the strategy its `skill.md` describes (specific channels, named
architectures/loss terms, stated preprocessing/training procedures), not just whether
it runs. A flagged mismatch is fed back for a targeted rewrite
(`revise_pipeline_for_fidelity`, which can search the literature for the exact
technique named) rather than a generic regeneration; this shares the same
`max_debug_iters` budget as the runtime-error debug loop
([`src/agents/runner.py`](src/agents/runner.py)). This judge deliberately runs on a
separate, typically cheaper model from the one that writes skills/code (see LLM roles
in Setup below), since judging faithfulness is a narrower task than generating the
code itself.

## Dataset

By default, pipelines are trained and scored on the **Landslide4Sense** benchmark
(Ghorbanzadeh et al., 2022): 128×128 multispectral + DEM-derived tiles (14 channels)
with binary landslide masks, expected under `archive/` in the same layout as the
original dataset release (`TrainData/{img,mask}/*.h5`, etc.). A fixed, seeded 70/15/15
train/val/test split is carved out of the labeled tiles once per machine and reused
across every run.

The dataset layer is dataset-agnostic by design: everything specific to Landslide4Sense
(its description text, array shapes/channels, file layout, and loading code) lives
behind a single `DatasetSpec`
([`src/data/spec.py`](src/data/spec.py),
[`src/data/datasets/landslide4sense.py`](src/data/datasets/landslide4sense.py)), looked
up by `settings.dataset_name` via a registry
([`src/data/registry.py`](src/data/registry.py)). Every LLM-facing description of the
dataset (`DATASET_DESCRIPTION`, the pipeline contract's array-shape doc-string,
retrieval query, fidelity judge) and the actual loading code
([`src/data/dataset.py`](src/data/dataset.py)) are all derived from that one spec —
switching datasets never means editing prompts or loader code by hand.

### Adding a new dataset

1. Create `src/data/datasets/<name>.py` implementing:
   - `load_split(split, dataset_dir, max_tiles=None) -> (X, y, ids)` — loads one of
     `"train"/"val"/"test"` as stacked arrays. If your dataset is indexed by id (like
     Landslide4Sense's tile ids), reuse
     [`split_ids_by_ratio`](src/data/dataset.py) from `data.dataset` for the
     seeded train/val/test cut instead of reimplementing it.
   - `load_smoke_sample(dataset_dir, ids) -> (X, y)` — loads a handful of specific
     samples by id, used for the cheap pre-flight smoke test.
   - A `DESCRIPTION` string: free-text prose for the LLM covering tile/scene size,
     channel count and semantics, whether the data is multi-temporal, and label
     semantics — this is what an agent actually sees when deciding its strategy.
   - A module-level `SPEC = DatasetSpec(...)` (see
     [`src/data/spec.py`](src/data/spec.py)) wiring up `name`, `description`,
     `input_shape_doc`/`label_shape_doc` (short shape annotations spliced into the
     pipeline contract, e.g. `"(N,H,W,C)"` or `"(N,T,H,W,C)"` for multi-temporal data),
     `smoke_tile_ids`, and the two loader functions above.
2. Hand-verify `smoke_tile_ids`: a few ids from the train split that include at least
   one positive (label present) and one negative sample — this is dataset-specific and
   must be re-checked for every new dataset, it does not carry over from
   Landslide4Sense's `[1, 4, 8, 10]`.
3. Register it in [`src/data/registry.py`](src/data/registry.py) (one line: add the
   module's `SPEC` to `_REGISTRY`).
4. Point `dataset_name` (in [`src/config.py`](src/config.py) or `.env`) at your
   dataset's `name`, and `dataset_dir` at where its files live on disk.

Nothing outside `src/data/` needs to change — `preprocess`/`build_model`/`train`'s
contract shapes, the skill/pipeline prompts, and the fidelity judge all pick up the new
dataset automatically through the spec.

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
  round/plateau/epsilon hyperparameters, `dataset_name`/`dataset_dir` and corpus paths,
  retrieval settings.
- **`src/agents/`** — everything about one agent's own generate/run/debug cycle:
  - [`codegen.py`](src/agents/codegen.py) — LLM calls that produce a `skill.md`
    (from retrieved papers), compile a `skill.md` into `pipeline.py`, fix a pipeline
    that crashed, or rewrite one flagged by the fidelity judge (optionally searching
    the literature for the exact technique named).
  - [`fidelity.py`](src/agents/fidelity.py) — the LLM-judge guardrail that checks a
    generated `pipeline.py` actually implements the strategy its `skill.md` describes,
    before the pipeline is ever run.
  - [`driver.py`](src/agents/driver.py) — standalone subprocess entry point that
    actually executes one `pipeline.py` end-to-end (preprocess → train → evaluate) in
    isolation from the orchestrator, auto-installing any library the agent imports.
  - [`runner.py`](src/agents/runner.py) — the per-agent loop that alternates the
    fidelity check with the generate → run → debug cycle: a fidelity mismatch triggers
    a targeted rewrite, a runtime crash feeds its traceback back for a fix, and either
    kind of retry shares the same `max_debug_iters` budget before the round gives up.
  - [`pipeline_contract.py`](src/agents/pipeline_contract.py) — the fixed interface
    every generated pipeline must implement, plus the smoke test that checks it cheaply
    before a full run.
  - [`prompts.py`](src/agents/prompts.py) — system/user prompt templates for skill and
    pipeline generation.
  - [`skill.py`](src/agents/skill.py) — per-run/round/agent file I/O
    (`<runs_dir>/<run_id>/round_<t>/agent_<i>/{skill.md,pipeline.py}`).
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
- **`src/data/`** — dataset-agnostic loading, dispatched by `settings.dataset_name`
  (see Dataset above):
  - [`spec.py`](src/data/spec.py) — the `DatasetSpec` dataclass every dataset module
    builds: description text, contract shape doc-strings, smoke-test ids, and loader
    functions.
  - [`registry.py`](src/data/registry.py) — `get_dataset_spec(name)`, the name → spec
    lookup every other module goes through.
  - [`datasets/landslide4sense.py`](src/data/datasets/landslide4sense.py) — the
    Landslide4Sense `DatasetSpec`: its description, h5py-based tile loading, and
    hand-verified smoke-test ids. Template for adding another dataset.
  - [`dataset.py`](src/data/dataset.py) — the generic entry points (`load_split`,
    `load_smoke_sample`, `build_data_cache`) that dispatch to the active spec, plus
    `split_ids_by_ratio`, the seeded train/val/test split logic shared by any
    id-indexed dataset. Also builds the per-run `.npz` cache so the dataset is read
    from disk only once per run.
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

The swarm uses up to three independently-configured LLMs
([`src/config.py`](src/config.py)), each free to be a different provider/model:

| Role | Settings | Used for |
|---|---|---|
| Primary | `llm_provider` / `model_name` | writing initial skills and the peer-review chain (`Reflect → GroundReflection → VelocityUpdate → SkillUpdate`) |
| Judge | `llm_provider_2` / `model_name_2` | the fidelity guardrail (`agents/fidelity.py`) that checks `pipeline.py` against `skill.md` — deliberately cheap/small, since judging is a narrower task |
| Code | `llm_provider_3` / `model_name_3` | generating/fixing/revising `pipeline.py` (`agents/codegen.py`) |

Add the required API key(s) to a `.env` file at the repo root — one per distinct
provider used above (e.g. `ANTHROPIC_API_KEY`, `GOOGLE_GENAI_API_KEY`); if two roles
share a provider they share its key. `corpus_json_dir` and `publications_dir` in the
same file point at the literature corpus's JSON summaries and source PDFs on disk —
update them to wherever your copy of the corpus lives.

## Usage

```bash
python main.py [--agents N] [--rounds T] [--max-train-tiles N] [--max-val-tiles N]
                [--plateau-patience N] [--rebuild-index] [--resume RUN_ID]
```

| Flag | Description |
|---|---|
| `--agents N` | Number of agents in the swarm (default from config: 2). Ignored with `--resume`. |
| `--rounds T` | Number of rounds to train for (default from config: 2). Required alongside `--resume`, where it is the target round to train up to. |
| `--max-train-tiles N` | Cap on labeled tiles used for training (a cheap-proxy subset). Ignored with `--resume` — data is reused from the original run. |
| `--max-val-tiles N` | Cap on labeled tiles used for validation scoring. Ignored with `--resume`. |
| `--plateau-patience N` | Stop early after this many consecutive rounds with no global-best Dice improvement (default from config: 3). |
| `--rebuild-index` | Force a rebuild of the RAG literature index even if one already exists. |
| `--resume RUN_ID` | Keep training an existing run (`<runs_dir>/<RUN_ID>`) instead of starting a new one, reusing its cached dataset split and current swarm state. |

Each run writes to `<runs_dir>/<run_id>/` (`runs_dir` in
[`src/config.py`](src/config.py), default `runs_landslide4sense/`): per-round,
per-agent `skill.md`/`pipeline.py`/`driver_stdout.json`, an append-only `ledger.jsonl`
of every round's scores, a `state_snapshot.json` checkpoint (used by `--resume`), and a
final `summary.json`.

## License

GPL-3.0 — see [LICENSE.txt](LICENSE.txt).

## References

Hwang, H., Kim, J., Kim, C., Chang, H., & Ye, J. C. (2026). *AgentPSO: Evolving Agent
Reasoning Skill via Multi-agent Particle Swarm Optimization*. arXiv:2605.08704.
https://arxiv.org/abs/2605.08704v2
