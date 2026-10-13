# AutoLSM-Swarm

A swarm of LLM agents that search the landslide-mapping literature, write their own
landslide detection/segmentation pipelines, and **talk to each other** about how to improve
them. Each agent is a LangChain agent (`create_agent`) with tools for the literature corpus
(RAG) and the web. After every round the agents compare results and discuss strategies on a
shared blackboard. They can ask a hypothesis checker to test a claim against everyone's
predictions. Then each one writes its next pipeline.

Hundreds of landslide-mapping methods have been published, no single one generalizes
across regions/sensors, and reproducing them one at a time to find the best combination
for a given dataset does not scale — only ~20% of papers on the topic even release code
(see the corpus survey below). Instead of a human replicating papers sequentially, a
population of agents searches the published methodology space in parallel, each
developing its own pipeline while learning from its peers through conversation.

> Earlier versions of this project adapted AgentPSO's particle-swarm loop (peer review →
> reflect → velocity → apply, with a fidelity judge). That design is preserved in git history;
> this branch (`dev_conversational_agents`)
> replaces it with the conversational design below.

## Methodology

Every round runs through one LangGraph `StateGraph`
([`src/orchestration/graph.py`](src/orchestration/graph.py)). All prompts live in
[`src/prompts.yaml`](src/prompts.yaml).

```
START → seed → build_0 → … → build_{N-1} → update_bests → observer ──(last round / plateau)──→ finalize → END
                 ▲                                            │
                 │                                            ▼
                 └──(cycles done / all PASS)── next_speaker ⇄ talk_i ──(check request)──→ hypothesis_checker ─┐
                                                    ▲                                                          │
                                                    └──────────────────────────────────────────────────────────┘
```

1. **Seeding** (`seed`, round 0 only). Each agent gets its own papers from the literature
   corpus. `retrieval_strategy` (`shard` / `mmr` / `cluster`) keeps agents from all starting
   from the same top-k papers. With `use_rag=false` there is no corpus at all: agents start
   from the dataset description alone.
2. **Build** (`build_i`, one node per agent, run in sequence). Agent *i* writes its
   `pipeline.py`. In round 0 it works from its seed papers. In later rounds it works from the
   observer's report, the global-best pipeline, the finished blackboard conversation, and its
   own current pipeline, personal-best pipeline and notebook. It may call `search_corpus` and
   `web_search` while writing. The pipeline is run right away: one `driver.py` subprocess per
   evaluation seed, and a crash goes back to the same agent to fix, up to `max_debug_iters`
   times ([`src/agents/runner.py`](src/agents/runner.py)). Its score is the **mean Dice over
   the evaluation seeds**. Along with the code the agent writes a short private **notebook**.
   The notebook and its pipelines are all it carries into the next round; its context is
   otherwise rebuilt fresh every time.
3. **Best tracking** (`update_bests`). A pipeline replaces an agent's personal-best, or the
   swarm's global-best, only if its Dice improves on it by more than `p_best_epsilon`.
4. **Observer** (`observer`). Deterministic code
   ([`src/eval/observe.py`](src/eval/observe.py), plus
   [`observe_detail.py`](src/eval/observe_detail.py) when `observation_detail="detailed"`)
   turns every agent's predictions on the same validation tiles into tables of how each
   pipeline *behaves*:
   - precision/recall and over- vs under-segmentation
   - tile- and object-level detection, and false-alarm tiles
   - results by landslide size
   - complementarity between agents
   - Dice across seeds
   - fit: each pipeline's Dice on its own training tiles next to its validation Dice (train ≫ val
     suggests overfitting, both low underfitting). The driver scores every pipeline on both
     splits.

   The OBSERVER agent (role-2 LLM) writes a comparative report on those tables. It describes
   and does not prescribe. Report and tables go into every agent's **shared** context.
5. **Conversation** (`next_speaker` ⇄ `talk_i`). The agents take turns round-robin on the
   **blackboard**, a shared list of messages. The order rotates every round, and there are
   `conversation_cycles` cycles.
   - **What an agent reads.** Every agent reads the same shared context: last round's scores,
     the observer's report and tables, the global-best pipeline and the board so far. It also
     reads its own private context: its current pipeline, its personal-best pipeline and its
     notebook.
   - **What an agent does on its turn.** It posts one message or answers `PASS`. It may first
     use `search_corpus` / `web_search`; tool results stay private unless it chooses to share
     them.
   - **Early stop.** A cycle in which every agent passes ends the conversation early.
   - **Goal.** The prompts aim the discussion at the swarm's global Dice while asking each
     agent to keep a direction of its own.
6. **Hypothesis checks** (`talk_i → hypothesis_checker`).
   - **Asking.** During its turn an agent may call `request_hypothesis_check(question)`. After
     the turn, the graph hands control to the HYPOTHESIS CHECKER
     ([`src/agents/checker.py`](src/agents/checker.py), role-3 LLM).
   - **Running.** The checker writes Python analysis scripts and runs them (`run_analysis`)
     in a subprocess over the validation split only:
     - every agent's predictions this round and the ground-truth masks
     - the observer's difficulty group of each tile (`tile_group`: solved / contested /
       persistently hard, and clean / intermittent / persistent false alarm)
     - the validation images `X_val` (bands, slope, DEM, so e.g. NDVI). These are read from the
       run's `_data_cache.npz` only when a script first uses them.

     It never sees the train or test split.
   - **Answering.** It then posts its numbers on the blackboard.
   - **Limits.** At most `max_checks_per_round` checks per round, one per turn.
7. After the conversation the next round starts at step 2. The run stops after `n_rounds`,
   or earlier if the global-best Dice has not improved for `plateau_patience` consecutive
   rounds.

Generated pipelines must satisfy a fixed contract
([`src/agents/pipeline_contract.py`](src/agents/pipeline_contract.py)):
- `preprocess(X) -> X`, `build_model(config) -> model` and `train(model, X, y) -> model`
- a `.predict(X)` that returns binary masks
- a JSON-serializable `HPARAMS` dict

A pipeline is smoke-tested on a handful of fixed tiles before it is trusted to run on the
full split. Agents never write the scoring code: Dice/IoU and every observer metric come
from [`src/eval/`](src/eval/).

## Status

| Milestone | State |
|---|---|
| Dataset layer, pipeline contract, subprocess runner, deterministic metrics | done (carried over) |
| Literature corpus RAG (seeding strategies, metadata filter) | done (carried over) |
| Conversational swarm: create_agent agents, observer, blackboard, hypothesis checker | implemented; offline graph tests pass (`tests/`), first live runs pending |
| SQLite checkpointing with mid-conversation resume | implemented; tested offline |
| Run viewer (`frontend/`): browse runs, agent inputs/prompts, pipelines, observer, blackboard, checks | done |
| Analysis notebooks for conversational runs | not started (the notebooks in `runs_pipeline_optimization/` read the old PSO run layout) |

## Dataset

By default, pipelines are trained and scored on the **Landslide4Sense** benchmark
(Ghorbanzadeh et al., 2022): 128×128 multispectral + DEM-derived tiles (14 channels)
with binary landslide masks, expected under `archive/` as `archive/img/image_N.h5` and
`archive/mask/mask_N.h5` (only the originally labeled `TrainData` tiles). A fixed, seeded
train/val/test split (`split_ratios` in [`src/config.py`](src/config.py), default 70/15/15)
is carved out of these tiles and reused across every run.

The dataset layer is dataset-agnostic by design: everything specific to Landslide4Sense
(its description text, array shapes/channels, file layout, and loading code) lives
behind a single `DatasetSpec`
([`src/data/spec.py`](src/data/spec.py),
[`src/data/datasets/landslide4sense.py`](src/data/datasets/landslide4sense.py)), looked
up by `settings.dataset_name` via a registry
([`src/data/registry.py`](src/data/registry.py)). Every LLM-facing description of the
dataset (`DATASET_DESCRIPTION`, the pipeline contract's array-shape doc-string,
agent and observer prompts, retrieval query) and the actual loading code
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
contract shapes and the agent prompts all pick up the new
dataset automatically through the spec.


## Literature corpus

Agents ground their pipelines in retrieved papers rather than generic knowledge. They are
seeded with papers in round 0, and they can query the corpus at any time through their
`search_corpus` tool, while writing a pipeline and during the conversation. The corpus is a
set of ~300 landslide-mapping papers.
- **Format.** Each paper is pre-summarized into a structured JSON (methods, datasets,
  reported novelty, reproducibility status) and indexed into a local Chroma vector store
  ([`src/corpus/ingest.py`](src/corpus/ingest.py),
  [`src/corpus/retrieve.py`](src/corpus/retrieve.py)).
- **Browsing.** The corpus is browsable at
  **[fhrochaf.github.io/LSM_ReproViewer](https://fhrochaf.github.io/LSM_ReproViewer/)**,
  including the reproducibility survey mentioned above (only ~20% of 2020–2025 automated
  landslide-mapping papers publish code).
- **Full text.** Retrieved summaries are hydrated on demand with an excerpt of each paper's
  actual full text, extracted once via `pypdf` and then disk-cached
  ([`src/corpus/pdf_extract.py`](src/corpus/pdf_extract.py)). Excerpts are capped at
  `full_text_char_cap` per paper for seeding and `rag_tool_char_cap` per paper for tool
  calls.

## Module overview

- [`main.py`](main.py) — CLI entry point. It wires config, the RAG index, the cached dataset
  and the swarm graph together, then runs, resumes or extends a run.
- [`src/config.py`](src/config.py) — the single `Settings` object (pydantic-settings, loaded
  from `.env`):
  - LLM roles, swarm size, rounds
  - conversation and checker limits
  - tool settings, retrieval settings
  - dataset and corpus paths
- **`src/agents/`** — the agents and what they run:
  - [`swarm_agent.py`](src/agents/swarm_agent.py) — the `create_agent()` factory every LLM
    actor goes through. It adds retry and tool-call-limit middleware. It provides
    `build_pipeline` (pipeline.py + notebook), `take_turn` (a blackboard post / PASS / check
    request) and `fix_pipeline` (the debug loop).
  - [`context.py`](src/agents/context.py) — builds an agent's **shared** and **private**
    context from the swarm state, fresh for every invocation.
  - [`tools.py`](src/agents/tools.py) — `search_corpus` (RAG), `web_search` (Tavily) and
    `request_hypothesis_check`, with a recorder that logs what each agent looked up.
  - [`checker.py`](src/agents/checker.py) / [`checker_driver.py`](src/agents/checker_driver.py)
    — the HYPOTHESIS CHECKER agent and the subprocess that runs its analysis scripts over a
    read-only `DATA` mapping (val ground truth, every agent's val predictions, tile groups, and
    `X_val` loaded lazily from the run's data cache).
  - [`runner.py`](src/agents/runner.py) / [`driver.py`](src/agents/driver.py) — run one
    `pipeline.py` end-to-end in a subprocess per seed. Missing libraries are auto-installed,
    and crashes go through the agent's debug loop.
  - [`pipeline_contract.py`](src/agents/pipeline_contract.py) — the interface every
    pipeline must implement, and its smoke test.
  - [`prompts.py`](src/agents/prompts.py) — loads and validates the prompts YAML and renders
    its `<<placeholders>>`.
  - [`artifacts.py`](src/agents/artifacts.py) and [`text_utils.py`](src/agents/text_utils.py)
    — run-dir file I/O, and parsing of LLM output (code fence, notebook, citations).
- **`src/orchestration/`** — the swarm loop:
  - [`graph.py`](src/orchestration/graph.py) — the LangGraph `StateGraph` (nodes above), the
    ledgers, the SQLite checkpointer, and finalization.
  - [`state.py`](src/orchestration/state.py) — `AgentState`, `Post`, `PendingCheck` and
    `SwarmState`.
  - [`observer.py`](src/orchestration/observer.py) — metric tables and the OBSERVER's report.
  - [`blackboard.py`](src/orchestration/blackboard.py) — formatting and persistence of the
    blackboard.
  - [`registry.py`](src/orchestration/registry.py) — the cross-run results registry.
- **`src/corpus/`** — the literature RAG layer:
  - index building: [`ingest.py`](src/corpus/ingest.py)
  - similarity search and round-0 diverse seeding: [`retrieve.py`](src/corpus/retrieve.py)
  - full-text excerpts: [`pdf_extract.py`](src/corpus/pdf_extract.py)
- **`src/data/`** — dataset-agnostic loading through a `DatasetSpec` (see Dataset above).
- **`src/eval/`** — the fixed scoring code agents cannot override:
  - [`metrics.py`](src/eval/metrics.py): Dice/IoU
  - [`infer.py`](src/eval/infer.py): inference
  - [`observe.py`](src/eval/observe.py) and [`observe_detail.py`](src/eval/observe_detail.py):
    the behavioral tables behind the observer
- [`frontend/`](frontend/) — the run viewer (see "Run viewer" below).
- [`tests/`](tests/) — offline tests of the graph with every LLM call and pipeline run
  stubbed, covering:
  - turn order and rotation
  - the all-PASS early stop
  - the check handoff and its cap
  - mid-conversation resume
  - the checker's analysis subprocess

  Run them with `uv run pytest tests`.

## Setup

```bash
uv sync
```

The swarm's LLMs are configured independently ([`src/config.py`](src/config.py)), and each
can be a different provider/model:

| Role | Settings | Used for |
|---|---|---|
| Agents | `agents` — a list of groups | the swarm agents: writing and fixing `pipeline.py`, and talking on the blackboard |
| Observer | `llm_provider_2` / `model_name_2` | the once-per-round comparative report on the metric tables (a cheap model is fine) |
| Checker | `llm_provider_3` / `model_name_3` | the HYPOTHESIS CHECKER, which writes and runs analysis scripts |

`agents` lets agents on different models coexist in one swarm, which gives it more diverse
priors. Each group is `n_agents` agents on one model:

```python
agents = [
    {"n_agents": 5, "llm_provider": "google_genai", "model_name": "gemini-3.1-flash-lite", "max_tokens": 300000},
    {"n_agents": 3, "llm_provider": "openai", "model_name": "gpt-6-luna", "max_tokens": 300000},
]
```

- **Numbering.** Agents are numbered in list order: here agent_0–4 run on Gemini and agent_5–7
  on GPT. The swarm size is the sum of the groups' `n_agents`.
- **Setting it per run.** `agents` can also be given as JSON in `.env` / the environment
  (`AGENTS='[...]'`).
- **What records the model.** Each agent's model is recorded in `run_config.json`, the
  ledger, the agents' score table and the viewer. A resumed run keeps every agent on the
  model it started with.
- **Per-model options.** A group's `model_kwargs` is passed to the model's LangChain class, e.g.
  `{"reasoning_effort": "low"}`. `None` removes a default, e.g. `{"temperature": None}` for
  reasoning models that reject a temperature.
- **OpenAI models use the Responses API** (`use_responses_api=True`) by default. Chat
  Completions rejects function tools together with reasoning on recent OpenAI models, and
  agents always have tools. Override with `model_kwargs` if needed.
- **Provider packages.** A provider needs its LangChain package installed (e.g.
  `uv add langchain-openai` for `openai`). Every model is instantiated at start-up, so a
  missing package or API key stops the run before anything is paid for.

Add one API key per distinct provider to a `.env` file at the repo root (e.g.
`GOOGLE_GENAI_API_KEY`, `ANTHROPIC_API_KEY`). Also add `TAVILY_API_KEY` for the agents'
`web_search` tool; without it the tool is simply not offered (`web_search=false` turns it
off).

`corpus_json_dir`, `corpus_metadata_csv` and `publications_dir` point at the literature
corpus on disk. `dataset_dir` points at the Landslide4Sense tiles.

> **Local index vs `retrieval_filter`.** The default filter uses `method_class_name`, which
> exists only in an index built with the Scopus CSV join. An older local index without it
> makes every corpus search come back empty. The defaults point at the server
> (`/mnt/pool/...`). On another machine, set `CORPUS_JSON_DIR`, `CORPUS_METADATA_CSV` and
> `PUBLICATIONS_DIR` in `.env` to your local copy of the corpus and run once with
> `--rebuild-index`. `--rebuild-index` drops the old collection first, so only run it where
> the corpus JSONs exist. Alternatively, `RETRIEVAL_FILTER=null` searches the whole corpus
> without the filter.

## Usage

```bash
python main.py [--agents N] [--rounds T] [--cycles C] [--max-train-tiles N] [--n-seeds N]
               [--plateau-patience N] [--rebuild-index] [--resume RUN_ID] [--show-graph]
```

| Flag | Description |
|---|---|
| `--agents N` | Run N agents, all on the model of the first group in `agents` (default: the groups as configured). Ignored with `--resume`. |
| `--rounds T` | Number of rounds. With `--resume`, the new target (extends a finished run). |
| `--cycles C` | Conversation cycles per round (`conversation_cycles`; 0 = no conversation). |
| `--max-train-tiles N` | Cap on training tiles (a cheap-proxy subset). Ignored with `--resume`. |
| `--n-seeds N` | Evaluate every pipeline under up to N of `eval_seeds`. The score is the mean Dice. |
| `--plateau-patience N` | Stop after this many rounds without global-best improvement. |
| `--rebuild-index` | Rebuild the RAG literature index. |
| `--resume RUN_ID` | Continue a run from its last checkpoint, which can be mid-conversation. |
| `--show-graph` | Print the swarm graph as mermaid and exit. |

Main conversation/tool settings in [`src/config.py`](src/config.py), each also settable as an
environment variable:

| Setting | Controls |
|---|---|
| `conversation_cycles` | cycles per round |
| `max_checks_per_round`, `checker_max_runs`, `checker_timeout_s`, `checker_output_char_cap` | the checker |
| `agent_max_tool_calls` | tool calls per agent invocation |
| `rag_tool_k`, `rag_tool_char_cap` | corpus tool |
| `web_search`, `web_search_max_results` | Tavily |
| `notebook_char_cap` | the agent's notebook |

Each run writes to `<runs_dir>/<run_id>/`:

- `run_config.json`, `prompts_used.yaml`, `_data_cache.npz`
- `checkpoints.sqlite` — the LangGraph checkpoint after every node, used by `--resume`
- `ledger.jsonl` — one row per agent per round:
  - Dice per seed, IoU, train Dice (same pipeline on its training tiles) and behavior metrics
  - debug iterations, cited papers and changed `HPARAMS`
  - what the agent searched while building
- `conversation_ledger.jsonl` — one row per agent per round: posts, passes, check requests,
  tool use
- `round_<t>/`:
  - `observer.md`, `observer_tables.md`
  - `blackboard.{jsonl,md}`
  - `checker/` (inputs, scripts, transcripts)
- `round_<t>/agent_<i>/`:
  - `pipeline.py`, `notebook.md`, `seed_papers.md` (round 0)
  - `build_transcript.md`, `talk_transcript.md` (tool calls and answers)
  - `driver_stdout.json`, `val_preds_seed<S>.npz`
- `g_best_pipeline.py`, `summary.json`

Every finished run is also registered in `<runs_dir>/registry.jsonl` and a flattened
`registry.csv`: one row per run. Each row holds:
- configuration: models, agents, rounds, cycles, seeds, retrieval settings, git commit
- outcome: global-best Dice, best agent and round, stop reason
- per-round curves
- conversation totals: posts, checks, tool calls

`orchestration.registry.load_registry(runs_dir)` returns the registry as a DataFrame.

### Run viewer

```bash
uv run python frontend/server.py [RUNS_DIR ...] [--port 8765]
```

A local, read-only web page (standard-library server on `127.0.0.1`) for browsing runs.
Without arguments it shows `runs_dir` and every `runs*/` folder at the repo root; old PSO runs
are listed too, as generic files. Per run it shows:
- **Overview** — Dice per agent and round, and conversation activity.
- **Rounds** — for every round:
  - **Agents** — each agent's pipeline, notebook, and full build conversation: system prompt,
    shared + private input, every tool call and result, and the answer. Also its debug fixes,
    run log (tracebacks per attempt) and blackboard turns.
  - **Observer** — the observer's report, tables and prompt.
  - **Blackboard** — the blackboard as a chat.
  - **Hypothesis checks** — every check with its analysis scripts.

Tick *auto-refresh* to follow a run while it is going. Transcripts split into messages on the
`===== [kind] =====` lines that `agents/swarm_agent.py::transcript` writes.

### Prompts

Every prompt lives in one YAML file, [`src/prompts.yaml`](src/prompts.yaml) by default
(`prompts_file` / `PROMPTS_FILE` selects another):

| Key | Prompt |
|---|---|
| `agent.system` | the system prompt of every swarm agent |
| `agent.build_initial_user[_no_rag]`, `agent.build_user`, `agent.fix_user` | writing and fixing pipelines |
| `conversation.turn_user` | a blackboard turn |
| `observer.{system,user}` | the OBSERVER |
| `checker.{system,user}` | the HYPOTHESIS CHECKER |

The file is validated at start-up. A missing prompt stops the run. A placeholder the code
does not supply only produces a warning. Each run keeps a copy of the file (`prompts_used.yaml`)
and the registry stores its hash. The PSO-era prompt variants are in git history (before this
branch); they do not work with this code.

### Metadata filter on retrieval

At index time ([`src/corpus/ingest.py`](src/corpus/ingest.py)) each paper JSON is joined, on its
filename stem == the CSV's `EID`, to the Scopus-export CSV at `corpus_metadata_csv` (config.py;
`;`-delimited; `None` skips the join), adding `method_class_name`, `year`, `cited_by`,
`source_title` and `document_type` to the metadata already stored (`paper_id`,
`reproducibility_status`, `n_methods`, `n_datasets`). Unmatched or blank values become `"UNKNOWN"`
(text) or `0` (numbers). `retrieval_filter` (config / `RETRIEVAL_FILTER` as JSON) is a Chroma
`where` dict applied to every corpus retrieval — round-0 seeding under all three strategies and every
`search_corpus` tool call; `None` searches the whole corpus. E.g. `{"method_class_name": "CNN-based Semantic
Segmentation"}` or `{"$and": [{"year": {"$gte": 2022}}, {"reproducibility_status": "REPRODUCIBLE"}]}`.
Changing the CSV or its path needs `--rebuild-index` (which now drops the old collection first
instead of appending duplicates); changing only the filter does not. A filter matching fewer than
`retrieval_k × n_agents` papers gives agents fewer papers, and one matching none will fail.

### Round-0 retrieval strategies

`retrieval_strategy` (config / `RETRIEVAL_STRATEGY`) controls how round-0 papers are handed to the
agents ([`src/corpus/retrieve.py`](src/corpus/retrieve.py)): `shard` (default) deals the top
`retrieval_k × n_agents` papers round-robin; `mmr` picks them by maximal marginal relevance over a
`retrieval_pool_size` pool (`mmr_lambda` trades relevance against diversity) and deals them out the
same way; `cluster` k-means-clusters that pool on the stored embeddings into one group per agent and
gives each agent the top papers of its group. The prompts do not change between strategies.

### Observation detail

`observation_detail` (config / `OBSERVATION_DETAIL`) sets what the OBSERVER is shown about each
round. `basic` (default) is the aggregate tables of [`src/eval/observe.py`](src/eval/observe.py).
`detailed` appends the tile-level section of [`src/eval/observe_detail.py`](src/eval/observe_detail.py),
which both the OBSERVER and every agent (through the shared tables) then see:

- **Stratified results**: tile Dice on landslide tiles, and the false-alarm share on landslide-free
  tiles, by landslide size, number of landslide objects, and terciles of mean slope / elevation /
  NDVI of the tile's input (the terrain strata come from the dataset spec's optional
  `tile_descriptors`, computed once per run and cached as `_tile_descriptors.npz`).
- **Object-level error types**: the fate of every landslide object (missed, under-covered,
  fragmented, detected; loss rate by object size) and of every predicted blob (spurious, merged).
- **Persistent difficulty**: landslide tiles grouped as solved by most pipelines, contested (some
  pipeline solves them, most do not) or persistently hard, and landslide-free tiles as clean,
  intermittent or persistent false alarms, from every earlier non-collapsed pipeline of the run
  (per-tile stats cached as `tile_stats.npz` next to each pipeline and back-filled from the saved
  predictions when absent). Groups are described by their make-up, never by tile id, so agents learn
  what makes tiles hard rather than tuning to named validation tiles; it also shows whether the swarm's
  Dice on the hard groups is improving round by round.

The extra text is about 2k tokens with 10 agents, added to the observer prompt and to every agent's shared context; computing it takes about 10 s per round (the first observation of a run also back-fills the per-tile cache from the saved predictions).

## License

GPL-3.0 — see [LICENSE.txt](LICENSE.txt).

## References

Hwang, H., Kim, J., Kim, C., Chang, H., & Ye, J. C. (2026). *AgentPSO: Evolving Agent
Reasoning Skill via Multi-agent Particle Swarm Optimization*. arXiv:2605.08704.
https://arxiv.org/abs/2605.08704v2
