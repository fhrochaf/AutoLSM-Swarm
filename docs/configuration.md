# Configuration guide

All settings are fields of `Settings` in [`src/config.py`](../src/config.py). You can
override any of them in `.env` or with an environment variable of the same name in upper
case. Each run saves its resolved settings, without API keys, as `<run>/config_used.json`.
A `--resume` writes `config_used_resume_r<round>.json` and leaves the original untouched.

- [Prompt variants](#prompt-variants)
- [No-literature baseline](#no-literature-baseline)
- [Retrieval](#retrieval)
- [Reflection and grounding](#reflection-and-grounding)
- [Observation detail](#observation-detail)
- [Fidelity gate](#fidelity-gate)
- [Structured outputs](#structured-outputs)
- [Scoring and stopping](#scoring-and-stopping)

---

## Prompt variants

All prompts live in one YAML file. The `prompts_file` setting selects it, either as a name
resolved under `src/` or as a path:

```bash
PROMPTS_FILE=prompts_velocity_update_cot.yaml python main.py --agents 5 --rounds 10
```

| File | Behavior |
|---|---|
| `prompts.yaml` | Baseline: one change per velocity (`CHANGE <HPARAMS key>: old -> new` or one structural mechanism). |
| `prompts_diversity_enhancement.yaml` | Pushes agents to keep their own approach. |
| `prompts_divers_enhanc_onVeloc.yaml` | Like the diversity variant, with a looser velocity: one coherent change, which may be a mechanism plus its companion hyperparameters. It also prefers adopting literature candidates over further tuning. |
| `prompts_velocity_update_cot.yaml` | Chain-of-thought velocity in five numbered steps: last move, diagnosis, 2–4 candidates, elimination, decision. |
| `prompts_velocity_cot_multichanges.yaml` | Chain of thought, and a velocity may hold several related changes. Run it with `fidelity_check=False`, because its judge prompts still assume a single change. |
| `prompts_singleagent_noRag.yaml` | Single-agent, no-literature baseline (see below). |

**Validation.** The file is checked at start-up, before any paid call. Only two problems
stop a run:

- the file is missing
- a prompt needed by an enabled feature is missing

Placeholders never stop a run. If a prompt omits a placeholder, that value is simply not
shown. If a prompt uses a placeholder the code does not supply, you get a warning and the
text stays as written.

**Traceability.** Each run stores its prompts as `<run>/prompts_used.yaml`. The registry
records the file name and a short hash (`prompts_file`, `prompts_sha`). `--resume` warns if
the file has changed since the run started.

---

## No-literature baseline

`use_rag=false` switches corpus retrieval off:

- Round 0 is written from the dataset description alone, using the prompt
  `pipeline.initial_user_no_rag`.
- Later rounds never run an enriched reflection.

The prompt file must define `pipeline.initial_user_no_rag`.
`prompts_singleagent_noRag.yaml` does, and it also removes every mention of peers and
the global best:

```bash
USE_RAG=false PROMPTS_FILE=prompts_singleagent_noRag.yaml python main.py --agents 1 --rounds 10
```

---

## Retrieval

### Round-0 strategies

`retrieval_strategy` controls how round-0 papers are shared out among the agents
([`retrieve.py`](../src/corpus/retrieve.py)). The prompts are the same for every strategy.

| Strategy | Behavior |
|---|---|
| `shard` | Takes the top `retrieval_k × n_agents` papers and deals them out round-robin. |
| `mmr` | Picks papers by maximal marginal relevance from a pool of `retrieval_pool_size`, then deals them out round-robin. `mmr_lambda` trades relevance against diversity. |
| `cluster` | Runs k-means on the pool's embeddings with one cluster per agent. Each agent gets the top papers of its cluster. |

### Metadata filter

At index time ([`ingest.py`](../src/corpus/ingest.py)), each paper is joined to the Scopus
CSV at `corpus_metadata_csv`, matching the file stem to `EID`. The join adds
`method_class_name`, `year`, `cited_by`, `source_title` and `document_type` to the
metadata. Missing values become `"UNKNOWN"` or `0`.

`retrieval_filter` is a Chroma `where` dict applied to every retrieval. For example:

```python
{"method_class_name": "CNN-based Semantic Segmentation"}
{"$and": [{"year": {"$gte": 2022}}, {"reproducibility_status": "REPRODUCIBLE"}]}
```

- Changing the CSV requires `--rebuild-index`. Changing only the filter does not.
- If the filter matches fewer than `retrieval_k × n_agents` papers, agents get fewer
  papers. If it matches none, retrieval fails.

`retrieval_l4s_usable_only` (default `True`) adds an allowlist from
`utils.l4s_usable_paper_ids`. The allowlist excludes papers the human review left
unclassified, and papers that need input Landslide4Sense cannot supply: SAR/InSAR, LiDAR
or multi-temporal data. It is applied at query time.

---

## Reflection and grounding

An agent's reflection can be **enriched** with literature. The reflection then ends with a
`GROUNDING QUERY: ...` line. That line is run unchanged as a corpus search, and a second
pass revises the reflection against the retrieved papers, citing them as `[paper_id]`.

| Setting | Effect |
|---|---|
| `enriched_reflection_all` | Enrich every agent's reflection. |
| `random_enriched_reflection` | Probability of enriching each agent's reflection. |
| `enriched_leader` | Always enrich the current global-best holder. |
| `velocity_reflection` | What the velocity step reads: `original+enriched` (both) or `enriched` (enriched only). |

Enrichment tends to drop the behavioral numbers, which is why `original+enriched` exists.
`reflection.md` holds the enriched text. The original is kept as
`reflection_original.md`.

---

## Observation detail

`observation_detail` sets what the peer review and the reflections see.

- `basic`: the aggregate tables from [`observe.py`](../src/eval/observe.py).
- `detailed`: adds the tile-level analysis from
  [`observe_detail.py`](../src/eval/observe_detail.py):
  - **Stratified results**: Dice and false-alarm share by landslide size, object count,
    and slope / elevation / NDVI terciles.
  - **Object-level errors**: missed, under-covered, fragmented, spurious and merged objects.
  - **Persistent difficulty**: tiles grouped as solved, contested or persistently hard
    across all earlier pipelines. Groups are described by their make-up and never by tile
    id, so agents cannot tune to specific validation tiles.

`detailed` adds about 2k tokens per prompt with 10 agents, and about 10 s of compute per
round.

---

## Fidelity gate

When `fidelity_check=True`, a judge LLM ([`fidelity.py`](../src/agents/fidelity.py))
reviews each velocity application. It reads the velocity, the diff and the new code, then
returns `FAITHFUL` or `NOT_FAITHFUL`.

| Setting | Effect |
|---|---|
| `max_fidelity_iters` | Maximum number of redo attempts after a `NOT_FAITHFUL` verdict. |
| `fidelity_web_search` | Lets the code LLM use Tavily during a redo (needs `TAVILY_API_KEY`). |
| `run_unfaithful_pipeline` | If the last attempt is still flagged: `True` runs it anyway, `False` re-runs the previous pipeline. |

The gate does not check round-0 pipelines or crash fixes. Each check is logged in
`fidelity.md`. The ledger marks failed checks with `fidelity_faithful: false`.

---

## Structured outputs

Two calls return JSON whose shape is fixed by code
([`structured.py`](../src/agents/structured.py)) rather than by prompt wording:

- **Velocity update**: `{reasoning, final_velocity}`. `final_velocity` is the directive
  that gets applied, judged and recorded. `reasoning` is saved to
  `velocity_reasoning.md` and passed to the code LLM, but is hidden from the judge.
- **Fidelity judge**: `{violations, verdict}`.

The code first tries the provider's native structured output. If that fails, it asks for
JSON in plain text. As a last resort, it uses the raw reply. This means any prompt file
works, even one that never mentions JSON.

Every pipeline declares its tunables in a module-level `HPARAMS` dict. The ledger can
therefore measure the size of each move (`changed_lines`, `hparams_changed`).

---

## Scoring and stopping

| Setting | Effect |
|---|---|
| `eval_seeds`, `n_seeds` | Seeds each pipeline is trained under. The score is the mean Dice over these seeds. |
| `p_best_epsilon` | Minimum Dice gain needed to replace a personal or global best. |
| `plateau_patience` | Rounds without a global-best improvement before the run stops early. |
| `max_debug_iters` | Crash-fix attempts per pipeline. |
| `exec_timeout_s` | Wall-clock limit for one pipeline run. |
| `split_ratios`, `seed` | Train/val/test split and the seed that fixes it. |
