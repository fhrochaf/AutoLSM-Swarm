"""Central run configuration. Everything else imports Settings from here."""
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent

# Populate os.environ from .env so provider SDKs (anthropic, openai, ...) and other
# libraries (e.g. huggingface_hub) that read their own env vars directly also see it
load_dotenv(REPO_ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore")

    # Any provider LangChain's init_chat_model supports (anthropic, openai, ollama,
    # google_genai, ...).
    llm_provider: str = "google_genai"
    model_name: str = "gemini-3.1-flash-lite"
    api_key: str = ""
    max_tokens: int = 100000

    # llm_provider: str = "google_genai"
    # model_name: str = "gemini-3.1-flash-lite"

    # llm_provider: str = "deepsee  k"
    # model_name: str = "deepseek-v4-pro"

    ################################
    # Used for the peer review (orchestration/peer_review.py::run_peer_review): one call per round that
    # turns the code-computed metric tables of all agents into the comparative report agents reflect on,
    # and for the fidelity judge (agents/fidelity.py) when fidelity_check is on. Both are narrow
    # read-and-summarize/judge tasks, so a cheap model is fine.
    # llm_provider_2: str = "ollama"
    # model_name_2: str = "qwen3"

    llm_provider_2: str = "google_genai"
    model_name_2: str = "gemini-3.1-flash-lite"
    api_key_2: str = ""
    max_tokens_2: int = 100000

    ################################
    # Used only for pipeline.py creation (agents/codegen.py's generate_initial_pipeline,
    # apply_velocity, revise_pipeline_for_fidelity, fix_pipeline)
    llm_provider_3: str = "anthropic"
    model_name_3: str = "claude-sonnet-5"
    api_key_3: str = ""
    max_tokens_3: int = 100000

    temperature: float = 0.0

    # Retries (exponential backoff+jitter) applied to every LLM call, so a transient
    # provider error doesn't crash a whole round.
    llm_retry_attempts: int = 5

    @model_validator(mode="after")
    def _fill_api_key(self) -> "Settings":
        if not self.api_key:
            self.api_key = os.environ.get(f"{self.llm_provider.upper()}_API_KEY", "")
        if not self.api_key_2:
            self.api_key_2 = os.environ.get(f"{self.llm_provider_2.upper()}_API_KEY", "")
        if not self.api_key_3:
            self.api_key_3 = os.environ.get(f"{self.llm_provider_3.upper()}_API_KEY", "")
        return self

    n_agents: int = 2
    n_rounds: int = 2
    # Stop early if the swarm's global-best Dice hasn't improved by more than
    # p_best_epsilon for this many consecutive rounds.
    plateau_patience: int = 4

    # Selects a DatasetSpec from data/registry.py -- everything dataset-specific
    # (description text, array shapes, loading logic, smoke-test tile ids) flows from
    # this one name; see data/spec.py.
    dataset_name: str = "landslide4sense"
    dataset_dir: Path = REPO_ROOT / "archive/TrainData"
    runs_dir: Path = REPO_ROOT / "runs_landslide4sense"

    # The YAML holding every LLM prompt (see agents/prompts.py): a file name resolved under src/, or a path.
    # Keep one file per prompt variant to compare; override per run without editing code with
    # PROMPTS_FILE=prompts.yaml python main.py ...  (or in .env). A run stores a
    # copy of the file it used as <run>/prompts_used.yaml.
    prompts_file: str = "prompts_diversity_enhancement.yaml"

    corpus_json_dir: Path =  Path(
        "/mnt/pool/landslide/AutoLSM-Swarm/AutoLSM-Swarm/output_fullPDF_with_guardrails_singlePDFread"
    )
    # Scopus-export CSV (';'-delimited) with per-paper columns the JSONs lack. Joined to the JSONs on
    # its `EID` column == the JSON filename stem (paper_id) at index time (corpus/ingest.py), and
    # stored as filterable metadata: method_class_name, year, cited_by, source_title, document_type.
    # Missing/unmatched papers get "UNKNOWN" (text) or 0 (numbers). Changing the CSV or this path needs
    # --rebuild-index. None skips the join.
    corpus_metadata_csv: Path | None = (
        corpus_json_dir / "analysis" / "scopus_query1_postPipeline_postHumanReview.csv"
    )
    # Metadata filter applied to EVERY corpus retrieval (round 0 and enriched reflection) as a Chroma
    # `where` dict; None = search the whole corpus. Keys: paper_id, reproducibility_status, n_methods,
    # n_datasets, method_class_name, year, cited_by, source_title, document_type. Operators: $eq $ne
    # $gt $gte $lt $lte $in $nin; combine several conditions with {"$and": [...]} / {"$or": [...]}.
    # Examples:
    #   {"method_class_name": "CNN-based Semantic Segmentation"}
    #   {"$and": [{"year": {"$gte": 2022}}, {"reproducibility_status": {"$in": ["REPRODUCIBLE", "PARTIALLY_REPRODUCIBLE"]}}]}
    # Also settable per run as JSON: RETRIEVAL_FILTER='{"year": {"$gte": 2022}}'
    retrieval_filter: dict | None = {"method_class_name": {"$in": ["CNN-based Semantic Segmentation", "Hybrid & Advanced Architectures"]}}
    # The actual paper PDFs -- same paper_id stems as corpus_json_dir's *.json files.
    # The JSON summaries are a cheap pre-filter; retrieval hydrates the top matches
    # with an excerpt of the real full text from here (see corpus/pdf_extract.py).
    publications_dir: Path = Path(
        "/mnt/pool/landslide/LSM_ReproChecker/publications"
    )
    pdf_text_cache_dir: Path = REPO_ROOT / "database_vector_store" / "pdf_text_cache"
    # Per-paper cap when a full-text excerpt is stuffed into a prompt (~2k tokens).
    full_text_char_cap: int = 80000
    vector_store_dir: Path = REPO_ROOT / "database_vector_store" / "vector_store"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    retrieval_k: int = 3
    # Off: no corpus retrieval anywhere -- round 0 is written from the dataset description
    # alone (prompt `pipeline.initial_user_no_rag`) and later rounds never run an enriched
    # reflection. The no-literature baseline; pair with n_agents=1 for a single-agent run.
    use_rag: bool = True
    # How round 0 hands papers to the agents (corpus/retrieve.py::retrieve_diverse):
    #   "shard"   top retrieval_k * n_agents papers by similarity, dealt out round-robin (original behavior)
    #   "mmr"     the same number of papers picked by maximal marginal relevance over a retrieval_pool_size
    #             candidate pool (relevant to the query but dissimilar to each other), dealt out round-robin
    #   "cluster" the retrieval_pool_size most similar papers are k-means clustered into n_agents groups and
    #             each agent gets the top retrieval_k papers of one group
    retrieval_strategy: Literal["shard", "mmr", "cluster"] = "mmr"
    # Candidate pool for "mmr" / "cluster" (raised to retrieval_k * n_agents if smaller).
    retrieval_pool_size: int = 50
    # MMR trade-off: 1.0 = pure similarity to the query, 0.0 = pure diversity.
    mmr_lambda: float = 0.6

    split_ratios: dict[str, float] = {"train": 0.7, "val": 0.15, "test": 0.15}
    seed: int = 42

    # Seeds every pipeline is evaluated under: each one is exported to the pipeline as AUTOLSM_SEED and
    # the pipeline is trained + scored once per seed. The score behind personal-best / global-best /
    # the plateau rule is the mean Dice over these. n_seeds caps how many of them are used ("up to N";
    # --n-seeds on the CLI); None uses all. A single seed reproduces single-run scoring.
    eval_seeds: list[int] = [42, 43, 44]
    n_seeds: int | None = 1

    def active_seeds(self) -> list[int]:
        return self.eval_seeds[: self.n_seeds] if self.n_seeds else list(self.eval_seeds)
    
    max_train_tiles: int = 1500
    max_debug_iters: int = 5

    # Budget for one driver.py subprocess call: package auto-install (agents can now
    # import any library, e.g. torch) + train + evaluate, combined. Was 300s when only
    # numpy/scikit-learn were allowed; widened since a first-time heavy install alone
    # (e.g. torch) can take several minutes.
    exec_timeout_s: int = 3600

    # Margin: a new score only replaces personal-best /
    # global-best if it improves by more than this, to ignore noisy fluctuations.
    p_best_epsilon: float = 0.01

    # Enrich reflection: If True, all agents will update their reflections after each round based, aditionally on their peer's reflections, on
    # on retrieved context over the corpus via RAG and a query extracted for the initial reflection
    enriched_reflection_all: bool = True
    # If > 0.0, random enriched reflection will assign enriched reflections through CORPUS retrieval to a fraction of the agents
    random_enriched_reflection: float = 0.0
    # If true, the agent with the best global score will have an enriched reflection in that round.
    enriched_leader: bool = True

    # How many past velocities (with the Dice before/after applying each) the velocity
    # update sees as a "recent trajectory". 0 disables it.
    velocity_history_len: int = 2

    # Fidelity gate after the velocity is applied (agents/fidelity.py): a judge LLM (role 2 above) checks that
    # the updated pipeline.py implements exactly the one change the velocity chose and nothing else. If it
    # answers NOT_FAITHFUL, the update goes back to the code LLM (role 3) to redo, up to max_fidelity_iters
    # times; if it is still flagged after that, see run_unfaithful_pipeline. Either way it is flagged in
    # the ledger. Off: the updated pipeline is run as the code LLM returned it.
    fidelity_check: bool = True
    max_fidelity_iters: int = 4
    # What happens to an update still NOT_FAITHFUL after max_fidelity_iters revisions:
    #   True   the latest (flagged) attempt is run as it is
    #   False  the update is rejected: the agent keeps its previous pipeline.py, which is re-run and
    #          re-scored this round (the round's velocity is still recorded in the agent's history)
    run_unfaithful_pipeline: bool = True
    # While redoing a flagged update the code LLM may look things up with Tavily web search (needs
    # TAVILY_API_KEY in .env; without it the revision just runs without the tool).
    fidelity_web_search: bool = True


settings = Settings()
