"""Central run configuration. Everything else imports Settings from here."""
import os
from pathlib import Path

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
    # Used only for the peer review (orchestration/peer_review.py::run_peer_review): one call per round that
    # turns the code-computed metric tables of all agents into the comparative report agents reflect on.
    # llm_provider_2: str = "ollama"
    # model_name_2: str = "qwen3"

    llm_provider_2: str = "google_genai"
    model_name_2: str = "gemini-3.1-flash-lite"
    api_key_2: str = ""
    max_tokens_2: int = 100000

    ################################
    # Used only for pipeline.py creation (agents/codegen.py's generate_initial_pipeline,
    # apply_velocity, fix_pipeline)
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

    n_agents: int = 5
    n_rounds: int = 5
    # Stop early if the swarm's global-best Dice hasn't improved by more than
    # p_best_epsilon for this many consecutive rounds.
    plateau_patience: int = 4

    # Selects a DatasetSpec from data/registry.py -- everything dataset-specific
    # (description text, array shapes, loading logic, smoke-test tile ids) flows from
    # this one name; see data/spec.py.
    dataset_name: str = "landslide4sense"
    dataset_dir: Path = REPO_ROOT / "archive"
    runs_dir: Path = REPO_ROOT / "runs_landslide4sense"

    corpus_json_dir: Path =  Path(
        "/mnt/pool/landslide/AutoLSM-Swarm/AutoLSM-Swarm/output_fullPDF_with_guardrails_singlePDFread"
    )
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

    split_ratios: dict[str, float] = {"train": 0.7, "val": 0.15, "test": 0.15}
    seed: int = 42

    # Seeds every pipeline is evaluated under: each one is exported to the pipeline as AUTOLSM_SEED and
    # the pipeline is trained + scored once per seed. The score behind personal-best / global-best /
    # the plateau rule is the mean Dice over these. n_seeds caps how many of them are used ("up to N";
    # --n-seeds on the CLI); None uses all. A single seed reproduces single-run scoring.
    eval_seeds: list[int] = [42, 43, 44]
    n_seeds: int | None = None

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
    enriched_reflection_all: bool = False
    # If > 0.0, random enriched reflection will assign enriched reflections through CORPUS retrieval to a fraction of the agents
    random_enriched_reflection: float = 0.0

    # How many past velocities (with the Dice before/after applying each) the velocity
    # update sees as a "recent trajectory". 0 disables it.
    velocity_history_len: int = 2


settings = Settings()
