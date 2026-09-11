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
    # google_genai, ...). Only "anthropic" is installed by default right now -- add the
    # matching langchain-<provider> package to switch.
    # llm_provider: str = "anthropic"
    # model_name: str = "claude-sonnet-5"

    llm_provider: str = "google_genai"
    model_name: str = "gemini-3.1-flash-lite"

    # Loaded from .env. Set API_KEY directly to override, otherwise falls back to the
    # provider's own convention, e.g. ANTHROPIC_API_KEY / OPENAI_API_KEY.
    api_key: str = ""
    temperature: float = 0.0

    @model_validator(mode="after")
    def _fill_api_key(self) -> "Settings":
        if not self.api_key:
            self.api_key = os.environ.get(f"{self.llm_provider.upper()}_API_KEY", "")
        return self

    n_agents: int = 3
    n_rounds: int = 5
    # Stop early if the swarm's global-best Dice hasn't improved by more than
    # p_best_epsilon for this many consecutive rounds (implementation_steps.md's
    # "stop ... once dice plateaus across the population").
    plateau_patience: int = 3

    dataset_dir: Path = REPO_ROOT / "archive"
    runs_dir: Path = REPO_ROOT / "runs"

    corpus_json_dir: Path = Path(
        r"D:\Flávio Rocha USER\OneDrive\University of Twente\MSc 2025-2027"
        r"\Thesis_LAReprod\LSM_ReproChecker\output_fullPDF_with_guardrails_singlePDFread"
    )
    # The actual paper PDFs -- same paper_id stems as corpus_json_dir's *.json files.
    # The JSON summaries are a cheap pre-filter; retrieval hydrates the top matches
    # with an excerpt of the real full text from here (see corpus/pdf_extract.py).
    publications_dir: Path = Path(
        r"D:\Flávio Rocha USER\OneDrive\University of Twente\MSc 2025-2027"
        r"\Thesis_LAReprod\LSM_ReproChecker\publications"
    )
    pdf_text_cache_dir: Path = REPO_ROOT / "database_vector_store" / "pdf_text_cache"
    # Per-paper cap when a full-text excerpt is stuffed into a prompt (~2k tokens).
    full_text_char_cap: int = 8000
    vector_store_dir: Path = REPO_ROOT / "database_vector_store" / "vector_store"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    retrieval_k: int = 3

    split_ratios: dict[str, float] = {"train": 0.7, "val": 0.15, "test": 0.15}
    seed: int = 42
    
    max_train_tiles: int = 200
    max_val_tiles: int = 100
    max_debug_iters: int = 5
    # Budget for one driver.py subprocess call: package auto-install (agents can now
    # import any library, e.g. torch) + train + evaluate, combined. Was 300s when only
    # numpy/scikit-learn were allowed; widened since a first-time heavy install alone
    # (e.g. torch) can take several minutes.
    exec_timeout_s: int = 1800

    # AgentPSO Algorithm 1's margin: a new score only replaces personal-best /
    # global-best if it improves by more than this, to ignore noisy fluctuations.
    p_best_epsilon: float = 0.01


settings = Settings()
