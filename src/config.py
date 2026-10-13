"""Central run configuration. Everything else imports Settings from here."""
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent

# Populate os.environ from .env so provider SDKs (anthropic, openai, ...) and other
# libraries (e.g. huggingface_hub) that read their own env vars directly also see it
load_dotenv(REPO_ROOT / ".env")


class AgentGroup(BaseModel):
    """`n_agents` swarm agents that all run on the same chat model."""

    n_agents: int = 1
    # Any provider LangChain's init_chat_model supports (google_genai, anthropic, openai, ollama,
    # deepseek, ...); the provider's LangChain package must be installed (e.g. langchain-openai).
    llm_provider: str
    model_name: str
    api_key: str = ""  # empty: read from <LLM_PROVIDER>_API_KEY in .env / the environment
    max_tokens: int = 120000
    # Extra keyword arguments for this model's LangChain chat class (init_chat_model), e.g.
    # {"reasoning_effort": "low"}. A value of None removes a default -- e.g. {"temperature": None}
    # for reasoning models that reject a temperature. OpenAI models already go through the
    # Responses API (use_responses_api=True, needed for tools + reasoning); override it here.
    model_kwargs: dict = {}

    @property
    def label(self) -> str:
        return f"{self.llm_provider}/{self.model_name}"


class Settings(BaseSettings):
    # env_parse_none_str: "null" in an env var / .env means None (e.g. RETRIEVAL_FILTER=null searches
    # the whole corpus, CORPUS_METADATA_CSV=null skips the CSV join).
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", extra="ignore", env_parse_none_str="null")

    # Role 1 -- the swarm agents themselves (agents/swarm_agent.py): they write pipeline.py, fix it
    # when it crashes, and talk on the blackboard. A list of groups, each `n_agents` agents on one
    # model, so agents of different models can coexist in one swarm (more diverse priors). Agents
    # are numbered in list order: with the example below agent_0..4 run Gemini and agent_5..7 GPT.
    # The swarm size is the sum of the groups' n_agents. Also settable as JSON, e.g.
    # AGENTS='[{"n_agents": 2, "llm_provider": "google_genai", "model_name": "gemini-3.1-flash-lite"}]'
    #
    #   agents = [
    #       {"n_agents": 5, "llm_provider": "google_genai", "model_name": "gemini-3.1-flash-lite", "max_tokens": 300000},
    #       {"n_agents": 3, "llm_provider": "openai", "model_name": "gpt-6-luna", "max_tokens": 300000},
    #   ]
    agents: list[AgentGroup] = Field(default_factory=lambda: [
        AgentGroup(n_agents=6, llm_provider="google_genai", model_name="gemini-3.1-flash-lite"),
        AgentGroup(n_agents=3, llm_provider="openai", model_name="gpt-6-luna"),
        #AgentGroup(n_agents=1, llm_provider="anthropic", model_name="claude-sonnet-5")
        AgentGroup(n_agents=1, llm_provider="deepseek", model_name="deepseek-v4-pro")
    ])

    ################################
    # Role 2 -- the OBSERVER (orchestration/observer.py): one call per round that turns the
    # code-computed metric tables of every agent into the comparative report all agents read
    # before the conversation. A narrow read-and-summarize task, so a cheap model is fine.
    llm_provider_2: str = "google_genai"
    model_name_2: str = "gemini-3.1-flash-lite"
    api_key_2: str = ""
    max_tokens_2: int = 300000

    ################################
    # Role 3 -- the HYPOTHESIS CHECKER (agents/checker.py): writes and runs analysis scripts over
    # the agents' validation predictions when an agent asks it to check a hypothesis.
    llm_provider_3: str = "google_genai"
    model_name_3: str = "gemini-3.1-flash-lite"
    api_key_3: str = ""
    max_tokens_3: int = 300000

    temperature: float = 0.0

    # Retries (exponential backoff+jitter) applied to every LLM call, so a transient
    # provider error doesn't crash a whole round.
    llm_retry_attempts: int = 5
    # Hard cap on how many tool calls (corpus search, web search, analysis runs, ...) one agent
    # invocation -- one pipeline write, one conversation turn -- may make. Past it the agent is
    # told to answer with what it has.
    agent_max_tool_calls: int = 8

    @model_validator(mode="after")
    def _fill_api_key(self) -> "Settings":
        self.fill_agent_api_keys()
        if not self.api_key_2:
            self.api_key_2 = os.environ.get(f"{self.llm_provider_2.upper()}_API_KEY", "")
        if not self.api_key_3:
            self.api_key_3 = os.environ.get(f"{self.llm_provider_3.upper()}_API_KEY", "")
        return self

    def fill_agent_api_keys(self) -> None:
        for group in self.agents:
            if not group.api_key:
                group.api_key = os.environ.get(f"{group.llm_provider.upper()}_API_KEY", "")

    @property
    def n_agents(self) -> int:
        return sum(g.n_agents for g in self.agents)

    def agent_models(self) -> list[AgentGroup]:
        """One entry per agent (index = agent_idx): the group whose model that agent runs on."""
        return [g for g in self.agents for _ in range(g.n_agents)]

    n_rounds: int = 2
    # Stop early if the swarm's global-best Dice hasn't improved by more than
    # p_best_epsilon for this many consecutive rounds.
    plateau_patience: int = 10

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
    prompts_file: str = "prompts.yaml"

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
    # Metadata filter applied to EVERY corpus retrieval (round-0 seeding and the agents' search_corpus tool) as a Chroma
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
    # alone (prompt `agent.build_initial_user_no_rag`) and the agents get no search_corpus tool.
    # The no-literature baseline.
    use_rag: bool = True
    # The agents' search_corpus tool (agents/tools.py): papers per query, and the per-paper cap on
    # the full-text excerpt it returns.
    rag_tool_k: int = 3
    rag_tool_char_cap: int = 80000
    # The agents' web_search tool (Tavily; needs TAVILY_API_KEY in .env -- without it the tool is
    # simply not offered).
    web_search: bool = True
    web_search_max_results: int = 3
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

    # What the OBSERVER is shown about each round (eval/observe.py, eval/observe_detail.py):
    #   "basic"    the aggregate tables (pixel / tile / object level, by landslide size, across agents)
    #   "detailed" the basic tables plus stratified results (landslide size, object count, terrain), the
    #              fate of every landslide object and predicted blob, and a summary of the tiles the whole
    #              swarm keeps failing on (described by their characteristics, never by tile id)
    observation_detail: Literal["basic", "detailed"] = "detailed"

    # The conversation on the shared blackboard (orchestration/graph.py), after the OBSERVER and
    # before the agents write their next pipelines: round-robin over the agents (the order rotates
    # every round) for this many cycles. An agent may PASS; a cycle in which every agent passes
    # ends the conversation early.
    conversation_cycles: int = 3
    # The HYPOTHESIS CHECKER (agents/checker.py): at most this many check requests per round
    # (across the whole swarm); per request, at most checker_max_runs analysis scripts, each with
    # a checker_timeout_s budget and its printed output capped at checker_output_char_cap.
    max_checks_per_round: int = 10
    checker_max_runs: int = 4
    checker_timeout_s: int = 600
    checker_output_char_cap: int = 10000
    # Each agent's private notebook -- the only thing it carries from one round to the next besides
    # its pipelines -- is capped at this many characters.
    notebook_char_cap: int = 10000

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
    
    max_train_tiles: int = 3000
    max_debug_iters: int = 5

    # Budget for one driver.py subprocess call: package auto-install (agents can now
    # import any library, e.g. torch) + train + evaluate, combined. Was 300s when only
    # numpy/scikit-learn were allowed; widened since a first-time heavy install alone
    # (e.g. torch) can take several minutes.
    exec_timeout_s: int = 3600

    # Margin: a new score only replaces personal-best /
    # global-best if it improves by more than this, to ignore noisy fluctuations.
    p_best_epsilon: float = 0.01


settings = Settings()
