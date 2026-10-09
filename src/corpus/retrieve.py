"""RAG retrieval over the corpus index built by ingest.py."""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import Settings
from corpus.ingest import get_embeddings
from corpus.pdf_extract import get_full_text


def get_vectorstore(settings: Settings) -> Chroma:
    return Chroma(
        embedding_function=get_embeddings(settings),
        persist_directory=str(settings.vector_store_dir),
        collection_name="landslide_auto_mapping_corpus",
    )


def retrieve(
    query: str, settings: Settings, k: int | None = None, filters: dict | None = None
) -> list[Document]:
    """Top-k papers for `query`. `filters` (a Chroma `where` dict) defaults to
    settings.retrieval_filter, so every corpus retrieval honours the configured filter."""
    store = get_vectorstore(settings)
    return store.similarity_search(
        query, k=k or settings.retrieval_k, filter=filters or settings.retrieval_filter
    )


def _shard(query: str, settings: Settings, n_agents: int, k: int) -> list[list[Document]]:
    """The top k * n_agents papers by similarity, dealt out round-robin."""
    store = get_vectorstore(settings)
    pool_size = min(k * n_agents, store._collection.count())
    candidates = store.similarity_search(query, k=pool_size, filter=settings.retrieval_filter)
    return [candidates[i::n_agents][:k] for i in range(n_agents)]


def _pool_size(settings: Settings, store: Chroma, n_agents: int, k: int) -> int:
    return min(max(settings.retrieval_pool_size, k * n_agents), store._collection.count())


def _mmr_order(query: str, settings: Settings, store: Chroma, n: int, fetch_k: int) -> list[Document]:
    """`n` papers picked by maximal marginal relevance among the `fetch_k` most similar ones:
    each next paper is relevant to the query and dissimilar to those already chosen, so the
    list comes out most-different-first."""
    return store.max_marginal_relevance_search(
        query, k=n, fetch_k=fetch_k, lambda_mult=settings.mmr_lambda, filter=settings.retrieval_filter
    )


def _mmr(query: str, settings: Settings, n_agents: int, k: int) -> list[list[Document]]:
    """MMR-selected papers dealt out round-robin: each agent's leading paper is one of the
    first, most mutually different picks."""
    store = get_vectorstore(settings)
    pool = _pool_size(settings, store, n_agents, k)
    picked = _mmr_order(query, settings, store, min(k * n_agents, pool), pool)
    return [picked[i::n_agents][:k] for i in range(n_agents)]


def _cluster(query: str, settings: Settings, n_agents: int, k: int) -> list[list[Document]]:
    """The most similar `retrieval_pool_size` papers (the relevance gate) are k-means
    clustered, on their stored embeddings, into one group per agent; each agent gets the
    top-k papers of one group, ranked by similarity to the query. Groups are assigned to
    agents in order of their best-ranked paper, so the assignment is deterministic. A group
    with fewer than k papers is topped up from the MMR order, never repeating a paper."""
    import numpy as np
    from sklearn.cluster import KMeans

    store = get_vectorstore(settings)
    pool = _pool_size(settings, store, n_agents, k)
    candidates = store.similarity_search(query, k=pool, filter=settings.retrieval_filter)  # best match first
    ids = [d.metadata["paper_id"] for d in candidates]
    stored = store.get(where={"paper_id": {"$in": ids}}, include=["embeddings", "metadatas"])
    emb_by_id = {m["paper_id"]: e for m, e in zip(stored["metadatas"], stored["embeddings"])}
    X = np.array([emb_by_id[i] for i in ids], dtype=float)

    n_clusters = min(n_agents, len(candidates))
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=settings.seed).fit_predict(X)
    groups = [[d for d, lab in zip(candidates, labels) if lab == c] for c in range(n_clusters)]
    groups.sort(key=lambda g: candidates.index(g[0]))               # group holding the best match first

    assigned = [g[:k] for g in groups] + [[] for _ in range(n_agents - n_clusters)]
    used = {d.metadata["paper_id"] for g in assigned for d in g}
    if any(len(g) < k for g in assigned):
        spare = [d for d in _mmr_order(query, settings, store, pool, pool) if d.metadata["paper_id"] not in used]
        for g in assigned:
            while len(g) < k and spare:
                g.append(spare.pop(0))
    return assigned


_STRATEGIES = {"shard": _shard, "mmr": _mmr, "cluster": _cluster}


def retrieve_diverse(
    query: str, settings: Settings, n_agents: int, k: int | None = None
) -> list[list[Document]]:
    """One list of papers per agent for round 0, chosen by `settings.retrieval_strategy`
    (see config.py): agents starting from the same dataset-derived query still end up
    grounded in different, non-overlapping papers instead of all converging on the same
    top-k."""
    return _STRATEGIES[settings.retrieval_strategy](query, settings, n_agents, k or settings.retrieval_k)


def format_for_prompt(docs: list[Document], settings: Settings) -> str:
    """Format retrieved papers for a prompt: each paper's JSON-summary text (the
    cheap match) plus an excerpt of its actual full text, hydrated on demand via
    pdf_extract.get_full_text (extracted once, then cached on disk)."""
    blocks = []
    for d in docs:
        paper_id = d.metadata.get("paper_id", "unknown")
        block = f"[{paper_id}]\n{d.page_content}"

        full_text = get_full_text(paper_id, settings)
        if full_text:
            excerpt = full_text[: settings.full_text_char_cap]
            if len(full_text) > settings.full_text_char_cap:
                excerpt += "\n...[truncated]"
            block += f"\n\nFull text excerpt:\n{excerpt}"
        else:
            block += "\n\n(full text PDF not found -- summary only)"

        blocks.append(block)
    return "\n\n---\n\n".join(blocks)
