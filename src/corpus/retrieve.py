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
    store = get_vectorstore(settings)
    return store.similarity_search(query, k=k or settings.retrieval_k, filter=filters)


def retrieve_diverse(
    query: str, settings: Settings, n_agents: int, k: int | None = None
) -> list[list[Document]]:
    """Retrieve one shared candidate pool for `query`, then split them across
    `n_agents` slots so agents starting from the same dataset-derived query still end up
    grounded in different, non-overlapping papers instead of all converging on the same
    top-k."""
    k = k or settings.retrieval_k
    store = get_vectorstore(settings)
    corpus_size = store._collection.count()
    pool_size = min(k * n_agents, corpus_size)
    candidates = store.similarity_search(query, k=pool_size)
    return [candidates[i::n_agents][:k] for i in range(n_agents)]


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
