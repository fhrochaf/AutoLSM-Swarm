"""Build the local RAG index from the per-paper JSON method/dataset summaries.

Each JSON (one per paper, keyed by its filename stem, e.g. a Scopus EID) has the
shape: {"datasets": [...], "methods": [{"name","method_type","summary",...}, ...],
"availability": {...}, "reproducibility_status": str, "reproducibility_assessment": str}.
We flatten each paper into one text blob (for embedding) plus flat metadata (for
filtering), and persist a Chroma collection agents can retrieve from.
"""
from __future__ import annotations

import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import Settings


def _paper_to_document(paper_id: str, data: dict, verbose: bool = False) -> Document:
    methods = data.get("methods") or []
    datasets = data.get("datasets") or []

    method_lines = [
        f"- {m.get('name', 'unnamed method')} ({m.get('method_type', 'unknown')}): "
        f"{m.get('summary', '')}"
        for m in methods
    ]
    dataset_lines = [
        f"- {d.get('name', 'unnamed dataset')} (source: {d.get('source', 'unknown')})"
        for d in datasets
    ]

    text = (
        f"Paper {paper_id}\n\n"
        f"Methods:\n" + ("\n".join(method_lines) or "(none extracted)") + "\n\n"
        f"Datasets used:\n" + ("\n".join(dataset_lines) or "(none extracted)") + "\n\n"
    )

    metadata = {
        "paper_id": paper_id,
        "method_names": "; ".join(m.get("name", "") for m in methods),
        "dataset_names": "; ".join(d.get("name", "") for d in datasets),
        "reproducibility_status": data.get("reproducibility_status", "UNKNOWN"),
        "n_methods": len(methods),
        "n_datasets": len(datasets),
    }
    if verbose:
        print(f"Doc {paper_id}:\n{text}")

    return Document(page_content=text, metadata=metadata)


def load_corpus_summaries(corpus_json_dir: Path, verbose: bool = False) -> list[Document]:
    docs = []
    for path in sorted(corpus_json_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        docs.append(_paper_to_document(path.stem, data, verbose))
    return docs


def get_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def build_index(settings: Settings, verbose: bool = False) -> int:
    """(Re)build the persisted Chroma collection from CORPUS_JSON_DIR. Returns doc count."""
    docs = load_corpus_summaries(settings.corpus_json_dir, verbose)
    if not docs:
        raise RuntimeError(
            f"No paper JSON summaries found under {settings.corpus_json_dir}. "
            "Check config.corpus_json_dir / the CORPUS_JSON_DIR env var."
        )

    settings.vector_store_dir.mkdir(parents=True, exist_ok=True)
    Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(settings),
        persist_directory=str(settings.vector_store_dir),
        collection_name="landslide_auto_mapping_corpus",
    )
    return len(docs)
