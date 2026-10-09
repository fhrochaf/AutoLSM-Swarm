"""Build the local RAG index from the per-paper JSON method/dataset summaries.

Each JSON (one per paper, keyed by its filename stem, e.g. a Scopus EID) has the
shape: {"datasets": [...], "methods": [{"name","method_type","summary",...}, ...],
"availability": {...}, "reproducibility_status": str, "reproducibility_assessment": str}.
We flatten each paper into one text blob (for embedding) plus flat metadata (for
filtering), and persist a Chroma collection agents can retrieve from.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import Settings


# CSV column -> (metadata key, default). Chroma rejects None metadata, so every paper gets a value:
# "UNKNOWN" for text, 0 for numbers (so e.g. {"year": {"$gte": 2022}} excludes unmatched papers).
_CSV_FIELDS = {
    "method_class_name": ("method_class_name", "UNKNOWN"),
    "Year": ("year", 0),
    "Cited by": ("cited_by", 0),
    "Source title": ("source_title", "UNKNOWN"),
    "Document Type": ("document_type", "UNKNOWN"),
}


def load_csv_metadata(csv_path: Path | None) -> dict[str, dict]:
    """EID -> flat metadata from the Scopus-export CSV (';'-delimited). {} if no CSV is configured."""
    if csv_path is None:
        return {}
    if not csv_path.exists():
        raise FileNotFoundError(
            f"corpus_metadata_csv not found: {csv_path} (set config.corpus_metadata_csv, or None to skip the join)"
        )
    try:
        text = csv_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = csv_path.read_text(encoding="latin-1")

    out: dict[str, dict] = {}
    for row in csv.DictReader(io.StringIO(text), delimiter=";"):
        eid = (row.get("EID") or "").strip()
        if not eid:
            continue
        meta = {}
        for col, (key, default) in _CSV_FIELDS.items():
            value = (row.get(col) or "").strip()
            if isinstance(default, int):
                try:
                    meta[key] = int(float(value))
                except ValueError:
                    meta[key] = default
            else:
                meta[key] = value or default
        out[eid] = meta
    return out


def _paper_to_document(
    paper_id: str, data: dict, csv_meta: dict[str, dict] | None = None, verbose: bool = False
) -> Document:
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
    if csv_meta is not None:
        metadata.update(csv_meta.get(paper_id) or {key: default for key, default in _CSV_FIELDS.values()})
    if verbose:
        print(f"Doc {paper_id}:\n{text}")

    return Document(page_content=text, metadata=metadata)


def load_corpus_summaries(
    corpus_json_dir: Path, csv_path: Path | None = None, verbose: bool = False
) -> list[Document]:
    csv_meta = load_csv_metadata(csv_path) if csv_path is not None else None
    docs = []
    for path in sorted(corpus_json_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        docs.append(_paper_to_document(path.stem, data, csv_meta, verbose))
    if csv_meta is not None:
        matched = sum(d.metadata["paper_id"] in csv_meta for d in docs)
        print(f"Joined {matched}/{len(docs)} paper JSONs to {csv_path.name} on EID.")
    return docs


def get_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def build_index(settings: Settings, verbose: bool = False) -> int:
    """(Re)build the persisted Chroma collection from CORPUS_JSON_DIR. Returns doc count."""
    docs = load_corpus_summaries(settings.corpus_json_dir, settings.corpus_metadata_csv, verbose)
    if not docs:
        raise RuntimeError(
            f"No paper JSON summaries found under {settings.corpus_json_dir}. "
            "Check config.corpus_json_dir / the CORPUS_JSON_DIR env var."
        )

    settings.vector_store_dir.mkdir(parents=True, exist_ok=True)
    # Drop any existing collection first: from_documents appends, so a rebuild (e.g. to pick up new
    # metadata) would otherwise duplicate every paper.
    Chroma(
        embedding_function=get_embeddings(settings),
        persist_directory=str(settings.vector_store_dir),
        collection_name="landslide_auto_mapping_corpus",
    ).delete_collection()
    Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(settings),
        persist_directory=str(settings.vector_store_dir),
        collection_name="landslide_auto_mapping_corpus",
    )
    return len(docs)
