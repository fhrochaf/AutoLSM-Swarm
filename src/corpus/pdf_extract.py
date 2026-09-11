"""One-time PDF text extraction + on-disk cache, keyed by paper_id -- the same ids
the JSON summaries in corpus/ingest.py use (paper PDFs and JSON summaries share the
same filename stem, e.g. 2-s2.0-105000159569.{pdf,json}).

The JSON summaries are a cheap metadata pre-filter (corpus/retrieve.py does the
similarity search over them); this module hydrates the top matches with an excerpt
of the actual paper text, extracted once via pypdf and cached so repeated rounds/
agents never re-parse the same PDF.
"""
from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from config import Settings


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 - a single unparsable page shouldn't lose the rest
            continue
    return "\n".join(pages).strip()


def get_full_text(paper_id: str, settings: Settings) -> str | None:
    """A paper's full extracted text, using (and populating) the on-disk cache.
    Returns None if no matching PDF exists under settings.publications_dir."""
    cache_path = settings.pdf_text_cache_dir / f"{paper_id}.txt"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    pdf_path = settings.publications_dir / f"{paper_id}.pdf"
    if not pdf_path.exists():
        return None

    text = extract_text(pdf_path)
    settings.pdf_text_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")
    return text


def build_full_text_cache(settings: Settings) -> int:
    """Eagerly extract+cache every PDF's text (a maintenance/warm-up utility, not
    called automatically -- extraction is otherwise lazy, on first retrieval).
    Returns the number of PDFs newly cached."""
    settings.pdf_text_cache_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for pdf_path in sorted(settings.publications_dir.glob("*.pdf")):
        cache_path = settings.pdf_text_cache_dir / f"{pdf_path.stem}.txt"
        if cache_path.exists():
            continue
        cache_path.write_text(extract_text(pdf_path), encoding="utf-8")
        count += 1
    return count
