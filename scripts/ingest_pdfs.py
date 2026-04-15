#!/usr/bin/env python3
"""Ingest PDF documents into Claw's ChromaDB memory.

Extracts text from PDF files using pymupdf. For pages with images but little
text (diagrams, screenshots, scanned content), uses Gemini vision to describe
the visual content. Stores everything as scoped facts in ChromaDB.

Usage:
    python scripts/ingest_pdfs.py [--dry-run] [DIR ...]
    python scripts/ingest_pdfs.py --scope elka /path/to/ElkaGates
    python scripts/ingest_pdfs.py --no-vision  # skip Gemini image analysis
    python scripts/ingest_pdfs.py  # uses default directories
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import sys
import time
from pathlib import Path

import pymupdf

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from claw.memory_engine.store import MemoryStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DEFAULT_DIRS = [
    Path.home() / "Downloads/ElkaGates",
    Path.home() / "Downloads/Dencar - Random PDFs For Claw",
]
DEFAULT_SCOPE = "dencar"
MIN_CONTENT_LENGTH = 50
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Gemini vision config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# gemini-2.0-flash: free tier 15 RPM / 1500 req/day (vs 2.5-flash: 20/day)
GEMINI_MODEL = "gemini-2.0-flash"
# Pages with fewer chars than this AND images will be sent to Gemini vision
IMAGE_TEXT_THRESHOLD = 100
# Rate limit: 15 RPM free tier -> 4s between requests (safe margin)
GEMINI_DELAY = 4.5
MAX_RETRIES = 3


def init_gemini():
    """Initialize Gemini client for vision analysis."""
    api_key = GEMINI_API_KEY
    if not api_key:
        try:
            from claw.secret_store import load as secret_load
            api_key = secret_load("gemini_api_key") or ""
        except Exception:
            pass
    if not api_key:
        log.warning("No Gemini API key found (set GEMINI_API_KEY env var or store in secret store)")
        return None
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        log.info("Gemini vision initialized (model: %s)", GEMINI_MODEL)
        return client
    except Exception as e:
        log.warning("Gemini unavailable, images will be skipped: %s", e)
        return None


def describe_page_image(client, pdf_path: Path, page_num: int, pdf_title: str) -> str | None:
    """Use Gemini vision to describe the visual content of a PDF page.

    Retries on 429 rate limit errors with exponential backoff.
    """
    from google import genai
    try:
        doc = pymupdf.open(str(pdf_path))
        page = doc[page_num - 1]  # page_num is 1-indexed
        # Render page to PNG at 150 DPI (good balance of quality vs size)
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        doc.close()

        prompt = (
            f"This is page {page_num} from a PDF document titled '{pdf_title}'. "
            "This page contains diagrams, screenshots, or other visual content. "
            "Describe ALL the information shown in detail — text in images, labels, "
            "diagrams, wiring details, UI elements, step-by-step instructions, tables, "
            "and any technical specifications visible. Be thorough and factual. "
            "Format as plain text paragraphs."
        )

        for attempt in range(MAX_RETRIES):
            try:
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[
                        prompt,
                        genai.types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                    ],
                )
                if response.text and len(response.text.strip()) > MIN_CONTENT_LENGTH:
                    return response.text.strip()
                return None
            except Exception as api_err:
                err_str = str(api_err)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait = GEMINI_DELAY * (2 ** attempt)
                    try:
                        if "retryDelay" in err_str:
                            # Parse retry delay suggestion
                            idx = err_str.find("retryDelay")
                            snippet = err_str[idx:idx+50]
                            secs = float(''.join(c for c in snippet.split("'")[1] if c.isdigit() or c == '.').rstrip('.'))
                            wait = max(wait, secs + 1)
                    except (ValueError, IndexError):
                        pass
                    if attempt < MAX_RETRIES - 1:
                        log.info("    Rate limited, waiting %.0fs (attempt %d/%d)...", wait, attempt + 1, MAX_RETRIES)
                        time.sleep(wait)
                        continue
                raise

    except Exception as e:
        log.warning("  Gemini vision failed for %s p%d: %s", pdf_path.name, page_num, e)
        return None


def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    """Extract text and image info from each page of a PDF.

    Returns list of dicts with keys: page_num, text, has_images, text_length.
    """
    pages = []
    try:
        doc = pymupdf.open(str(pdf_path))
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()
            images = page.get_images()
            pages.append({
                "page_num": page_num + 1,
                "text": text,
                "has_images": len(images) > 0,
                "text_length": len(text),
            })
        doc.close()
    except Exception as e:
        log.warning("Failed to read %s: %s", pdf_path.name, e)
    return pages


def clean_pdf_text(text: str) -> str:
    """Clean extracted PDF text for better embedding quality."""
    text = re.sub(r"[ \t]{3,}", "  ", text)
    text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.replace("\f", "")
    return text.strip()


def chunk_text(text: str, title: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, preserving paragraph boundaries."""
    if len(text) <= chunk_size:
        prefix = f"[{title}] " if title else ""
        return [f"{prefix}{text}"]

    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) + 2 > chunk_size and current_chunk:
            prefix = f"[{title}] " if title else ""
            chunks.append(f"{prefix}{current_chunk.strip()}")
            if overlap > 0:
                current_chunk = current_chunk[-overlap:] + "\n\n" + para
            else:
                current_chunk = para
        else:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para

    if current_chunk.strip():
        prefix = f"[{title}] " if title else ""
        chunks.append(f"{prefix}{current_chunk.strip()}")

    return chunks


def make_fact_id(file_path: str, page: int, chunk_idx: int) -> str:
    """Deterministic fact ID from file path, page number, and chunk index."""
    key = f"pdf:{file_path}:p{page}:c{chunk_idx}"
    return f"pdf-{hashlib.sha256(key.encode()).hexdigest()[:16]}"


def collect_pdfs(dirs: list[Path]) -> list[Path]:
    """Collect all PDF files from the given directories."""
    files = []
    for d in dirs:
        if not d.is_dir():
            log.warning("Directory not found, skipping: %s", d)
            continue
        for pdf in sorted(d.glob("*.pdf")):
            files.append(pdf)
    seen = set()
    unique = []
    for f in files:
        if f.name not in seen:
            seen.add(f.name)
            unique.append(f)
        else:
            log.info("Skipping duplicate: %s", f)
    return unique


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into Claw memory")
    parser.add_argument("dirs", nargs="*", type=Path, help="Directories containing PDFs")
    parser.add_argument("--scope", default=DEFAULT_SCOPE, help="Memory scope (default: dencar)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing to ChromaDB")
    parser.add_argument("--no-vision", action="store_true", help="Skip Gemini image analysis")
    args = parser.parse_args()

    source_dirs = args.dirs if args.dirs else DEFAULT_DIRS
    pdfs = collect_pdfs(source_dirs)
    log.info("Found %d PDFs across %d directories", len(pdfs), len(source_dirs))

    if not pdfs:
        log.warning("No PDFs to process")
        sys.exit(0)

    # Initialize Gemini for image-heavy pages
    gemini_client = None if args.no_vision else init_gemini()

    all_chunks: list[tuple[str, str, dict]] = []
    skipped_files = 0
    total_pages = 0
    vision_pages = 0

    for pdf_path in pdfs:
        title = pdf_path.stem
        log.info("Processing: %s", pdf_path.name)

        pages = extract_pdf_pages(pdf_path)
        if not pages:
            log.warning("  No pages found in %s", pdf_path.name)
            skipped_files += 1
            continue

        total_pages += len(pages)

        # Phase 1: Use Gemini vision for pages with images but little/no text
        vision_texts: dict[int, str] = {}  # page_num -> vision description
        if gemini_client:
            pages_needing_vision = [
                p for p in pages
                if p["has_images"] and p["text_length"] < IMAGE_TEXT_THRESHOLD
            ]
            if pages_needing_vision:
                log.info("  Sending %d image-heavy pages to Gemini vision...", len(pages_needing_vision))
                for p in pages_needing_vision:
                    desc = describe_page_image(gemini_client, pdf_path, p["page_num"], title)
                    if desc:
                        vision_texts[p["page_num"]] = desc
                        vision_pages += 1
                        log.info("    p%d: got %d chars from vision", p["page_num"], len(desc))
                    time.sleep(GEMINI_DELAY)

        # Phase 2: Combine text from all pages (text extraction + vision)
        page_contents: list[tuple[int, str]] = []
        for p in pages:
            text_parts = []
            # Add extracted text if meaningful
            if p["text_length"] >= MIN_CONTENT_LENGTH:
                text_parts.append(clean_pdf_text(p["text"]))
            # Add vision description if we got one
            if p["page_num"] in vision_texts:
                vision_desc = vision_texts[p["page_num"]]
                text_parts.append(f"[Visual content] {vision_desc}")
            combined = "\n\n".join(text_parts)
            if combined:
                page_contents.append((p["page_num"], combined))

        if not page_contents:
            log.warning("  No content extracted from %s", pdf_path.name)
            skipped_files += 1
            continue

        # Phase 3: Group nearby pages and chunk
        page_groups: list[tuple[int, str]] = []
        current_text = ""
        first_page = page_contents[0][0]

        for page_num, page_text in page_contents:
            if len(current_text) + len(page_text) > CHUNK_SIZE * 3:
                if current_text:
                    page_groups.append((first_page, current_text))
                current_text = page_text
                first_page = page_num
            else:
                current_text = current_text + "\n\n" + page_text if current_text else page_text

        if current_text:
            page_groups.append((first_page, current_text))

        for start_page, group_text in page_groups:
            chunks = chunk_text(group_text, title)
            for idx, chunk in enumerate(chunks):
                if len(chunk.strip()) < MIN_CONTENT_LENGTH:
                    continue
                fact_id = make_fact_id(str(pdf_path), start_page, idx)
                metadata = {
                    "source": f"pdf://{pdf_path.name}",
                    "title": title,
                    "page": start_page,
                    "chunk": idx,
                    "total_chunks": len(chunks),
                    "file": pdf_path.name,
                    "type": "pdf",
                }
                all_chunks.append((fact_id, chunk, metadata))

        file_chunks = sum(1 for _, _, m in all_chunks if m["file"] == pdf_path.name)
        log.info("  %d pages -> %d chunks", len(page_contents), file_chunks)

    log.info("Prepared %d total chunks from %d PDFs (%d pages, %d vision-analyzed, skipped %d files)",
             len(all_chunks), len(pdfs) - skipped_files, total_pages, vision_pages, skipped_files)

    if args.dry_run:
        log.info("DRY RUN — showing first 5 chunks:")
        for fact_id, text, meta in all_chunks[:5]:
            log.info("  [%s] p%s: %s... (%d chars)",
                     meta["file"][:30], meta["page"], text[:80], len(text))
        log.info("Would insert %d facts into scope '%s'", len(all_chunks), args.scope)
        return

    log.info("Initializing memory store...")
    store = MemoryStore()
    store.initialize()

    existing = store.facts.count()
    log.info("Existing facts in store: %d", existing)

    BATCH_SIZE = 100
    inserted = 0

    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i : i + BATCH_SIZE]
        ids = [c[0] for c in batch]
        documents = [c[1] for c in batch]
        metadatas = [{**c[2], "scope": args.scope} for c in batch]

        store.facts.upsert(ids=ids, documents=documents, metadatas=metadatas)
        inserted += len(batch)
        log.info("  Ingested %d / %d chunks", inserted, len(all_chunks))

    final_count = store.facts.count()
    log.info("Done! Facts: %d -> %d (added %d PDF chunks)", existing, final_count, final_count - existing)


if __name__ == "__main__":
    main()
