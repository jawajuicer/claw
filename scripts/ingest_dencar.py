#!/usr/bin/env python3
"""Ingest Dencar Technology website content into Claw's ChromaDB memory.

Reads markdown files from a crawled website export and stores them as
scoped facts in the 'dencar' scope. Content is chunked to stay within
embedding model limits (~256 tokens per chunk).

Usage:
    python scripts/ingest_dencar.py [--source PATH] [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
import uuid
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from claw.memory_engine.store import MemoryStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

DEFAULT_SOURCE = Path.home() / "Documents/Projects/website_crawlscrape/output_inhouse/www.dencartechnology.com"
SCOPE = "dencar"
MIN_CONTENT_LENGTH = 50  # skip files with less meaningful content
CHUNK_SIZE = 800  # characters per chunk (~200 tokens for sentence-transformers)
CHUNK_OVERLAP = 100  # overlap between chunks for context continuity

# Directories to skip
SKIP_DIRS = {"images", "low_value"}


def extract_metadata(content: str) -> tuple[str, str, str]:
    """Extract source URL, title, and body from a markdown file."""
    source = ""
    title = ""
    lines = content.split("\n")
    body_start = 0

    for i, line in enumerate(lines):
        if line.startswith("# Source:"):
            source = line.removeprefix("# Source:").strip()
            body_start = i + 1
        elif line.startswith("# Title:"):
            title = line.removeprefix("# Title:").strip()
            body_start = i + 1
        elif line.startswith("# ") and not title:
            title = line.removeprefix("# ").strip()
            body_start = i + 1
        else:
            break

    body = "\n".join(lines[body_start:]).strip()
    return source, title, body


def clean_text(text: str) -> str:
    """Clean markdown artifacts for better embedding quality."""
    # Remove image references
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # Remove excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove escaped characters
    text = text.replace("\\*", "*")
    return text.strip()


def chunk_text(text: str, title: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, preserving paragraph boundaries."""
    if len(text) <= chunk_size:
        prefix = f"[{title}] " if title else ""
        return [f"{prefix}{text}"]

    # Split on paragraph boundaries
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
            # Keep overlap from end of previous chunk
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


def collect_files(source_dir: Path) -> list[Path]:
    """Collect all meaningful markdown files from the source directory."""
    files = []
    for md_file in sorted(source_dir.rglob("*.md")):
        # Skip files in excluded directories
        rel_parts = md_file.relative_to(source_dir).parts
        if any(part in SKIP_DIRS for part in rel_parts):
            continue
        # Skip tiny files
        if md_file.stat().st_size < MIN_CONTENT_LENGTH:
            continue
        files.append(md_file)
    return files


def make_fact_id(file_path: str, chunk_idx: int) -> str:
    """Create a deterministic fact ID from file path and chunk index.

    Uses the file path (not source URL) to guarantee uniqueness.
    This allows re-running the script without creating duplicates.
    """
    key = f"dencar:{file_path}:{chunk_idx}"
    return f"dencar-{hashlib.sha256(key.encode()).hexdigest()[:16]}"


def main():
    parser = argparse.ArgumentParser(description="Ingest Dencar website into Claw memory")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Source directory")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing to ChromaDB")
    args = parser.parse_args()

    if not args.source.is_dir():
        log.error("Source directory not found: %s", args.source)
        sys.exit(1)

    files = collect_files(args.source)
    log.info("Found %d markdown files in %s", len(files), args.source)

    if not files:
        log.warning("No files to process")
        sys.exit(0)

    # Prepare all chunks
    all_chunks: list[tuple[str, str, dict]] = []  # (fact_id, text, metadata)
    skipped = 0

    for md_file in files:
        content = md_file.read_text(encoding="utf-8", errors="replace")
        source, title, body = extract_metadata(content)
        body = clean_text(body)

        if len(body) < MIN_CONTENT_LENGTH:
            skipped += 1
            continue

        chunks = chunk_text(body, title)
        for idx, chunk in enumerate(chunks):
            fact_id = make_fact_id(str(md_file), idx)
            metadata = {
                "source": source,
                "title": title,
                "chunk": idx,
                "total_chunks": len(chunks),
                "file": md_file.name,
            }
            all_chunks.append((fact_id, chunk, metadata))

    log.info("Prepared %d chunks from %d files (skipped %d empty/tiny)", len(all_chunks), len(files) - skipped, skipped)

    if args.dry_run:
        log.info("DRY RUN — showing first 5 chunks:")
        for fact_id, text, meta in all_chunks[:5]:
            log.info("  [%s] %s... (%d chars)", fact_id[:20], text[:80], len(text))
        log.info("Would insert %d facts into scope '%s'", len(all_chunks), SCOPE)
        return

    # Initialize memory store and ingest
    log.info("Initializing memory store...")
    store = MemoryStore()
    store.initialize()

    existing = store.facts.count()
    log.info("Existing facts in store: %d", existing)

    # Batch upsert for performance
    BATCH_SIZE = 100
    inserted = 0

    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i : i + BATCH_SIZE]
        ids = [c[0] for c in batch]
        documents = [c[1] for c in batch]
        metadatas = [{**c[2], "scope": SCOPE} for c in batch]

        store.facts.upsert(ids=ids, documents=documents, metadatas=metadatas)
        inserted += len(batch)
        log.info("  Ingested %d / %d chunks", inserted, len(all_chunks))

    final_count = store.facts.count()
    log.info("Done! Facts: %d -> %d (added %d Dencar chunks)", existing, final_count, final_count - existing)


if __name__ == "__main__":
    main()
