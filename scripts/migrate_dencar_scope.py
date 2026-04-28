#!/usr/bin/env python3
"""Re-tag facts from one scope to another in-place.

Default: dencar -> shared, so cross-profile bridge users can see Dencar facts.
Idempotent: re-running has no effect.

Usage:
    python scripts/migrate_dencar_scope.py --dry-run
    python scripts/migrate_dencar_scope.py --from-scope dencar --to-scope shared
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from claw.memory_engine.store import MemoryStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

QUERY_LIMIT = 10000  # 1,550 Dencar chunks fit comfortably in one call
UPDATE_BATCH_SIZE = 500  # safety: chunk update calls into batches


def count_scope(store: MemoryStore, scope: str) -> int:
    """Count facts at a given scope by paging through .get() results."""
    total = 0
    offset = 0
    page = 10000
    while True:
        result = store.facts.get(
            where={"scope": scope},
            limit=page,
            offset=offset,
            include=[],
        )
        ids = result.get("ids") or []
        total += len(ids)
        if len(ids) < page:
            break
        offset += page
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-tag facts in ChromaDB from one scope to another, in-place."
    )
    parser.add_argument(
        "--from-scope",
        default="dencar",
        help="Existing scope value to migrate from (default: dencar)",
    )
    parser.add_argument(
        "--to-scope",
        default="shared",
        help="New scope value to assign (default: shared)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the migration without writing to ChromaDB",
    )
    args = parser.parse_args()

    from_scope = args.from_scope
    to_scope = args.to_scope

    if from_scope == to_scope:
        log.error("--from-scope and --to-scope are identical (%r); nothing to do.", from_scope)
        sys.exit(1)

    log.info("Initializing memory store...")
    store = MemoryStore()
    store.initialize()

    log.info("Querying facts at scope=%r (limit=%d)...", from_scope, QUERY_LIMIT)
    result = store.facts.get(
        where={"scope": from_scope},
        limit=QUERY_LIMIT,
        include=["metadatas"],
    )
    ids = list(result.get("ids") or [])
    metadatas = list(result.get("metadatas") or [])

    if not ids:
        log.info("Nothing to migrate: 0 facts found at scope=%r.", from_scope)
        return

    if len(ids) >= QUERY_LIMIT:
        log.warning(
            "Query returned %d facts (>= limit %d). There may be more — increase QUERY_LIMIT.",
            len(ids),
            QUERY_LIMIT,
        )

    # Cache "before" counts for both scopes (single source of truth for reporting).
    before_from = len(ids)  # we already paged the from-scope above
    before_to = count_scope(store, to_scope)

    # Build new metadata dicts: preserve all existing fields, replace scope.
    new_metadatas: list[dict] = []
    for meta in metadatas:
        new_meta = dict(meta) if meta else {}
        new_meta["scope"] = to_scope
        new_metadatas.append(new_meta)

    if args.dry_run:
        log.info(
            "DRY RUN — would update %d facts: scope=%r -> scope=%r",
            len(ids),
            from_scope,
            to_scope,
        )
        log.info("Current counts: scope=%r -> %d, scope=%r -> %d", from_scope, before_from, to_scope, before_to)
        log.info("Sample (first 3 ids): %s", ids[:3])
        return

    log.info(
        "Updating %d facts in batches of %d: scope=%r -> scope=%r",
        len(ids),
        UPDATE_BATCH_SIZE,
        from_scope,
        to_scope,
    )
    updated = 0
    for i in range(0, len(ids), UPDATE_BATCH_SIZE):
        batch_ids = ids[i : i + UPDATE_BATCH_SIZE]
        batch_metas = new_metadatas[i : i + UPDATE_BATCH_SIZE]
        store.facts.update(ids=batch_ids, metadatas=batch_metas)
        updated += len(batch_ids)
        log.info("  Updated %d / %d", updated, len(ids))

    after_from = count_scope(store, from_scope)
    after_to = count_scope(store, to_scope)

    log.info("Migration complete.")
    log.info("  scope=%r: %d -> %d", from_scope, before_from, after_from)
    log.info("  scope=%r: %d -> %d", to_scope, before_to, after_to)


if __name__ == "__main__":
    main()
