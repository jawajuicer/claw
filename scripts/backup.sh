#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# backup.sh — create a timestamped backup of data/ + config.yaml
#
# Excludes data/chromadb/ (regenerable embedding cache).
# Keeps backups for 7 days, then prunes older ones.
#
# Auto-detects whether it's running locally or on the remote server:
#   - If ~/claw exists and contains config.yaml, uses ~/claw as project root
#   - Otherwise uses the parent directory of scripts/
#
# Usage:
#   ./scripts/backup.sh
###############################################################################

BACKUP_BASE="${HOME}/backups/claw"
TIMESTAMP="$(date '+%Y-%m-%d_%H%M%S')"
BACKUP_NAME="claw-backup-${TIMESTAMP}.tar.gz"
RETENTION_DAYS=7

# ── detect project root ─────────────────────────────────────────────────────

if [[ -f "${HOME}/claw/config.yaml" ]]; then
    PROJECT_ROOT="${HOME}/claw"
elif [[ -f "$(cd "$(dirname "$0")/.." && pwd)/config.yaml" ]]; then
    PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
else
    echo "[backup] ERROR: Cannot find project root (no config.yaml found)"
    echo "[backup] Expected either ~/claw/config.yaml or ../config.yaml relative to this script"
    exit 1
fi

echo "[backup] Project root: ${PROJECT_ROOT}"

# ── validate required paths ─────────────────────────────────────────────────

if [[ ! -d "${PROJECT_ROOT}/data" ]]; then
    echo "[backup] ERROR: ${PROJECT_ROOT}/data/ does not exist — nothing to back up"
    exit 1
fi

if [[ ! -f "${PROJECT_ROOT}/config.yaml" ]]; then
    echo "[backup] WARNING: config.yaml not found, backing up data/ only"
fi

# ── create backup directory ──────────────────────────────────────────────────

mkdir -p "${BACKUP_BASE}"
echo "[backup] Backup destination: ${BACKUP_BASE}/${BACKUP_NAME}"

# ── build tarball ────────────────────────────────────────────────────────────

TAR_ARGS=(
    -czf "${BACKUP_BASE}/${BACKUP_NAME}"
    -C "${PROJECT_ROOT}"
    --exclude='data/chromadb'
)

# Include data/ always, config.yaml if it exists
TAR_TARGETS=("data/")
if [[ -f "${PROJECT_ROOT}/config.yaml" ]]; then
    TAR_TARGETS+=("config.yaml")
fi

echo "[backup] Creating tarball (excluding data/chromadb/)..."
tar "${TAR_ARGS[@]}" "${TAR_TARGETS[@]}"

BACKUP_SIZE="$(du -h "${BACKUP_BASE}/${BACKUP_NAME}" | cut -f1)"
echo "[backup] Backup created: ${BACKUP_BASE}/${BACKUP_NAME} (${BACKUP_SIZE})"

# ── prune old backups ────────────────────────────────────────────────────────

echo "[backup] Pruning backups older than ${RETENTION_DAYS} days..."

PRUNED=0
while IFS= read -r old_backup; do
    echo "[backup]   Deleting: $(basename "${old_backup}")"
    rm -f "${old_backup}"
    PRUNED=$(( PRUNED + 1 ))
done < <(find "${BACKUP_BASE}" -name 'claw-backup-*.tar.gz' -mtime "+${RETENTION_DAYS}" -type f)

if (( PRUNED == 0 )); then
    echo "[backup] No old backups to prune"
else
    echo "[backup] Pruned ${PRUNED} old backup(s)"
fi

# ── summary ──────────────────────────────────────────────────────────────────

TOTAL="$(find "${BACKUP_BASE}" -name 'claw-backup-*.tar.gz' -type f | wc -l)"
echo "[backup] Done. ${TOTAL} backup(s) in ${BACKUP_BASE}/"
