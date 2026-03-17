#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# restore.sh — restore a claw backup tarball
#
# Extracts to a staging directory, shows a config diff, asks for confirmation,
# then applies the restore and restarts the claw service.
#
# Usage:
#   ./scripts/restore.sh ~/backups/claw/claw-backup-2026-03-13_120000.tar.gz
###############################################################################

# ── argument validation ─────────────────────────────────────────────────────

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <backup-tarball>"
    echo "Example: $0 ~/backups/claw/claw-backup-2026-03-13_120000.tar.gz"
    exit 1
fi

BACKUP_FILE="$1"

if [[ ! -f "${BACKUP_FILE}" ]]; then
    echo "[restore] ERROR: File not found: ${BACKUP_FILE}"
    exit 1
fi

# ── detect project root ─────────────────────────────────────────────────────

if [[ -f "${HOME}/claw/config.yaml" ]]; then
    PROJECT_ROOT="${HOME}/claw"
elif [[ -f "$(cd "$(dirname "$0")/.." && pwd)/config.yaml" ]]; then
    PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
else
    echo "[restore] ERROR: Cannot find project root (no config.yaml found)"
    exit 1
fi

echo "[restore] Project root: ${PROJECT_ROOT}"
echo "[restore] Backup file:  ${BACKUP_FILE}"

# ── extract to staging directory ─────────────────────────────────────────────

STAGING_DIR="$(mktemp -d /tmp/claw-restore-XXXXXX)"
trap 'rm -rf "${STAGING_DIR}"' EXIT

echo "[restore] Extracting backup to staging: ${STAGING_DIR}/"
tar -xzf "${BACKUP_FILE}" -C "${STAGING_DIR}"

echo "[restore] Staged contents:"
ls -la "${STAGING_DIR}/"

# ── config diff ──────────────────────────────────────────────────────────────

STAGED_CONFIG="${STAGING_DIR}/config.yaml"
CURRENT_CONFIG="${PROJECT_ROOT}/config.yaml"

if [[ -f "${STAGED_CONFIG}" ]]; then
    echo ""
    echo "[restore] ── config.yaml diff (current vs backup) ──"
    echo ""
    if diff --color=auto -u "${CURRENT_CONFIG}" "${STAGED_CONFIG}"; then
        echo "[restore] No differences in config.yaml"
    fi
    echo ""
else
    echo "[restore] No config.yaml in backup — current config will be preserved"
fi

# ── show data summary ────────────────────────────────────────────────────────

if [[ -d "${STAGING_DIR}/data" ]]; then
    STAGED_DATA_SIZE="$(du -sh "${STAGING_DIR}/data" | cut -f1)"
    echo "[restore] Backup data/ size: ${STAGED_DATA_SIZE}"
else
    echo "[restore] WARNING: No data/ directory in backup"
fi

# ── confirmation ─────────────────────────────────────────────────────────────

echo ""
echo "[restore] This will overwrite:"
if [[ -d "${STAGING_DIR}/data" ]]; then
    echo "           - ${PROJECT_ROOT}/data/ (except chromadb/)"
fi
if [[ -f "${STAGED_CONFIG}" ]]; then
    echo "           - ${PROJECT_ROOT}/config.yaml"
fi
echo ""

read -rp "[restore] Proceed with restore? [y/N] " confirm
if [[ "${confirm}" != "y" && "${confirm}" != "Y" ]]; then
    echo "[restore] Aborted by user"
    exit 0
fi

# ── apply restore ────────────────────────────────────────────────────────────

echo "[restore] Applying restore..."

if [[ -d "${STAGING_DIR}/data" ]]; then
    # Preserve chromadb/ if it exists in the current install (not in backups)
    echo "[restore] Restoring data/ (preserving existing data/chromadb/)..."
    rsync -a --delete \
        --exclude='chromadb/' \
        "${STAGING_DIR}/data/" \
        "${PROJECT_ROOT}/data/"
    echo "[restore] data/ restored"
fi

if [[ -f "${STAGED_CONFIG}" ]]; then
    echo "[restore] Restoring config.yaml..."
    cp "${STAGED_CONFIG}" "${CURRENT_CONFIG}"
    echo "[restore] config.yaml restored"
fi

# ── restart service ──────────────────────────────────────────────────────────

echo "[restore] Restarting claw service..."
if systemctl --user restart claw 2>/dev/null; then
    echo "[restore] Service restarted successfully"
else
    echo "[restore] WARNING: Could not restart claw service (may not be running as systemd user service here)"
    echo "[restore] If running on a different machine, restart manually on the target"
fi

echo "[restore] Restore complete"
