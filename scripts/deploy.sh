#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# deploy.sh — rsync claw source to remote server and restart the service
#
# Usage:
#   ./scripts/deploy.sh        # full deploy
#   ./scripts/deploy.sh -n     # dry-run (rsync --dry-run, no restart)
###############################################################################

REMOTE_USER="amd370"
REMOTE_HOST="10.7.8.245"
REMOTE_DIR="~/claw"
HEALTH_URL="http://${REMOTE_HOST}:8080/api/health"
HEALTH_TIMEOUT=30

SSH_OPTS="-o PreferredAuthentications=password -o PubkeyAuthentication=no -o StrictHostKeyChecking=no"
SSH_CMD="sshpass -p 'lolumadbro1' ssh ${SSH_OPTS} ${REMOTE_USER}@${REMOTE_HOST}"

# Resolve project root (parent of scripts/)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

DRY_RUN=false
if [[ "${1:-}" == "-n" ]]; then
    DRY_RUN=true
    echo "[deploy] DRY-RUN mode — no files will be transferred, no service restart"
fi

# ── rsync source to remote ──────────────────────────────────────────────────

RSYNC_EXCLUDES=(
    --exclude='.venv'
    --exclude='__pycache__'
    --exclude='*.pyc'
    --exclude='.git'
    --exclude='data/'
    --exclude='.env'
    --exclude='claw-android/'
)

RSYNC_FLAGS=(-avz --delete "${RSYNC_EXCLUDES[@]}")

if $DRY_RUN; then
    RSYNC_FLAGS+=(--dry-run)
fi

echo "[deploy] Syncing ${PROJECT_ROOT}/ -> ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

SSHPASS='lolumadbro1' rsync \
    -e "sshpass -e ssh ${SSH_OPTS}" \
    "${RSYNC_FLAGS[@]}" \
    "${PROJECT_ROOT}/" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

echo "[deploy] rsync complete"

if $DRY_RUN; then
    echo "[deploy] Dry-run finished — skipping service restart and health check"
    exit 0
fi

# ── restart claw service ────────────────────────────────────────────────────

echo "[deploy] Restarting claw service on remote..."
SSHPASS='lolumadbro1' sshpass -e ssh ${SSH_OPTS} \
    "${REMOTE_USER}@${REMOTE_HOST}" \
    "systemctl --user restart claw"

echo "[deploy] Service restart issued"

# ── health check ────────────────────────────────────────────────────────────

echo "[deploy] Polling ${HEALTH_URL} (timeout: ${HEALTH_TIMEOUT}s)..."

elapsed=0
while (( elapsed < HEALTH_TIMEOUT )); do
    if curl -sf --max-time 2 "${HEALTH_URL}" > /dev/null 2>&1; then
        echo "[deploy] Health check passed after ${elapsed}s"
        exit 0
    fi
    sleep 2
    elapsed=$(( elapsed + 2 ))
done

echo "[deploy] ERROR: Health check failed after ${HEALTH_TIMEOUT}s"
echo "[deploy] Check remote logs: journalctl --user -u claw -n 50"
exit 1
