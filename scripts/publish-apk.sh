#!/usr/bin/env bash
# Publish a new APK to the Claw server for auto-updates.
#
# Usage: bash scripts/publish-apk.sh [version_name] [changelog]
# Example: bash scripts/publish-apk.sh "1.1.0" "Added QR scanning and auto-updates"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
APK_SOURCE="${PROJECT_ROOT}/claw-android/app/build/outputs/apk/debug/app-debug.apk"
APP_DIR="${PROJECT_ROOT}/data/remote/app"
VERSION_NAME="${1:-$(date +%Y.%m.%d)}"
CHANGELOG="${2:-Bug fixes and improvements}"

# Build the APK
echo "Building APK..."
cd "${PROJECT_ROOT}/claw-android"
./gradlew assembleDebug --quiet

if [[ ! -f "$APK_SOURCE" ]]; then
    echo "ERROR: APK not found at $APK_SOURCE"
    exit 1
fi

# Create app directory
mkdir -p "$APP_DIR"

# Extract version code from APK (increment based on existing)
CURRENT_CODE=0
if [[ -f "${APP_DIR}/version.json" ]]; then
    CURRENT_CODE=$(python3 -c "import json; print(json.load(open('${APP_DIR}/version.json')).get('version_code', 0))" 2>/dev/null || echo 0)
fi
NEW_CODE=$((CURRENT_CODE + 1))

# Copy APK
cp "$APK_SOURCE" "${APP_DIR}/claw.apk"
APK_SIZE=$(stat -c%s "${APP_DIR}/claw.apk")

# Write version info
cat > "${APP_DIR}/version.json" << EOF
{
    "version_code": ${NEW_CODE},
    "version_name": "${VERSION_NAME}",
    "size": ${APK_SIZE},
    "changelog": "${CHANGELOG}",
    "published_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

echo "Published APK v${VERSION_NAME} (code: ${NEW_CODE}, size: $(numfmt --to=iec $APK_SIZE))"
echo "  APK: ${APP_DIR}/claw.apk"
echo "  Version: ${APP_DIR}/version.json"
