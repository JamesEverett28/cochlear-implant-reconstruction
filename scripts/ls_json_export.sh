#!/usr/bin/env bash

set -euo pipefail

LABEL_STUDIO_URL="http://localhost:8080"
PROJECT_ID=19
OUTPUT="datasets/side1-insertion/label-studio.json"

: "${LABEL_STUDIO_REFRESH_TOKEN:?LABEL_STUDIO_REFRESH_TOKEN is not set}"

ACCESS_TOKEN=$(
    curl --fail --silent --show-error \
        -X POST \
        "$LABEL_STUDIO_URL/api/token/refresh" \
        -H "Content-Type: application/json" \
        -d "{\"refresh\":\"$LABEL_STUDIO_REFRESH_TOKEN\"}" |
    python3 -c 'import json, sys; print(json.load(sys.stdin)["access"])'
)

curl --fail --silent --show-error \
    -H "Authorization: Bearer $ACCESS_TOKEN" \
    "$LABEL_STUDIO_URL/api/projects/$PROJECT_ID/export?exportType=JSON&interpolate_key_frames=true" \
    -o "$OUTPUT"

echo "Exported interpolated annotations to $OUTPUT"