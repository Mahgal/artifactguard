#!/usr/bin/env bash
# ArtifactGuard narrated demo (<90 seconds). Safe to run anywhere:
# the hostile fixtures are generated in the open and only ever scanned.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "### Preparing a folder of 'downloaded' artifacts (2 are hostile)..."
python3 demo/make_fixtures.py
sleep 1
python3 demo/run_demo.py
