#!/usr/bin/env bash
# verify-research.sh — the single verification command for the llmlab repo.
#
# Run locally before pushing, and it runs in CI independently (a pre-push
# hook alone is bypassable; CI is the enforcement point). It covers:
#   - syntax validation of the tooling (python compile)
#   - provenance: schema + arithmetic + cross-reference of published numbers
#   - docs: broken links, hardware specs vs inventory, superseded claims,
#     production-declaration consistency, report dating
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "== py_compile scripts/"
python3 -m py_compile scripts/*.py

echo "== validate-provenance.py"
python3 scripts/validate-provenance.py

echo "== check-docs.py"
python3 scripts/check-docs.py

echo "== leak sweep (if present)"
if [ -x scripts/sanitize-sweep.sh ]; then
  scripts/sanitize-sweep.sh
fi

echo "verify-research: PASS"
