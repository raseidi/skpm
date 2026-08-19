#!/usr/bin/env bash
#
# Rehearse the `tests` workflow locally, the way GitHub actually runs it.
#
#   ./scripts/rehearse-ci.sh [git-ref]      # defaults to HEAD
#
# A plain `uv run pytest` in your working tree can pass while CI fails, because
# CI differs from your shell in three ways this script reproduces:
#
#   1. it checks out only *committed* content — no gitignored data/ cache, no
#      stray files, nothing you forgot to `git add`;
#   2. it runs every cell of the Python matrix, not just your default version;
#   3. it builds the environment from scratch, so a package left over from an
#      earlier install cannot satisfy an import CI would not have.
#
# The matrix is read out of the workflow file rather than hardcoded, so this
# script cannot drift from what CI runs.
set -euo pipefail

WORKFLOW=".github/workflows/tests.yml"

cd "$(git rev-parse --show-toplevel)"
[[ -f "$WORKFLOW" ]] || { echo "error: $WORKFLOW not found" >&2; exit 1; }

REF="${1:-HEAD}"

# The rehearsal runs a committed ref, so uncommitted work is invisible to it.
if [[ -n "$(git status --porcelain)" ]]; then
  echo "warning: working tree is dirty — uncommitted changes are NOT rehearsed." >&2
fi

# `uv run` warns when an unrelated venv is active; the worktree gets its own.
unset VIRTUAL_ENV

# Static check first: actionlint type-checks ${{ }} expressions against the
# real context schema and validates runner labels and action refs.
echo "===== actionlint ====="
uvx --from actionlint-py actionlint .github/workflows/*.yml
echo "ok"

# Read the matrix out of the workflow: python: ['3.11', '3.12']
PYTHONS=$(uv run --no-project python - "$WORKFLOW" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
match = re.search(r"^\s*python:\s*\[([^\]]*)\]", text, re.MULTILINE)
if not match:
    sys.exit("could not find the python matrix in the workflow")
print(" ".join(v.strip().strip("'\"") for v in match.group(1).split(",") if v.strip()))
PY
)
echo "matrix: $PYTHONS"

WORKTREE="$(mktemp -d)/rehearsal"
cleanup() { git worktree remove --force "$WORKTREE" 2>/dev/null || true; }
trap cleanup EXIT

git worktree add --detach "$WORKTREE" "$REF" >/dev/null
echo "rehearsing $(git rev-parse --short "$REF") in a clean checkout"

# Mirror the workflow's `fail-fast: false`: run every cell, report at the end.
failed=()
for python in $PYTHONS; do
  echo
  echo "===== python $python ====="
  (
    cd "$WORKTREE"
    rm -rf .venv coverage.xml
    uv sync --locked --no-default-groups --group test -p "$python" >/dev/null
    uv run --no-sync pytest tests --cov=skpm --cov-report=term --cov-report=xml
    test -f coverage.xml || { echo "coverage.xml was not written" >&2; exit 1; }
  ) || failed+=("$python")
done

echo
if (( ${#failed[@]} )); then
  echo "FAILED on: ${failed[*]}"
  exit 1
fi
echo "all ${PYTHONS} cells green"
