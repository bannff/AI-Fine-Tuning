#!/usr/bin/env bash
#
# Repository convention checks.
#
# The previous version of this script could not fail: it assigned EXIT_CODE=1 in
# one branch, never read it, and always exited 0. Its LOC check also reported a
# false pass because `find . -maxdepth 4 -name "*.py" -o -name "*.md"` binds as
# `(-maxdepth 4 -name "*.py") -o (-name "*.md")`, and its colour constants
# (\033[0-1;31m) were not valid SGR sequences.
#
# Every check here records a failure, and the script exits non-zero if any fired.

set -uo pipefail

# Run against the repository root. Resolve it from git when possible so the
# script behaves the same whether invoked from the root, from scripts/, or copied
# somewhere else entirely.
if ROOT=$(git rev-parse --show-toplevel 2>/dev/null) && [ -n "$ROOT" ]; then
  cd "$ROOT" || exit 1
else
  cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
fi

# Colour only when attached to a terminal, so CI logs stay clean.
if [ -t 1 ]; then
  RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; BLUE=$'\033[0;34m'; NC=$'\033[0m'
else
  RED=''; GREEN=''; BLUE=''; NC=''
fi

FAILURES=0
MAX_LOC=200

pass() { printf '  %sok%s      %s\n' "$GREEN" "$NC" "$1"; }
fail() { printf '  %sFAILED%s  %s\n' "$RED" "$NC" "$1"; FAILURES=$((FAILURES + 1)); }

printf '%s== repository verification ==%s\n\n' "$BLUE" "$NC"

# --------------------------------------------------------------- required files

REQUIRED_FILES=(
  LICENSE
  NOTICE
  README.md
  pyproject.toml
  uv.lock
  .gitignore
  .github/workflows/ci.yml
  src/finetune/__init__.py
  scripts/verify_repo.sh
)
missing=()
for file in "${REQUIRED_FILES[@]}"; do
  [ -f "$file" ] || missing+=("$file")
done
if [ ${#missing[@]} -eq 0 ]; then
  pass "required files present (${#REQUIRED_FILES[@]})"
else
  fail "missing: ${missing[*]}"
fi

# ------------------------------------------------------------ no name collisions
# The bug this repository was restructured to fix: a top-level directory sharing
# a name with a PyPI dependency shadows it as a namespace package.

COLLIDING=(peft trl transformers datasets torch mlx mlx_lm accelerate)
collisions=()
for name in "${COLLIDING[@]}"; do
  [ -e "$name" ] && collisions+=("$name")
done
if [ ${#collisions[@]} -eq 0 ]; then
  pass "no top-level directory shadows a dependency"
else
  fail "these shadow installed packages: ${collisions[*]}"
fi

# -------------------------------------------------------------------- file size

if command -v git >/dev/null 2>&1 && git rev-parse --git-dir >/dev/null 2>&1; then
  # Applied to src/ only. Test files are naturally longer -- a single behaviour
  # often needs a fixture, a parametrise table and several assertions -- and
  # splitting them to satisfy a line count makes them harder to read, not easier.
  oversized=$(git ls-files -z 'src/*.py' \
    | xargs -0 -r wc -l 2>/dev/null \
    | awk -v max="$MAX_LOC" '$2 != "total" && $1 > max { print "    " $2 " (" $1 " lines)" }')
  if [ -z "$oversized" ]; then
    pass "every tracked file in src/ is under $MAX_LOC lines"
  else
    fail "files over $MAX_LOC lines:"
    printf '%s\n' "$oversized"
  fi

  # ------------------------------------------------------------- committed blobs
  # Model weights, tokenizers and training logs belong in releases or on the
  # Hub. A 3.5 MB tokenizer.json was previously 99% of this repository by bytes.
  # uv.lock is exempt: a lock file is meant to be committed, and it is text.
  blobs=$(git ls-files -z \
    | xargs -0 -r -I{} sh -c 'test -f "{}" && find "{}" -size +512k -print' 2>/dev/null \
    | grep -v '^uv\.lock$')
  if [ -z "$blobs" ]; then
    pass "no tracked file exceeds 512 KB (uv.lock exempt)"
  else
    fail "large tracked files:"
    printf '    %s\n' $blobs
  fi

  # --------------------------------------------------------------- empty files
  empties=$(git ls-files | while IFS= read -r f; do
    [ -f "$f" ] && [ ! -s "$f" ] && printf '%s\n' "$f"
  done)
  if [ -z "$empties" ]; then
    pass "no empty tracked files"
  else
    fail "empty tracked files:"
    printf '    %s\n' $empties
  fi
else
  fail "not a git repository"
fi

# ------------------------------------------------------------- example configs
# Guards against a config file that no code path can read -- the repository
# previously shipped three of those.

if [ -d configs ]; then
  count=$(find configs -name '*.yaml' | wc -l | tr -d ' ')
  if [ "$count" -gt 0 ]; then
    pass "$count example config(s) present (validated by tests/test_config.py)"
  else
    fail "configs/ contains no YAML"
  fi
else
  fail "configs/ directory missing"
fi

# ---------------------------------------------------------------------- summary

printf '\n'
if [ "$FAILURES" -eq 0 ]; then
  printf '%sall checks passed%s\n' "$GREEN" "$NC"
  exit 0
fi
printf '%s%d check(s) failed%s\n' "$RED" "$FAILURES" "$NC"
exit 1
