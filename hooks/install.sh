#!/usr/bin/env bash
# Install the repository's git hooks. Run once per clone.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# --path-format=absolute needs git >= 2.31; fall back to resolving by hand.
hooks_dir=$(git -C "$here" rev-parse --path-format=absolute --git-path hooks 2>/dev/null) ||
    hooks_dir="$(git -C "$here" rev-parse --absolute-git-dir)/hooks"

mkdir -p "$hooks_dir"
install -m 755 "$here/pre-commit" "$hooks_dir/pre-commit"
echo "Installed pre-commit hook -> $hooks_dir/pre-commit"
