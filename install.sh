#!/usr/bin/env bash
# NFTool (claude-rewrite) — standalone one-command installer for Linux/macOS.
#   curl -fsSL https://raw.githubusercontent.com/nm-z/NFTool/claude-rewrite/install.sh | bash
set -euo pipefail
tmp="$(mktemp -d)"
echo "==> Downloading NFTool (claude-rewrite) from GitHub..."
curl -fsSL https://github.com/nm-z/NFTool/archive/refs/heads/claude-rewrite.tar.gz | tar xz -C "$tmp"
cd "$tmp"/NFTool-claude-rewrite
echo "==> Running setup_and_run.sh"
exec bash setup_and_run.sh "$@"
