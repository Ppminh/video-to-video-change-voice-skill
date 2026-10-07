#!/bin/bash
# Chay thu cac link trong config/test_links.txt (khong can giao dien). Ket qua: logs/test_run.txt
cd "$(dirname "$0")" || exit 1
ROOT="$(pwd)"
PY="$HOME/.douyin_dubber/venv/bin/python"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
export PYTHONPATH="$ROOT/app"
mkdir -p "$ROOT/logs"
caffeinate -i "$PY" -m dubber.cli 2>&1 | tee "$ROOT/logs/test_console.txt"
echo "XONG. Co the dong cua so nay."
