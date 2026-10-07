#!/bin/bash
# Kiem tra cac cach tai Douyin. Ket qua: logs/douyin_debug/report.txt
cd "$(dirname "$0")" || exit 1
export PYTHONPATH="$(pwd)/app"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
"$HOME/.douyin_dubber/venv/bin/python" -m dubber.douyin_debug "$@" 2>&1 | tee "$(pwd)/logs/douyin_debug_console.txt"
echo "XONG."
