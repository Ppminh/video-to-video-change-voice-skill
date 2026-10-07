#!/bin/bash
# Cap nhat thu vien (nhanh, khong tai lai model). Ket qua: logs/update.log
cd "$(dirname "$0")" || exit 1
ROOT="$(pwd)"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
uv pip install --python "$HOME/.douyin_dubber/venv/bin/python" -r "$ROOT/app/requirements.txt" 2>&1 | tee "$ROOT/logs/update.log"
echo "UPDATE_DONE $(date)" >> "$ROOT/logs/update.log"
echo "XONG. Co the dong cua so nay."
