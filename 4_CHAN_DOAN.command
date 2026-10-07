#!/bin/bash
# Chan doan: ket noi dich (logs/diag.txt) + tach nhac + giong doc (logs/selftest_part.txt)
cd "$(dirname "$0")" || exit 1
ROOT="$(pwd)"
export PYTHONPATH="$ROOT/app"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
PY="$HOME/.douyin_dubber/venv/bin/python"
"$PY" -m dubber.diag 2>&1 | tee "$ROOT/logs/diag_console.txt"
"$PY" -m dubber.selftest --parts=separate,tts,voices,tts8 2>&1 | tee "$ROOT/logs/selftest_console.txt"
echo "XONG. Co the dong cua so nay."
