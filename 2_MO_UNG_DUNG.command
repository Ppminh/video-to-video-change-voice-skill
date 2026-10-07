#!/bin/bash
# =============================================================
#  MO UNG DUNG LONG TIENG
#  Nhap dup file nay -> trinh duyet tu mo trang dieu khien.
#  De ung dung chay: GIU cua so Terminal nay (co the thu nho).
#  Tat ung dung: dong cua so Terminal nay.
# =============================================================
cd "$(dirname "$0")" || exit 1
ROOT="$(pwd)"
APPHOME="$HOME/.douyin_dubber"
PY="$APPHOME/venv/bin/python"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
mkdir -p "$ROOT/logs"
if [ ! -x "$PY" ]; then
  echo "Chua cai dat. Hay nhap dup 1_CAI_DAT.command truoc."
  exit 1
fi
if lsof -i :7860 >/dev/null 2>&1; then
  echo "Ung dung dang chay roi -> mo trinh duyet."
  open "http://127.0.0.1:7860"
  exit 0
fi
export PYTHONPATH="$ROOT/app"
(sleep 3; open "http://127.0.0.1:7860") &
# caffeinate -i: giu Mac khong ngu trong luc ung dung chay (van tat man hinh duoc)
caffeinate -i "$PY" -m dubber.server 2>&1 | tee -a "$ROOT/logs/app.log"
