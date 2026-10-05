#!/bin/bash
# Chạy công cụ điều khiển: bash dub.sh <lệnh> [...]   (xem: bash dub.sh)
# Tự tìm thư mục dự án theo thứ tự: biến DUBBER_PROJECT -> skill nằm trong dự án (.agents/skills/...)
# -> file ~/.douyin_dubber/project_path (do 6_CAI_SKILL_ANTIGRAVITY.command ghi).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "$DUBBER_PROJECT" ]; then
  CAND="$(cd "$HERE/../../../.." 2>/dev/null && pwd)"
  if [ -f "$CAND/app/dubber/__init__.py" ]; then
    DUBBER_PROJECT="$CAND"
  elif [ -f "$HOME/.douyin_dubber/project_path" ]; then
    DUBBER_PROJECT="$(cat "$HOME/.douyin_dubber/project_path")"
  fi
fi
if [ -z "$DUBBER_PROJECT" ] || [ ! -f "$DUBBER_PROJECT/app/dubber/__init__.py" ]; then
  echo "KHÔNG TÌM THẤY DỰ ÁN. Đặt biến DUBBER_PROJECT=<thư mục dự án> hoặc chạy setup_project.sh để tạo mới."
  exit 2
fi
PY="$HOME/.douyin_dubber/venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "CHƯA CÀI MÔI TRƯỜNG: nhấp đúp $DUBBER_PROJECT/1_CAI_DAT.command (hoặc chạy: bash \"$DUBBER_PROJECT/1_CAI_DAT.command\")"
  exit 3
fi
export DUBBER_PROJECT
export PYTHONPATH="$DUBBER_PROJECT/app"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
exec "$PY" "$HERE/dub.py" "$@"
