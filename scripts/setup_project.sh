#!/bin/bash
# Tạo dự án lồng tiếng từ bản sao đi kèm skill (dùng khi cài trên máy mới).
# Dùng: bash setup_project.sh [thư_mục_đích]   (mặc định ~/Desktop/douyin-long-tieng)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZIP="$HERE/../bundle/project.zip"
DEST="${1:-$HOME/Desktop/douyin-long-tieng}"
if [ -f "$DEST/app/dubber/__init__.py" ]; then
  echo "Đã có dự án ở: $DEST"
else
  if [ ! -f "$ZIP" ]; then
    echo "Không có bản sao dự án (bundle/project.zip). Hãy cài skill bằng 6_CAI_SKILL_ANTIGRAVITY.command từ máy gốc."
    exit 1
  fi
  TMP="$(mktemp -d)"
  unzip -q "$ZIP" -d "$TMP" || exit 1
  mkdir -p "$DEST"
  cp -R "$TMP/project/." "$DEST/"
  rm -rf "$TMP"
  mkdir -p "$DEST/config" "$DEST/DAU_VAO" "$DEST/THANH_PHAM" "$DEST/logs"
  chmod +x "$DEST"/*.command
  echo "Đã tạo dự án ở: $DEST"
fi
mkdir -p "$HOME/.douyin_dubber"
printf '%s' "$DEST" > "$HOME/.douyin_dubber/project_path"
echo "Bước tiếp theo:"
echo "  1) Cài môi trường (15-30 phút): bash \"$DEST/1_CAI_DAT.command\""
echo "  2) Ghi API key miễn phí (https://aistudio.google.com/apikey) vào $DEST/config/.env dạng GEMINI_API_KEY=... rồi chmod 600"
