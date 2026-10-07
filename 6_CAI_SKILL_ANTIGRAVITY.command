#!/bin/bash
# Cai skill "douyin-long-tieng" cho Google Antigravity de dung o MOI du an (skill toan cuc).
# - Trong thu muc du an nay, Antigravity da tu thay skill o .agents/skills/ (khong can cai).
# - File nay chep skill vao ~/.gemini/config/skills/ (Antigravity 2.0 / IDE) kem 1 ban sao du an
#   (khong co API key, khong co video) de cai lai tren may khac.
cd "$(dirname "$0")" || exit 1
ROOT="$(pwd)"
NAME="douyin-long-tieng"
SRC="$ROOT/.agents/skills/$NAME"
LOG="$ROOT/logs/skill_install.txt"
mkdir -p "$ROOT/logs" "$HOME/.douyin_dubber"
printf '%s' "$ROOT" > "$HOME/.douyin_dubber/project_path"
{
echo "CAI SKILL ANTIGRAVITY - $(date)"
TMP="$(mktemp -d)"
mkdir -p "$TMP/project/config" "$TMP/project/.agents/skills"
cp -R app ./*.command HUONG_DAN.txt AGENTS.md "$TMP/project/"
cp config/test_links.txt "$TMP/project/config/" 2>/dev/null
cp -R "$SRC" "$TMP/project/.agents/skills/"
rm -rf "$TMP/project/.agents/skills/$NAME/bundle"
find "$TMP/project" \( -name '__pycache__' -o -name '.DS_Store' \) -prune -exec rm -rf {} +
(cd "$TMP" && zip -qr project.zip project)
DESTS=("$HOME/.gemini/config/skills")
if [ -d "$HOME/.gemini/antigravity" ] && [ ! -d "$HOME/.gemini/config" ]; then
  DESTS+=("$HOME/.gemini/antigravity/skills")      # ban Antigravity cu
fi
if [ -d "$HOME/.gemini/antigravity-cli" ]; then
  DESTS+=("$HOME/.gemini/antigravity-cli/skills")  # Antigravity CLI
fi
for D in "${DESTS[@]}"; do
  mkdir -p "$D"
  rm -rf "${D:?}/$NAME"
  cp -R "$SRC" "$D/$NAME"
  rm -rf "$D/$NAME/bundle"
  find "$D/$NAME" -name '__pycache__' -prune -exec rm -rf {} +
  mkdir -p "$D/$NAME/bundle"
  cp "$TMP/project.zip" "$D/$NAME/bundle/project.zip"
  chmod +x "$D/$NAME/scripts/"*.sh
  echo "Da cai: $D/$NAME"
done
rm -rf "$TMP"
echo "Kiem tra skill:"
bash "$HOME/.gemini/config/skills/$NAME/scripts/dub.sh" where
echo "XONG. Mo Antigravity, go: /$NAME  hoac noi 'long tieng video douyin nay ...'"
} 2>&1 | tee "$LOG"
