#!/bin/bash
# =============================================================
#  CAI DAT HE THONG LONG TIENG DOUYIN -> TIENG VIET
#  Nhap dup file nay. Chay 1 lan (khoang 10-25 phut, tuy mang).
#  Neu Mac hoi mat khau (de cai ffmpeg) thi go mat khau may.
# =============================================================
cd "$(dirname "$0")" || exit 1
ROOT="$(pwd)"
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/install.log"
STATUS="$ROOT/logs/install_status.txt"
exec > >(tee "$LOG") 2>&1

export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
APPHOME="$HOME/.douyin_dubber"
mkdir -p "$APPHOME"
PY="$APPHOME/venv/bin/python"

step() { echo; echo "================ $1"; date "+%H:%M:%S"; }
fail() { echo; echo "!!! LOI: $1"; echo "FAILED: $1" > "$STATUS"; echo "Xem chi tiet trong logs/install.log"; exit 1; }
echo "RUNNING" > "$STATUS"
echo "Bat dau cai dat: $(date)"
df -h "$HOME" | tail -1

step "1/7 ffmpeg (xu ly video)"
if ! command -v ffmpeg >/dev/null 2>&1; then
  command -v brew >/dev/null 2>&1 || fail "Chua co Homebrew de cai ffmpeg"
  brew install ffmpeg || fail "Khong cai duoc ffmpeg"
fi
ffmpeg -hide_banner -version | head -1

step "2/7 uv (bo cai Python nhanh)"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh || fail "Khong cai duoc uv"
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

step "3/7 Python 3.11 rieng cho ung dung (khong anh huong Python khac tren may)"
uv python install 3.11 || fail "Khong tai duoc Python 3.11"
if [ ! -x "$PY" ]; then
  uv venv --python 3.11 "$APPHOME/venv" || fail "Khong tao duoc moi truong Python"
fi
"$PY" --version

step "4/7 Thu vien AI (lan dau tai khoang 1-2GB)"
uv pip install --python "$PY" -r "$ROOT/app/requirements.txt" || fail "Cai thu vien loi"
echo "-- Thu cai demucs-mlx (tach nhac bang GPU chip Apple, khong bat buoc)"
if "$PY" -c "import demucs_mlx" >/dev/null 2>&1; then
  echo "demucs-mlx da co."
elif uv pip install --python "$PY" "demucs-mlx==1.5.3"; then
  echo "Cai demucs-mlx thanh cong."
else
  echo "Canh bao: khong cai duoc demucs-mlx -> se dung bo tach UVR (chay CPU, cham hon)."
fi

step "5/7 Model nhan dang giong noi, phan biet nguoi noi, font chu"
PYTHONPATH="$ROOT/app" "$PY" -m dubber.models || fail "Tai model loi"

step "6/7 Model tach nhac nen demucs-mlx (chuyen doi 1 lan, tam dung PyTorch roi xoa)"
if ! "$PY" -c "import demucs_mlx" >/dev/null 2>&1; then
  echo "Bo qua (khong co demucs-mlx). Dung UVR."
elif ls "$HOME/.cache/demucs-mlx/htdemucs.safetensors" >/dev/null 2>&1; then
  echo "Da co san."
else
  TMPV="$APPHOME/convert_venv"
  rm -rf "$TMPV"
  if uv venv --python 3.11 "$TMPV" \
     && uv pip install --python "$TMPV/bin/python" "demucs-mlx[convert]==1.5.3" \
     && "$TMPV/bin/python" -m demucs_mlx.mlx_convert htdemucs --output-dir "$HOME/.cache/demucs-mlx"; then
    echo "Chuyen doi demucs thanh cong."
  else
    echo "Canh bao: khong chuyen doi duoc demucs -> he thong se dung UVR (cham hon mot chut)."
  fi
  rm -rf "$TMPV"
  uv cache clean torch >/dev/null 2>&1
  uv cache prune >/dev/null 2>&1
fi

step "7/7 Kiem tra tung bo phan (tai them model giong doc lan dau)"
PYTHONPATH="$ROOT/app" "$PY" -m dubber.selftest
echo
du -sh "$APPHOME" 2>/dev/null
df -h "$HOME" | tail -1
echo "OK $(date)" > "$STATUS"
echo
echo "=========================================================="
echo " CAI DAT XONG. Ket qua kiem tra: logs/selftest.txt"
echo " Giong mau de nghe thu: logs/giong_mau/"
echo " Tiep theo: nhap dup 2_MO_UNG_DUNG.command"
echo "=========================================================="
