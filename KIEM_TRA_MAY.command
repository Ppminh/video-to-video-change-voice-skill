#!/bin/bash
# Kiem tra cau hinh Mac, ghi ket qua vao logs/mac_info.txt
cd "$(dirname "$0")"
mkdir -p logs
{
echo "== date"; date
echo "== sw_vers"; sw_vers
echo "== hw"; sysctl -n machdep.cpu.brand_string; echo "memsize: $(sysctl -n hw.memsize)"; echo "ncpu: $(sysctl -n hw.ncpu)"
system_profiler SPHardwareDataType 2>/dev/null | grep -E "Model Name|Model Identifier|Chip|Memory|Cores"
echo "== disk"; df -h ~ /
echo "== brew"; which brew; ls -d /opt/homebrew/bin/brew ~/.homebrew/bin/brew /usr/local/bin/brew 2>/dev/null
echo "== python"; which -a python3; python3 --version; which conda; ls -d ~/anaconda3 ~/opt/anaconda3 /opt/anaconda3 ~/miniconda3 2>/dev/null
echo "== ffmpeg"; which ffmpeg; ffmpeg -version 2>/dev/null | head -1
ffmpeg -hide_banner -encoders 2>/dev/null | grep -i videotoolbox
ffmpeg -hide_banner -filters 2>/dev/null | grep -E " subtitles | ass | drawtext "
echo "== uv"; which uv; ls ~/.local/bin 2>/dev/null
echo "== xcode clt"; xcode-select -p
echo "== net"; for u in https://pypi.org https://huggingface.co https://github.com https://generativelanguage.googleapis.com https://www.iesdouyin.com https://astral.sh https://formulae.brew.sh; do curl -s -o /dev/null -m 10 -w "%{http_code} $u\n" "$u"; done
echo "== power"; pmset -g batt | head -3
echo "== DONE"
} > logs/mac_info.txt 2>&1
echo ""
echo "XONG. Ket qua da ghi vao logs/mac_info.txt. Ban co the dong cua so nay."
