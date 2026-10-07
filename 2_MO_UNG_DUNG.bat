@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONPATH=%~dp0app
set PY=%USERPROFILE%\.douyin_dubber\venv\Scripts\python.exe

if not exist "%PY%" (
    echo [LOI] Chua co moi truong Python. Vui long chay 1_CAI_DAT.bat truoc.
    pause
    exit /b 1
)

start "" http://127.0.0.1:7860
echo ========================================================
echo   Dang khoi dong giao dien web: http://127.0.0.1:7860
echo   Giu cua so nay trong khi su dung.
echo   De tat: Dong cua so nay lai.
echo ========================================================
"%PY%" -m dubber.server
pause
