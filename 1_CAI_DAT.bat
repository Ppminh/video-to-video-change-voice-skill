@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONPATH=%~dp0app
set PY=%USERPROFILE%\.douyin_dubber\venv\Scripts\python.exe

if not exist "%PY%" (
    echo [LOI] Chua co moi truong Python.
    pause
    exit /b 1
)

echo Dang kiem tra va tai cac mo hinh AI...
"%PY%" -m dubber.models
echo.
echo Dang kiem tra he thong...
"%PY%" -m dubber.selftest
echo.
echo Hoan tat! Nhan phim bat ky de thoat.
pause
