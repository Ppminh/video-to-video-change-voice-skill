@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONPATH=%~dp0app
set PY=%USERPROFILE%\.douyin_dubber\venv\Scripts\python.exe

"%PY%" -m dubber.selftest %*
pause
