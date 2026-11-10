@echo off
rem Opens the status dashboard (scripts/status.py).
cd /d "%~dp0"
.venv\Scripts\python.exe -m scripts.status %*
