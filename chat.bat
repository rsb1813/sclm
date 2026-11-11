@echo off
rem Opens the terminal chat with the tiered model (scripts/chat.py).
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe -u -m scripts.chat %*
