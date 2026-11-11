@echo off
rem Opens the terminal chat with the earlier model, DeepSeek-V4.1 Flash (scripts/chat.py).
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe -u -m scripts.chat %*
