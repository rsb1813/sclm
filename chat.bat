@echo off
rem Opens the terminal chat with the tiered model (scripts/chat.py): DeepSeek-V4-Flash-0731. chat_v41.bat opens V4.1 Flash.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set SCLM_MODEL=v4
.venv\Scripts\python.exe -u -m scripts.chat %*
