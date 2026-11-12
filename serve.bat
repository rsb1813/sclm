@echo off
rem Starts the OpenAI-compatible server of the tiered model (scripts/serve.py): DeepSeek-V4-Flash-0731 on
rem http://127.0.0.1:8000/v1. Options are passed on (--port, --context, --max-tokens, --slice, --cache-dir,
rem --cache-gb; -h lists them).
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set SCLM_MODEL=v4
.venv\Scripts\python.exe -u -m scripts.serve %*
