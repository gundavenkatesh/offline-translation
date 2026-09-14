@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
echo === Offline Document Chatbot - Setup (bilingual) ===

echo [1/7] Checking Python...
python --version 2>nul | findstr /R "3\.1[0-9]" >nul
if errorlevel 1 (
    py -3.12 --version >nul 2>&1
    if errorlevel 1 (
        echo   ERROR: Python 3.10+ not found. Install 3.12 from python.org ^(tick Add to PATH^).
        pause & exit /b 1
    )
    set "PY=py -3.12"
) else ( set "PY=python" )
echo   OK: using !PY!

echo [2/7] Checking Ollama...
ollama --version >nul 2>&1
if errorlevel 1 ( echo   ERROR: install Ollama from ollama.com/download/windows, reopen terminal, re-run. & pause & exit /b 1 )
ollama ps >nul 2>&1
if errorlevel 1 ( start "" /min ollama serve & timeout /t 5 /nobreak >nul )
echo   OK

echo [3/7] GPU check...
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>nul

echo [4/7] Pulling Ollama models...
ollama list | findstr /C:"qwen3:4b" >nul || ollama pull qwen3:4b
ollama list | findstr /C:"bge-m3" >nul || ollama pull bge-m3
echo   OK

echo [5/7] Python venv + deps...
if not exist ".venv\Scripts\python.exe" ( !PY! -m venv .venv )
call .venv\Scripts\activate
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
if errorlevel 1 ( echo   ERROR: pip failed - check internet/proxy. & pause & exit /b 1 )
echo   OK

echo [6/7] Pre-downloading Opus-MT models (both directions)...
python -c "from transformers import MarianMTModel, MarianTokenizer; [ (MarianTokenizer.from_pretrained(m), MarianMTModel.from_pretrained(m)) for m in ('Helsinki-NLP/opus-mt-de-en','Helsinki-NLP/opus-mt-en-de') ]" && echo   OK: both translation models cached

echo [7/7] start.bat is ready.
echo.
echo ============================================================
echo  SETUP COMPLETE
echo   1. Firewall (admin, once): netsh advfirewall firewall add rule name="Chatbot UI" dir=in action=allow protocol=TCP localport=8501
echo   2. Start: start.bat
echo   3. Local: http://localhost:8501   LAN: http://%COMPUTERNAME%:8501
echo ============================================================
pause
