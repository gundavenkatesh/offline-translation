@echo off
REM ============================================================
REM  Offline Document Chatbot - Office PC Setup
REM  Run this from inside the unzipped project folder.
REM  Safe to re-run: every step checks before acting.
REM ============================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"
echo.
echo === Offline Document Chatbot - Setup ===
echo Project folder: %CD%
echo.

REM ---------- 1. Check Python ----------
echo [1/7] Checking Python...
python --version 2>nul | findstr /R "3\.1[0-9]" >nul
if errorlevel 1 (
    py -3.12 --version >nul 2>&1
    if errorlevel 1 (
        echo   ERROR: Python 3.10+ not found.
        echo   Install Python 3.12 from https://www.python.org/downloads/
        echo   IMPORTANT: tick "Add python.exe to PATH" during install.
        pause & exit /b 1
    )
    set "PY=py -3.12"
) else (
    set "PY=python"
)
echo   OK: using !PY!

REM ---------- 2. Check Ollama ----------
echo [2/7] Checking Ollama...
ollama --version >nul 2>&1
if errorlevel 1 (
    echo   ERROR: Ollama not found.
    echo   Install from https://ollama.com/download/windows
    echo   Then open a NEW terminal and re-run this script.
    pause & exit /b 1
)
echo   OK: Ollama installed

REM Ollama server usually auto-runs on Windows. Verify it responds:
ollama ps >nul 2>&1
if errorlevel 1 (
    echo   Ollama server not responding - starting it...
    start "" /min ollama serve
    timeout /t 5 /nobreak >nul
)
echo   OK: Ollama server running

REM ---------- 3. GPU info ----------
echo [3/7] GPU check...
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>nul
if errorlevel 1 echo   WARNING: nvidia-smi not found - GPU drivers may be missing.

REM ---------- 4. Pull models ----------
echo [4/7] Pulling models (skips if already present)...
ollama list | findstr /C:"qwen3:4b" >nul || ollama pull qwen3:4b
ollama list | findstr /C:"bge-m3" >nul || ollama pull bge-m3
echo   OK: models ready

REM ---------- 5. Python venv + dependencies ----------
echo [5/7] Python environment...
if not exist ".venv\Scripts\python.exe" (
    !PY! -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo   ERROR: pip install failed. Check internet/proxy and re-run.
    pause & exit /b 1
)
echo   OK: dependencies installed

REM Pre-download the Opus-MT translation model (cached locally afterwards)
python -c "from transformers import MarianMTModel, MarianTokenizer; MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-de-en'); MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-de-en')" && echo   OK: Opus-MT model cached

REM ---------- 6. Smoke test ----------
echo [6/7] Smoke test (model responds in English)...
ollama run qwen3:4b "/no_think Uebersetze ins Englische und antworte nur mit der Uebersetzung: Der Vertrag endet am 31. Dezember."
echo   ^^ If you see an English sentence above, the model works.

REM ---------- 7. Create start.bat ----------
echo [7/7] Creating start.bat...
(
echo @echo off
echo cd /d "%%~dp0"
echo call .venv\Scripts\activate
echo set LLM_MODEL=qwen3:4b
echo set EMBED_MODEL=bge-m3
echo set OLLAMA_KEEP_ALIVE=30m
echo streamlit run app/main.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
) > start.bat
echo   OK: start.bat created

echo.
echo ============================================================
echo  SETUP COMPLETE
echo.
echo  Next steps:
echo   1. Firewall rule (run as ADMIN, once):
echo      netsh advfirewall firewall add rule name="Chatbot UI" dir=in action=allow protocol=TCP localport=8501
echo   2. Start the app:   start.bat
echo   3. Test locally:    http://localhost:8501
echo   4. Test from LAN:   http://%COMPUTERNAME%:8501
echo   5. Auto-start on boot: Win+R, shell:startup, put a shortcut to start.bat there
echo ============================================================
pause
