@echo off
REM ====================================================
REM  Offline Document Chatbot - Start (office notebook)
REM  Model + context tuned for GTX 1650 Ti 4GB VRAM
REM ====================================================
cd /d "%~dp0"
call .venv\Scripts\activate

set LLM_MODEL=qwen3:4b
set EMBED_MODEL=bge-m3
set OLLAMA_KEEP_ALIVE=30m

streamlit run app/main.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
