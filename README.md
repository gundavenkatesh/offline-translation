# Offline Document Chatbot (bilingual)

Fully offline. Upload documents, ask questions in German or English (answered in
the same language), and translate documents both DE→EN and EN→DE.

## Stack
- LLM (Q&A): Qwen3 via Ollama
- Translation: Opus-MT (de-en + en-de), fast on CPU, cached in data/translations/
- Embeddings: bge-m3 (via Ollama) · Vector DB: Chroma · Pipeline: LangChain · UI: Streamlit

## Features
- Auto-ingest on upload (no button), skips already-indexed files
- Bilingual Q&A: German question → German answer, English question → English answer
- Bidirectional translation with a direction toggle (default DE→EN)
- Translation cache: re-translating a file is instant
- Set TRANSLATE_ENGINE=llm to force the Qwen3 translation path

## First-time setup (needs internet once)
Run setup-office-pc.bat, or manually: install Python 3.12 + Ollama,
`ollama pull qwen3:4b bge-m3`, create venv, `pip install -r requirements.txt`,
then pre-download both Opus models:
```
python -c "from transformers import MarianMTModel, MarianTokenizer; [ (MarianTokenizer.from_pretrained(m), MarianMTModel.from_pretrained(m)) for m in ('Helsinki-NLP/opus-mt-de-en','Helsinki-NLP/opus-mt-en-de') ]"
```

## Run
start.bat → http://localhost:8501 (LAN: http://<hostname>:8501)
