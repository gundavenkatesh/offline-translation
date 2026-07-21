# Offline Document Chatbot (2026 stack)

Fully offline RAG chatbot. Upload documents in any language, ask questions in any language, get answers in English. Optional full-document translation.

## Stack (current as of July 2026)

- **Translation:** Opus-MT de->en (default engine, fast on CPU) with automatic
  fallback to the LLM; `TRANSLATE_ENGINE=llm` forces the LLM path.
  Translations are cached in `data/translations/` (re-translating a file is instant).

- **LLM:** `qwen3:30b-a3b` — MoE, 30B total / ~3.3B active per token. Big-model quality, small-model speed. Apache 2.0.
- **Embeddings:** `bge-m3` served by Ollama (multilingual, ideal for German docs + English queries)
- **Vector DB:** Chroma (persisted on disk)
- **Pipeline:** LangChain · **UI:** Streamlit

Everything runs through Ollama — no separate Python ML dependencies for embeddings.

## Setup (one-time, needs internet)

```bash
# 1. Install Ollama: https://ollama.com/download
ollama pull qwen3:30b-a3b     # ~19 GB download
ollama pull bge-m3            # embeddings
# fallback for tight VRAM: ollama pull qwen3:14b

# 2. Python deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Fully offline afterwards.

## Run

```bash
ollama serve                      # if not already running
streamlit run app/main.py
```

Open http://localhost:8501

## Configuration (env vars)

| Var | Default | Notes |
|-----|---------|-------|
| `LLM_MODEL` | `qwen3:30b-a3b` | Any Ollama model tag |
| `EMBED_MODEL` | `bge-m3` | Ollama embedding model |
| `CHROMA_DIR` | `data/chroma` | Vector store location |
| `UPLOAD_DIR` | `data/uploads` | Originals kept for translation |

## Hardware fit (your machine: 32 GB RAM + dev GPU)

| Model | Q4 size | Fits |
|-------|---------|------|
| qwen3:30b-a3b | ~19 GB | 24 GB VRAM GPU, or CPU+RAM hybrid (12-15 tok/s) |
| qwen3:14b | ~9 GB | 12 GB VRAM |
| qwen3:8b | ~5 GB | 8 GB VRAM |

Tips:
- `OLLAMA_KEEP_ALIVE=30m` — model stays warm but frees VRAM for Android builds later
- `OLLAMA_KV_CACHE_TYPE=q8_0` — halves KV-cache memory if context runs long

## Notes / next steps

- Scanned PDFs need OCR — add Tesseract or PaddleOCR in `extract_text()`
- Air-gapped deploy: copy `~/.ollama` + pip wheels (`pip download -r requirements.txt -d wheels/`)
- Chat history is per-session; no conversation memory in retrieval yet
