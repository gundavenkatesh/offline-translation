"""Unified translation entry point: disk cache + engine selection.

Engines:
  opus (default) - fast dedicated MT model (Opus-MT de->en), CPU-friendly
  llm            - Qwen3 via Ollama (any language, slower, higher fluency)

Set TRANSLATE_ENGINE=llm to force the LLM path. If the opus dependencies
(transformers/torch) are missing, we fall back to the LLM automatically.
"""
import hashlib
import os
from pathlib import Path

from ingest import extract_text
import rag
import translator

CACHE_DIR = Path(os.environ.get("TRANSLATION_CACHE_DIR", "data/translations"))
ENGINE = os.environ.get("TRANSLATE_ENGINE", "opus")


def _file_hash(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _cache_path(path: str, engine: str) -> Path:
    return CACHE_DIR / f"{Path(path).stem}_{engine}_{_file_hash(path)}.txt"


def active_engine() -> str:
    if ENGINE == "opus" and translator.available():
        return "opus"
    return "llm"


def get_translation(path: str, progress_cb=None, text_cb=None,
                    force_refresh: bool = False) -> tuple[str, bool]:
    """Returns (english_text, from_cache)."""
    engine = active_engine()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(path, engine)

    if cache.exists() and not force_refresh:
        return cache.read_text(encoding="utf-8"), True

    if engine == "opus":
        result = translator.translate_document_opus(
            path, extract_text, progress_cb=progress_cb, text_cb=text_cb
        )
    else:
        result = rag.translate_document(
            path, progress_cb=progress_cb, text_cb=text_cb
        )

    cache.write_text(result, encoding="utf-8")
    return result, False
