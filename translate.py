"""Unified translation entry: disk cache + engine selection + direction.

direction: 'de-en' (default) or 'en-de'
engine:    'opus' (default, fast CPU) or 'llm' (Qwen3 fallback)
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


def _cache_path(path: str, engine: str, direction: str) -> Path:
    return CACHE_DIR / f"{Path(path).stem}_{engine}_{direction}_{_file_hash(path)}.txt"


def active_engine() -> str:
    if ENGINE == "opus" and translator.available():
        return "opus"
    return "llm"


def get_translation(path: str, direction: str = "de-en",
                    progress_cb=None, text_cb=None,
                    force_refresh: bool = False) -> tuple[str, bool]:
    """Returns (translated_text, from_cache). direction: 'de-en' or 'en-de'."""
    engine = active_engine()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(path, engine, direction)

    if cache.exists() and not force_refresh:
        return cache.read_text(encoding="utf-8"), True

    if engine == "opus":
        result = translator.translate_document_opus(
            path, extract_text, direction=direction,
            progress_cb=progress_cb, text_cb=text_cb,
        )
    else:
        target = "en" if direction == "de-en" else "de"
        result = rag.translate_document(
            path, progress_cb=progress_cb, text_cb=text_cb, target=target,
        )

    cache.write_text(result, encoding="utf-8")
    return result, False
