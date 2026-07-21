"""Fast dedicated translation engine (Opus-MT, German->English, CPU-friendly).

Used by translate.py when TRANSLATE_ENGINE=opus (default).
Falls back gracefully if transformers/torch are not installed.
"""
import os
import re

MT_MODEL = os.environ.get("MT_MODEL", "Helsinki-NLP/opus-mt-de-en")
MAX_PIECE_CHARS = 400   # Marian models handle ~512 tokens; keep pieces short
BATCH_SIZE = 8

_model = None
_tokenizer = None


def available() -> bool:
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _load():
    global _model, _tokenizer
    if _model is None:
        from transformers import MarianMTModel, MarianTokenizer
        _tokenizer = MarianTokenizer.from_pretrained(MT_MODEL)
        _model = MarianMTModel.from_pretrained(MT_MODEL)
        _model.eval()
    return _model, _tokenizer


def _split_sentences(text: str) -> list[str]:
    """Naive sentence split, good enough for MT piece sizing."""
    parts = re.split(r"(?<=[.!?:;])\s+", text)
    return [p for p in parts if p.strip()]


def _pieces_for(paragraph: str) -> list[str]:
    """Split a paragraph into MT-sized pieces on sentence boundaries."""
    if len(paragraph) <= MAX_PIECE_CHARS:
        return [paragraph]
    pieces, current = [], ""
    for s in _split_sentences(paragraph):
        if len(current) + len(s) > MAX_PIECE_CHARS and current:
            pieces.append(current)
            current = s
        else:
            current = f"{current} {s}" if current else s
    if current:
        pieces.append(current)
    return pieces


def translate_document_opus(path: str, extract_text,
                            progress_cb=None, text_cb=None) -> str:
    """Translate a whole document with Opus-MT, preserving paragraph structure."""
    import torch

    model, tokenizer = _load()
    text = extract_text(path)

    # Build flat list of pieces, remembering paragraph boundaries
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    pieces, para_end_idx = [], []
    for para in paragraphs:
        ps = _pieces_for(para.strip())
        pieces.extend(ps)
        para_end_idx.append(len(pieces) - 1)  # index of last piece of this paragraph

    total = len(pieces)
    if progress_cb:
        progress_cb(0, total)

    translated: list[str] = []
    with torch.no_grad():
        for start in range(0, total, BATCH_SIZE):
            batch = pieces[start:start + BATCH_SIZE]
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=512)
            out = model.generate(**enc, max_length=512, num_beams=2)
            translated.extend(
                tokenizer.decode(t, skip_special_tokens=True) for t in out
            )
            if progress_cb:
                progress_cb(min(start + BATCH_SIZE, total), total)
            if text_cb:
                text_cb(" ".join(translated)[-2000:])

    # Reassemble: join pieces, restore paragraph breaks
    result_paras, cursor = [], 0
    for end in para_end_idx:
        result_paras.append(" ".join(translated[cursor:end + 1]))
        cursor = end + 1
    return "\n\n".join(result_paras)
