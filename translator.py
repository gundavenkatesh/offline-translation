"""Fast dedicated translation engine (Opus-MT), both directions, CPU-friendly.

Directions:
  de->en  Helsinki-NLP/opus-mt-de-en
  en->de  Helsinki-NLP/opus-mt-en-de
"""
import os
import re

MODELS = {
    "de-en": os.environ.get("MT_MODEL_DE_EN", "Helsinki-NLP/opus-mt-de-en"),
    "en-de": os.environ.get("MT_MODEL_EN_DE", "Helsinki-NLP/opus-mt-en-de"),
}
MAX_PIECE_CHARS = 400
BATCH_SIZE = 8

_cache = {}   # direction -> (model, tokenizer)


def available() -> bool:
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _load(direction: str):
    if direction not in _cache:
        from transformers import MarianMTModel, MarianTokenizer
        name = MODELS[direction]
        tok = MarianTokenizer.from_pretrained(name)
        mdl = MarianMTModel.from_pretrained(name)
        mdl.eval()
        _cache[direction] = (mdl, tok)
    return _cache[direction]


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?:;])\s+", text)
    return [p for p in parts if p.strip()]


def _pieces_for(paragraph: str) -> list[str]:
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


def translate_document_opus(path: str, extract_text, direction: str = "de-en",
                            progress_cb=None, text_cb=None) -> str:
    """Translate a document with Opus-MT in the given direction."""
    import torch

    model, tokenizer = _load(direction)
    text = extract_text(path)

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    pieces, para_end_idx = [], []
    for para in paragraphs:
        ps = _pieces_for(para.strip())
        pieces.extend(ps)
        para_end_idx.append(len(pieces) - 1)

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

    result_paras, cursor = [], 0
    for end in para_end_idx:
        result_paras.append(" ".join(translated[cursor:end + 1]))
        cursor = end + 1
    return "\n\n".join(result_paras)
