"""RAG chain: retrieve relevant chunks, answer in English via local Ollama model."""
import os
import re

from langchain_ollama import ChatOllama

from ingest import get_vectorstore, extract_text

LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3:30b-a3b")
# MoE: 30B knowledge, ~3.3B active/token -> fast. ~15-17GB VRAM at Q4.
# fallback if VRAM-tight: "qwen3:14b" or "qwen3:8b"

SYSTEM_PROMPT = """You are a document assistant. Answer the user's question using ONLY the provided context.

Rules:
- ALWAYS respond in English, even if the source documents or the question are in another language.
- If the context does not contain the answer, say so clearly. Do not invent information.
- Cite the source file name when referencing information.
- Be concise and direct. 
/no_think"""

TRANSLATE_PROMPT = """Translate the following text to English. Preserve structure, headings, and lists. Output only the translation, nothing else.

Text:
{text}
/no_think"""


def get_llm() -> ChatOllama:
    return ChatOllama(model=LLM_MODEL, temperature=0.1)

def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def answer_question(question: str, k: int = 5) -> tuple[str, list]:
    """Returns (answer, source_documents)."""
    vs = get_vectorstore()
    docs = vs.similarity_search(question, k=k)

    context = "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
    )

    llm = get_llm()
    messages = [
        ("system", SYSTEM_PROMPT),
        ("user", f"Context:\n{context}\n\nQuestion: {question}"),
    ]
    response = llm.invoke(messages)
    return strip_think(response.content), docs


def translate_document(path: str, chunk_chars: int = 3000) -> str:
    """Full document translation to English, chunk by chunk."""
    text = extract_text(path)
    llm = get_llm()

    # split on paragraph boundaries to keep chunks coherent
    paragraphs = text.split("\n\n")
    chunks, current = [], ""
    for p in paragraphs:
        if len(current) + len(p) > chunk_chars and current:
            chunks.append(current)
            current = p
        else:
            current = f"{current}\n\n{p}" if current else p
    if current:
        chunks.append(current)

    translated = []
    for chunk in chunks:
        resp = llm.invoke([("user", TRANSLATE_PROMPT.format(text=chunk))])
        translated.append(strip_think(resp.content))
    return "\n\n".join(translated)
