"""RAG chain: retrieve relevant chunks, answer in English via local Ollama model."""
import os
import re

from langchain_ollama import ChatOllama

from ingest import get_vectorstore, extract_text

LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3:4b")
# alternatives: "qwen3:8b" (better quality, hybrid speed on 4GB VRAM),
#               "qwen3:14b" / "qwen3:30b-a3b" (needs bigger GPU)

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


def strip_think(text: str) -> str:
    """Remove qwen3 <think>...</think> reasoning blocks from output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def get_llm() -> ChatOllama:
    return ChatOllama(model=LLM_MODEL, temperature=0.1)


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


def translate_document(path: str, chunk_chars: int = 3000,
                       progress_cb=None, text_cb=None) -> str:
    """Full document translation to English, chunk by chunk, streaming.

    progress_cb(done, total) fires after each block (and once at start with done=0).
    text_cb(partial_text) fires continuously while the model generates.
    """
    text = extract_text(path)
    llm = get_llm()

    # Pack paragraphs into blocks of up to chunk_chars, keeping paragraphs intact.
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

    if progress_cb:
        progress_cb(0, len(chunks))

    translated = []
    for i, chunk in enumerate(chunks):
        partial = ""
        for piece in llm.stream([("user", TRANSLATE_PROMPT.format(text=chunk))]):
            partial += piece.content
            if text_cb:
                text_cb("\n\n".join(translated) + "\n\n" + partial)
        translated.append(strip_think(partial))
        if progress_cb:
            progress_cb(i + 1, len(chunks))
    return "\n\n".join(translated)
