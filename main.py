"""RAG chain: retrieve relevant chunks, answer in the user's language via Ollama."""
import os
import re

from langchain_ollama import ChatOllama

from ingest import get_vectorstore, extract_text

LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3:4b")

# Feature 3: mirror the question's language instead of forcing English.
SYSTEM_PROMPT = """You are a document assistant. Answer the user's question using ONLY the provided context.

Rules:
- Respond in the SAME language as the user's question. If they ask in German, answer in German; if they ask in English, answer in English.
- If the context does not contain the answer, say so clearly in that same language. Do not invent information.
- Cite the source file name when referencing information.
- Be concise and direct.
/no_think"""

TRANSLATE_PROMPT = """Translate the following text to English. Preserve structure, headings, and lists. Output only the translation, nothing else.

Text:
{text}
/no_think"""


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def get_llm() -> ChatOllama:
    return ChatOllama(model=LLM_MODEL, temperature=0.1)


def answer_question(question: str, k: int = 5) -> tuple[str, list]:
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
                       progress_cb=None, text_cb=None, target: str = "en") -> str:
    """LLM fallback translation. target: 'en' or 'de'."""
    text = extract_text(path)
    llm = get_llm()

    lang = "English" if target == "en" else "German"
    prompt = (f"Translate the following text to {lang}. Preserve structure, "
              f"headings, and lists. Output only the translation, nothing else.\n\n"
              f"Text:\n{{text}}\n/no_think")

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
        for piece in llm.stream([("user", prompt.format(text=chunk))]):
            partial += piece.content
            if text_cb:
                text_cb("\n\n".join(translated) + "\n\n" + partial)
        translated.append(strip_think(partial))
        if progress_cb:
            progress_cb(i + 1, len(chunks))
    return "\n\n".join(translated)
