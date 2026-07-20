"""Streamlit UI: upload, chat, streaming translation with live preview."""
import os
from pathlib import Path

import streamlit as st

from ingest import ingest_file, list_sources
from rag import answer_question, translate_document

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "data/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Offline Document Chatbot", page_icon="📄")
st.title("📄 Offline Document Chatbot")

# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Sidebar: upload & manage ----------
with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploaded and st.button("Ingest", type="primary"):
        with st.spinner("Processing..."):
            for f in uploaded:
                dest = UPLOAD_DIR / f.name
                dest.write_bytes(f.getbuffer())
                n = ingest_file(str(dest))
                st.success(f"{f.name}: {n} chunks indexed")

    sources = list_sources()
    if sources:
        st.subheader("Indexed files")
        for s in sources:
            st.text(f"• {s}")

        st.divider()
        st.subheader("Full translation")
        sel = st.selectbox("Translate a file to English", sources)

        translating = st.session_state.get("translating", False)

        if st.button("Translate", disabled=translating):
            st.session_state.translating = True
            st.session_state.translate_file = sel
            st.rerun()

        if translating:
            fname = st.session_state.translate_file
            path = UPLOAD_DIR / fname
            if path.exists():
                with st.status("Translating...", expanded=True) as status:
                    bar = st.progress(0)
                    preview = st.empty()

                    def update(done, total):
                        bar.progress(done / total if total else 0.0)
                        status.update(label=f"Translating block {done}/{total}")

                    def show_text(t):
                        preview.markdown(t[-1500:])

                    result = translate_document(
                        str(path), progress_cb=update, text_cb=show_text
                    )
                    status.update(label="Translation complete", state="complete")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"**Translation of {fname}:**\n\n{result}",
                })
                st.session_state.last_translation = (fname, result)
            else:
                st.error("Original file not found on disk.")
            st.session_state.translating = False
            st.rerun()

# ---------- Chat history ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- Download button for last translation ----------
if "last_translation" in st.session_state:
    name, text = st.session_state.last_translation
    st.download_button(
        "⬇️ Download last translation",
        text,
        file_name=f"{Path(name).stem}_EN.txt",
    )

# ---------- Chat input ----------
if prompt := st.chat_input("Ask about your documents (any language, answer in English)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not list_sources():
            reply = "No documents indexed yet. Upload files in the sidebar first."
            st.markdown(reply)
        else:
            with st.spinner("Thinking..."):
                reply, docs = answer_question(prompt)
            st.markdown(reply)
            with st.expander("Sources"):
                for d in docs:
                    st.caption(
                        f"**{d.metadata.get('source')}** (chunk {d.metadata.get('chunk')})"
                    )
                    st.text(d.page_content[:300])
    st.session_state.messages.append({"role": "assistant", "content": reply})
