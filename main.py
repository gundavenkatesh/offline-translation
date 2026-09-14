"""Streamlit UI: auto-ingest, bilingual chat, bidirectional translation."""
import os
from pathlib import Path

import streamlit as st

from ingest import ingest_file, list_sources, is_indexed
from rag import answer_question
from translate import get_translation, active_engine

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "data/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Offline Document Chatbot", page_icon="📄")
st.title("📄 Offline Document Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "handled_uploads" not in st.session_state:
    st.session_state.handled_uploads = set()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Documents")

    # Feature 1: auto-ingest on upload (no Ingest button), with dedup + status.
    uploaded = st.file_uploader(
        "Upload documents (indexed automatically)",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploaded:
        for f in uploaded:
            key = f"{f.name}:{f.size}"
            if key in st.session_state.handled_uploads:
                continue
            dest = UPLOAD_DIR / f.name
            dest.write_bytes(f.getbuffer())
            if is_indexed(f.name):
                st.info(f"↩︎ {f.name} already indexed — skipped")
            else:
                with st.spinner(f"Indexing {f.name}…"):
                    n = ingest_file(str(dest))
                st.success(f"✓ {f.name}: {n} chunks indexed")
            st.session_state.handled_uploads.add(key)

    sources = list_sources()
    if sources:
        st.subheader("Indexed files")
        for s in sources:
            st.text(f"• {s}")

        st.divider()
        st.subheader("Full translation")

        # Feature 2: direction toggle, default DE -> EN.
        direction_label = st.radio(
            "Direction",
            ["German → English", "English → German"],
            index=0,
            horizontal=True,
        )
        direction = "de-en" if direction_label.startswith("German") else "en-de"

        sel = st.selectbox("File to translate", sources)

        translating = st.session_state.get("translating", False)
        if st.button("Translate", disabled=translating):
            st.session_state.translating = True
            st.session_state.translate_file = sel
            st.session_state.translate_dir = direction
            st.rerun()

        if translating:
            fname = st.session_state.translate_file
            tdir = st.session_state.translate_dir
            path = UPLOAD_DIR / fname
            if path.exists():
                with st.status("Translating…", expanded=True) as status:
                    bar = st.progress(0)
                    preview = st.empty()

                    def update(done, total):
                        bar.progress(done / total if total else 0.0)
                        status.update(label=f"Translating piece {done}/{total}")

                    def show_text(t):
                        preview.markdown(t[-1500:])

                    result, cached = get_translation(
                        str(path), direction=tdir,
                        progress_cb=update, text_cb=show_text,
                    )
                    label = ("Loaded from cache" if cached
                             else f"Translation complete ({active_engine()})")
                    status.update(label=label, state="complete")

                arrow = "DE→EN" if tdir == "de-en" else "EN→DE"
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"**Translation of {fname} ({arrow}):**\n\n{result}",
                })
                st.session_state.last_translation = (fname, tdir, result)
            else:
                st.error("Original file not found on disk.")
            st.session_state.translating = False
            st.rerun()

# ---------- Chat history ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- Download last translation ----------
if "last_translation" in st.session_state:
    name, tdir, text = st.session_state.last_translation
    suffix = "EN" if tdir == "de-en" else "DE"
    st.download_button(
        f"⬇️ Download last translation ({suffix})",
        text,
        file_name=f"{Path(name).stem}_{suffix}.txt",
    )

# ---------- Chat input ----------
if prompt := st.chat_input("Ask about your documents (German or English)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not list_sources():
            reply = "No documents indexed yet. Upload files in the sidebar first."
            st.markdown(reply)
        else:
            with st.spinner("Thinking…"):
                reply, docs = answer_question(prompt)
            st.markdown(reply)
            with st.expander("Sources"):
                for d in docs:
                    st.caption(
                        f"**{d.metadata.get('source')}** (chunk {d.metadata.get('chunk')})"
                    )
                    st.text(d.page_content[:300])
    st.session_state.messages.append({"role": "assistant", "content": reply})
