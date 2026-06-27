"""
app.py — Streamlit UI for the Offline RAG Framework
====================================================
This is the frontend of the application. It provides a generic, domain-
agnostic chat interface for querying the knowledge base.

To run:
    streamlit run app.py

Features:
    - Sidebar with database stats and LLM health checks.
    - Streaming LLM responses (typewriter effect).
    - Expandable "Retrieved Context" section showing exact source
      documents, metadata, and similarity scores.
    - Expandable "Debug Info" section showing the raw prompt sent
      to the LLM (useful for prompt engineering and debugging).
    - Session state management so the retriever and LLM are only
      initialised once per session, not on every re-render.

This module relies on:
    - config.py      (for titles, labels, and settings)
    - retriever.py   (for semantic search)
    - llm.py         (for Ollama inference)
    - prompts.py     (for prompt construction)
"""

import time
from typing import Any

import streamlit as st

from config import config
from llm import OllamaLLM
from logger import get_logger
from prompts import build_rag_prompt, format_prompt_preview, NO_ANSWER_PHRASE
from retriever import RAGRetriever, load_database

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Initialisation / Session State
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    """
    Initialise heavy objects (Retriever and LLM) once per user session.
    Streamlit re-runs the entire script on every user interaction, so
    we must cache these objects in st.session_state.
    """
    if "retriever" not in st.session_state:
        try:
            st.session_state.retriever = load_database()
            log.info("Retriever initialised in session state.")
        except Exception as exc:
            st.session_state.retriever_error = str(exc)
            log.error("Failed to initialise retriever: %s", exc)

    if "llm" not in st.session_state:
        try:
            st.session_state.llm = OllamaLLM()
            log.info("LLM initialised in session state.")
        except Exception as exc:
            st.session_state.llm_error = str(exc)
            log.error("Failed to initialise LLM: %s", exc)

    # Chat history: list of dicts {"role": "user"|"assistant", "content": str, ...}
    if "messages" not in st.session_state:
        st.session_state.messages = []


# ---------------------------------------------------------------------------
# Sidebar (Status and Settings)
# ---------------------------------------------------------------------------

def _render_sidebar() -> None:
    """
    Render the sidebar containing database statistics, LLM status,
    and optional settings.
    """
    with st.sidebar:
        st.title(config.APP_TITLE)
        st.markdown(config.APP_DESCRIPTION)
        st.divider()

        # Database Status
        st.subheader("📊 Knowledge Base")
        if "retriever" in st.session_state and hasattr(st.session_state.retriever, "get_info"):
            retriever: RAGRetriever = st.session_state.retriever
            info = retriever.get_info()

            if info.is_empty:
                st.error(f"Collection '{info.collection_name}' is empty.")
                st.markdown("Run `python ingest.py` in your terminal.")
            else:
                st.success(f"{info.document_count:,} documents indexed")
                st.caption(f"Collection: `{info.collection_name}`")
                st.caption(f"Model: `{info.embedding_model}`")
                st.caption(f"Top-K: `{config.TOP_K}`")
                st.caption(f"Threshold: `{config.SIMILARITY_THRESHOLD}`")
        elif "retriever_error" in st.session_state:
            st.error("Database Error")
            st.caption(st.session_state.retriever_error)
        else:
            st.warning("Database not loaded.")

        st.divider()

        # LLM Status
        st.subheader("🤖 Local LLM (Ollama)")
        if "llm" in st.session_state:
            llm: OllamaLLM = st.session_state.llm
            is_ok, msg = llm.is_available()

            if is_ok:
                st.success(f"Connected: `{config.OLLAMA_MODEL}`")
                st.caption(f"Host: `{config.OLLAMA_HOST}`")
                st.caption(f"Temperature: `{config.OLLAMA_TEMPERATURE}`")
            else:
                st.error("Ollama Unavailable")
                st.caption(msg)
        else:
            st.warning("LLM not initialised.")

        st.divider()

        # Settings
        st.subheader("⚙️ Settings")
        st.session_state.stream_output = st.checkbox(
            "Stream LLM output",
            value=True,
            help="Show text as it is generated (typewriter effect).",
        )

        st.divider()
        st.caption("Offline RAG Framework — Fully local, zero cloud dependencies.")


# ---------------------------------------------------------------------------
# Main Chat Interface
# ---------------------------------------------------------------------------

def _render_chat_message(msg: dict[str, Any]) -> None:
    """
    Render a single chat message from the session history.

    Args:
        msg: Message dict containing role, content, and optional context.
    """
    role = msg["role"]
    avatar = "👤" if role == "user" else "🤖"

    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["content"])

        # If this is an assistant message with context, show the expanders
        if role == "assistant" and "context_docs" in msg:
            _render_context_expander(msg["context_docs"], msg.get("retrieval_time", 0))

            if "prompt_preview" in msg:
                with st.expander("🛠️ Debug: Raw Prompt Sent to LLM", expanded=False):
                    st.code(msg["prompt_preview"], language="text")

            # Show inference time if available
            if "inference_time" in msg:
                st.caption(f"⏱️ Inference time: {msg['inference_time']:.2f}s")


def _render_context_expander(docs: list[dict[str, Any]], elapsed: float) -> None:
    """
    Render an expander showing the exact documents retrieved from ChromaDB.

    Args:
        docs: List of result dicts from retriever.retrieve().
        elapsed: Time taken for the retrieval step.
    """
    title = f"📚 Retrieved Context ({len(docs)} documents in {elapsed:.2f}s)"
    with st.expander(title, expanded=False):
        if not docs:
            st.warning("No documents met the similarity threshold.")
            return

        for idx, doc in enumerate(docs, 1):
            text = doc.get("text", "")
            meta = doc.get("metadata", {})
            sim  = doc.get("similarity", 0.0)

            # Build a nice header for each document
            source = meta.get("source", "Unknown")
            st.markdown(f"**[{idx}] Source: {source} | Similarity: {sim * 100:.1f}%**")

            # Show custom metadata if it exists (e.g., CSV row details)
            custom_meta = {k: v for k, v in meta.items() if k not in ["source", "format", "chunk_index"]}
            if custom_meta:
                meta_str = " | ".join(f"{k.capitalize()}: {v}" for k, v in custom_meta.items())
                st.caption(meta_str)

            # Show the actual text
            st.info(text)


def _handle_user_input(prompt: str) -> None:
    """
    Process a new user query:
        1. Display user message.
        2. Retrieve context from ChromaDB.
        3. Build system/user prompts.
        4. Call Ollama (streaming or block).
        5. Display response and save to history.

    Args:
        prompt: The user's query string.
    """
    # 1. Append and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Prevent processing if critical components failed to load
    if "retriever" not in st.session_state or "llm" not in st.session_state:
        st.error("System is not fully initialised. Check sidebar for errors.")
        return

    retriever: RAGRetriever = st.session_state.retriever
    llm: OllamaLLM          = st.session_state.llm

    # Assistant response container
    with st.chat_message("assistant", avatar="🤖"):
        # 2. Retrieval Phase
        with st.spinner("🔍 Searching knowledge base..."):
            t0 = time.perf_counter()
            context_docs = retriever.retrieve(prompt)
            retrieval_time = time.perf_counter() - t0

        # 3. Prompt Construction
        sys_prompt, user_prompt = build_rag_prompt(context_docs, prompt)
        prompt_preview = format_prompt_preview(sys_prompt, user_prompt)

        # 4. LLM Inference Phase
        answer_placeholder = st.empty()
        full_answer = ""
        inference_time = 0.0

        if st.session_state.stream_output:
            # Streaming mode
            t0 = time.perf_counter()
            try:
                # stream_generate yields chunks of text
                for chunk in llm.stream_generate(sys_prompt, user_prompt):
                    full_answer += chunk
                    answer_placeholder.markdown(full_answer + "▌")

                inference_time = time.perf_counter() - t0
                answer_placeholder.markdown(full_answer)
            except Exception as exc:
                full_answer = f"⚠️ Error generating response: {exc}"
                answer_placeholder.error(full_answer)
        else:
            # Blocking mode
            with st.spinner("🤖 Generating answer..."):
                response = llm.generate(sys_prompt, user_prompt)
                full_answer = response.answer
                inference_time = response.elapsed_seconds
                if response.error:
                    answer_placeholder.error(response.error)
                else:
                    answer_placeholder.markdown(full_answer)

        # Highlight if it's the fallback phrase
        if NO_ANSWER_PHRASE.lower() in full_answer.lower():
            st.warning("The LLM indicates the answer is not in the knowledge base.")

        # Display context expanders inline for the current message
        _render_context_expander(context_docs, retrieval_time)

        with st.expander("🛠️ Debug: Raw Prompt Sent to LLM", expanded=False):
            st.code(prompt_preview, language="text")

        st.caption(f"⏱️ Inference time: {inference_time:.2f}s")

    # 5. Save assistant message to history
    st.session_state.messages.append({
        "role":           "assistant",
        "content":        full_answer,
        "context_docs":   context_docs,
        "retrieval_time": retrieval_time,
        "inference_time": inference_time,
        "prompt_preview": prompt_preview,
    })


# ---------------------------------------------------------------------------
# Main Application Layout
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Main Streamlit application entry point.
    """
    # Configure page
    st.set_page_config(
        page_title=config.APP_TITLE,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialise backend (run once per session)
    _init_session_state()

    # Render UI components
    _render_sidebar()

    st.title("Offline RAG Assistant")
    st.caption("Ask questions about the indexed documents.")

    # Render chat history
    for msg in st.session_state.messages:
        _render_chat_message(msg)

    # Chat input
    if prompt := st.chat_input("Ask a question based on the knowledge base..."):
        _handle_user_input(prompt)


if __name__ == "__main__":
    main()
