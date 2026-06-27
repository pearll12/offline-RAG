"""
prompts.py — Reusable Prompt Templates for the Offline RAG Framework
=====================================================================
This module contains ALL prompt engineering logic for the framework.

Design principles:
    - Every prompt template is a pure function that takes arguments and
      returns a formatted string. No side effects, no external calls.
    - Prompts are domain-agnostic — they work identically whether the
      knowledge base contains agriculture records, legal documents,
      military manuals, or research papers.
    - Hallucination is prevented by explicit, repeated instruction within
      the prompt. The LLM is told multiple times to use ONLY the context.
    - A strict fallback phrase is defined as a constant so it can be
      detected programmatically in llm.py and displayed distinctly in
      the UI.

Usage:
    from prompts import build_rag_prompt, NO_ANSWER_PHRASE
    prompt = build_rag_prompt(context_docs, user_question)
"""

from typing import Any

# ---------------------------------------------------------------------------
# Sentinel phrase
# ---------------------------------------------------------------------------
# This exact phrase is what the LLM is instructed to return when it cannot
# find sufficient information. It is defined here as a constant so that:
#   1. llm.py can detect it and set a flag for the UI.
#   2. app.py can render it with a distinct warning style.
#   3. It can be changed in ONE place if needed — no string hunting.
NO_ANSWER_PHRASE: str = (
    "I couldn't find sufficient information in the indexed dataset."
)


# ---------------------------------------------------------------------------
# System prompt — defines the LLM's role and hard constraints
# ---------------------------------------------------------------------------
SYSTEM_PROMPT: str = """You are a precise, factual question-answering assistant integrated into an Offline Retrieval-Augmented Generation (RAG) system.

Your role is to answer the user's question using ONLY the information provided in the retrieved context passages below.

STRICT RULES you must follow without exception:
1. Answer ONLY using the information explicitly present in the retrieved context.
2. Do NOT use any of your pre-trained knowledge, assumptions, or general world knowledge.
3. Do NOT fabricate, invent, or extrapolate any facts, figures, names, or dates.
4. Do NOT assume facts that are not stated in the context.
5. If the retrieved context does not contain enough information to answer the question, respond with EXACTLY this phrase and nothing else:
   "{no_answer_phrase}"
6. Keep your answer concise, factual, and well-structured.
7. If the context contains partial information, state only what is explicitly confirmed — do not fill gaps with assumptions.
8. When numerical values are mentioned in the context, repeat them exactly as stated.
9. Do not apologise or explain why you cannot answer beyond the specified fallback phrase.

You are a knowledge retrieval system, not a creative assistant. Accuracy and faithfulness to the source material are your only priorities.""".format(
    no_answer_phrase=NO_ANSWER_PHRASE
)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_rag_prompt(
    retrieved_docs: list[dict[str, Any]],
    user_question: str,
) -> tuple[str, str]:
    """
    Build a complete RAG prompt from retrieved documents and the user question.

    This function constructs a two-part prompt:
        - system: Defines the LLM's role and hard constraints.
        - user:   Contains the numbered retrieved context passages followed
                  by the user's question.

    Prompt structure (user message):
        ┌─────────────────────────────────────┐
        │ RETRIEVED CONTEXT                   │
        │ ─────────────────                   │
        │ [1] <document text>                 │
        │     Source: <filename> | Score: X%  │
        │                                     │
        │ [2] <document text>                 │
        │     ...                             │
        │                                     │
        │ QUESTION                            │
        │ ────────                            │
        │ <user question>                     │
        │                                     │
        │ INSTRUCTIONS                        │
        │ ────────────                        │
        │ Answer using ONLY the context ...   │
        └─────────────────────────────────────┘

    Args:
        retrieved_docs: List of dicts from retriever.retrieve(), each with:
                            - "text":       str   — the passage text
                            - "metadata":   dict  — source info
                            - "similarity": float — cosine similarity score
        user_question:  The user's raw query string.

    Returns:
        A tuple of (system_prompt, user_prompt) strings, ready to be passed
        directly to the Ollama chat API.
    """
    if not retrieved_docs:
        # No context available — instruct LLM to return the fallback phrase
        user_prompt = _build_no_context_prompt(user_question)
        return SYSTEM_PROMPT, user_prompt

    # Build numbered context block
    context_lines: list[str] = ["RETRIEVED CONTEXT", "─" * 40]

    for idx, doc in enumerate(retrieved_docs, start=1):
        text        = doc.get("text", "").strip()
        metadata    = doc.get("metadata", {})
        similarity  = doc.get("similarity", 0.0)

        source      = metadata.get("source", "Unknown source")
        score_pct   = f"{similarity * 100:.1f}%"

        context_lines.append(f"[{idx}] {text}")
        context_lines.append(f"     ↳ Source: {source} | Relevance: {score_pct}")
        context_lines.append("")  # blank line between passages

    # Build the user prompt
    user_prompt = "\n".join([
        *context_lines,
        "",
        "QUESTION",
        "─" * 40,
        user_question.strip(),
        "",
        "INSTRUCTIONS",
        "─" * 40,
        "Using ONLY the retrieved context passages above, provide a clear and",
        "accurate answer to the question.",
        f"If the context does not contain enough information, respond with exactly:",
        f'"{NO_ANSWER_PHRASE}"',
        "Do not use any knowledge outside of the provided context.",
    ])

    return SYSTEM_PROMPT, user_prompt


def _build_no_context_prompt(user_question: str) -> str:
    """
    Build a user prompt for the case where no documents were retrieved
    (e.g., retrieval returned zero results or all fell below the
    similarity threshold).

    This happens when the query is completely unrelated to the knowledge base.
    The LLM is instructed to return the exact fallback phrase.

    Args:
        user_question: The user's raw query string.

    Returns:
        A user prompt string instructing the LLM to return the fallback phrase.
    """
    return "\n".join([
        "RETRIEVED CONTEXT",
        "─" * 40,
        "[No relevant documents were found in the knowledge base for this query.]",
        "",
        "QUESTION",
        "─" * 40,
        user_question.strip(),
        "",
        "INSTRUCTIONS",
        "─" * 40,
        "No relevant context was found for this question.",
        f"You MUST respond with exactly this phrase and nothing else:",
        f'"{NO_ANSWER_PHRASE}"',
    ])


def build_summary_prompt(texts: list[str]) -> tuple[str, str]:
    """
    Build a prompt for summarising a collection of retrieved passages.

    This is an optional utility — not used in the main RAG pipeline but
    available for future features like "Summarise this topic" buttons in the UI.

    Args:
        texts: List of text passages to summarise.

    Returns:
        A tuple of (system_prompt, user_prompt) for the summarisation task.
    """
    system = (
        "You are a precise summarisation assistant. "
        "Summarise the provided passages concisely and factually. "
        "Do not add information that is not present in the passages."
    )
    combined = "\n\n".join(
        f"[{i + 1}] {t.strip()}" for i, t in enumerate(texts)
    )
    user = f"Please summarise the following passages:\n\n{combined}"
    return system, user


def format_prompt_preview(system: str, user: str, max_chars: int = 500) -> str:
    """
    Return a truncated preview of a prompt for debug display in the UI.

    Args:
        system:    The system prompt string.
        user:      The user prompt string.
        max_chars: Maximum characters to show from each section.

    Returns:
        A formatted multi-line string suitable for display in a Streamlit
        expander or code block.
    """
    def truncate(s: str, n: int) -> str:
        return s[:n] + "..." if len(s) > n else s

    return (
        f"=== SYSTEM PROMPT ===\n{truncate(system, max_chars)}\n\n"
        f"=== USER PROMPT ===\n{truncate(user, max_chars)}"
    )
