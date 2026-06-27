"""
llm.py — Local LLM Integration via Ollama
==========================================
This module is the sole interface between the RAG framework and the
locally running Ollama inference server.

Responsibilities:
    - Verify Ollama is running and the configured model is available.
    - Send constructed prompts to the model and return structured responses.
    - Measure and return inference latency.
    - Handle all Ollama-specific errors with user-friendly messages.

This module has NO knowledge of:
    - ChromaDB or embeddings (retriever.py handles that)
    - How prompts are constructed (prompts.py handles that)
    - The Streamlit UI (app.py handles that)
    - The source dataset format (utils.py handles that)

Usage:
    from llm import OllamaLLM
    llm = OllamaLLM()
    result = llm.generate(system_prompt, user_prompt)
    print(result["answer"])
"""

import time
from dataclasses import dataclass, field
from typing import Any

import ollama

from config import config
from logger import get_logger, log_llm_inference
from prompts import NO_ANSWER_PHRASE

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """
    Structured container for a single LLM inference result.

    Using a dataclass instead of a plain dict ensures that callers
    (app.py) get IDE autocompletion and type safety on every field.

    Attributes:
        answer:          The final text answer from the model.
        model:           The Ollama model name that produced this answer.
        elapsed_seconds: Wall-clock time for the full inference call.
        found_answer:    True if the model returned a real answer;
                         False if it returned the NO_ANSWER_PHRASE fallback.
        prompt_tokens:   Approximate token count of the input prompt
                         (from Ollama's usage metadata, if available).
        answer_tokens:   Approximate token count of the generated answer
                         (from Ollama's usage metadata, if available).
        raw_response:    The full raw response dict from Ollama, stored for
                         debug/audit purposes.
        error:           Error message if inference failed; None on success.
    """
    answer:          str
    model:           str
    elapsed_seconds: float
    found_answer:    bool
    prompt_tokens:   int | None = None
    answer_tokens:   int | None = None
    raw_response:    dict[str, Any] = field(default_factory=dict)
    error:           str | None = None


# ---------------------------------------------------------------------------
# Main LLM class
# ---------------------------------------------------------------------------

class OllamaLLM:
    """
    Wrapper around the official Ollama Python client for the RAG framework.

    This class manages:
        - Connection verification at startup
        - Model availability checking
        - Prompt formatting and submission
        - Response parsing and latency measurement
        - Graceful error handling

    The class is instantiated once in app.py and cached in Streamlit's
    session state — avoiding repeated connection checks per query.

    Example:
        llm = OllamaLLM()
        if llm.is_available():
            response = llm.generate(system_prompt, user_prompt)
            print(response.answer)
    """

    def __init__(self) -> None:
        """
        Initialise the Ollama client with the host from config.

        The client is created here but the actual connection to the Ollama
        server is only established when a request is made. Use is_available()
        to perform an explicit health check before the first query.
        """
        self.model:  str = config.OLLAMA_MODEL
        self.host:   str = config.OLLAMA_HOST
        self.client: ollama.Client = ollama.Client(host=self.host)

        log.info("OllamaLLM initialised — model: %s, host: %s", self.model, self.host)

    def is_available(self) -> tuple[bool, str]:
        """
        Check whether the Ollama server is running and the configured model
        is available locally (i.e. has been pulled).

        Returns:
            A tuple of (is_ok: bool, message: str).
                - (True,  "OK") if server is reachable and model is present.
                - (False, "<reason>") if either check fails.

        This method is called once at Streamlit startup to give the user a
        clear, actionable error message if Ollama is not ready.
        """
        # Step 1: Check server reachability
        try:
            models_response = self.client.list()
        except Exception as exc:
            msg = (
                f"Cannot connect to Ollama at {self.host}. "
                f"Please ensure Ollama is running: `ollama serve`. "
                f"Error: {exc}"
            )
            log.error(msg)
            return False, msg

        # Step 2: Check model availability
        try:
            # models_response.models is a list of Model objects
            available_models = [m.model for m in models_response.models]
            # Ollama model names may include a ":latest" suffix — normalise
            normalised = [m.split(":")[0] for m in available_models]
            target     = self.model.split(":")[0]

            if target not in normalised and self.model not in available_models:
                msg = (
                    f"Model '{self.model}' is not available in Ollama. "
                    f"Pull it first with: `ollama pull {self.model}`. "
                    f"Available models: {available_models}"
                )
                log.error(msg)
                return False, msg

        except Exception as exc:
            msg = f"Error listing Ollama models: {exc}"
            log.error(msg)
            return False, msg

        log.info("Ollama health check passed — model '%s' is available.", self.model)
        return True, "OK"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMResponse:
        """
        Send a system + user prompt pair to the Ollama model and return
        a structured LLMResponse.

        Uses the Ollama chat API (messages format) rather than the raw
        generate API because:
            - Chat format supports system/user role separation natively.
            - It is compatible with all modern instruction-tuned models.
            - It allows cleaner prompt construction without manual role tokens.

        Generation parameters:
            - temperature:  From config (default 0.1 for factual accuracy)
            - num_ctx:      From config (default 4096 tokens)
            - num_predict:  512 tokens max — sufficient for factual answers,
                            prevents runaway generation

        Args:
            system_prompt: The system instruction string from prompts.py.
            user_prompt:   The user message string (context + question).

        Returns:
            An LLMResponse dataclass with the answer and metadata.
            On failure, returns an LLMResponse with error set and
            found_answer = False.
        """
        log.info("Sending prompt to Ollama model: %s", self.model)
        start_time = time.perf_counter()

        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                options={
                    "temperature": config.OLLAMA_TEMPERATURE,
                    "num_ctx":     config.OLLAMA_NUM_CTX,
                    "num_predict": 512,
                },
            )

            elapsed = time.perf_counter() - start_time

            # Extract the answer text
            answer: str = response.message.content.strip()

            # Determine if the model found a real answer or returned the fallback
            found_answer: bool = NO_ANSWER_PHRASE.lower() not in answer.lower()

            # Extract token usage if available
            prompt_tokens: int | None = None
            answer_tokens: int | None = None
            try:
                if hasattr(response, "prompt_eval_count"):
                    prompt_tokens = response.prompt_eval_count
                if hasattr(response, "eval_count"):
                    answer_tokens = response.eval_count
            except Exception:
                pass  # Token counts are optional metadata — never fatal

            log_llm_inference(
                log,
                model=self.model,
                elapsed_seconds=elapsed,
                token_estimate=answer_tokens,
            )

            return LLMResponse(
                answer=answer,
                model=self.model,
                elapsed_seconds=elapsed,
                found_answer=found_answer,
                prompt_tokens=prompt_tokens,
                answer_tokens=answer_tokens,
                raw_response=dict(response),
                error=None,
            )

        except ollama.ResponseError as exc:
            elapsed = time.perf_counter() - start_time
            error_msg = (
                f"Ollama model error for '{self.model}': {exc.error}. "
                f"Ensure the model is pulled: `ollama pull {self.model}`"
            )
            log.error(error_msg)
            return LLMResponse(
                answer=NO_ANSWER_PHRASE,
                model=self.model,
                elapsed_seconds=elapsed,
                found_answer=False,
                error=error_msg,
            )

        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            error_msg = (
                f"Unexpected error during LLM inference: {type(exc).__name__}: {exc}"
            )
            log.error(error_msg)
            return LLMResponse(
                answer=NO_ANSWER_PHRASE,
                model=self.model,
                elapsed_seconds=elapsed,
                found_answer=False,
                error=error_msg,
            )

    def stream_generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ):
        """
        Generator variant of generate() that yields answer tokens as they
        are produced by the model (streaming mode).

        Used by app.py when the user enables streaming mode in the UI sidebar.
        Streaming provides a much better perceived response time for slow models.

        Args:
            system_prompt: The system instruction string from prompts.py.
            user_prompt:   The user message string (context + question).

        Yields:
            str — individual text chunks as they are streamed from Ollama.

        Raises:
            RuntimeError: If the Ollama stream cannot be established.
        """
        log.info("Starting streaming generation — model: %s", self.model)

        try:
            stream = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                options={
                    "temperature": config.OLLAMA_TEMPERATURE,
                    "num_ctx":     config.OLLAMA_NUM_CTX,
                    "num_predict": 512,
                },
                stream=True,
            )
            for chunk in stream:
                token = chunk.message.content
                if token:
                    yield token

        except Exception as exc:
            log.error("Streaming error: %s", exc)
            yield f"\n\n[Streaming error: {exc}]"

    def get_model_info(self) -> dict[str, Any]:
        """
        Retrieve metadata about the currently configured Ollama model.

        Returns:
            A dict with model metadata (parameter size, quantisation, etc.)
            or an empty dict if the information cannot be retrieved.
        """
        try:
            info = self.client.show(self.model)
            return {
                "model":        self.model,
                "parameter_size": getattr(info, "details", {}).get("parameter_size", "N/A"),
                "quantisation": getattr(info, "details", {}).get("quantization_level", "N/A"),
                "family":       getattr(info, "details", {}).get("family", "N/A"),
            }
        except Exception as exc:
            log.warning("Could not retrieve model info: %s", exc)
            return {"model": self.model}
