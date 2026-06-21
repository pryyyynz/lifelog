"""LLM client wrappers for chit-chat reply generation.

Currently supports Ollama (local). The client is optional — if Ollama is not
installed or not reachable, ``OllamaClient`` raises ``LLMUnavailableError``
at construction time so the caller can fall back to ``GENERIC_FALLBACK``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class LLMUnavailableError(RuntimeError):
    """Raised when the LLM backend cannot be reached or is not installed."""


class OllamaClient:
    """Thin wrapper around the ``ollama`` Python package.

    Implements the ``generate(prompt: str) -> str`` interface expected by
    ``IntentClassifier``.

    Args:
        model: Ollama model name, e.g. ``"llama3"`` or ``"mistral"``.
        host: Ollama server URL, defaults to ``"http://localhost:11434"``.
        system_prompt: System-level instruction prepended to every request.
        timeout: Request timeout in seconds.

    Raises:
        LLMUnavailableError: If the ``ollama`` package is not installed or the
            server is not reachable at construction time.
    """

    _SYSTEM_PROMPT = (
        "You are a concise, friendly assistant for a personal life log search app. "
        "The app lets users search their own photos, notes, audio, video, emails, "
        "and calendar events stored locally on their machine. "
        "Respond naturally and briefly (1-3 sentences) to the user's message. "
        "Do not make up information about the user's data. "
        "If the user seems to be asking to search their data, encourage them to ask "
        "a search question instead."
    )

    def __init__(
        self,
        model: str = "llama3",
        host: str = "http://localhost:11434",
        system_prompt: str = _SYSTEM_PROMPT,
        timeout: int = 30,
    ) -> None:
        try:
            import ollama  # noqa: PLC0415
        except ImportError as exc:
            raise LLMUnavailableError(
                "The 'ollama' package is not installed. "
                "Run: pip install 'ollama>=0.3'"
            ) from exc

        self._ollama = ollama
        self._model = model
        self._host = host
        self._system_prompt = system_prompt
        self._timeout = timeout

        # Probe the server once at construction — fail fast if unreachable.
        self._client = ollama.Client(host=host)
        try:
            self._client.list()
        except Exception as exc:  # noqa: BLE001
            raise LLMUnavailableError(
                f"Ollama server not reachable at {host}: {exc}"
            ) from exc

        logger.info("OllamaClient ready — model=%s host=%s", model, host)

    def generate(self, prompt: str, *, system: str | None = None, num_predict: int = 150) -> str:
        """Send *prompt* to Ollama and return the generated text.

        ``system`` overrides the default system prompt for this call (used by the
        RAG answer/decomposition jobs); ``num_predict`` caps the response length.
        Falls back to an empty string on any error so the caller can decide
        whether to use ``GENERIC_FALLBACK``.
        """
        try:
            response = self._client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": system or self._system_prompt},
                    {"role": "user", "content": prompt},
                ],
                options={"num_predict": num_predict},
            )
            return response["message"]["content"].strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama generate failed: %s", exc)
            return ""


def build_ollama_client(
    model: str,
    host: str,
) -> OllamaClient | None:
    """Try to build an ``OllamaClient``; return ``None`` on failure.

    This is the safe factory used at startup — it logs a warning instead of
    crashing the server when Ollama is unavailable.
    """
    try:
        return OllamaClient(model=model, host=host)
    except LLMUnavailableError as exc:
        logger.warning(
            "Ollama unavailable — chit-chat replies will use generic fallback. Reason: %s", exc
        )
        return None
