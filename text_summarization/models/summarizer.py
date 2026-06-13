"""
models/summarizer.py
--------------------
Low-level inference wrapper around the T5 model.

This module is the only place in the codebase that calls
``model.generate()``.  It accepts pre-tokenized tensors and returns decoded
text, keeping the generation logic isolated from business rules.

Author: Gerges Emad
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from config import DEVICE, GENERATION_CONFIG, GenerationConfig, MODEL_CONFIG
from models.model_loader import load_model_and_tokenizer
from utils.logger import get_logger
from utils.preprocessing import prepare_input

logger = get_logger(__name__)


class Summarizer:
    """
    Thin inference wrapper for T5-based abstractive summarisation.

    The class owns the tokenizer and model references, handles tokenisation,
    calls ``model.generate()``, and returns decoded summary text together with
    timing information.

    Parameters
    ----------
    tokenizer:
        Loaded T5 tokenizer.  If ``None``, loaded automatically via
        :func:`~models.model_loader.load_model_and_tokenizer`.
    model:
        Loaded T5 model in eval mode.  If ``None``, loaded automatically.
    device:
        Target device for inference.
    """

    def __init__(
        self,
        tokenizer: Optional[T5Tokenizer] = None,
        model: Optional[T5ForConditionalGeneration] = None,
        device: str = DEVICE,
    ) -> None:
        self._device = device

        if tokenizer is None or model is None:
            self._tokenizer, self._model = load_model_and_tokenizer(
                MODEL_CONFIG, device
            )
        else:
            self._tokenizer = tokenizer
            self._model = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summarize(
        self,
        text: str,
        generation_config: GenerationConfig = GENERATION_CONFIG,
    ) -> Tuple[str, float]:
        """
        Preprocess *text*, tokenise it, generate a summary, and decode it.

        Parameters
        ----------
        text:
            Raw article text (preprocessing is applied internally).
        generation_config:
            Beam-search and length parameters; defaults to global
            ``GENERATION_CONFIG``.

        Returns
        -------
        Tuple[str, float]
            ``(summary_text, elapsed_seconds)``

        Raises
        ------
        RuntimeError
            If the model forward pass raises an exception (e.g. OOM).
        ValueError
            If *text* is empty after stripping.
        """
        stripped = text.strip()
        if not stripped:
            raise ValueError("Input text must not be empty.")

        input_text = prepare_input(stripped)
        inputs = self._tokenize(input_text, generation_config.max_new_tokens)

        logger.debug(
            "Generating summary — tokens: %d, beams: %d, max_new: %d",
            inputs["input_ids"].shape[1],
            generation_config.num_beams,
            generation_config.max_new_tokens,
        )

        t0 = time.time()
        try:
            with torch.no_grad():
                output_ids = self._model.generate(
                    inputs["input_ids"],
                    max_new_tokens=generation_config.max_new_tokens,
                    min_length=generation_config.min_length,
                    num_beams=generation_config.num_beams,
                    length_penalty=generation_config.length_penalty,
                    early_stopping=generation_config.early_stopping,
                    no_repeat_ngram_size=generation_config.no_repeat_ngram_size,
                )
        except RuntimeError as exc:
            logger.exception("Model.generate() failed: %s", exc)
            raise

        elapsed = time.time() - t0
        summary = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)

        logger.info(
            "Summary generated in %.2fs — %d words",
            elapsed,
            len(summary.split()),
        )
        return summary, elapsed

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _tokenize(self, input_text: str, max_length: int) -> dict:
        """
        Tokenise *input_text* and return tensors on the configured device.

        Parameters
        ----------
        input_text:
            Preprocessed text with T5 ``summarize:`` prefix already applied.
        max_length:
            Maximum number of input tokens (encoder side).

        Returns
        -------
        dict
            BatchEncoding with ``input_ids`` (and optionally
            ``attention_mask``) as tensors on ``self._device``.
        """
        return self._tokenizer(
            input_text,
            return_tensors="pt",
            max_length=MODEL_CONFIG.max_input_length,
            truncation=True,
            padding="longest",
        ).to(self._device)
