"""
models/model_loader.py
----------------------
Responsible for downloading, caching, and returning the T5 tokenizer and
model.  Loading is separated from inference so that:

  * Unit tests can mock the loader without touching the GPU.
  * The service layer never needs to know about HuggingFace internals.
  * Models are loaded lazily (on first request) and cached in memory.

Author: Gerges Emad
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from config import DEVICE, MODEL_CONFIG, ModelConfig
from utils.logger import get_logger

logger = get_logger(__name__)

# Module-level cache so repeated calls to load_model() are instant.
_tokenizer_cache: Optional[T5Tokenizer] = None
_model_cache: Optional[T5ForConditionalGeneration] = None


def load_tokenizer(config: ModelConfig = MODEL_CONFIG) -> T5Tokenizer:
    """
    Load (or return cached) T5 tokenizer from HuggingFace Hub.

    Parameters
    ----------
    config:
        Model configuration containing the model name and optional
        ``cache_dir``.

    Returns
    -------
    T5Tokenizer
        Loaded tokenizer instance.

    Raises
    ------
    OSError
        If the model cannot be downloaded and is not found in the local cache.
    """
    global _tokenizer_cache
    if _tokenizer_cache is not None:
        logger.debug("Returning cached tokenizer.")
        return _tokenizer_cache

    logger.info("Loading tokenizer: %s", config.name)
    try:
        _tokenizer_cache = T5Tokenizer.from_pretrained(
            config.name,
            cache_dir=config.cache_dir,
        )
        logger.info(
            "Tokenizer ready — vocab size: %d", _tokenizer_cache.vocab_size
        )
    except OSError as exc:
        logger.exception("Failed to load tokenizer '%s': %s", config.name, exc)
        raise

    return _tokenizer_cache


def load_model(
    config: ModelConfig = MODEL_CONFIG,
    device: str = DEVICE,
) -> T5ForConditionalGeneration:
    """
    Load (or return cached) T5 model, move it to *device*, and set it to
    inference mode.

    Parameters
    ----------
    config:
        Model configuration containing the model name and optional
        ``cache_dir``.
    device:
        Target device string, e.g. ``'cpu'`` or ``'cuda'``.

    Returns
    -------
    T5ForConditionalGeneration
        Model in ``eval()`` mode on the requested device.

    Raises
    ------
    OSError
        If the model cannot be downloaded and is not found in the local cache.
    RuntimeError
        If moving the model to the target device fails.
    """
    global _model_cache
    if _model_cache is not None:
        logger.debug("Returning cached model.")
        return _model_cache

    logger.info("Loading model: %s  →  device: %s", config.name, device.upper())
    try:
        model = T5ForConditionalGeneration.from_pretrained(
            config.name,
            cache_dir=config.cache_dir,
        )
        model = model.to(device)
        model.eval()
    except (OSError, RuntimeError) as exc:
        logger.exception("Failed to load model '%s': %s", config.name, exc)
        raise

    num_params = sum(p.numel() for p in model.parameters())
    logger.info("Model ready — %.1fM parameters on %s", num_params / 1e6, device.upper())

    _model_cache = model
    return _model_cache


def load_model_and_tokenizer(
    config: ModelConfig = MODEL_CONFIG,
    device: str = DEVICE,
) -> Tuple[T5Tokenizer, T5ForConditionalGeneration]:
    """
    Convenience wrapper: load both tokenizer and model in one call.

    Parameters
    ----------
    config:
        Model configuration.
    device:
        Target device string.

    Returns
    -------
    Tuple[T5Tokenizer, T5ForConditionalGeneration]
        ``(tokenizer, model)`` — both ready for inference.
    """
    tokenizer = load_tokenizer(config)
    model = load_model(config, device)
    return tokenizer, model


def clear_cache() -> None:
    """
    Evict the in-memory tokenizer and model caches.

    Useful in test teardown or when switching between models at runtime.
    Frees GPU memory after clearing the Python references.
    """
    global _tokenizer_cache, _model_cache
    _tokenizer_cache = None
    _model_cache = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model and tokenizer caches cleared.")
