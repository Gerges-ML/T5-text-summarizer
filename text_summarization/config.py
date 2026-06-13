"""
config.py
---------
Centralized configuration for the Text Summarization application.

All tunable constants, model parameters, generation defaults, and
environment-specific settings live here so that no magic numbers are
scattered across the codebase.

Author: Gerges Emad
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def _resolve_device() -> str:
    """Return 'cuda' if a CUDA-capable GPU is available, otherwise 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


DEVICE: str = os.environ.get("SUMMARIZER_DEVICE", _resolve_device())


# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """Immutable settings for the T5 model and tokenizer."""

    name: str = "t5-small"
    """HuggingFace model identifier (default: t5-small, 60 M parameters)."""

    max_input_length: int = 512
    """Maximum number of tokens accepted by the encoder."""

    cache_dir: str | None = None
    """Optional local directory for caching downloaded model weights."""


# ---------------------------------------------------------------------------
# Generation / inference settings
# ---------------------------------------------------------------------------

@dataclass
class GenerationConfig:
    """Default beam-search generation parameters (all overridable at runtime)."""

    max_new_tokens: int = 120
    """Upper bound on the number of output tokens to generate."""

    min_length: int = 20
    """Lower bound on output length (forces the decoder to keep writing)."""

    num_beams: int = 4
    """Number of beams for beam-search decoding (1 = greedy)."""

    length_penalty: float = 2.0
    """Exponential penalty applied to sequence length during beam ranking.
    Values > 1.0 favour longer summaries; values < 1.0 favour shorter ones."""

    early_stopping: bool = True
    """Stop beam search once ``num_beams`` finished sequences have been found."""

    no_repeat_ngram_size: int = 3
    """Prevent any n-gram of this size from appearing more than once in output."""


# ---------------------------------------------------------------------------
# ROUGE evaluation settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvaluationConfig:
    """Settings for ROUGE-based evaluation."""

    metrics: List[str] = field(default_factory=lambda: ["rouge1", "rouge2", "rougeL"])
    """ROUGE variants to compute."""

    use_stemmer: bool = True
    """Apply Porter stemming before computing ROUGE overlap."""

    batch_size: int = 10
    """Number of test samples used in the batch evaluation pipeline."""


# ---------------------------------------------------------------------------
# Dataset settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetConfig:
    """Settings for the CNN/DailyMail dataset loader."""

    name: str = "cnn_dailymail"
    version: str = "3.0.0"
    article_column: str = "article"
    highlights_column: str = "highlights"


# ---------------------------------------------------------------------------
# Gradio / UI settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UIConfig:
    """Settings for the Gradio web interface."""

    title: str = "T5 Text Summarizer"
    share: bool = False
    debug: bool = False
    server_port: int = 7860
    server_name: str = "0.0.0.0"

    slider_max_tokens_range: tuple[int, int] = (40, 200)
    slider_num_beams_range: tuple[int, int] = (1, 8)
    slider_length_penalty_range: tuple[float, float] = (0.5, 3.0)
    slider_no_repeat_range: tuple[int, int] = (0, 5)

    min_input_words: int = 20
    """Minimum number of words required in the input text."""


# ---------------------------------------------------------------------------
# Logging settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LoggingConfig:
    """Settings for the application-wide logger."""

    level: str = os.environ.get("LOG_LEVEL", "INFO")
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    log_to_file: bool = False
    log_file_path: str = "logs/summarizer.log"


# ---------------------------------------------------------------------------
# Top-level singleton instances (import these elsewhere)
# ---------------------------------------------------------------------------

MODEL_CONFIG = ModelConfig()
GENERATION_CONFIG = GenerationConfig()
EVALUATION_CONFIG = EvaluationConfig()
DATASET_CONFIG = DatasetConfig()
UI_CONFIG = UIConfig()
LOGGING_CONFIG = LoggingConfig()
