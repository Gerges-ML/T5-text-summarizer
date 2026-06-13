"""
utils/preprocessing.py
-----------------------
Text cleaning and normalisation utilities used before tokenisation.

All preprocessing logic that was previously inline in the notebook is
consolidated here so it can be reused by the service layer, the batch
evaluator, and the UI without duplication.

Author: Gerges Emad
"""

from __future__ import annotations

import re
from typing import List

from utils.logger import get_logger

logger = get_logger(__name__)

# T5 requires this prefix on every input sequence.
T5_SUMMARIZE_PREFIX: str = "summarize: "


def clean_text(text: str) -> str:
    """
    Clean and normalise raw article text before feeding it to the model.

    The pipeline applies four transformations in order:

    1. Remove HTTP/HTTPS URLs and bare ``www.`` links.
    2. Collapse consecutive whitespace (including newlines and tabs) to a
       single space and strip leading/trailing whitespace.
    3. Remove characters that are not alphanumeric, sentence-ending
       punctuation, apostrophes, or hyphens.
    4. Lowercase the result.

    Parameters
    ----------
    text:
        Raw article text, potentially containing URLs, newlines, and
        special characters.

    Returns
    -------
    str
        Cleaned, lower-cased text ready for tokenisation.

    Raises
    ------
    ValueError
        If *text* is not a string.

    Examples
    --------
    >>> from utils.preprocessing import clean_text
    >>> clean_text("Visit https://example.com for more!\\n\\nThis is a test.")
    'visit  for more! this is a test.'
    """
    if not isinstance(text, str):
        raise ValueError(f"Expected str, got {type(text).__name__}")

    # 1. Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # 2. Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # 3. Strip non-linguistic special characters
    text = re.sub(r"[^\w\s.,!?'\-]", "", text)
    # 4. Lowercase
    text = text.lower()

    return text


def add_t5_prefix(text: str) -> str:
    """
    Prepend the ``summarize:`` task prefix required by T5.

    Parameters
    ----------
    text:
        Cleaned article text (output of :func:`clean_text`).

    Returns
    -------
    str
        Text prefixed with ``'summarize: '``.
    """
    return T5_SUMMARIZE_PREFIX + text


def prepare_input(text: str) -> str:
    """
    Full preprocessing pipeline: clean then add T5 prefix.

    Convenience wrapper that combines :func:`clean_text` and
    :func:`add_t5_prefix` into one call.

    Parameters
    ----------
    text:
        Raw article text.

    Returns
    -------
    str
        Cleaned, lower-cased text with T5 summarisation prefix.
    """
    cleaned = clean_text(text)
    logger.debug("Preprocessing: %d → %d chars", len(text), len(cleaned))
    return add_t5_prefix(cleaned)


def count_words(text: str) -> int:
    """
    Return the number of whitespace-delimited tokens in *text*.

    Parameters
    ----------
    text:
        Any string.

    Returns
    -------
    int
        Word count (0 for empty or whitespace-only input).
    """
    stripped = text.strip()
    return len(stripped.split()) if stripped else 0


def compute_compression_ratio(original: str, summary: str) -> float:
    """
    Compute the percentage of words removed by summarisation.

    .. math::
        \\text{compression} = \\left(1 - \\frac{|\\text{summary}|}{|\\text{original}|}\\right) \\times 100

    Parameters
    ----------
    original:
        Source article text.
    summary:
        Generated summary text.

    Returns
    -------
    float
        Compression percentage in [0, 100].  Returns 0.0 if *original* is
        empty to avoid division by zero.
    """
    orig_words = count_words(original)
    if orig_words == 0:
        return 0.0
    summary_words = count_words(summary)
    ratio = (1 - summary_words / orig_words) * 100
    return max(0.0, ratio)


def batch_prepare_inputs(texts: List[str]) -> List[str]:
    """
    Apply :func:`prepare_input` to a list of article texts.

    Parameters
    ----------
    texts:
        List of raw article strings.

    Returns
    -------
    List[str]
        List of preprocessed, prefixed strings ready for tokenisation.
    """
    return [prepare_input(t) for t in texts]
