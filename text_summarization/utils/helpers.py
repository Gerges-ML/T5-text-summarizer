"""
utils/helpers.py
----------------
General-purpose helper utilities that do not belong to a specific domain layer.

Includes ROUGE score interpretation, result formatting, and UI-oriented
word-count colouring logic extracted from the notebook.

Author: Gerges Emad
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class RougeScores:
    """Container for a single document's ROUGE scores."""

    rouge1: float
    rouge2: float
    rougeL: float

    def as_dict(self) -> Dict[str, float]:
        """Return scores as a plain dictionary."""
        return {"rouge1": self.rouge1, "rouge2": self.rouge2, "rougeL": self.rougeL}


@dataclass
class SummaryResult:
    """Holds all artefacts produced by a single summarisation call."""

    summary: str
    scores: RougeScores
    compression_pct: float
    input_word_count: int
    output_word_count: int
    elapsed_seconds: float
    interpretation: str


# ---------------------------------------------------------------------------
# ROUGE interpretation
# ---------------------------------------------------------------------------

# Thresholds align with the notebook's original interpretation comments.
_ROUGE1_STRONG_THRESHOLD: float = 0.40
_ROUGE1_MODERATE_THRESHOLD: float = 0.25


def interpret_rouge1(rouge1_score: float, lang: str = "en") -> str:
    """
    Return a human-readable interpretation of a ROUGE-1 F1 score.

    Parameters
    ----------
    rouge1_score:
        ROUGE-1 F-measure in [0, 1].
    lang:
        ``'en'`` for English or ``'ar'`` for Arabic interpretation strings.

    Returns
    -------
    str
        Interpretation message.
    """
    messages: Dict[str, Dict[str, str]] = {
        "en": {
            "strong": "Strong overlap with reference — excellent coverage.",
            "moderate": "Moderate overlap — acceptable for a small abstractive model.",
            "low": "Low overlap — summary is highly abstractive or diverges from reference.",
        },
        "ar": {
            "strong": "تداخل قوي مع المرجع — تغطية ممتازة.",
            "moderate": "تداخل معتدل — مقبول لنموذج استخلاصي صغير.",
            "low": "تداخل منخفض — الملخص مجرد جداً أو يختلف عن المرجع.",
        },
    }

    lang_msgs = messages.get(lang, messages["en"])

    if rouge1_score >= _ROUGE1_STRONG_THRESHOLD:
        return lang_msgs["strong"]
    elif rouge1_score >= _ROUGE1_MODERATE_THRESHOLD:
        return lang_msgs["moderate"]
    else:
        return lang_msgs["low"]


# ---------------------------------------------------------------------------
# Batch statistics
# ---------------------------------------------------------------------------

def compute_average_scores(score_list: list[RougeScores]) -> RougeScores:
    """
    Compute the mean ROUGE scores over a list of :class:`RougeScores`.

    Parameters
    ----------
    score_list:
        Non-empty list of per-document ROUGE scores.

    Returns
    -------
    RougeScores
        Averaged ROUGE-1, ROUGE-2, and ROUGE-L values.

    Raises
    ------
    ValueError
        If *score_list* is empty.
    """
    if not score_list:
        raise ValueError("score_list must not be empty.")

    n = len(score_list)
    avg_r1 = sum(s.rouge1 for s in score_list) / n
    avg_r2 = sum(s.rouge2 for s in score_list) / n
    avg_rl = sum(s.rougeL for s in score_list) / n

    logger.debug(
        "Batch averages (n=%d): ROUGE-1=%.4f  ROUGE-2=%.4f  ROUGE-L=%.4f",
        n, avg_r1, avg_r2, avg_rl,
    )
    return RougeScores(rouge1=avg_r1, rouge2=avg_r2, rougeL=avg_rl)


def interpret_batch_averages(averages: RougeScores) -> Dict[str, str]:
    """
    Produce interpretations for all three averaged ROUGE metrics.

    Parameters
    ----------
    averages:
        Averaged ROUGE scores from :func:`compute_average_scores`.

    Returns
    -------
    Dict[str, str]
        Keys: ``'rouge1'``, ``'rouge2'``, ``'rougeL'``.
    """
    return {
        "rouge1": (
            "Good unigram coverage."
            if averages.rouge1 >= 0.35
            else "Moderate unigram coverage."
        ),
        "rouge2": (
            "Good bigram fluency."
            if averages.rouge2 >= 0.15
            else "Low bigram overlap; summaries are abstractive."
        ),
        "rougeL": (
            "Strong sequence preservation."
            if averages.rougeL >= 0.30
            else "Moderate sequence overlap."
        ),
    }


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def word_count_color(word_count: int) -> str:
    """
    Return a CSS hex colour representing input readiness.

    * Grey  → 0 words (empty)
    * Red   → fewer than 20 words (below minimum)
    * Amber → 20–49 words (short but valid)
    * Green → 50+ words (good length)

    Parameters
    ----------
    word_count:
        Number of words in the input textarea.

    Returns
    -------
    str
        CSS colour string, e.g. ``'#34d399'``.
    """
    if word_count == 0:
        return "#334155"
    elif word_count < 20:
        return "#f87171"
    elif word_count < 50:
        return "#fbbf24"
    else:
        return "#34d399"


def format_stats_line(
    input_words: int,
    output_words: int,
    num_beams: int,
    max_tokens: int,
    length_penalty: float,
    lang: str = "en",
) -> str:
    """
    Build the compact statistics line shown beneath the summary output.

    Parameters
    ----------
    input_words:    Word count of the original input.
    output_words:   Word count of the generated summary.
    num_beams:      Beam-search width used.
    max_tokens:     Maximum output tokens configured.
    length_penalty: Length penalty applied during generation.
    lang:           ``'en'`` or ``'ar'`` for localised formatting.

    Returns
    -------
    str
        Formatted statistics string.
    """
    templates: Dict[str, str] = {
        "en": (
            "Input: {in_w} words → Summary: {out_w} words"
            "  |  Beams: {beams}  |  Max tokens: {mt}  |  Length penalty: {lp}"
        ),
        "ar": (
            "المدخل: {in_w} كلمة ← الملخص: {out_w} كلمة"
            "  |  الأشعة: {beams}  |  الرموز: {mt}  |  معامل الطول: {lp}"
        ),
    }
    template = templates.get(lang, templates["en"])
    return template.format(
        in_w=input_words,
        out_w=output_words,
        beams=num_beams,
        mt=max_tokens,
        lp=length_penalty,
    )
