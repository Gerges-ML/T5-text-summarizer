"""
services/summarization_service.py
----------------------------------
Orchestration layer that composes the Summarizer, ROUGE scoring, and
dataset access into coherent use-cases consumed by the UI and CLI.

This is the single entry-point that higher layers (``interface/ui.py``,
``app.py``) should call.  It never imports Gradio, Torch, or HuggingFace
directly — those details stay in the model and utils layers.

Use-cases exposed:

  * :meth:`SummarizationService.summarize_text`  — single article
  * :meth:`SummarizationService.evaluate_batch`  — batch ROUGE evaluation
  * :meth:`SummarizationService.load_dataset_sample` — fetch a CNN/DailyMail
    article by index for demo purposes

Author: Gerges Emad
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from rouge_score import rouge_scorer

from config import (
    DEVICE,
    EVALUATION_CONFIG,
    EvaluationConfig,
    GENERATION_CONFIG,
    GenerationConfig,
)
from models.summarizer import Summarizer
from utils.helpers import (
    RougeScores,
    SummaryResult,
    compute_average_scores,
    interpret_rouge1,
)
from utils.logger import get_logger
from utils.preprocessing import compute_compression_ratio, count_words

logger = get_logger(__name__)


class SummarizationService:
    """
    Coordinates summarisation, evaluation, and dataset access.

    Parameters
    ----------
    summarizer:
        Pre-built :class:`~models.summarizer.Summarizer` instance.  If
        ``None`` a new instance is created (which triggers model loading).
    evaluation_config:
        ROUGE scorer configuration.
    """

    def __init__(
        self,
        summarizer: Optional[Summarizer] = None,
        evaluation_config: EvaluationConfig = EVALUATION_CONFIG,
    ) -> None:
        self._summarizer = summarizer or Summarizer(device=DEVICE)
        self._eval_cfg = evaluation_config
        self._rouge_scorer = rouge_scorer.RougeScorer(
            evaluation_config.metrics,
            use_stemmer=evaluation_config.use_stemmer,
        )
        logger.info("SummarizationService ready.")

    # ------------------------------------------------------------------
    # Primary use-case: single article summarisation
    # ------------------------------------------------------------------

    def summarize_text(
        self,
        text: str,
        reference: Optional[str] = None,
        generation_config: GenerationConfig = GENERATION_CONFIG,
        lang: str = "en",
    ) -> SummaryResult:
        """
        Summarise a single article and optionally score against a reference.

        Parameters
        ----------
        text:
            Raw article text (preprocessing handled internally).
        reference:
            Optional gold-standard summary used as the ROUGE reference.
            If omitted the *text* itself is used as the reference (which
            gives a rough self-overlap score).
        generation_config:
            Beam-search parameters.
        lang:
            Language code for the interpretation string (``'en'`` / ``'ar'``).

        Returns
        -------
        SummaryResult
            Dataclass containing the summary, scores, compression, and timing.

        Raises
        ------
        ValueError
            If *text* has fewer than 20 words.
        """
        input_word_count = count_words(text)
        if input_word_count < 20:
            raise ValueError(
                f"Input text is too short ({input_word_count} words). "
                "Please provide at least 20 words."
            )

        logger.info("Summarising article (%d words)…", input_word_count)
        summary, elapsed = self._summarizer.summarize(text, generation_config)

        ref = reference if reference is not None else text
        scores = self._compute_rouge(ref, summary)
        compression = compute_compression_ratio(text, summary)
        interpretation = interpret_rouge1(scores.rouge1, lang=lang)
        output_word_count = count_words(summary)

        result = SummaryResult(
            summary=summary,
            scores=scores,
            compression_pct=compression,
            input_word_count=input_word_count,
            output_word_count=output_word_count,
            elapsed_seconds=elapsed,
            interpretation=interpretation,
        )

        logger.info(
            "Done — ROUGE-1: %.4f  ROUGE-2: %.4f  ROUGE-L: %.4f  "
            "Compression: %.1f%%  Time: %.2fs",
            scores.rouge1,
            scores.rouge2,
            scores.rougeL,
            compression,
            elapsed,
        )
        return result

    # ------------------------------------------------------------------
    # Secondary use-case: batch ROUGE evaluation
    # ------------------------------------------------------------------

    def evaluate_batch(
        self,
        n_samples: Optional[int] = None,
        generation_config: GenerationConfig = GENERATION_CONFIG,
    ) -> Tuple[List[RougeScores], RougeScores]:
        """
        Evaluate the model on *n_samples* articles from the CNN/DailyMail
        test split and return per-sample and averaged ROUGE scores.

        Parameters
        ----------
        n_samples:
            Number of test articles to process.  Defaults to
            ``EVALUATION_CONFIG.batch_size``.
        generation_config:
            Generation parameters applied to every sample.

        Returns
        -------
        Tuple[List[RougeScores], RougeScores]
            ``(per_sample_scores, averaged_scores)``

        Raises
        ------
        ImportError
            If the ``datasets`` package is not installed.
        RuntimeError
            If the CNN/DailyMail dataset cannot be loaded.
        """
        from datasets import load_dataset  # noqa: PLC0415  (lazy import)

        n = n_samples or self._eval_cfg.batch_size
        logger.info("Starting batch evaluation on %d samples…", n)

        try:
            dataset = load_dataset("cnn_dailymail", "3.0.0")
            test_split = dataset["test"]
        except Exception as exc:
            logger.exception("Failed to load CNN/DailyMail dataset: %s", exc)
            raise RuntimeError("Could not load the evaluation dataset.") from exc

        per_sample: List[RougeScores] = []

        for i in range(n):
            sample = test_split[i]
            article = sample["article"]
            reference = sample["highlights"]

            try:
                summary, _ = self._summarizer.summarize(article, generation_config)
                scores = self._compute_rouge(reference, summary)
                per_sample.append(scores)
                logger.debug(
                    "Sample %3d/%d — R1: %.4f  R2: %.4f  RL: %.4f",
                    i + 1, n, scores.rouge1, scores.rouge2, scores.rougeL,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping sample %d due to error: %s", i, exc)

        if not per_sample:
            raise RuntimeError("All evaluation samples failed. Check logs.")

        averages = compute_average_scores(per_sample)
        logger.info(
            "Batch done — Avg R1: %.4f  R2: %.4f  RL: %.4f",
            averages.rouge1, averages.rouge2, averages.rougeL,
        )
        return per_sample, averages

    # ------------------------------------------------------------------
    # Dataset helper: fetch a sample for the UI demo
    # ------------------------------------------------------------------

    def load_dataset_sample(self, index: int = 0) -> Dict[str, str]:
        """
        Fetch a single article+highlights pair from the CNN/DailyMail
        test split for use in the interface demo.

        Parameters
        ----------
        index:
            Zero-based index into the test split.

        Returns
        -------
        Dict[str, str]
            ``{'article': ..., 'highlights': ...}``

        Raises
        ------
        RuntimeError
            If the dataset cannot be loaded.
        IndexError
            If *index* is out of range.
        """
        from datasets import load_dataset  # noqa: PLC0415

        try:
            dataset = load_dataset("cnn_dailymail", "3.0.0")
            sample = dataset["test"][index]
            return {
                "article": sample["article"],
                "highlights": sample["highlights"],
            }
        except Exception as exc:
            logger.exception("Could not load dataset sample %d: %s", index, exc)
            raise RuntimeError(f"Failed to load sample at index {index}.") from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_rouge(self, reference: str, hypothesis: str) -> RougeScores:
        """
        Compute ROUGE scores between *reference* and *hypothesis*.

        Parameters
        ----------
        reference:
            Gold-standard text.
        hypothesis:
            Generated summary.

        Returns
        -------
        RougeScores
        """
        raw = self._rouge_scorer.score(reference, hypothesis)
        return RougeScores(
            rouge1=raw["rouge1"].fmeasure,
            rouge2=raw["rouge2"].fmeasure,
            rougeL=raw["rougeL"].fmeasure,
        )
