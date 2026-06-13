"""
app.py
------
Application entry point for the T5 Text Summarisation project.

Running this file launches the Gradio web interface.  An optional
``--evaluate`` flag bypasses the UI and runs batch ROUGE evaluation on
CNN/DailyMail test samples instead.

Usage
-----
Launch the web interface::

    python app.py

Run batch evaluation on 10 test samples::

    python app.py --evaluate --n-samples 10

Override default device::

    SUMMARIZER_DEVICE=cuda python app.py

Author: Gerges Emad
"""

from __future__ import annotations

import argparse
import sys

from config import UI_CONFIG
from interface.ui import build_interface
from services.summarization_service import SummarizationService
from utils.helpers import compute_average_scores, interpret_batch_averages
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="text_summarizer",
        description="T5-small Text Summarisation — web interface or batch evaluation.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run batch ROUGE evaluation instead of launching the web UI.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        metavar="N",
        help="Number of CNN/DailyMail test samples for batch evaluation (default from config).",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a temporary public Gradio share link.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=UI_CONFIG.server_port,
        help=f"Port for the Gradio server (default: {UI_CONFIG.server_port}).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def run_evaluation(service: SummarizationService, n_samples: int | None) -> None:
    """
    Execute batch ROUGE evaluation and print a formatted report to stdout.

    Parameters
    ----------
    service:
        Shared service instance.
    n_samples:
        Number of test articles to evaluate.
    """
    logger.info("Starting batch evaluation…")
    per_sample, averages = service.evaluate_batch(n_samples=n_samples)

    sep = "-" * 36
    header = f"{'#':>3}  {'ROUGE-1':>8}  {'ROUGE-2':>8}  {'ROUGE-L':>8}"
    print(sep)
    print(header)
    print(sep)
    for i, s in enumerate(per_sample, start=1):
        print(f"{i:>3}  {s.rouge1:>8.4f}  {s.rouge2:>8.4f}  {s.rougeL:>8.4f}")

    print(sep)
    print(
        f"{'AVG':>3}  {averages.rouge1:>8.4f}  "
        f"{averages.rouge2:>8.4f}  {averages.rougeL:>8.4f}"
    )
    print(sep)

    interpretations = interpret_batch_averages(averages)
    print("\nBatch Interpretation:")
    for metric, msg in interpretations.items():
        print(f"  {metric.upper():<10}: {msg}")
    print("\nBatch evaluation complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """
    Parse CLI arguments and either launch the web UI or run evaluation.

    Parameters
    ----------
    argv:
        Argument list (defaults to ``sys.argv[1:]`` when ``None``).

    Returns
    -------
    int
        Exit code (0 = success).
    """
    args = _parse_args(argv)

    logger.info("Initialising SummarizationService…")
    service = SummarizationService()

    if args.evaluate:
        run_evaluation(service, n_samples=args.n_samples)
        return 0

    logger.info("Building Gradio interface…")
    demo = build_interface(service=service)

    logger.info(
        "Launching on http://0.0.0.0:%d  (share=%s)",
        args.port,
        args.share,
    )
    demo.launch(
        share=args.share or UI_CONFIG.share,
        debug=UI_CONFIG.debug,
        server_name=UI_CONFIG.server_name,
        server_port=args.port,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
