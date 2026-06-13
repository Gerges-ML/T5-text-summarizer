# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] — 2025

### Added
- Abstractive summarization pipeline using `t5-small` (60.5 M parameters)
- Bilingual Gradio web interface with full English ↔ Arabic toggle and RTL layout
- Real-time ROUGE-1, ROUGE-2, ROUGE-L F1 scoring per summary
- Animated visual metric progress bars rendered inline
- Adjustable generation parameters: beam width, max tokens, length penalty, no-repeat n-gram
- Three built-in example articles (Climate, AI, Health)
- Batch ROUGE evaluation across CNN/DailyMail test samples with tabular report
- GPU auto-detection (CUDA / CPU fallback)
- `InputValidator` class — pure validation logic, independently testable without Gradio
- `ValidationResult` dataclass — typed, immutable validation outcome
- `SummarizationService` orchestration layer decoupled from model internals
- `Summarizer` class — single responsibility for `model.generate()` calls
- Lazy-cached model and tokenizer loading via `models/model_loader.py`
- Centralised configuration via `config.py` (all constants, zero magic numbers)
- Project-wide structured logging factory (`utils/logger.py`) with optional file rotation
- Full text preprocessing pipeline: URL removal, whitespace normalisation, lowercasing
- 26-class pytest test suite (100+ tests) covering all layers; most run without PyTorch
- GitHub Actions CI workflow — runs tests on every push and pull request
- `conftest.py` for automatic `sys.path` setup in pytest
- MIT License

---

## [Unreleased]

### Planned
- Fine-tuning T5-small on CNN/DailyMail for improved ROUGE scores
- Upgrade path to T5-base / T5-large
- Arabic text summarization via AraT5 / AraBERT
- PDF and plain-text file upload support in the Gradio interface
- HuggingFace Spaces / Docker deployment
- Multilingual extension via mT5 or mBART
