# 🧠 T5 Text Summarizer

> Abstractive text summarization powered by **T5-small**, fine-tuned on the CNN/DailyMail dataset, with an interactive bilingual (English / Arabic) Gradio interface.

[![CI](https://github.com/Gerges-ML/t5-text-summarizer/actions/workflows/ci.yml/badge.svg)](https://github.com/Gerges-ML/t5-text-summarizer/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers)
[![Gradio](https://img.shields.io/badge/Gradio-4.20%2B-green.svg)](https://gradio.app/)

---

## Overview

This project implements an **abstractive text summarization** system using the pre-trained `t5-small` model (60.5 M parameters). Unlike extractive approaches that copy sentences verbatim, abstractive summarization generates new, concise text that captures the core meaning — much like how a human would paraphrase an article.

The system exposes a polished **Gradio web interface** that supports both English and Arabic UI modes, real-time ROUGE scoring, and adjustable generation parameters.

---

## Features

| Feature | Description |
|---|---|
| 🔤 Abstractive Summarization | Generates fluent, novel summaries via T5 beam search |
| 📊 Real-time ROUGE Scoring | ROUGE-1, ROUGE-2, ROUGE-L F1 scores per summary |
| 📉 Visual Metric Bars | Animated progress bars render scores inline |
| 🌍 Bilingual UI | Full English ↔ Arabic toggle with RTL layout support |
| ⚙️ Adjustable Parameters | Beam width, max tokens, length penalty, no-repeat n-gram |
| 📚 Built-in Examples | Three ready-made articles (Climate, AI, Health) |
| 🔬 Batch Evaluation | Automated ROUGE benchmark across CNN/DailyMail test samples |
| ⚡ GPU Acceleration | Auto-detects and uses CUDA when available |
| 🧪 Full Test Suite | 100+ unit tests with pytest, zero torch required for most |

---

## Project Structure

```
t5-text-summarizer/
│
├── app.py                          # Entry point — web UI or batch evaluation
├── config.py                       # Centralised configuration (all constants)
├── requirements.txt                # Python dependencies
├── pytest.ini                      # Pytest configuration
├── conftest.py                     # Pytest path setup
├── .gitignore
│
├── interface/
│   ├── __init__.py
│   └── ui.py                       # Gradio interface, InputValidator, HTML renderers
│
├── services/
│   ├── __init__.py
│   └── summarization_service.py    # Orchestration: summarise + evaluate
│
├── models/
│   ├── __init__.py
│   ├── model_loader.py             # Lazy-cached T5 tokenizer + model loading
│   └── summarizer.py               # Low-level inference wrapper (model.generate)
│
├── utils/
│   ├── __init__.py
│   ├── helpers.py                  # RougeScores, SummaryResult, formatting helpers
│   ├── logger.py                   # Project-wide logging factory
│   └── preprocessing.py            # Text cleaning + T5 prefix pipeline
│
├── tests/
│   ├── __init__.py
│   └── test_summarization.py       # Full test suite (26 test classes)
│
└── docs/
    ├── Project_Documentation.md    # Full technical documentation
    ├── System_Architecture.md      # Architecture and design decisions
    └── User_Guide.md               # Step-by-step user guide
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/Gerges-ML/t5-text-summarizer.git
cd t5-text-summarizer
pip install -r requirements.txt
```

### 2. Launch the web interface

```bash
python app.py
```

Open `http://localhost:7860` in your browser.

### 3. Run batch ROUGE evaluation

```bash
python app.py --evaluate --n-samples 10
```

### 4. Run the test suite

```bash
pytest tests/ -v
# Skip tests that require PyTorch:
pytest tests/ -v -m "not requires_torch"
# With coverage:
pytest tests/ -v --cov=. --cov-report=term-missing
```

---

## Usage

### Web Interface

1. Paste any English article into the text box (minimum 20 words)
2. Optionally pick a built-in example from the dropdown
3. Adjust generation parameters with the sliders
4. Click **▶ Summarize**
5. View the summary, ROUGE metrics, progress bars, and compression statistics

### Generation Parameters

| Parameter | Default | Range | Effect |
|---|---|---|---|
| Max tokens | 120 | 40 – 200 | Maximum summary length |
| Beam width | 4 | 1 – 8 | Higher = better quality, slower |
| Length penalty | 2.0 | 0.5 – 3.0 | Higher = encourages longer summaries |
| No-repeat n-gram | 3 | 0 – 5 | Prevents repetitive phrases |

### CLI Options

```
python app.py [--evaluate] [--n-samples N] [--share] [--port PORT]

  --evaluate       Run batch ROUGE evaluation instead of launching the web UI
  --n-samples N    Number of CNN/DailyMail test samples (default: 10)
  --share          Create a temporary public Gradio share link
  --port PORT      Gradio server port (default: 7860)
```

### Environment Variables

```bash
SUMMARIZER_DEVICE=cuda python app.py    # force GPU
LOG_LEVEL=DEBUG python app.py           # verbose logging
```

---

## Architecture

```
app.py  ──────────────────────────────────────────────────────┐
                                                               │
  ┌──────────────────┐    ┌─────────────────────────────────┐  │
  │  interface/ui.py │    │  services/summarization_service │  │
  │  ─────────────── │───▶│  ──────────────────────────────  │  │
  │  InputValidator  │    │  summarize_text()                │  │
  │  build_interface │    │  evaluate_batch()               │  │
  └──────────────────┘    └────────────┬────────────────────┘  │
                                       │                        │
                          ┌────────────▼────────────────────┐  │
                          │  models/summarizer.py            │  │
                          │  ──────────────────────────────  │  │
                          │  Summarizer.summarize()          │  │
                          └────────────┬────────────────────┘  │
                                       │                        │
                          ┌────────────▼────────────────────┐  │
                          │  models/model_loader.py          │  │
                          │  ──────────────────────────────  │  │
                          │  load_tokenizer()  (cached)      │  │
                          │  load_model()      (cached)      │  │
                          └─────────────────────────────────┘  │
                                                                │
  config.py ─── utils/ (logger, preprocessing, helpers) ───────┘
```

---

## Evaluation Results (10 CNN/DailyMail test samples)

| Sample | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| 1 | 0.2286 | 0.0294 | 0.1714 |
| 2 | 0.4045 | 0.0690 | 0.2472 |
| 3 | 0.3544 | 0.1039 | 0.2532 |
| 4 | 0.2105 | 0.0270 | 0.1316 |
| 5 | 0.4054 | 0.1389 | 0.2703 |
| 6 | 0.3226 | 0.1000 | 0.2258 |
| 7 | 0.4630 | 0.2075 | 0.3148 |
| 8 | 0.3656 | 0.0659 | 0.2796 |
| 9 | 0.3846 | 0.1579 | 0.2308 |
| 10 | 0.3600 | 0.1250 | 0.2400 |
| **AVG** | **0.3499** | **0.1025** | **0.2365** |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | T5-small (`t5-small`) — 60.5 M parameters |
| Deep Learning | PyTorch 2.0+ |
| NLP | HuggingFace Transformers |
| Dataset | CNN/DailyMail 3.0.0 (287 K articles) |
| Evaluation | rouge-score |
| Interface | Gradio 4.20+ |
| Testing | pytest + pytest-cov |
| Language | Python 3.10+ |

---

## Future Improvements

- **Fine-tuning** on CNN/DailyMail to substantially improve ROUGE scores
- **Larger models** — T5-base (220 M) or T5-large (770 M) for better abstractive quality
- **Arabic summarization** — integrate AraT5 / AraBERT for actual Arabic text
- **PDF / file upload** support in the Gradio interface
- **HuggingFace Spaces / Docker** deployment for public access
- **Multilingual extension** via mT5 or mBART

---

## Author

**Gerges Emad** — NLP Final Project  
T5 Text Summarization · CNN/DailyMail · HuggingFace Transformers · Gradio

---

*Built with PyTorch, HuggingFace Transformers, and Gradio.*
