# System Architecture

**Project:** T5 Text Summarizer
**Author:** Gerges Emad

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Data Flow](#2-data-flow)
3. [Component Diagram Description](#3-component-diagram-description)
4. [Module Responsibilities](#4-module-responsibilities)
5. [Design Decisions](#5-design-decisions)

---

## 1. Architecture Overview

The T5 Text Summarizer is a **single-notebook, inference-first NLP pipeline** built inside Google Colab. It follows a linear execution model where each notebook cell corresponds to a distinct architectural layer — from environment setup through data loading, model loading, inference, evaluation, and finally the interactive interface.

The system is structured around three major subsystems:

```
┌──────────────────────────────────────────────────────────────────┐
│                         SUBSYSTEM MAP                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐   ┌──────────────────┐   ┌──────────────┐  │
│  │  DATA SUBSYSTEM  │   │  MODEL SUBSYSTEM  │   │  UI SUBSYSTEM│  │
│  │                 │   │                  │   │              │  │
│  │ • CNN/DailyMail │   │ • T5Tokenizer    │   │ • Gradio app │  │
│  │   Dataset       │   │ • T5-small model │   │ • Bilingual  │  │
│  │ • preprocess()  │   │ • Beam decoder   │   │   toggle     │  │
│  │ • Tokenizer     │   │ • CUDA device    │   │ • ROUGE viz  │  │
│  │   pipeline      │   │   management     │   │ • Sliders    │  │
│  └────────┬────────┘   └────────┬─────────┘   └──────┬───────┘  │
│           │                     │                     │          │
│           └─────────────────────┼─────────────────────┘          │
│                                 │                                 │
│                    ┌────────────▼────────────┐                   │
│                    │  EVALUATION SUBSYSTEM    │                   │
│                    │  • rouge_scorer          │                   │
│                    │  • Batch evaluation      │                   │
│                    │  • ROUGE-1/2/L F1        │                   │
│                    │  • Compression stats     │                   │
│                    └─────────────────────────┘                   │
└──────────────────────────────────────────────────────────────────┘
```

### Execution Environment

| Layer | Technology |
|---|---|
| Compute | Google Colab (NVIDIA T4 GPU, CUDA 12.x) |
| Runtime | Python 3.x, PyTorch 2.10.0+cu128 |
| Model Hub | HuggingFace Hub (model + tokenizer + dataset downloads) |
| Interface Server | Gradio local server, proxied through Colab kernel at port 7860 |

---

## 2. Data Flow

The complete data flow from raw user input (or dataset article) to a rendered summary with metrics is described below.

### 2.1 Dataset Evaluation Path

```
HuggingFace Hub
      │
      │  load_dataset("cnn_dailymail", "3.0.0")
      ▼
┌─────────────────────────────────────┐
│         CNN/DailyMail Dataset        │
│  train: 287,113 │ test: 11,490      │
│  columns: article, highlights, id   │
└──────────────┬──────────────────────┘
               │  test_data[i]["article"]
               ▼
┌─────────────────────────────────────┐
│         preprocess(article)          │
│  • remove URLs                       │
│  • normalize whitespace              │
│  • strip special characters          │
│  • lowercase                         │
└──────────────┬──────────────────────┘
               │  "summarize: " + clean_text
               ▼
┌─────────────────────────────────────┐
│           T5Tokenizer                │
│  • SentencePiece BPE encoding        │
│  • max_length=512, truncation=True   │
│  • return_tensors="pt"               │
└──────────────┬──────────────────────┘
               │  input_ids tensor → CUDA
               ▼
┌─────────────────────────────────────┐
│     T5ForConditionalGeneration       │
│  • Encoder: 6-layer transformer      │
│  • Decoder: autoregressive beam scan │
│  • num_beams=4, max_new_tokens=120   │
│  • length_penalty=2.0                │
│  • no_repeat_ngram_size=3            │
└──────────────┬──────────────────────┘
               │  output_ids tensor
               ▼
┌─────────────────────────────────────┐
│       tokenizer.decode()             │
│  • skip_special_tokens=True          │
│  → plain text summary string         │
└──────────────┬──────────────────────┘
               │  (summary, reference)
               ▼
┌─────────────────────────────────────┐
│     rouge_scorer.RougeScorer         │
│  • ROUGE-1 F1 (unigram overlap)      │
│  • ROUGE-2 F1 (bigram overlap)       │
│  • ROUGE-L F1 (LCS overlap)          │
│  • compression = 1 - out/in words    │
└─────────────────────────────────────┘
```

### 2.2 Interactive Interface Path

```
Gradio UI (browser)
      │
      │  text, max_tokens, num_beams,
      │  length_penalty, no_repeat, lang
      ▼
┌─────────────────────────────────────┐
│        summarize() function          │
│  • validate: len(words) >= 20        │
│  • preprocess(text)                  │
│  • tokenize with task prefix         │
│  • model.generate(...)               │
│  • tokenizer.decode(output_ids)      │
│  • rouge_scorer.score(text, summary) │
│  • compression, timing stats         │
│  • build HTML output string          │
└──────────────┬──────────────────────┘
               │  HTML string
               ▼
┌─────────────────────────────────────┐
│        Gradio gr.HTML component      │
│  • Summary text box                  │
│  • Metric pills (ROUGE + time)       │
│  • Animated progress bars            │
│  • Interpretation badge              │
│  • Stats footer line                 │
└─────────────────────────────────────┘
```

---

## 3. Component Diagram Description

The system is composed of the following logical components:

```
┌────────────────────────────────────────────────────────────────────┐
│                      COMPONENT DIAGRAM                             │
│                                                                    │
│  ┌──────────────┐        ┌──────────────────────────────────────┐  │
│  │  HuggingFace │        │          Notebook Kernel              │  │
│  │     Hub      │        │                                      │  │
│  │              │ fetch  │  ┌─────────────┐  ┌───────────────┐  │  │
│  │  t5-small    │───────▶│  │  T5Tokenizer│  │  T5 Model     │  │  │
│  │  tokenizer   │        │  │  (32K vocab)│  │  (60.5M params│  │  │
│  │  config      │        │  └──────┬──────┘  └──────┬────────┘  │  │
│  │              │        │         │                 │           │  │
│  │  cnn_daily   │ fetch  │  ┌──────▼─────────────────▼────────┐  │  │
│  │  mail 3.0.0  │───────▶│  │        Inference Engine          │  │  │
│  └──────────────┘        │  │  preprocess → tokenize →        │  │  │
│                           │  │  generate → decode              │  │  │
│                           │  └──────────────┬──────────────────┘  │  │
│                           │                 │                     │  │
│                           │  ┌──────────────▼──────────────────┐  │  │
│                           │  │       Evaluation Engine          │  │  │
│                           │  │  rouge_scorer (R1, R2, RL)       │  │  │
│                           │  │  compression ratio               │  │  │
│                           │  └──────────────┬──────────────────┘  │  │
│                           │                 │                     │  │
│                           └─────────────────┼─────────────────────┘  │
│                                             │                        │
│  ┌──────────────────────────────────────────▼──────────────────────┐ │
│  │                     Gradio Interface Layer                       │ │
│  │                                                                  │ │
│  │  ┌───────────────┐  ┌─────────────┐  ┌───────────────────────┐  │ │
│  │  │  Input Panel   │  │  Param Panel │  │    Output Panel        │  │ │
│  │  │  • Textbox     │  │  • Sliders   │  │    • Summary HTML      │  │ │
│  │  │  • Examples DD │  │  • Run btn   │  │    • Metric pills      │  │ │
│  │  │  • Lang toggle │  │             │  │    • Bar charts         │  │ │
│  │  └───────────────┘  └─────────────┘  └───────────────────────┘  │ │
│  │                                                                  │ │
│  │  ┌──────────────────────────────────────────────────────────┐   │ │
│  │  │              LANG Dictionary (en / ar)                    │   │ │
│  │  │  All UI labels, placeholders, button text, interpretations│   │ │
│  │  └──────────────────────────────────────────────────────────┘   │ │
│  └──────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
```

### Component Interactions

| Source Component | Target Component | Interaction |
|---|---|---|
| HuggingFace Hub | T5Tokenizer | `from_pretrained("t5-small")` download |
| HuggingFace Hub | T5 Model | `from_pretrained("t5-small")` download (242MB) |
| HuggingFace Hub | Dataset loader | `load_dataset("cnn_dailymail", "3.0.0")` |
| Dataset loader | Inference Engine | raw article text |
| Gradio Input Panel | Inference Engine | text + generation parameters |
| Inference Engine | T5Tokenizer | tokenize input text |
| Inference Engine | T5 Model | `model.generate()` call |
| Inference Engine | Evaluation Engine | (summary, reference) pair |
| Evaluation Engine | Gradio Output Panel | ROUGE scores + stats dict |
| LANG Dictionary | Gradio Interface | label strings (EN or AR) |

---

## 4. Module Responsibilities

The notebook is organized into 8 logical cells, each with a single responsibility:

### Cell 1 — Dependency Installation

**File:** `NLP_FINAL_nlp.ipynb` (Cell 1)
**Responsibility:** Install all third-party libraries into the Colab runtime.

Libraries installed: `transformers`, `datasets`, `sentencepiece`, `rouge-score`, `ipywidgets`, `gradio`

```python
!pip install transformers datasets sentencepiece rouge-score ipywidgets -q
```

### Cell 2 — Imports and Device Setup

**Responsibility:** Import all required modules and detect compute device.

Key decisions made here:
- `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"` — all tensors and the model are subsequently moved to this device

### Cell 3a — Dataset Loading

**Responsibility:** Download and cache the CNN/DailyMail dataset via the HuggingFace Datasets API.

Outputs: `train_data` (287,113 samples), `test_data` (11,490 samples)

### Cell 3b — Text Preprocessing

**Responsibility:** Define and validate the `preprocess()` function using a sample article.

The function is stateless and pure — it takes a string and returns a string with no side effects.

### Cell 4a — Tokenizer Loading

**Responsibility:** Load the T5 SentencePiece tokenizer from HuggingFace Hub.

Outputs: `tokenizer` object with vocabulary size 32,100

### Cell 4b — Model Loading

**Responsibility:** Load the T5ForConditionalGeneration model, transfer it to the compute device, and set it to evaluation mode.

```python
model = T5ForConditionalGeneration.from_pretrained("t5-small")
model = model.to(DEVICE)
model.eval()
```

The `.eval()` call disables dropout layers and ensures deterministic inference. `torch.no_grad()` context is used in all generate calls to suppress gradient computation.

### Cells 5a–5c — Single-Sample Inference

**Responsibility:** Tokenize a single test article, run inference with beam search, and display the raw output alongside the reference highlights and timing statistics.

### Cells 6a–6b — ROUGE Evaluation (Single Sample)

**Responsibility:** Compute and display ROUGE-1, ROUGE-2, and ROUGE-L F1 scores for the single-sample inference result, along with a threshold-based interpretation.

### Cells 7a–7b — Batch Evaluation

**Responsibility:** Loop over 10 test samples, run full inference on each, accumulate ROUGE scores, and print per-sample and aggregate average scores.

### Cell 8 — Gradio Interface

**Responsibility:** Define and launch the complete interactive web application.

Sub-components within Cell 8:
- `LANG` dict — all UI strings in English and Arabic
- `EXAMPLES` dict — three pre-loaded sample articles
- `CSS` string — full custom dark-theme stylesheet
- `summarize()` function — the core callback triggered by the Summarize button
- `build_ui()` function — constructs the Gradio Blocks layout
- `toggle_lang()` function — switches UI language state
- `load_example()` function — populates the text box from examples dropdown
- `gr.Blocks(...).launch()` — starts the Gradio server

---

## 5. Design Decisions

### Decision 1: Pre-trained Inference Only (No Fine-tuning)

**Choice:** Use the T5-small checkpoint directly without fine-tuning on CNN/DailyMail.

**Rationale:** Fine-tuning T5-small on 287K articles requires approximately 4–8 hours of GPU time and significant memory management. For a demonstration project running in a free Colab session (which resets after ~12 hours), zero-shot inference with the already CNN/DailyMail-pretrained weights achieves acceptable results while keeping the notebook self-contained and runnable in under 5 minutes.

**Trade-off:** ROUGE scores are approximately 0.06–0.08 lower than fine-tuned baselines.

---

### Decision 2: T5-small Over Larger Variants

**Choice:** T5-small (60.5M params) rather than T5-base (220M), T5-large (770M), or T5-3B.

**Rationale:** T5-small generates a 40-word summary from a 512-token input in ~2–3 seconds on a Colab T4 GPU. This makes the interactive Gradio interface feel responsive. Larger models would require 8–30 seconds per inference call, degrading the interactive experience. T5-small's 242MB weight file also downloads quickly on first run.

**Trade-off:** Summary quality and ROUGE scores are lower than larger variants.

---

### Decision 3: Gradio Over ipywidgets

**Choice:** Gradio for the final interactive interface (Cell 8) rather than ipywidgets.

**Rationale:** The notebook markdown header mentions ipywidgets (used in earlier exploration), but the final cell uses Gradio. Gradio provides a production-quality HTML/CSS interface that is significantly more polished, supports custom CSS theming, renders HTML output (enabling metric pills and animated bars), and provides better support for the bilingual layout (RTL/LTR switching).

**Trade-off:** Requires an additional `pip install gradio` step and uses a local server proxied through Colab.

---

### Decision 4: Bilingual Interface (EN / AR)

**Choice:** Full English ↔ Arabic UI toggle using a `LANG` dictionary.

**Rationale:** The interface targets an Arabic-speaking academic audience. Rather than maintaining two separate UIs, all user-visible strings are centralized in a `LANG = {"en": {...}, "ar": {...}}` dictionary. The toggle button switches the active language key and rebuilds all component labels. RTL layout is applied via CSS `direction: rtl` on the root container.

**Trade-off:** The underlying model is English-only; the Arabic UI is cosmetic only and does not enable Arabic summarization.

---

### Decision 5: HTML Output Rendering in Gradio

**Choice:** Use a `gr.HTML` output component rather than `gr.Textbox` or `gr.Markdown`.

**Rationale:** The output panel requires rich visual elements: metric pills with color coding, animated progress bars for ROUGE scores, an interpretation badge, and a statistics footer. These cannot be expressed in plain text or basic Markdown. Returning a raw HTML string from the `summarize()` function and rendering it in `gr.HTML` allows arbitrary CSS-styled components.

**Trade-off:** The HTML string is assembled by string interpolation, which requires careful escaping of user text to avoid XSS-like issues in the summary display.

---

### Decision 6: In-notebook ROUGE Scoring

**Choice:** Compute ROUGE scores live for every summary rather than reporting only pre-computed benchmark numbers.

**Rationale:** For the interactive use case, real-time ROUGE feedback lets users immediately see how their custom text compares to the model's summarization quality on a reference document. For the interface, ROUGE is computed against the raw input text itself (input as the "reference"), serving as a rough self-similarity measure when no external reference is available.

**Trade-off:** ROUGE against the source article is not a meaningful summarization metric — it measures compression, not quality. The batch evaluation (Cells 7a–7b) uses proper reference highlights for meaningful ROUGE computation.

---

*Document prepared by Gerges Emad — NLP Final Project*
