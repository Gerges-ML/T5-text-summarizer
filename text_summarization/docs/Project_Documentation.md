# Project Documentation

**Project:** T5 Text Summarizer — Abstractive Summarization with T5-small
**Author:** Gerges Emad
**Domain:** Natural Language Processing (NLP)
**Task:** Abstractive Text Summarization

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Objectives](#2-objectives)
3. [Dataset Description](#3-dataset-description)
4. [Data Preprocessing](#4-data-preprocessing)
5. [Methodology](#5-methodology)
6. [Model Architecture](#6-model-architecture)
7. [Training Process](#7-training-process)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Results](#9-results)
10. [Limitations](#10-limitations)
11. [Future Work](#11-future-work)

---

## 1. Problem Statement

The exponential growth of digital text — news articles, research papers, reports, and social media — has made it increasingly difficult for individuals to consume information efficiently. Manually summarizing long documents is time-consuming and impractical at scale.

**Automatic text summarization** addresses this by condensing long documents into shorter representations while preserving the essential meaning. There are two primary paradigms:

- **Extractive summarization** — selects and concatenates important sentences verbatim from the source text. It is reliable but produces summaries that can feel disjointed.
- **Abstractive summarization** — generates entirely new sentences that paraphrase the source, producing more natural and coherent summaries, but requiring deeper language understanding and generation capability.

This project focuses on **abstractive summarization** using a modern sequence-to-sequence transformer model, applied to English news articles from the CNN/DailyMail corpus — a standard benchmark in the summarization literature.

---

## 2. Objectives

The project has the following concrete objectives:

**Primary Objectives**

- Implement an end-to-end abstractive text summarization pipeline using a pre-trained T5-small model
- Apply a text preprocessing pipeline to clean raw news articles before model input
- Generate summaries using beam search decoding with configurable generation parameters
- Evaluate the quality of generated summaries using ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L)

**Secondary Objectives**

- Build an interactive Gradio web interface allowing users to summarize custom text in real time
- Expose generation hyperparameters (beam width, max tokens, length penalty, n-gram penalty) as tunable UI controls
- Provide instant ROUGE feedback per generated summary when reference text is available
- Support a bilingual interface (English and Arabic) to enhance accessibility
- Demonstrate the compression capabilities of the model through word-count statistics

---

## 3. Dataset Description

### CNN/DailyMail Dataset (Version 3.0.0)

The CNN/DailyMail dataset is one of the most widely used benchmarks in abstractive text summarization research. It was originally created by pairing CNN and Daily Mail news articles with their associated bullet-point story highlights.

| Attribute | Value |
|---|---|
| Source | HuggingFace Datasets Hub (`cnn_dailymail`, version `3.0.0`) |
| Train split | 287,113 articles |
| Validation split | 13,368 articles |
| Test split | 11,490 articles |
| Columns | `article`, `highlights`, `id` |
| Language | English |
| Article length | ~700–900 words (average) |
| Summary length | ~50–80 words (average, multi-sentence bullet highlights) |

### Sample Record

**`article`** (truncated):
> LONDON, England (Reuters) -- Harry Potter star Daniel Radcliffe gains access to a reported £20 million ($41.1 million) fortune as he turns 18 on Monday, but he insists the money won't cast a spell on him...

**`highlights`** (reference summary):
> Daniel Radcliffe gets £20M fortune as he turns 18 Monday. Young actor says he has no plans to "blow" the money. Radcliffe's earnings from first five Potter films have been held in trust.

### Dataset Loading

```python
from datasets import load_dataset

dataset    = load_dataset("cnn_dailymail", "3.0.0")
train_data = dataset["train"]   # 287,113 samples
test_data  = dataset["test"]    # 11,490  samples
```

---

## 4. Data Preprocessing

Raw news articles contain noise that can degrade model performance. A preprocessing function is applied to every article before tokenization.

### Preprocessing Pipeline

```python
def preprocess(text):
    text = re.sub(r"http\S+|www\S+", "", text)       # 1. Remove URLs
    text = re.sub(r"\s+", " ", text).strip()          # 2. Normalize whitespace
    text = re.sub(r"[^\w\s.,!?\'‑]", "", text)       # 3. Remove special characters
    text = text.lower()                               # 4. Lowercase normalization
    return text
```

### Step-by-Step Explanation

**Step 1 — URL Removal**
News articles often contain embedded links (`http://...`, `www...`). These are tokenized as unknown or rare tokens and carry no semantic value for summarization.

**Step 2 — Whitespace Normalization**
Articles scraped from HTML sources frequently contain multiple consecutive spaces, tabs, and newline characters. This step reduces all whitespace sequences to a single space.

**Step 3 — Special Character Removal**
Characters such as `£`, `©`, `—`, curly quotes, and other non-ASCII symbols are stripped. Sentence-delimiting punctuation (`.`, `,`, `!`, `?`, `'`) is retained to preserve syntactic structure.

**Step 4 — Lowercase Conversion**
All text is lowercased. T5's SentencePiece tokenizer is case-aware, but lowercasing reduces vocabulary fragmentation and makes the preprocessing consistent regardless of source capitalization styles.

### Verification Example

| Stage | Text (first 150 chars) |
|---|---|
| Raw | `LONDON, England (Reuters) -- Harry Potter star Daniel Radcliffe gains access to a reported £20 million ($41.1 million)...` |
| Cleaned | `london, england reuters -- harry potter star daniel radcliffe gains access to a reported 20 million 41.1 million...` |

### T5 Task Prefix

After preprocessing, the task-specific prefix `"summarize: "` is prepended to the text before passing it to the tokenizer. This is required by T5's multi-task training format:

```python
input_text = "summarize: " + article_clean
```

---

## 5. Methodology

### Approach: Transfer Learning with Pre-trained T5-small

The project uses **zero-shot inference** with a pre-trained T5-small model. Rather than training from scratch — which would require hundreds of GPU-hours on the full CNN/DailyMail corpus — the pre-trained weights learned during T5's original multi-task training (which included summarization on CNN/DailyMail) are used directly for inference.

### Inference Pipeline

```
Raw Article
     ↓
  preprocess()     ← URL removal, whitespace, lowercase
     ↓
"summarize: " + clean_text
     ↓
  T5Tokenizer      ← SentencePiece, max_length=512, truncation=True
     ↓
  input_ids tensor → GPU
     ↓
  model.generate() ← beam search decoding
     ↓
  tokenizer.decode() ← skip_special_tokens=True
     ↓
  Generated Summary
     ↓
  rouge_scorer     ← ROUGE-1, ROUGE-2, ROUGE-L vs reference highlights
```

### Beam Search Decoding Parameters

The generation step uses several parameters to control output quality:

| Parameter | Value | Rationale |
|---|---|---|
| `max_new_tokens` | 120 | Sufficient for a 2–4 sentence summary |
| `min_length` | 20–30 | Prevents degenerate one-sentence outputs |
| `num_beams` | 4 | Balances quality and speed |
| `length_penalty` | 2.0 | Encourages longer, more complete summaries |
| `early_stopping` | True | Stops beam search when all beams reach EOS |
| `no_repeat_ngram_size` | 3 | Prevents 3-gram repetition in output |

---

## 6. Model Architecture

### T5: Text-To-Text Transfer Transformer

T5 (Raffel et al., 2020) is an encoder-decoder transformer that frames every NLP task as a text-to-text problem. Rather than using task-specific classification heads, it generates free-form text for every task — translation, summarization, question answering, classification — all using the same architecture and training objective.

### T5-small Specifications

| Property | Value |
|---|---|
| Architecture | Encoder-Decoder Transformer |
| Parameters | 60.5M |
| Encoder layers | 6 |
| Decoder layers | 6 |
| Attention heads | 8 |
| d_model (hidden size) | 512 |
| d_ff (feed-forward size) | 2,048 |
| Vocabulary size | 32,100 (SentencePiece) |
| Max input tokens | 512 |
| Positional encoding | Relative position biases |
| Pre-training objective | Span corruption (masked language modeling) |

### Input/Output Flow

```
Input:  "summarize: [article text up to 512 tokens]"
           ↓
    T5 Encoder (6 layers)
    → contextualized token embeddings
           ↓
    T5 Decoder (6 layers, autoregressive)
    → token-by-token generation via beam search
           ↓
Output: "[generated summary text]"
```

### Why T5-small?

T5-small was chosen because it is the lightest T5 variant that still produces acceptable quality summaries. On a free Colab GPU (NVIDIA T4), it generates a 40-word summary from a 512-token input in approximately 2–3 seconds, making real-time interactive use feasible.

---

## 7. Training Process

### Pre-training (External, by Google)

T5-small was pre-trained by Google on the **C4 (Colossal Clean Crawled Corpus)** dataset — approximately 750GB of clean English web text. The pre-training objective was a **masked span prediction** task, where spans of tokens are masked and the model must predict the missing spans.

T5's multi-task fine-tuning included **summarization on CNN/DailyMail**, which means the model already has knowledge of news summarization baked into its weights.

### Project Training

This project does **not** perform additional fine-tuning. The pre-trained checkpoint `t5-small` is loaded directly from HuggingFace Hub and used in `eval()` mode with `torch.no_grad()` for all inference:

```python
model = T5ForConditionalGeneration.from_pretrained("t5-small")
model = model.to(DEVICE)
model.eval()
```

Gradient computation is disabled during inference to reduce memory usage and improve speed.

### Tokenization Details

```python
inputs = tokenizer(
    input_text,
    return_tensors = "pt",
    max_length     = 512,
    truncation     = True,
    padding        = "longest",
)
```

- Articles longer than 512 tokens are **truncated** — a known limitation of T5-small
- The tokenizer uses SentencePiece byte-pair encoding with a 32,100-token vocabulary
- Padding strategy is `"longest"` (pads to the longest sequence in the batch)

---

## 8. Evaluation Metrics

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

ROUGE is the standard evaluation framework for text summarization. It measures the overlap between a generated summary (hypothesis) and one or more reference summaries.

All three variants are computed with **stemming enabled** via the `rouge_scorer` library:

```python
scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeL"],
    use_stemmer=True
)
scores = scorer.score(reference, prediction)
```

### ROUGE-1 (Unigram Overlap)

Measures the overlap of individual words between generated and reference summary.

```
ROUGE-1 Recall    = (matching unigrams) / (total reference unigrams)
ROUGE-1 Precision = (matching unigrams) / (total generated unigrams)
ROUGE-1 F1        = 2 × (P × R) / (P + R)
```

ROUGE-1 F1 is the primary reported metric. A score of ≥0.40 is considered strong for abstractive models.

### ROUGE-2 (Bigram Overlap)

Measures the overlap of consecutive word pairs. Lower scores are expected for abstractive models because they rarely reproduce exact two-word sequences from the reference.

### ROUGE-L (Longest Common Subsequence)

Measures the longest common subsequence between generated and reference text, rewarding summaries that maintain the order of information even if phrased differently.

### Compression Ratio

Additionally reported as a quality signal:

```
Compression = (1 - summary_words / article_words) × 100
```

A compression of ~90–95% is typical for single-paragraph summaries of full news articles.

---

## 9. Results

### Single-Sample Result (Test Sample 1)

| Metric | Score |
|---|---|
| ROUGE-1 F1 | 0.2105 |
| ROUGE-2 F1 | 0.0270 |
| ROUGE-L F1 | 0.1579 |
| Input words | 567 |
| Summary words | 39 |
| Compression | 93.1% |
| Generation time | 2.35 seconds |

### Batch Evaluation Results (10 Test Samples)

| Metric | Average | Min | Max |
|---|---|---|---|
| ROUGE-1 F1 | **0.3499** | 0.2105 | 0.4630 |
| ROUGE-2 F1 | **0.1025** | 0.0270 | 0.2075 |
| ROUGE-L F1 | **0.2365** | 0.1316 | 0.3148 |

### Interpretation

**ROUGE-1 (avg 0.3499)** — Moderate unigram coverage. The generated summaries share roughly 35% of their unigram vocabulary with the reference highlights. This is expected for a zero-shot abstractive model; fine-tuned T5-base models typically achieve ROUGE-1 of 0.42–0.44 on this benchmark.

**ROUGE-2 (avg 0.1025)** — Low bigram overlap, consistent with abstractive behavior. The model paraphrases rather than copying two-word phrases from the source, which is the desired behavior but penalized by ROUGE-2.

**ROUGE-L (avg 0.2365)** — Moderate longest common subsequence. The model preserves the general order of information presentation even while paraphrasing.

**Compression (avg ~93%)** — The model successfully condenses articles of ~500–600 words to summaries of approximately 35–45 words, a compression ratio in line with the reference highlights.

---

## 10. Limitations

**Token Length Constraint**
T5-small supports a maximum of 512 input tokens. Articles longer than this are silently truncated, meaning the model only summarizes the first half of very long articles. Many CNN/DailyMail articles exceed 512 tokens.

**Zero-shot Performance Gap**
Because no fine-tuning is performed on CNN/DailyMail in this project, scores are lower than the published state-of-the-art for T5-small (~ROUGE-1 0.43 when fine-tuned). The model uses its general pre-training rather than task-specific learned behavior.

**Lowercase Output**
The preprocessing step lowercases all input text, which causes the model to produce lowercased summaries. This does not affect ROUGE scores (stemming is enabled) but reduces readability for end users.

**English Only**
Despite the bilingual UI, the underlying model only handles English text. Submitting Arabic or other non-English text will produce garbled or empty output.

**No Factual Verification**
As with all abstractive models, T5 may occasionally generate plausible-sounding but factually incorrect statements (hallucinations). There is no fact-checking layer in this pipeline.

**Small Evaluation Set**
The batch evaluation covers only 10 test samples. This is sufficient for demonstration but not statistically robust enough to draw firm conclusions about model quality.

---

## 11. Future Work

**Fine-tuning**
Training T5-small on the CNN/DailyMail training split (287K articles) using a sequence-to-sequence cross-entropy loss would close the performance gap with published benchmarks. Expected improvement: ROUGE-1 from ~0.35 to ~0.43.

**Larger Model Variants**
Switching to `t5-base` (220M params) or `t5-large` (770M params) would improve summary quality at the cost of inference speed. Both are available as HuggingFace checkpoints.

**Extended Evaluation**
Running evaluation over 500–1,000 test samples would yield statistically meaningful ROUGE averages and enable confidence interval reporting.

**Arabic Summarization**
Integrating a dedicated Arabic NLP model (AraT5, AraBERT) would extend the system to genuinely summarize Arabic text, matching the bilingual UI promise.

**Length-Controlled Generation**
Implementing dynamic length targets (e.g., "summarize in 50 words") using constrained beam search or controllable generation techniques.

**Deployment**
Packaging the model as a HuggingFace Space using Gradio for permanent public hosting, or as a Docker container for on-premise deployment.

---

*Document prepared by Gerges Emad — NLP Final Project*
