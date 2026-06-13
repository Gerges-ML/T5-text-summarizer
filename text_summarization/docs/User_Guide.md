# User Guide

**Project:** T5 Text Summarizer
**Author:** Gerges Emad

---

## Table of Contents

1. [Installation Guide](#1-installation-guide)
2. [Configuration](#2-configuration)
3. [Running the Application](#3-running-the-application)
4. [Using the Interface](#4-using-the-interface)
5. [Troubleshooting](#5-troubleshooting)
6. [FAQ](#6-faq)

---

## 1. Installation Guide

### Prerequisites

Before you begin, ensure you have the following:

| Requirement | Details |
|---|---|
| Google Account | Required to access Google Colab |
| Web Browser | Chrome, Firefox, or Edge (latest version recommended) |
| Internet Connection | Required to download models (~242MB) and dataset (~800MB) |
| Google Drive (optional) | To save a persistent copy of the notebook |

No local Python installation, `pip`, `conda`, or GPU hardware is required. Everything runs in the cloud on Google Colab.

---

### Step 1 — Open Google Colab

Navigate to [https://colab.research.google.com](https://colab.research.google.com) and sign in with your Google account.

---

### Step 2 — Upload the Notebook

**Option A — Upload directly:**
1. On the Colab welcome screen, click **Upload**
2. Select `NLP_FINAL_nlp.ipynb` from your local machine
3. The notebook will open automatically

**Option B — From Google Drive:**
1. Upload `NLP_FINAL_nlp.ipynb` to your Google Drive
2. Right-click the file → **Open with** → **Google Colaboratory**

---

### Step 3 — Enable GPU Runtime

This is the most important setup step. Without a GPU, inference will be significantly slower.

1. In the Colab menu, click **Runtime**
2. Click **Change runtime type**
3. Under **Hardware accelerator**, select **GPU**
4. Click **Save**

> Recommended GPU tier: **T4 GPU** (free tier). If T4 is unavailable, the notebook will fall back to CPU automatically — inference will take 20–60 seconds per article instead of 2–3 seconds.

---

### Step 4 — Install Dependencies (Cell 1)

Run the first cell by clicking the ▶ play button to its left, or pressing `Shift + Enter`:

```bash
!pip install transformers datasets sentencepiece rouge-score ipywidgets gradio -q
```

You will see progress bars as the packages download and install. This takes approximately **30–60 seconds**.

Expected output:
```
  Preparing metadata (setup.py) ... done
  Building wheel for rouge-score (setup.py) ... done
```

---

### Step 5 — Verify Installation (Cell 2)

Run Cell 2. The expected output confirms the environment is ready:

```
PyTorch version : 2.10.0+cu128
Device          : CUDA
All imports done!
```

If the device shows `CPU` instead of `CUDA`, return to Step 3 and verify the GPU runtime is enabled.

---

## 2. Configuration

The notebook contains several configurable constants. These are not required to change for normal use, but are documented here for reference.

### Model Configuration

Located in **Cell 4a:**

```python
MODEL_NAME = "t5-small"   # Options: "t5-small", "t5-base", "t5-large"
```

Changing to `"t5-base"` (220M params) or `"t5-large"` (770M params) will improve summary quality but increase download time and inference latency.

### Tokenizer Configuration

Located in **Cell 5a:**

```python
inputs = tokenizer(
    input_text,
    return_tensors = "pt",
    max_length     = 512,       # Maximum input tokens
    truncation     = True,      # Truncate articles longer than max_length
    padding        = "longest",
)
```

Increasing `max_length` beyond 512 is not supported by T5-small's positional encoding.

### Batch Evaluation Size

Located in **Cell 7a:**

```python
N_SAMPLES = 10   # Number of test articles to evaluate
```

Increase this value to evaluate over more samples. Note that each sample takes ~2–3 seconds on GPU, so 100 samples ≈ 5 minutes.

### Generation Defaults

Located in **Cell 5b** (and mirrored in the Gradio interface):

```python
output_ids = model.generate(
    inputs["input_ids"],
    max_new_tokens       = 120,   # Maximum summary length in tokens
    min_length           = 30,    # Minimum output length
    num_beams            = 4,     # Beam search width
    length_penalty       = 2.0,   # Encourages longer outputs
    early_stopping       = True,  # Stop when all beams reach <EOS>
    no_repeat_ngram_size = 3,     # Prevents 3-gram repetition
)
```

---

## 3. Running the Application

### Full Run (Recommended)

The quickest way to run the entire notebook is:

```
Runtime → Run all   (or Ctrl + F9)
```

This executes all 8 cells sequentially. The process takes approximately **3–5 minutes** on the first run (downloading models and dataset), and about **1–2 minutes** on subsequent runs (cached downloads).

---

### Cell-by-Cell Execution

If you prefer to run cells individually, execute them in this order:

| Cell | Action | Expected Duration |
|---|---|---|
| Cell 1 | Install libraries | ~45 seconds |
| Cell 2 | Import modules + device check | ~5 seconds |
| Cell 3a | Load CNN/DailyMail dataset | ~60–120 seconds |
| Cell 3b | Define + test `preprocess()` | ~2 seconds |
| Cell 4a | Load T5 tokenizer | ~10 seconds |
| Cell 4b | Load T5 model (242MB download) | ~30–60 seconds |
| Cell 5a | Tokenize test article | ~2 seconds |
| Cell 5b | Generate summary | ~3 seconds |
| Cell 5c | Display results | ~1 second |
| Cell 6a | Compute ROUGE scores | ~1 second |
| Cell 6b | Display ROUGE + interpretation | ~1 second |
| Cell 7a | Batch evaluate 10 samples | ~25–30 seconds |
| Cell 7b | Display batch averages | ~1 second |
| Cell 8 | Launch Gradio interface | ~5 seconds |

---

### Accessing the Interface

After Cell 8 executes, look for the following output:

```
* To create a public link, set `share=True` in `launch()`.
Running on local URL: https://localhost:7860
```

An **iframe** will appear directly below the cell, embedding the full Gradio application inside the Colab notebook. You do not need to open a separate browser tab.

---

## 4. Using the Interface

### Interface Layout

The Gradio interface is divided into three panels:

```
┌─────────────────────────────────────────────────────────────┐
│               🧠 T5 Text Summarizer                          │
│     Abstractive summarization powered by T5-small            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────┐  ┌─────────────────────┐  │
│  │       INPUT PANEL            │  │    PARAMETERS        │  │
│  │                             │  │                     │  │
│  │  [Text area — paste text]   │  │  Max tokens: 120    │  │
│  │                             │  │  Beam width: 4      │  │
│  │  [📋 Load an example ▼]    │  │  Length pen: 2.0    │  │
│  │                             │  │  No-repeat: 3       │  │
│  │  [🌐 العربية]  [▶ Summarize]│  │                     │  │
│  └─────────────────────────────┘  └─────────────────────┘  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                   OUTPUT PANEL                       │   │
│  │                                                     │   │
│  │  📝 Generated Summary                               │   │
│  │  [Summary text box]                                 │   │
│  │                                                     │   │
│  │  📊 ROUGE Metrics                                   │   │
│  │  [ROUGE-1] [ROUGE-2] [ROUGE-L] [Compression] [Time]│   │
│  │                                                     │   │
│  │  ROUGE Overlap Bars                                 │   │
│  │  ROUGE-1 ████████████░░░░ 0.35                     │   │
│  │  ROUGE-2 ████░░░░░░░░░░░░ 0.10                     │   │
│  │  ROUGE-L ███████░░░░░░░░░ 0.24                     │   │
│  │                                                     │   │
│  │  [Interpretation badge]                             │   │
│  │  Input: 567 words → Summary: 39 words | Beams: 4   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

### Step-by-Step: Summarizing Your Own Text

**Step 1 — Enter text**

Click inside the text area labeled **"Enter your text here"** and paste or type any English article or paragraph. The text must be at least **20 words** long.

**Step 2 — (Optional) Load an example**

Click the **📋 Load an example** dropdown to select one of three pre-loaded articles:
- **CNN — Climate**: a climate change news article
- **Tech — AI**: an artificial intelligence industry article
- **Health — Vaccines**: an mRNA vaccine study article

Selecting an example automatically fills the text area with the full article text.

**Step 3 — (Optional) Adjust generation parameters**

Use the four sliders in the right panel to control summary generation:

| Slider | What it does |
|---|---|
| **Max tokens** (40–200) | The maximum number of tokens the model can generate. Higher values allow longer summaries but may produce repetition. Default: 120. |
| **Beam search width** (1–8) | Number of candidate sequences explored simultaneously. Higher values improve quality at the cost of speed. Default: 4. Greedy decoding = 1. |
| **Length penalty** (0.5–4.0) | Values > 1.0 encourage longer summaries; values < 1.0 encourage shorter ones. Default: 2.0. |
| **No-repeat n-gram** (0–5) | Blocks the model from repeating any n-gram of this size. 0 = disabled. Default: 3 (prevents 3-word phrase repetition). |

**Step 4 — Click Summarize**

Click the **▶ Summarize** button. The output panel will update within 2–10 seconds depending on the text length and GPU availability.

**Step 5 — Read the results**

The output panel shows:
- **Generated Summary** — the model's abstractive summary of your text
- **Metric pills** — ROUGE-1, ROUGE-2, ROUGE-L F1 scores, compression percentage, and generation time
- **Bar chart** — visual comparison of the three ROUGE scores on a 0–1 scale
- **Interpretation badge** — a green label explaining whether the ROUGE-1 score indicates strong, moderate, or low overlap with the reference
- **Statistics line** — word count comparison, beam width, max tokens, and length penalty used

---

### Switching UI Language

Click the **🌐 العربية** button in the top-right of the input panel to switch all UI labels, placeholders, button text, and interpretation messages to Arabic (RTL layout).

Click **🌐 English** to switch back.

> **Note:** The language toggle changes the interface language only. The model always processes English text regardless of UI language selection.

---

### Understanding ROUGE Scores

The interface displays three ROUGE F1 scores after every summarization:

| Score | Good | Moderate | Low |
|---|---|---|---|
| **ROUGE-1** | ≥ 0.40 | 0.25 – 0.40 | < 0.25 |
| **ROUGE-2** | ≥ 0.15 | 0.08 – 0.15 | < 0.08 |
| **ROUGE-L** | ≥ 0.30 | 0.18 – 0.30 | < 0.18 |

For T5-small in zero-shot mode, scores in the **Moderate** range are expected and acceptable. Low scores do not always mean a bad summary — they may indicate the model is being highly abstractive (paraphrasing differently from the reference).

---

## 5. Troubleshooting

### The interface does not appear after running Cell 8

**Cause:** Gradio server failed to start or the Colab kernel proxy timed out.

**Solution:**
1. Re-run Cell 8 (click the cell's ▶ button directly)
2. If the issue persists, restart the runtime: **Runtime → Restart runtime**, then run all cells again

---

### "Device: CPU" is shown in Cell 2 output

**Cause:** The Colab runtime was not set to GPU before running the notebook.

**Solution:**
1. Go to **Runtime → Change runtime type**
2. Set **Hardware accelerator** to **T4 GPU**
3. Click **Save**
4. Re-run from Cell 1 (the runtime will restart automatically)

---

### Inference is extremely slow (30+ seconds per summary)

**Cause:** Running on CPU fallback, or using a high beam width on a slow GPU.

**Solutions:**
- Ensure GPU runtime is active (see above)
- Reduce **Beam search width** to 2 or 1 in the interface for faster inference
- Reduce **Max tokens** to 80 for shorter documents

---

### ⚠️ "Please enter at least 20 words" error

**Cause:** The text area contains fewer than 20 words when the Summarize button is clicked.

**Solution:** Enter a longer article. Single sentences or short paragraphs are too short for meaningful summarization.

---

### HuggingFace download warnings

```
Warning: You are sending unauthenticated requests to the HF Hub.
Please set a HF_TOKEN to enable higher rate limits and faster downloads.
```

**Cause:** This warning appears when downloading models and datasets without a HuggingFace authentication token.

**Solution:** This is a warning only — not an error. Downloads will proceed normally. To suppress it and get higher rate limits:
1. Create a free account at [https://huggingface.co](https://huggingface.co)
2. Generate an access token at https://huggingface.co/settings/tokens
3. In Colab, add the following before Cell 3a:

```python
import os
os.environ["HF_TOKEN"] = "your_token_here"
```

---

### Dataset download is very slow or stalls

**Cause:** The CNN/DailyMail dataset is approximately 800MB. Download speed depends on Colab's network.

**Solution:** Wait patiently. The dataset is cached after the first download, so subsequent runs of the notebook will not re-download it (within the same Colab session). If the session resets, the download will repeat.

---

### The Gradio iframe shows "This site can't be reached"

**Cause:** Colab's kernel proxy link has expired (typically after ~1 hour of inactivity).

**Solution:** Re-run Cell 8 to restart the Gradio server and generate a fresh proxy URL.

---

### Output summary is empty or very short

**Cause:** Either the input was too short, the model failed to generate tokens above `min_length`, or beam search terminated early.

**Solutions:**
- Increase the **Max tokens** slider to 150–200
- Decrease the **No-repeat n-gram** slider to 2 or 0
- Ensure the input text is a proper paragraph with complete sentences

---

## 6. FAQ

**Q: Can I summarize text in languages other than English?**

A: The model (`t5-small`) was trained on English text and will produce poor or empty results for other languages. The Arabic toggle changes only the UI labels — it does not enable Arabic text summarization. For Arabic summarization, a model like AraT5 would be required.

---

**Q: How long can my input text be?**

A: The tokenizer truncates input to a maximum of **512 tokens**, which corresponds to roughly **350–450 words** of English text. Longer articles will be silently truncated before being passed to the model, meaning the summary will only reflect the beginning of the article.

---

**Q: Why are the ROUGE scores different every time I summarize the same text?**

A: They should not be, because beam search with a fixed seed is deterministic. If you observe variation, it may be due to different preprocessing (e.g., whitespace differences) or, rarely, numerical non-determinism on certain GPU configurations. Results should be consistent for identical inputs and parameter settings.

---

**Q: Can I save the generated summary?**

A: Yes — simply select the text in the summary box and copy it (Ctrl+C / Cmd+C). The interface does not currently provide a dedicated download button.

---

**Q: What happens if I don't have a GPU available?**

A: The notebook will run on CPU automatically. Inference will take approximately 20–60 seconds per summary instead of 2–3 seconds. All functionality remains the same, just slower.

---

**Q: Can I use a better model (T5-base, T5-large) instead of T5-small?**

A: Yes. In Cell 4a, change:

```python
MODEL_NAME = "t5-small"
```

to:

```python
MODEL_NAME = "t5-base"   # 220M params — better quality, ~8s per summary
# or
MODEL_NAME = "t5-large"  # 770M params — best quality, ~25s per summary
```

Note that T5-large may exceed the 16GB GPU VRAM available on Colab's free T4, causing an out-of-memory error.

---

**Q: Is this using fine-tuned weights or the base pre-trained model?**

A: This project uses the **pre-trained T5-small checkpoint** from HuggingFace without any additional fine-tuning in this notebook. The model's weights already encode summarization capability from Google's multi-task pre-training, which included CNN/DailyMail summarization. Fine-tuning on the training set would improve ROUGE scores by approximately 0.06–0.08 points.

---

**Q: Why is the ROUGE-2 score so low compared to ROUGE-1?**

A: Low ROUGE-2 is expected and actually desirable for abstractive summarization. ROUGE-2 measures bigram (two-word phrase) overlap. An abstractive model paraphrases rather than copying exact phrases from the reference, so bigram overlap is inherently low. High ROUGE-2 would suggest the model is extracting text verbatim rather than truly abstracting.

---

**Q: Can I run this notebook outside of Google Colab?**

A: Yes, with modifications. You would need:
- Python 3.8+ with PyTorch, Transformers, Datasets, Gradio, and rouge-score installed
- The cell magic commands (`!pip install ...`) replaced with standard `pip install` in a terminal
- A compatible NVIDIA GPU with CUDA drivers, or acceptance of CPU-only performance

The notebook code itself is standard Python and will run in Jupyter Lab, VS Code with Jupyter extension, or any other Jupyter-compatible environment.

---

*Document prepared by Gerges Emad — NLP Final Project*
