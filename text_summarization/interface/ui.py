"""
interface/ui.py
---------------
Gradio-based bilingual (English / Arabic) web interface for the T5
Text Summariser.

Architecture
~~~~~~~~~~~~
The module is organised in four sections:

1. **Constants** — i18n dictionaries, example texts, CSS.
2. **InputValidator** — pure validation logic with no Gradio dependency,
   making it independently testable.
3. **HTML renderers** — stateless functions that build output card strings.
4. **build_interface()** — assembles the Gradio Blocks layout and wires
   all events.  Business logic is delegated entirely to
   :class:`~services.summarization_service.SummarizationService`.

Author: Gerges Emad
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gradio as gr

from config import GENERATION_CONFIG, GenerationConfig, UI_CONFIG
from services.summarization_service import SummarizationService
from utils.helpers import format_stats_line, word_count_color
from utils.logger import get_logger
from utils.preprocessing import count_words

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Section 1 — i18n strings
# ---------------------------------------------------------------------------

LANG: dict[str, dict[str, str]] = {
    "en": {
        "dir": "ltr",
        "title": "🧠 T5 Text Summarizer",
        "subtitle": "Abstractive summarization powered by T5-small · CNN/DailyMail",
        "input_label": "Enter your text here",
        "input_placeholder": "Paste or type any English article (at least 20 words)...",
        "example_label": "📋 Load an example",
        "params_title": "⚙️ Generation Parameters",
        "max_tokens": "Max tokens",
        "num_beams": "Beam search width",
        "length_penalty": "Length penalty",
        "no_repeat": "No repeat n-gram",
        "run_btn": "▶ Summarize",
        "summary_title": "📝 Generated Summary",
        "metrics_title": "📊 ROUGE Metrics",
        "rouge1": "ROUGE-1",
        "rouge2": "ROUGE-2",
        "rougel": "ROUGE-L",
        "compression": "Compression",
        "time_label": "Time",
        "bars_title": "ROUGE Overlap Bars",
        "compression_suffix": "Compression",
        "err_short": "⚠️ Please enter at least 20 words.",
        "err_empty": "⚠️ Input text cannot be empty.",
        "err_param_range": "⚠️ One or more parameters are out of the allowed range.",
        "err_generic": "❌ An error occurred during summarization. Please try again.",
        "lang_btn": "🌐 العربية",
        "placeholder_output": "Output will appear here after summarization.",
    },
    "ar": {
        "dir": "rtl",
        "title": "🧠 ملخص النصوص T5",
        "subtitle": "تلخيص استخلاصي بواسطة T5-small · CNN/DailyMail",
        "input_label": "أدخل النص هنا",
        "input_placeholder": "الصق أو اكتب أي مقال إنجليزي (20 كلمة على الأقل)...",
        "example_label": "📋 تحميل مثال",
        "params_title": "⚙️ معاملات التوليد",
        "max_tokens": "الحد الأقصى للرموز",
        "num_beams": "عرض البحث الشعاعي",
        "length_penalty": "معامل الطول",
        "no_repeat": "عدم تكرار N-gram",
        "run_btn": "▶ تلخيص",
        "summary_title": "📝 الملخص المُولَّد",
        "metrics_title": "📊 مقاييس ROUGE",
        "rouge1": "ROUGE-1",
        "rouge2": "ROUGE-2",
        "rougel": "ROUGE-L",
        "compression": "نسبة الضغط",
        "time_label": "الوقت",
        "bars_title": "أشرطة التداخل ROUGE",
        "compression_suffix": "نسبة الضغط",
        "err_short": "⚠️ الرجاء إدخال 20 كلمة على الأقل.",
        "err_empty": "⚠️ لا يمكن أن يكون النص المدخل فارغاً.",
        "err_param_range": "⚠️ أحد المعاملات خارج النطاق المسموح به.",
        "err_generic": "❌ حدث خطأ أثناء التلخيص. يرجى المحاولة مرة أخرى.",
        "lang_btn": "🌐 English",
        "placeholder_output": ".سيظهر الناتج هنا بعد التلخيص",
    },
}

# ---------------------------------------------------------------------------
# Section 1 — Built-in example texts
# ---------------------------------------------------------------------------

EXAMPLES: dict[str, str] = {
    "CNN — Climate": (
        "Scientists have warned that climate change is accelerating faster than previously "
        "predicted, with global temperatures rising more than 1.5 degrees Celsius above "
        "pre-industrial levels. New research shows that Arctic ice is melting at record rates, "
        "contributing to rising sea levels that threaten coastal communities worldwide. "
        "Governments are being urged to take immediate and drastic action to reduce carbon "
        "emissions and transition to renewable energy sources before the damage becomes "
        "irreversible. International agreements such as the Paris Accord are being revisited "
        "and strengthened in light of the new scientific data."
    ),
    "Tech — AI": (
        "Artificial intelligence is transforming industries at an unprecedented pace. "
        "From healthcare to finance, AI systems are being deployed to automate complex tasks, "
        "analyze massive datasets, and generate human-like text and images. However, experts "
        "warn that the rapid adoption of AI also raises serious ethical concerns, including "
        "bias in decision-making systems, job displacement, and the potential misuse of "
        "generative models. Governments and technology companies are working together to "
        "establish regulatory frameworks that balance innovation with safety and accountability."
    ),
    "Health — Vaccines": (
        "A new study published in a leading medical journal confirms that mRNA vaccines "
        "remain highly effective against severe illness and hospitalization from respiratory "
        "viruses, even as new variants emerge. The research followed over 50,000 participants "
        "across multiple countries over a two-year period. Scientists noted that booster doses "
        "significantly enhanced protection, particularly among elderly populations and those "
        "with underlying health conditions. Public health officials are encouraging continued "
        "vaccination efforts as part of a broader strategy to manage infectious disease outbreaks."
    ),
}

# ---------------------------------------------------------------------------
# Section 1 — CSS
# ---------------------------------------------------------------------------

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Tajawal:wght@400;500;700&display=swap');

body, .gradio-container {
    background: linear-gradient(145deg, #0f172a 0%, #1e293b 100%) !important;
    min-height: 100vh;
    font-family: 'Inter', 'Tajawal', sans-serif !important;
}

#nlp-header { text-align: center; padding: 28px 20px 10px; }
#nlp-header h1 {
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #c4b5fd, #67e8f9);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 6px;
}
#nlp-header p { color: #64748b; font-size: 0.85rem; margin: 0; }

.nlp-input textarea {
    background: #0f172a !important; color: #e2e8f0 !important;
    border: 1.5px solid rgba(99,102,241,0.3) !important;
    border-radius: 10px !important; font-size: 0.9rem !important; padding: 12px !important;
}
.nlp-input textarea:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 3px rgba(129,140,248,0.15) !important;
}
label[for] span, label span { color: #94a3b8 !important; font-size: 0.82rem !important; }

#run-btn {
    background: linear-gradient(135deg, #6366f1, #818cf8) !important;
    color: white !important; font-weight: 600 !important; font-size: 1rem !important;
    border: none !important; border-radius: 10px !important;
    padding: 12px 32px !important; transition: all 0.2s !important;
}
#run-btn:hover { filter: brightness(1.12) !important; transform: translateY(-1px) !important; }

#lang-btn {
    background: rgba(30,41,59,0.9) !important; color: #818cf8 !important;
    border: 1.5px solid rgba(99,102,241,0.4) !important;
    border-radius: 8px !important; font-weight: 600 !important; transition: all 0.2s !important;
}
#lang-btn:hover { border-color: #818cf8 !important; }
input[type=range] { accent-color: #818cf8 !important; }

.nlp-output-wrap {
    background: rgba(15,23,42,0.9); border: 1px solid rgba(99,102,241,0.25);
    border-radius: 14px; padding: 22px; color: #e2e8f0;
    font-family: 'Inter', 'Tajawal', sans-serif;
}
.nlp-summary-box {
    background: rgba(99,102,241,0.08); border-left: 3px solid #818cf8;
    border-radius: 8px; padding: 14px 16px; font-size: 0.95rem;
    line-height: 1.7; color: #cbd5e1; margin-bottom: 18px;
}
.nlp-metrics-row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 18px; }
.nlp-metric-pill {
    flex: 1; min-width: 80px; background: rgba(15,23,42,0.6);
    border: 1px solid rgba(99,102,241,0.2); border-radius: 10px;
    padding: 10px 8px; text-align: center;
}
.nlp-metric-pill .val { font-size: 1.25rem; font-weight: 700; }
.nlp-metric-pill .lbl {
    font-size: 0.68rem; color: #64748b; margin-top: 2px;
    text-transform: uppercase; letter-spacing: .06em;
}
.nlp-bar-section { margin-bottom: 14px; }
.nlp-bar-row { display: flex; align-items: center; gap: 10px; margin-bottom: 7px; }
.nlp-bar-name { font-size: 0.75rem; color: #64748b; width: 56px; text-align: right; }
.nlp-bar-track {
    flex: 1; height: 7px; background: rgba(255,255,255,.06);
    border-radius: 99px; overflow: hidden;
}
.nlp-bar-fill { height: 100%; border-radius: 99px; transition: width .5s ease; }
.nlp-bar-val { font-size: 0.78rem; font-weight: 600; width: 38px; }
.nlp-interp {
    background: rgba(52,211,153,0.08); border: 1px solid rgba(52,211,153,0.2);
    border-radius: 8px; padding: 10px 14px; font-size: 0.82rem;
    color: #6ee7b7; margin-bottom: 10px;
}
.nlp-stats { font-size: 0.75rem; color: #334155; line-height: 1.8; }
.nlp-error {
    background: rgba(248,113,113,0.08); border: 1px solid rgba(248,113,113,0.25);
    border-radius: 8px; padding: 12px 16px; color: #f87171; font-size: 0.88rem;
}
"""


# ---------------------------------------------------------------------------
# Section 2 — Input validation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidationResult:
    """
    Outcome of an :class:`InputValidator` check.

    Attributes
    ----------
    is_valid:
        ``True`` when validation passed; ``False`` otherwise.
    error_key:
        Key into the i18n ``LANG`` dict for the appropriate error message,
        or ``None`` when ``is_valid`` is ``True``.
    """

    is_valid: bool
    error_key: Optional[str] = None


class InputValidator:
    """
    Pure validation logic for the summarisation UI inputs.

    All methods are stateless and have no Gradio dependency, making this
    class independently testable without spinning up the interface.

    Parameters
    ----------
    min_words:
        Minimum number of words required in the input text.
    max_tokens_range:
        ``(min, max)`` for the max-tokens slider.
    num_beams_range:
        ``(min, max)`` for the beam-width slider.
    length_penalty_range:
        ``(min, max)`` for the length-penalty slider.
    no_repeat_range:
        ``(min, max)`` for the no-repeat-ngram slider.
    """

    def __init__(
        self,
        min_words: int = UI_CONFIG.min_input_words,
        max_tokens_range: tuple[int, int] = UI_CONFIG.slider_max_tokens_range,
        num_beams_range: tuple[int, int] = UI_CONFIG.slider_num_beams_range,
        length_penalty_range: tuple[float, float] = UI_CONFIG.slider_length_penalty_range,
        no_repeat_range: tuple[int, int] = UI_CONFIG.slider_no_repeat_range,
    ) -> None:
        self._min_words = min_words
        self._max_tokens_range = max_tokens_range
        self._num_beams_range = num_beams_range
        self._length_penalty_range = length_penalty_range
        self._no_repeat_range = no_repeat_range

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_text(self, text: str) -> ValidationResult:
        """
        Validate the user-supplied input text.

        Checks performed (in order):

        1. Not ``None`` and is a string.
        2. Not empty after stripping.
        3. Meets the minimum word-count threshold.

        Parameters
        ----------
        text:
            Raw text from the Gradio textbox.

        Returns
        -------
        ValidationResult
            ``is_valid=True`` when all checks pass; otherwise ``is_valid=False``
            with the appropriate ``error_key``.
        """
        if not isinstance(text, str) or not text.strip():
            return ValidationResult(is_valid=False, error_key="err_empty")

        if count_words(text) < self._min_words:
            return ValidationResult(is_valid=False, error_key="err_short")

        return ValidationResult(is_valid=True)

    def validate_generation_params(
        self,
        max_tokens: int | float,
        num_beams: int | float,
        length_penalty: float,
        no_repeat: int | float,
    ) -> ValidationResult:
        """
        Validate that all generation slider values are within their configured
        bounds.

        Parameters
        ----------
        max_tokens:     Value from the max-tokens slider.
        num_beams:      Value from the beam-width slider.
        length_penalty: Value from the length-penalty slider.
        no_repeat:      Value from the no-repeat-ngram slider.

        Returns
        -------
        ValidationResult
        """
        checks = [
            (max_tokens, self._max_tokens_range),
            (num_beams, self._num_beams_range),
            (length_penalty, self._length_penalty_range),
            (no_repeat, self._no_repeat_range),
        ]
        for value, (lo, hi) in checks:
            if not (lo <= value <= hi):
                return ValidationResult(is_valid=False, error_key="err_param_range")

        return ValidationResult(is_valid=True)

    def validate_all(
        self,
        text: str,
        max_tokens: int | float,
        num_beams: int | float,
        length_penalty: float,
        no_repeat: int | float,
    ) -> ValidationResult:
        """
        Run text validation then parameter validation; return the first failure.

        Parameters
        ----------
        text:           Raw input text.
        max_tokens:     Max-tokens slider value.
        num_beams:      Beam-width slider value.
        length_penalty: Length-penalty slider value.
        no_repeat:      No-repeat-ngram slider value.

        Returns
        -------
        ValidationResult
            First failing check, or a passing result if all checks pass.
        """
        text_result = self.validate_text(text)
        if not text_result.is_valid:
            return text_result

        return self.validate_generation_params(
            max_tokens, num_beams, length_penalty, no_repeat
        )


# ---------------------------------------------------------------------------
# Section 3 — HTML renderers
# ---------------------------------------------------------------------------

def _error_html(message: str, direction: str = "ltr") -> str:
    """
    Render a styled error card.

    Parameters
    ----------
    message:   Human-readable error string.
    direction: CSS ``dir`` attribute value (``'ltr'`` or ``'rtl'``).

    Returns
    -------
    str
        HTML string.
    """
    return (
        f"<div class='nlp-output-wrap' dir='{direction}'>"
        f"<div class='nlp-error'>{message}</div>"
        f"</div>"
    )


def _rouge_bar(value: float, gradient: str, label: str) -> str:
    """
    Render a single ROUGE progress-bar row as HTML.

    Parameters
    ----------
    value:    ROUGE F-measure in [0, 1].
    gradient: CSS ``linear-gradient`` string for the bar fill.
    label:    Short metric name displayed to the left of the bar.

    Returns
    -------
    str
        HTML string for one bar row.
    """
    pct = min(value / 0.5 * 100, 100)
    try:
        label_color = gradient.split(",")[1].strip().rstrip(")")
    except IndexError:
        label_color = "#818cf8"

    return (
        f"<div class='nlp-bar-row'>"
        f"<span class='nlp-bar-name'>{label}</span>"
        f"<div class='nlp-bar-track'>"
        f"<div class='nlp-bar-fill' style='width:{pct:.1f}%;background:{gradient}'></div>"
        f"</div>"
        f"<span class='nlp-bar-val' style='color:{label_color}'>{value:.3f}</span>"
        f"</div>"
    )


def _build_output_html(
    summary: str,
    rouge1: float,
    rouge2: float,
    rougeL: float,
    compression: float,
    elapsed: float,
    stats_line: str,
    interpretation: str,
    lang_dict: dict[str, str],
) -> str:
    """
    Render the full summarisation result card as an HTML string.

    Parameters
    ----------
    summary:        Generated summary text.
    rouge1:         ROUGE-1 F-measure.
    rouge2:         ROUGE-2 F-measure.
    rougeL:         ROUGE-L F-measure.
    compression:    Compression percentage (0–100).
    elapsed:        Inference wall-clock time in seconds.
    stats_line:     Pre-formatted statistics string.
    interpretation: ROUGE-1 interpretation sentence.
    lang_dict:      Language-specific label dictionary from :data:`LANG`.

    Returns
    -------
    str
        Self-contained HTML card string.
    """
    L = lang_dict
    bars = (
        _rouge_bar(rouge1, "linear-gradient(90deg,#6366f1,#818cf8)", L["rouge1"])
        + _rouge_bar(rouge2, "linear-gradient(90deg,#7c3aed,#a78bfa)", L["rouge2"])
        + _rouge_bar(rougeL, "linear-gradient(90deg,#4f46e5,#c4b5fd)", L["rougel"])
    )

    return f"""
<div class='nlp-output-wrap' dir='{L["dir"]}'>
  <div style='font-size:.75rem;color:#6366f1;font-weight:600;text-transform:uppercase;
              letter-spacing:.08em;margin-bottom:8px'>{L["summary_title"]}</div>
  <div class='nlp-summary-box'>{summary}</div>

  <div style='font-size:.75rem;color:#64748b;font-weight:600;text-transform:uppercase;
              letter-spacing:.08em;margin-bottom:10px'>{L["metrics_title"]}</div>
  <div class='nlp-metrics-row'>
    <div class='nlp-metric-pill'>
      <div class='val' style='color:#818cf8'>{rouge1:.3f}</div>
      <div class='lbl'>{L["rouge1"]}</div>
    </div>
    <div class='nlp-metric-pill'>
      <div class='val' style='color:#a78bfa'>{rouge2:.3f}</div>
      <div class='lbl'>{L["rouge2"]}</div>
    </div>
    <div class='nlp-metric-pill'>
      <div class='val' style='color:#c4b5fd'>{rougeL:.3f}</div>
      <div class='lbl'>{L["rougel"]}</div>
    </div>
    <div class='nlp-metric-pill'>
      <div class='val' style='color:#34d399'>{compression:.1f}%</div>
      <div class='lbl'>{L["compression"]}</div>
    </div>
    <div class='nlp-metric-pill'>
      <div class='val' style='color:#64748b;font-size:1.1rem'>{elapsed:.2f}s</div>
      <div class='lbl'>{L["time_label"]}</div>
    </div>
  </div>

  <div class='nlp-bar-section'>
    <div style='font-size:.68rem;color:#475569;text-transform:uppercase;
                letter-spacing:.07em;margin-bottom:8px'>{L["bars_title"]}</div>
    {bars}
  </div>

  <div class='nlp-interp'>{interpretation} {L["compression_suffix"]}: {compression:.1f}%</div>
  <div class='nlp-stats'>{stats_line}</div>
</div>
"""


# ---------------------------------------------------------------------------
# Section 4 — Gradio layout
# ---------------------------------------------------------------------------

def build_interface(
    service: Optional[SummarizationService] = None,
    validator: Optional[InputValidator] = None,
) -> gr.Blocks:
    """
    Construct and return the Gradio ``Blocks`` application.

    Parameters
    ----------
    service:
        Pre-built :class:`~services.summarization_service.SummarizationService`.
        If ``None``, a new instance is created (triggering model loading).
    validator:
        Pre-built :class:`InputValidator`.  If ``None``, a default instance
        is created from ``UI_CONFIG``.  Inject a custom validator in tests
        to control validation behaviour without starting Gradio.

    Returns
    -------
    gr.Blocks
        Ready-to-launch Gradio application.
    """
    _service = service or SummarizationService()
    _validator = validator or InputValidator()

    # ── Callbacks ────────────────────────────────────────────────────

    def _load_example(key: str) -> str:
        """Populate the text input from the example dropdown."""
        return EXAMPLES.get(key, "")

    def _count_words_html(text: str) -> str:
        """Return a coloured word-count badge as an HTML snippet."""
        n = count_words(text) if isinstance(text, str) else 0
        color = word_count_color(n)
        return f"<span style='font-size:11px;color:{color};font-weight:600'>{n} words</span>"

    def _toggle_lang(current_lang: str) -> tuple[str, str, str]:
        """
        Flip the interface language between English and Arabic.

        Returns
        -------
        tuple[str, str, str]
            ``(new_lang_state, new_header_html, new_lang_btn_label)``
        """
        new_lang = "ar" if current_lang == "en" else "en"
        L = LANG[new_lang]
        header = (
            f"<div id='nlp-header'><h1>{L['title']}</h1>"
            f"<p>{L['subtitle']}</p></div>"
        )
        return new_lang, header, L["lang_btn"]

    def _run_summary(
        text: str,
        max_tokens: int,
        num_beams: int,
        length_penalty: float,
        no_repeat: int,
        lang: str,
    ) -> str:
        """
        Gradio click handler: validate inputs, call the service, render HTML.

        All validation is delegated to :class:`InputValidator`.  Any
        exception raised by the service is caught and rendered as an error
        card so the UI never crashes.

        Parameters
        ----------
        text:           Raw text from the input textbox.
        max_tokens:     Max-tokens slider value.
        num_beams:      Beam-width slider value.
        length_penalty: Length-penalty slider value.
        no_repeat:      No-repeat-ngram slider value.
        lang:           Current language state (``'en'`` or ``'ar'``).

        Returns
        -------
        str
            HTML string for the output panel.
        """
        L = LANG.get(lang, LANG["en"])
        direction = L["dir"]

        # --- validation ---------------------------------------------------
        validation = _validator.validate_all(
            text, max_tokens, num_beams, length_penalty, no_repeat
        )
        if not validation.is_valid:
            error_msg = L.get(validation.error_key or "err_generic", L["err_generic"])
            return _error_html(error_msg, direction)

        # --- build generation config from validated slider values ---------
        gen_cfg = GenerationConfig(
            max_new_tokens=int(max_tokens),
            num_beams=int(num_beams),
            length_penalty=float(length_penalty),
            no_repeat_ngram_size=int(no_repeat),
        )

        # --- service call -------------------------------------------------
        try:
            result = _service.summarize_text(
                text=text,
                generation_config=gen_cfg,
                lang=lang,
            )
        except ValueError as exc:
            logger.warning("Validation error from service: %s", exc)
            return _error_html(str(exc), direction)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Summarization failed in UI callback: %s", exc)
            return _error_html(L["err_generic"], direction)

        # --- render result ------------------------------------------------
        stats = format_stats_line(
            input_words=result.input_word_count,
            output_words=result.output_word_count,
            num_beams=int(num_beams),
            max_tokens=int(max_tokens),
            length_penalty=float(length_penalty),
            lang=lang,
        )

        return _build_output_html(
            summary=result.summary,
            rouge1=result.scores.rouge1,
            rouge2=result.scores.rouge2,
            rougeL=result.scores.rougeL,
            compression=result.compression_pct,
            elapsed=result.elapsed_seconds,
            stats_line=stats,
            interpretation=result.interpretation,
            lang_dict=L,
        )

    # ── Layout ───────────────────────────────────────────────────────

    min_t, max_t   = UI_CONFIG.slider_max_tokens_range
    min_b, max_b   = UI_CONFIG.slider_num_beams_range
    min_lp, max_lp = UI_CONFIG.slider_length_penalty_range
    min_nr, max_nr = UI_CONFIG.slider_no_repeat_range

    with gr.Blocks(css=_CSS, title=UI_CONFIG.title) as demo:
        lang_state = gr.State("en")

        header_html = gr.HTML(
            f"<div id='nlp-header'>"
            f"<h1>{LANG['en']['title']}</h1>"
            f"<p>{LANG['en']['subtitle']}</p></div>"
        )

        with gr.Row():
            with gr.Column(scale=5):
                pass
            with gr.Column(scale=1):
                lang_btn = gr.Button(
                    LANG["en"]["lang_btn"], elem_id="lang-btn", size="sm"
                )

        with gr.Row():
            # ── Left column: inputs ──────────────────────────────────
            with gr.Column(scale=3):
                example_dd = gr.Dropdown(
                    choices=[""] + list(EXAMPLES.keys()),
                    value="",
                    label=LANG["en"]["example_label"],
                    interactive=True,
                )
                text_input = gr.Textbox(
                    lines=8,
                    label=LANG["en"]["input_label"],
                    placeholder=LANG["en"]["input_placeholder"],
                    elem_classes=["nlp-input"],
                )
                word_count_html = gr.HTML(
                    "<span style='font-size:11px;color:#475569'>0 words</span>"
                )

                with gr.Accordion(LANG["en"]["params_title"], open=False):
                    max_tokens_slider = gr.Slider(
                        min_t, max_t,
                        value=GENERATION_CONFIG.max_new_tokens,
                        step=10,
                        label=LANG["en"]["max_tokens"],
                    )
                    num_beams_slider = gr.Slider(
                        min_b, max_b,
                        value=GENERATION_CONFIG.num_beams,
                        step=1,
                        label=LANG["en"]["num_beams"],
                    )
                    len_penalty_slider = gr.Slider(
                        min_lp, max_lp,
                        value=GENERATION_CONFIG.length_penalty,
                        step=0.5,
                        label=LANG["en"]["length_penalty"],
                    )
                    no_repeat_slider = gr.Slider(
                        min_nr, max_nr,
                        value=GENERATION_CONFIG.no_repeat_ngram_size,
                        step=1,
                        label=LANG["en"]["no_repeat"],
                    )

                run_btn = gr.Button(
                    LANG["en"]["run_btn"], elem_id="run-btn", variant="primary"
                )

            # ── Right column: output ─────────────────────────────────
            with gr.Column(scale=4):
                output_html = gr.HTML(
                    f"<div class='nlp-output-wrap' style='min-height:200px;display:flex;"
                    f"align-items:center;justify-content:center;"
                    f"color:#334155;font-size:.85rem'>"
                    f"{LANG['en']['placeholder_output']}</div>"
                )

        # ── Event wiring ─────────────────────────────────────────────
        example_dd.change(_load_example, inputs=example_dd, outputs=text_input)
        text_input.change(_count_words_html, inputs=text_input, outputs=word_count_html)
        lang_btn.click(
            _toggle_lang,
            inputs=lang_state,
            outputs=[lang_state, header_html, lang_btn],
        )
        run_btn.click(
            _run_summary,
            inputs=[
                text_input,
                max_tokens_slider,
                num_beams_slider,
                len_penalty_slider,
                no_repeat_slider,
                lang_state,
            ],
            outputs=output_html,
        )

    logger.info("Gradio interface built successfully.")
    return demo
