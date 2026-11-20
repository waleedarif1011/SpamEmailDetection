"""Gradio live demo — paste SMS/email text for spam vs ham + trigger tokens."""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from src.predict import load_pipeline, predict
from src.train import METRICS_PATH, PIPELINE_PATH

ROOT = Path(__file__).resolve().parent

PRESETS = [
    {
        "label": "Phishing prize SMS",
        "text": (
            "Congratulations! You have WON a £1000 cash prize. "
            "Call 09061701461 now to claim. Free entry. T&Cs apply. Reply STOP to opt out."
        ),
    },
    {
        "label": "Normal work email",
        "text": (
            "Hi team, please review the Q3 roadmap draft before Thursday's standup. "
            "I've attached the slides and the action items from last week. Thanks!"
        ),
    },
    {
        "label": "Urgent bank scam",
        "text": (
            "URGENT: Your bank account will be SUSPENDED today. "
            "Verify your identity immediately at http://secure-bank-login.example "
            "or your funds will be locked. Act now!"
        ),
    },
    {
        "label": "Casual text",
        "text": "Hey, running 10 minutes late — grab a table if you get there first?",
    },
]

_PIPELINE = None


def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        if not PIPELINE_PATH.is_file():
            raise FileNotFoundError(
                "artifacts/spam_pipeline.joblib missing — run `python -m src.train` first."
            )
        _PIPELINE = load_pipeline(PIPELINE_PATH)
    return _PIPELINE


def _metrics_blurb() -> str:
    if not METRICS_PATH.is_file():
        return "_Train with `python -m src.train` to populate metrics._"
    m = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    best = m.get("best", {})
    rows = "\n".join(
        f"| `{r['name']}` | {r['accuracy']:.3f} | {r['precision']:.3f} | "
        f"{r['recall']:.3f} | {r['f1']:.3f} | {r['roc_auc']:.3f} |"
        for r in m.get("models", [])
    )
    return (
        f"**Best model:** `{best.get('name', '—')}` "
        f"(F1={best.get('f1', float('nan')):.3f}, "
        f"ROC-AUC={best.get('roc_auc', float('nan')):.3f})\n\n"
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |\n"
        "|-------|----------|-----------|--------|----|---------|\n"
        f"{rows}"
    )


def classify(message: str) -> tuple[str, dict[str, float], str]:
    result = predict(message or "", pipeline=_get_pipeline())
    label = result["label"]
    spam_p = float(result["spam_probability"])
    ham_p = 1.0 - spam_p
    verdict = (
        f"### Verdict: **{label.upper()}**\n\n"
        f"Spam probability: **{spam_p:.1%}**"
    )
    conf = {"ham": ham_p, "spam": spam_p}
    tokens = result.get("top_tokens") or []
    if tokens:
        lines = "\n".join(
            f"- `{t['token']}` (score {t['score']:+.3f})" for t in tokens if t["score"] > 0
        ) or "_No strong spam-trigger tokens in this message._"
    else:
        lines = "_No tokens extracted._"
    return verdict, conf, f"### Top spam-trigger tokens\n\n{lines}"


def apply_preset(name: str) -> str:
    match = next((p for p in PRESETS if p["label"] == name), PRESETS[0])
    return match["text"]


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Spam Email Detection") as demo:
        gr.Markdown(
            "# Spam Email / SMS Detector\n\n"
            "Paste a message to classify it as **spam** or **ham**, "
            "with confidence and the tokens that most push toward spam.\n\n"
            + _metrics_blurb()
        )
        with gr.Row():
            with gr.Column(scale=3):
                preset = gr.Dropdown(
                    choices=[p["label"] for p in PRESETS],
                    value=PRESETS[0]["label"],
                    label="Preset examples",
                )
                message = gr.Textbox(
                    lines=8,
                    label="Message",
                    placeholder="Paste an email or SMS…",
                    value=PRESETS[0]["text"],
                )
                run = gr.Button("Classify", variant="primary")
            with gr.Column(scale=2):
                verdict = gr.Markdown()
                confidence = gr.Label(label="Confidence")
                tokens = gr.Markdown()

        preset.change(fn=apply_preset, inputs=preset, outputs=message)
        run.click(fn=classify, inputs=message, outputs=[verdict, confidence, tokens])
        demo.load(fn=classify, inputs=message, outputs=[verdict, confidence, tokens])
    return demo


def main() -> None:
    demo = build_ui()
    demo.launch()


if __name__ == "__main__":
    main()
