# ============================================================
#  gradio_app.py — News Classifier Gradio Interface
#  Run with: python gradio_app.py
#  Or deploy free on: huggingface.co/spaces
# ============================================================

import gradio as gr
import sys, os
sys.path.append(os.path.dirname(__file__))
from news_classifier import predict_demo, LABELS, SAMPLE_HEADLINES
import random

def classify_headline(headline):
    if not headline.strip():
        return "⚠️ Please enter a headline.", {}

    result  = predict_demo(headline)
    name    = result["predicted_name"]
    emoji   = result["predicted_emoji"]
    conf    = result["confidence"]
    probs   = result["probabilities"]

    label_probs = {
        f"{LABELS[k]['emoji']} {LABELS[k]['name']}": v
        for k, v in probs.items()
    }
    output_text = (
        f"**{emoji} {name}**\n\n"
        f"Confidence: **{conf}%**"
    )
    return output_text, label_probs

# ── GRADIO INTERFACE ─────────────────────────────────────────
examples = [
    ["UN Security Council meets to discuss Middle East crisis"],
    ["Pakistan defeats India in thrilling T20 World Cup final"],
    ["Apple reports record quarterly earnings beating estimates"],
    ["OpenAI releases GPT-5 with advanced reasoning capabilities"],
]

demo = gr.Interface(
    fn=classify_headline,
    inputs=gr.Textbox(
        label="📰 News Headline",
        placeholder="Enter a news headline to classify...",
        lines=2
    ),
    outputs=[
        gr.Markdown(label="🎯 Prediction"),
        gr.Label(label="📊 Category Probabilities", num_top_classes=4),
    ],
    title="📰 News Topic Classifier (BERT)",
    description=(
        "Fine-tuned BERT on AG News Dataset. "
        "Classifies news headlines into: "
        "🌍 World | ⚽ Sports | 💼 Business | 🔬 Sci/Tech"
    ),
    examples=examples,
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch(share=True)
