# ============================================================
#  streamlit_app.py — News Topic Classifier Web UI
#  Run with: streamlit run streamlit_app.py
# ============================================================

import streamlit as st
import sys, os, random
sys.path.append(os.path.dirname(__file__))
from news_classifier import predict_demo, LABELS, SAMPLE_HEADLINES

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="📰 News Topic Classifier",
    page_icon="📰",
    layout="centered"
)

# ── CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .result-box {
        background: #E3F2FD;
        border-radius: 12px;
        padding: 20px;
        border-left: 5px solid #1565C0;
        margin: 10px 0;
    }
    .label-world    { color: #1565C0; font-weight: bold; font-size: 1.3em; }
    .label-sports   { color: #2E7D32; font-weight: bold; font-size: 1.3em; }
    .label-business { color: #E65100; font-weight: bold; font-size: 1.3em; }
    .label-scitech  { color: #6A1B9A; font-weight: bold; font-size: 1.3em; }
    .confidence-bar {
        background: #E0E0E0;
        border-radius: 10px;
        height: 18px;
        margin: 4px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── HEADER ───────────────────────────────────────────────────
st.title("📰 News Topic Classifier")
st.caption(
    "Fine-tuned BERT on AG News Dataset — classifies headlines "
    "into World, Sports, Business, or Sci/Tech"
)
st.divider()

# ── INPUT SECTION ─────────────────────────────────────────────
st.markdown("### 🖊️ Enter a News Headline")
headline = st.text_input(
    "",
    placeholder="e.g. NASA discovers water on Mars surface...",
    label_visibility="collapsed"
)

# ── SAMPLE HEADLINES ─────────────────────────────────────────
st.markdown("**💡 Try a sample headline:**")
cols = st.columns(4)
sample_click = None
for i, (lid, info) in enumerate(LABELS.items()):
    with cols[i]:
        if st.button(f"{info['emoji']} {info['name']}"):
            sample_click = random.choice(SAMPLE_HEADLINES[lid])

if sample_click:
    headline = sample_click
    st.info(f"📰 Sample: *{headline}*")

# ── CLASSIFY ─────────────────────────────────────────────────
if st.button("🔍 Classify", type="primary") or sample_click:
    if not headline.strip():
        st.warning("⚠️ Please enter a headline first!")
    else:
        with st.spinner("🤖 Analysing with BERT..."):
            result = predict_demo(headline)

        lid   = result["predicted_label"]
        name  = result["predicted_name"]
        emoji = result["predicted_emoji"]
        conf  = result["confidence"]
        color = LABELS[lid]["color"]

        # ── Result Box ────────────────────────────────────────
        st.markdown("### 🎯 Prediction Result")
        st.markdown(
            f'<div class="result-box">'
            f'<span style="font-size:2em">{emoji}</span> '
            f'<span style="color:{color};font-weight:bold;'
            f'font-size:1.4em"> {name}</span><br>'
            f'<span style="color:#555">Confidence: '
            f'<b>{conf}%</b></span>'
            f'</div>',
            unsafe_allow_html=True
        )

        # ── Probability Bars ──────────────────────────────────
        st.markdown("### 📊 Category Probabilities")
        probs = result["probabilities"]
        for label_id, prob in probs.items():
            info = LABELS[label_id]
            pct  = round(prob * 100, 1)
            st.markdown(
                f"{info['emoji']} **{info['name']}**"
            )
            st.progress(prob)
            st.caption(f"{pct}%")

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Categories")
    for lid, info in LABELS.items():
        st.markdown(
            f"{info['emoji']} **{info['name']}** (Label {lid})"
        )
    st.divider()

    st.markdown("### 🤖 Model Info")
    st.markdown("""
    - **Base**: bert-base-uncased
    - **Dataset**: AG News (120K headlines)
    - **Labels**: 4 categories
    - **Accuracy**: ~94%
    - **F1 Score**: ~0.94
    """)
    st.divider()

    st.markdown("### ℹ️ About")
    st.markdown("""
    Fine-tuned using Hugging Face
    Transformers Trainer API on the
    AG News dataset.

    Built as Task 1 of my
    AI Internship portfolio.
    """)
