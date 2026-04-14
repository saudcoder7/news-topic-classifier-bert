# 📰 News Topic Classifier Using BERT

> **Task 1 | AI Internship Portfolio Project**
> Fine-tune BERT on AG News to classify headlines into
> World, Sports, Business, and Sci/Tech categories.

---

## 📌 About This Project

This is **Task 1** of my AI Internship at CodeAlpha.

The goal is to fine-tune a BERT transformer model on the
AG News dataset to classify news headlines into 4 topic
categories and deploy it via Streamlit and Gradio.

---

## 📂 Project Structure

```
news-classifier/
│
├── news_classifier.py     # Main classifier + CLI demo
├── fine_tune_bert.py      # BERT fine-tuning script
├── streamlit_app.py       # Streamlit web interface
├── gradio_app.py          # Gradio web interface
├── requirements.txt       # All dependencies
└── README.md              # Project documentation
```

---

## 🎯 Objective

Build a text classification pipeline that:
- Tokenizes AG News headlines using BERT tokenizer
- Fine-tunes bert-base-uncased using Hugging Face Trainer API
- Evaluates with accuracy and F1-score
- Deploys via Streamlit and Gradio for live interaction

---

## 📊 Dataset Overview

| Property     | Details                              |
|--------------|--------------------------------------|
| Name         | AG News Dataset                      |
| Source       | Hugging Face Datasets Hub            |
| Train Size   | 120,000 headlines                    |
| Test Size    | 7,600 headlines                      |
| Categories   | 4 (World, Sports, Business, Sci/Tech)|
| Task         | Multi-class text classification      |

---

## 🏷️ Categories

| Label | Category  | Example Headline |
|-------|-----------|-----------------|
| 0 🌍 | World    | UN Security Council meets on crisis |
| 1 ⚽ | Sports   | Pakistan defeats India in T20 final |
| 2 💼 | Business | Apple reports record earnings |
| 3 🔬 | Sci/Tech | NASA discovers water on Mars |

---

## 🤖 Model Architecture

```
Input Headline (text)
       ↓
BERT Tokenizer (max_length=128)
       ↓
bert-base-uncased
(12 layers, 768 hidden, 110M params)
       ↓
Classification Head (Linear layer)
       ↓
Softmax → 4 class probabilities
       ↓
Predicted Category
```

---

## 🔧 Fine-Tuning Setup

| Parameter     | Value              |
|---------------|--------------------|
| Base Model    | bert-base-uncased  |
| Max Length    | 128 tokens         |
| Batch Size    | 16                 |
| Epochs        | 3                  |
| Learning Rate | 2e-5               |
| Optimizer     | AdamW              |
| Warmup Ratio  | 0.1                |
| Weight Decay  | 0.01               |

---

## 📈 Model Results

| Metric    | Score   |
|-----------|:-------:|
| Accuracy  | ~94%    |
| F1 Score  | ~0.94   |

> ✅ BERT achieves 94% accuracy on AG News —
> a strong result for 4-class news classification

---

## 🚀 How to Run

**Step 1 — Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/news-classifier.git
cd news-classifier
```

**Step 2 — Install dependencies**
```bash
pip install transformers datasets torch accelerate
           scikit-learn streamlit gradio
```

**Step 3 — Run demo instantly (no training needed)**
```bash
python news_classifier.py
```

**Step 4 — Fine-tune BERT (optional, needs GPU)**
```bash
python fine_tune_bert.py
```

**Step 5 — Launch Streamlit app**
```bash
streamlit run streamlit_app.py
```

**Step 6 — Launch Gradio app**
```bash
python gradio_app.py
```

---

## 🛠️ Tech Stack

| Tool                     | Purpose                        |
|--------------------------|--------------------------------|
| bert-base-uncased        | Pre-trained transformer model  |
| Hugging Face Transformers| Fine-tuning & inference        |
| Hugging Face Datasets    | AG News dataset loading        |
| PyTorch                  | Deep learning backend          |
| scikit-learn             | Accuracy & F1 metrics          |
| Streamlit                | Web interface                  |
| Gradio                   | Alternative web interface      |

---

## 💡 Key Learnings

- How to fine-tune BERT for text classification
- Tokenization and preprocessing for transformers
- Using Hugging Face Trainer API for model training
- Evaluating NLP models with accuracy and F1-score
- Deploying ML models with Streamlit and Gradio
- Transfer learning for NLP tasks

---

## 👤 Author

**Saood Faisal Sheikh**

LinkedIn — https://www.linkedin.com/in/saood-faisal-259b40316/
GitHub  — https://github.com/saudcoder7

---

## 📄 License

MIT License — free to use and adapt for your portfolio.
