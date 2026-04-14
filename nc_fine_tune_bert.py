# ============================================================
#  fine_tune_bert.py
#  Fine-tune bert-base-uncased on AG News dataset
#  Run ONCE to train — then use streamlit_app.py / gradio_app.py
# ============================================================
#
#  Requirements:
#    pip install transformers datasets torch accelerate scikit-learn
#
#  Expected training time:
#    CPU  : ~2–3 hours
#    GPU  : ~15–20 minutes
# ============================================================

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import torch

# ── CONFIG ───────────────────────────────────────────────────
MODEL_NAME  = "bert-base-uncased"
OUTPUT_DIR  = "./news_classifier_model"
MAX_LENGTH  = 128
BATCH_SIZE  = 16
EPOCHS      = 3
LR          = 2e-5
NUM_LABELS  = 4

LABEL_NAMES = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}

def main():
    print("=" * 55)
    print("   BERT FINE-TUNING — AG NEWS CLASSIFIER")
    print("=" * 55)

    # ── Step 1: Load tokenizer & model ───────────────────────
    print(f"\n[1/5] Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=LABEL_NAMES,
        label2id={v: k for k, v in LABEL_NAMES.items()},
    )
    print("      ✅ Model loaded")

    # ── Step 2: Load AG News dataset ─────────────────────────
    print("\n[2/5] Loading AG News dataset ...")
    dataset = load_dataset("ag_news")
    print(f"      ✅ Train: {len(dataset['train'])} | "
          f"Test: {len(dataset['test'])} rows")

    # ── Step 3: Tokenize ─────────────────────────────────────
    print("\n[3/5] Tokenizing dataset ...")

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )
    print("      ✅ Tokenization complete")

    # ── Step 4: Metrics ──────────────────────────────────────
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        f1  = f1_score(labels, preds, average="weighted")
        return {"accuracy": acc, "f1": f1}

    # ── Step 5: Training Arguments ───────────────────────────
    print("\n[4/5] Setting up Trainer ...")
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        warmup_ratio=0.1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    print("      ✅ Trainer ready")

    # ── Train ─────────────────────────────────────────────────
    print("\n[5/5] Training BERT ...")
    trainer.train()

    # ── Save ──────────────────────────────────────────────────
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # ── Final Evaluation ─────────────────────────────────────
    print("\n📊 Final Evaluation on Test Set:")
    results = trainer.evaluate()
    print(f"   Accuracy : {results['eval_accuracy']*100:.2f}%")
    print(f"   F1 Score : {results['eval_f1']:.4f}")

    print(f"\n✅ Model saved to {OUTPUT_DIR}")
    print("   Run: streamlit run streamlit_app.py")
    print("   Or : python gradio_app.py")

if __name__ == "__main__":
    main()
