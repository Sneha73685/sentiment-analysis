"""BERT (DistilBERT) depression model, retrained on the leakage-free
persisted split (results/splits/).

Trains ONLY on results/splits/depression_train.csv. Uses
results/splits/depression_val.csv ONLY for evaluation / early stopping /
best-checkpoint selection (load_best_model_at_end). Same architecture and
hyperparameters as src/train_depression_bert.py -- only the data source,
id2label/label2id, and output directories changed.

results/splits/depression_test.csv is never loaded here.

Does not overwrite depression_bert/ or depression_bert_model/ -- writes to
depression_bert_v2/ (checkpoints) and depression_bert_model_v2/ (final
exported model) instead.
"""

import inspect
import json
import os
import subprocess

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    DataCollatorWithPadding,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
import evaluate

TRAIN_PATH = "results/splits/depression_train.csv"
VAL_PATH = "results/splits/depression_val.csv"
CHECKPOINT_DIR = "depression_bert_v2"
FINAL_MODEL_DIR = "depression_bert_model_v2"
RANDOM_SEED = 42
MAX_LENGTH = 256

ID2LABEL = {0: "not_depressed", 1: "depressed"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
metric = evaluate.load("accuracy")


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return metric.compute(predictions=preds, references=labels)


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)[["clean_text", "is_depression"]].rename(
        columns={"clean_text": "text", "is_depression": "labels"}
    )
    val_df = pd.read_csv(VAL_PATH)[["clean_text", "is_depression"]].rename(
        columns={"clean_text": "text", "is_depression": "labels"}
    )
    return train_df, val_df


def tokenize(batch, tokenizer):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)


def _build_training_args():
    kwargs = {
        "output_dir": CHECKPOINT_DIR,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "num_train_epochs": 4,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "save_strategy": "epoch",
        "logging_steps": 50,
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
        "greater_is_better": True,
        "report_to": "none",
        "gradient_accumulation_steps": 2,
        "seed": RANDOM_SEED,
    }
    if "evaluation_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
        kwargs["evaluation_strategy"] = "epoch"
    else:
        kwargs["eval_strategy"] = "epoch"
    return TrainingArguments(**kwargs)


def main():
    set_seed(RANDOM_SEED)
    print("Starting BERT Depression Training (v2, leakage-free split)...")

    train_df, val_df = load_data()

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    val_ds = Dataset.from_pandas(val_df, preserve_index=False)

    train_ds = train_ds.map(lambda x: tokenize(x, tokenizer), batched=True)
    val_ds = val_ds.map(lambda x: tokenize(x, tokenizer), batched=True)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    training_args = _build_training_args()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    print("Trainer initialized. Beginning training loop...")
    trainer.train()

    results = trainer.evaluate()
    print("Final Evaluation (validation set only):", results)

    trainer.save_model(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)

    config = {
        "script": "src/train_depression_bert_v2.py",
        "base_model": "distilbert-base-uncased",
        "train_path": TRAIN_PATH,
        "val_path": VAL_PATH,
        "test_path_used": None,
        "random_seed": RANDOM_SEED,
        "max_length": MAX_LENGTH,
        "training_args": {
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "num_train_epochs": 4,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "gradient_accumulation_steps": 2,
        },
        "early_stopping_patience": 1,
        "id2label": ID2LABEL,
        "label2id": LABEL2ID,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "final_validation_metrics": results,
        "git_commit": _git_commit(),
    }
    with open(os.path.join(FINAL_MODEL_DIR, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved model + tokenizer + training_config.json to {FINAL_MODEL_DIR}/")


if __name__ == "__main__":
    main()
