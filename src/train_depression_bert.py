import inspect
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed,
)
import evaluate


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return metric.compute(predictions=preds, references=labels)


def load_data():
    df = pd.read_csv("data/depression_dataset.csv")

    df = df[["clean_text", "is_depression"]]
    df = df.rename(columns={"clean_text": "text", "is_depression": "labels"})

    train_df, val_df = train_test_split(
        df,
        test_size=0.1,
        random_state=42,
        stratify=df["labels"],
    )

    return train_df, val_df


def tokenize(batch, tokenizer):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=256,
    )


def _build_training_args():
    kwargs = {
        "output_dir": "depression_bert",
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
    }
    if "evaluation_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
        kwargs["evaluation_strategy"] = "epoch"
    else:
        kwargs["eval_strategy"] = "epoch"
    return TrainingArguments(**kwargs)


def main():
    set_seed(42)
    print("Starting BERT Depression Training...")

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
    print("Final Evaluation:", results)

    trainer.save_model("depression_bert_model")
    tokenizer.save_pretrained("depression_bert_model")


if __name__ == "__main__":
    main()
