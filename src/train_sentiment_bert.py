import inspect
import pandas as pd
import numpy as np
import evaluate
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import DataCollatorWithPadding
from transformers import DistilBertForSequenceClassification
from transformers import DistilBertTokenizerFast
from transformers import Trainer
from transformers import TrainingArguments
from transformers import set_seed


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_data():
    df = pd.read_csv(
        "data/sentiment140.csv",
        encoding="latin-1",
        header=None,
        names=["target", "id", "date", "flag", "user", "text"],
    )

    df = df[df["target"].isin([0, 4])]
    df["label"] = df["target"].map({0: 0, 4: 1})
    df = df[["text", "label"]]

    sample_size = min(len(df), 100000)
    df = df.sample(sample_size, random_state=42)

    return train_test_split(
        df,
        test_size=0.1,
        random_state=42,
        stratify=df["label"],
    )


def tokenize(batch, tokenizer):
    return tokenizer(batch["text"], truncation=True)


metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return metric.compute(predictions=preds, references=labels)


def _build_training_args():
    kwargs = {
        "output_dir": "sentiment_bert",
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "num_train_epochs": 2,
        "save_strategy": "epoch",
        "logging_steps": 100,
        "logging_dir": "logs",
        "load_best_model_at_end": True,
        "report_to": "none",
        "metric_for_best_model": "accuracy",
        "greater_is_better": True,
    }
    if "evaluation_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
        kwargs["evaluation_strategy"] = "epoch"
    else:
        kwargs["eval_strategy"] = "epoch"
    return TrainingArguments(**kwargs)


def main():
    set_seed(42)
    print("Starting BERT sentiment training...")

    train_df, test_df = load_data()

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    test_ds = Dataset.from_pandas(test_df, preserve_index=False)

    train_ds = train_ds.rename_column("label", "labels")
    test_ds = test_ds.rename_column("label", "labels")

    train_ds = train_ds.map(lambda x: tokenize(x, tokenizer), batched=True)
    test_ds = test_ds.map(lambda x: tokenize(x, tokenizer), batched=True)

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,
    ).to(device)

    training_args = _build_training_args()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Trainer initialized. Beginning training loop...")

    trainer.train()
    results = trainer.evaluate()
    print("Final Evaluation:", results)

    trainer.save_model("sentiment_bert_model")
    tokenizer.save_pretrained("sentiment_bert_model")


if __name__ == "__main__":
    main()
