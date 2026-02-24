# Sentiment + Depression Text Analysis

This project trains NLP models to analyze free-text input for:

- **Sentiment** (`Positive` / `Negative`)
- **Mental health risk proxy** (`Depressed` / `Not Depressed`)

It includes:

- Traditional ML pipelines (TF-IDF + Logistic Regression)
- DistilBERT training scripts for sentiment and depression
- An interactive CLI predictor

## Project Structure

```text
sentiment-analysis/
├── data/
│   ├── amazon_cells_labelled.txt
│   ├── imdb_labelled.txt
│   ├── yelp_labelled.txt
│   ├── depression_dataset.csv
│   └── sentiment140.csv
├── src/
│   ├── preprocess.py
│   ├── train_sentiment.py
│   ├── train_depression.py
│   ├── train_sentiment_bert.py
│   └── predict.py
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
├── depression_model.pkl
├── depression_vectorizer.pkl
└── README.md
```

## Datasets Used

- **Sentiment (small labeled text):** Amazon, IMDB, Yelp labeled sentence files
- **Depression classification:** `depression_dataset.csv`
- **Large-scale sentiment (BERT):** `sentiment140.csv`

## Setup

### 1) Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Download NLTK resources (required)

```bash
python -m nltk.downloader punkt stopwords wordnet omw-1.4
```

## Train Models

### Train sentiment model (TF-IDF + Logistic Regression)

```bash
python -m src.train_sentiment
```

Outputs:

- `sentiment_model.pkl`
- `tfidf_vectorizer.pkl`

### Train depression model (TF-IDF + Logistic Regression)

```bash
python -m src.train_depression
```

Outputs:

- `depression_model.pkl`
- `depression_vectorizer.pkl`

### Train DistilBERT sentiment model (optional)

```bash
python -m src.train_sentiment_bert
```

Expected output directory:

- `sentiment_bert/` (training artifacts)
- `sentiment_bert_model/` (exported model for inference)

### Train DistilBERT depression model (optional)

```bash
python -m src.train_depression_bert
```

Expected output directory:

- `depression_bert/` (training artifacts)
- `depression_bert_model/` (exported model for inference)

## Run Predictions (CLI)

After training both classical models, run:

```bash
python -m src.predict
```

Then enter text interactively:

```text
Enter a sentence (or type 'exit'): I feel amazing today!
Sentiment: Positive
Mental Health Risk: Not Depressed
```

Optional BERT CLIs:

```bash
python -m src.predict_bert
python -m src.predict_depression_bert
```

## Model Pipeline Summary

1. Text is normalized in `src/preprocess.py` (lowercasing, URL removal, punctuation cleanup, tokenization, stopword removal, lemmatization).
2. Feature extraction uses TF-IDF.
3. Classification uses Logistic Regression for the classical pipeline.
4. Inference combines both trained models for a dual output.

## Notes

- `app.py` and `pipeline.sh` are currently placeholders.
- Pretrained `.pkl` files are present in the repository, so you can run predictions directly if they are compatible with your environment.

## Troubleshooting

- **`LookupError` from NLTK**: run the NLTK downloader command in Setup step 3.
- **Model file not found**: train sentiment and depression models first, or ensure the `.pkl` files exist in the project root.
- **MPS / Apple Silicon**: verify with:

	```bash
	python -c "import torch; print(torch.backends.mps.is_available())"
	```


changes need to be made
