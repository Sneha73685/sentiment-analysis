#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage: ./pipeline.sh [options]

Runs the sentiment-analysis project end-to-end.

Options:
  --skip-install       Skip dependency installation
  --skip-nltk          Skip NLTK resource download
  --skip-classical     Skip classical model training
  --train-bert         Train both BERT models
  --run-cli            Run classical interactive CLI after training
  --run-bert-cli       Run both BERT interactive CLIs after training
  --help               Show this help

Examples:
  ./pipeline.sh
  ./pipeline.sh --train-bert
  ./pipeline.sh --skip-install --run-cli
EOF
}

SKIP_INSTALL=0
SKIP_NLTK=0
SKIP_CLASSICAL=0
TRAIN_BERT=0
RUN_CLI=0
RUN_BERT_CLI=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-install) SKIP_INSTALL=1 ;;
    --skip-nltk) SKIP_NLTK=1 ;;
    --skip-classical) SKIP_CLASSICAL=1 ;;
    --train-bert) TRAIN_BERT=1 ;;
    --run-cli) RUN_CLI=1 ;;
    --run-bert-cli) RUN_BERT_CLI=1 ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
  shift
done

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but was not found in PATH."
  exit 1
fi

if [[ $SKIP_INSTALL -eq 0 ]]; then
  echo "Installing dependencies from requirements.txt..."
  python3 -m pip install --upgrade pip
  python3 -m pip install -r requirements.txt
fi

if [[ $SKIP_NLTK -eq 0 ]]; then
  echo "Downloading NLTK resources..."
  python3 -m nltk.downloader punkt stopwords wordnet omw-1.4
fi

if [[ $SKIP_CLASSICAL -eq 0 ]]; then
  echo "Training classical sentiment model..."
  python3 -m src.train_sentiment

  echo "Training classical depression model..."
  python3 -m src.train_depression
fi

if [[ $TRAIN_BERT -eq 1 ]]; then
  echo "Training BERT sentiment model..."
  python3 -m src.train_sentiment_bert

  echo "Training BERT depression model..."
  python3 -m src.train_depression_bert
fi

if [[ $RUN_CLI -eq 1 ]]; then
  echo "Launching classical combined predictor CLI..."
  python3 -m src.predict
fi

if [[ $RUN_BERT_CLI -eq 1 ]]; then
  echo "Launching BERT sentiment CLI..."
  python3 -m src.predict_bert

  echo "Launching BERT depression CLI..."
  python3 -m src.predict_depression_bert
fi

echo "Pipeline finished successfully."
