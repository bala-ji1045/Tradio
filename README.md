# Tradio NIFTY Intraday Up/Down Prediction

This project builds ML models to predict whether the next candle's closing price will be higher (1) or lower (0) using NIFTY intraday OHLC data.

## Setup

1. Ensure Python 3.9+ is installed.
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Place your CSV (e.g., `nifty50_ticks.csv`) into `data/`.

https://github.com/user-attachments/assets/a9e5148e-ef2e-453f-abce-3151c71af879



## Run

Train, evaluate, and generate signals + cumulative PnL:
```
python src/train_and_eval.py --data data/nifty50_ticks.csv --time-split 0.7 --out output/test_predictions.csv
```

- `--time-split` is the fraction of data used for training via time-based split (default 0.7).
- The script prints Accuracy, Precision, Recall for each model and for the best model on the test set.

## Output

- Final testing dataset with additional columns saved to `output/test_predictions.csv`:
  - `target`, `pred_best`, `model_call`, `model_pnl`

## Best Model

A brief note is printed after training indicating which model performed best by accuracy (between Logistic Regression and Random Forest by default). In typical OHLC return-based features, Random Forest often wins due to non-linear patterns and robustness to noisy features.
