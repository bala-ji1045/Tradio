import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.dummy import DummyClassifier


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize column names
    cols = {c.lower().strip(): c for c in df.columns}
    # Expected names: Timestamp, Open, High, Low, Close (case-insensitive)
    rename_map = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc in ("timestamp", "time", "date"):
            rename_map[c] = "Timestamp"
        elif lc == "open":
            rename_map[c] = "Open"
        elif lc == "high":
            rename_map[c] = "High"
        elif lc == "low":
            rename_map[c] = "Low"
        elif lc == "close":
            rename_map[c] = "Close"
    if rename_map:
        df = df.rename(columns=rename_map)
    # Parse timestamp and sort
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])  # raises if missing
    df = df.sort_values("Timestamp").reset_index(drop=True)
    # Keep only required columns
    df = df[["Timestamp", "Open", "High", "Low", "Close"]]
    return df


def generate_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["next_close"] = df["Close"].shift(-1)
    df["target"] = (df["next_close"] > df["Close"]).astype(int)
    df = df.dropna().reset_index(drop=True)
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Basic returns and lags
    df["ret_close"] = df["Close"].pct_change()
    df["ret_high"] = df["High"].pct_change()
    df["ret_low"] = df["Low"].pct_change()
    df["ret_open"] = df["Open"].pct_change()

    # Price range and body
    df["range"] = (df["High"] - df["Low"]) / df["Close"].shift(1)
    df["body"] = (df["Close"] - df["Open"]) / df["Open"].replace(0, np.nan)

    # Rolling stats (short-term momentum/volatility)
    for w in (3, 5, 10):
        df[f"roll_close_mean_{w}"] = df["Close"].rolling(w).mean()
        df[f"roll_close_std_{w}"] = df["Close"].rolling(w).std()
        df[f"roll_ret_mean_{w}"] = df["ret_close"].rolling(w).mean()
        df[f"roll_ret_std_{w}"] = df["ret_close"].rolling(w).std()

    df = df.dropna().reset_index(drop=True)
    return df


def time_based_split(df: pd.DataFrame, train_frac: float):
    n = len(df)
    n_train = int(n * train_frac)
    train = df.iloc[:n_train].copy()
    test = df.iloc[n_train:].copy()
    return train, test


def build_models():
    models = {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000))
        ]),
        "rf": RandomForestClassifier(n_estimators=200, random_state=42),
        "baseline": DummyClassifier(strategy="most_frequent")
    }
    return models


def evaluate_model(name: str, model, X_test: np.ndarray, y_test: np.ndarray):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    print(f"Model {name}: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f}")
    return acc, prec, rec, y_pred


def add_signal_and_pnl(test_df: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    df = test_df.copy()
    df["pred_best"] = y_pred
    df["model_call"] = np.where(df["pred_best"] == 1, "buy", "sell")
    # Cumulative running PnL per given rule:
    # - If buy: subtract current close from cumulative
    # - If sell: add current close to cumulative
    pnl = []
    cum = 0.0
    for c, call in zip(df["Close"].values, df["model_call"].values):
        if call == "buy":
            cum -= float(c)
        else:
            cum += float(c)
        pnl.append(cum)
    df["model_pnl"] = pnl
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/nifty50_ticks.csv")
    parser.add_argument("--time-split", type=float, default=0.7, help="Fraction of data for training (time-based split)")
    parser.add_argument("--out", type=str, default="output/test_predictions.csv")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file not found: {args.data}")

    df = load_data(args.data)
    df = generate_labels(df)
    df = make_features(df)

    feature_cols = [c for c in df.columns if c not in ("Timestamp", "target", "next_close")]
    X = df[feature_cols].values
    y = df["target"].values

    train_df, test_df = time_based_split(df, args.time_split)
    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["target"].values

    models = build_models()

    best_name = None
    best_model = None
    best_acc = -1
    best_pred = None

    # Handle single-class training gracefully by allowing only baseline
    unique_classes = np.unique(y_train)
    for name, model in models.items():
        if name != "baseline" and unique_classes.size < 2:
            print(f"Skipping {name}: training data has a single class")
            continue
        try:
            model.fit(X_train, y_train)
            acc, prec, rec, y_pred = evaluate_model(name, model, X_test, y_test)
            if acc > best_acc:
                best_acc = acc
                best_name = name
                best_model = model
                best_pred = y_pred
        except Exception as e:
            print(f"Model {name} failed: {e}")

    print(f"Best model: {best_name} with acc={best_acc:.4f}")
    # Evaluate best explicitly
    acc, prec, rec, _ = evaluate_model(best_name, best_model, X_test, y_test)

    result_df = add_signal_and_pnl(test_df[["Timestamp", "Open", "High", "Low", "Close", "target"]], best_pred)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    result_df.to_csv(args.out, index=False)
    print(f"Saved test predictions to {args.out}")


if __name__ == "__main__":
    main()
