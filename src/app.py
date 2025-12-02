import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.dummy import DummyClassifier

st.set_page_config(page_title="NIFTY Up/Down Prediction", layout="wide")


def load_data_from_df(df: pd.DataFrame) -> pd.DataFrame:
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
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])  # will raise if missing
    df = df.sort_values("Timestamp").reset_index(drop=True)
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
    df["ret_close"] = df["Close"].pct_change()
    df["ret_high"] = df["High"].pct_change()
    df["ret_low"] = df["Low"].pct_change()
    df["ret_open"] = df["Open"].pct_change()
    df["range"] = (df["High"] - df["Low"]) / df["Close"].shift(1)
    df["body"] = (df["Close"] - df["Open"]) / df["Open"].replace(0, np.nan)
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
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000))
        ]),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Baseline (Most Frequent)": DummyClassifier(strategy="most_frequent")
    }
    return models


def add_signal_and_pnl(test_df: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    df = test_df.copy()
    df["pred_best"] = y_pred
    df["model_call"] = np.where(df["pred_best"] == 1, "buy", "sell")
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


st.title("NIFTY Next-Candle Direction Prediction")
st.markdown("Upload OHLC CSV or use sample, set split, run models, and view metrics, signals, and PnL.")

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("Upload CSV (Timestamp, Open, High, Low, Close)", type=["csv"])
    use_sample = st.checkbox("Use built-in sample data", value=uploaded is None)
    train_frac = st.slider("Train fraction (time-based)", min_value=0.5, max_value=0.9, value=0.7, step=0.05)
    run_btn = st.button("Run Prediction")

if run_btn:
    try:
        if uploaded is not None:
            raw_df = pd.read_csv(uploaded)
        elif use_sample:
            # Small synthetic sample for demo
            dates = pd.date_range("2025-01-01 09:15", periods=200, freq="min")
            prices = 20000 + np.cumsum(np.random.randn(len(dates)))
            opens = prices + np.random.randn(len(dates))*0.5
            highs = np.maximum(opens, prices) + np.abs(np.random.randn(len(dates))*0.5)
            lows = np.minimum(opens, prices) - np.abs(np.random.randn(len(dates))*0.5)
            raw_df = pd.DataFrame({
                "Timestamp": dates,
                "Open": opens,
                "High": highs,
                "Low": lows,
                "Close": prices,
            })
        else:
            st.error("Please upload a CSV or enable sample data.")
            st.stop()

        df = load_data_from_df(raw_df)
        df = generate_labels(df)
        df = make_features(df)

        feature_cols = [c for c in df.columns if c not in ("Timestamp", "target", "next_close")]
        train_df, test_df = time_based_split(df, train_frac)

        X_train = train_df[feature_cols].values
        y_train = train_df["target"].values
        X_test = test_df[feature_cols].values
        y_test = test_df["target"].values

        models = build_models()

        results = []
        best_acc = -1
        best_name = None
        best_pred = None
        best_model = None

        unique_classes = np.unique(y_train)
        for name, model in models.items():
            if name != "Baseline (Most Frequent)" and unique_classes.size < 2:
                results.append({"model": name, "accuracy": np.nan, "precision": np.nan, "recall": np.nan})
                continue
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                results.append({"model": name, "accuracy": acc, "precision": prec, "recall": rec})
                if acc > best_acc:
                    best_acc = acc
                    best_name = name
                    best_pred = y_pred
                    best_model = model
            except Exception as e:
                results.append({"model": name, "accuracy": np.nan, "precision": np.nan, "recall": np.nan})

        st.subheader("Model Comparison")
        st.dataframe(pd.DataFrame(results).round(4))
        st.write(f"Best model: {best_name} (accuracy={best_acc:.4f})")

        result_df = add_signal_and_pnl(test_df[["Timestamp", "Open", "High", "Low", "Close", "target"]], best_pred)
        st.subheader("Test Predictions + Signals + PnL")
        st.dataframe(result_df.head(50))

        st.subheader("PnL Over Time (Cumulative)")
        st.line_chart(result_df.set_index("Timestamp")["model_pnl"]) 

        st.subheader("Candlestick (Test Range)")
        cndl = alt.Chart(result_df).encode(x='Timestamp:T')
        candlestick_chart = (
            cndl.mark_rule().encode(y='Low:Q', y2='High:Q') +
            cndl.mark_bar().encode(
                y='Open:Q',
                y2='Close:Q',
                color=alt.condition('datum.Close > datum.Open', alt.value('green'), alt.value('red'))
            )
        ).interactive()
        st.altair_chart(candlestick_chart, use_container_width=True)

        st.subheader("Feature Correlations (Train)")
        corr = train_df[[c for c in train_df.columns if c not in ("Timestamp","target","next_close")]].corr(numeric_only=True)
        st.dataframe(corr.round(3))
        corr_melt = corr.reset_index().melt('index')
        heat = alt.Chart(corr_melt).mark_rect().encode(
            x=alt.X('index:N', title='Feature'),
            y=alt.Y('variable:N', title='Feature'),
            color=alt.Color('value:Q', scale=alt.Scale(scheme='redblue'), title='Corr'),
            tooltip=['index','variable',alt.Tooltip('value:Q', format='.3f')]
        )
        st.altair_chart(heat, use_container_width=True)

        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download test_predictions.csv", data=csv, file_name="test_predictions.csv", mime="text/csv")

    except Exception as e:
        st.exception(e)
        st.error("Failed to run prediction. Check CSV columns and formats.")
