import sqlite3
import sys

import pandas as pd
from scipy.stats import ks_2samp

FEATURES = ["AT", "V", "AP", "RH"]


def check_drift(train_df, new_df, alpha=0.05, min_rows=30):
    """Compare each input column of new data against the training data.
    Returns {column: {statistic, p_value, drift}}.

    alpha is split across the columns (Bonferroni), so that running four
    tests at once does not raise false alarms about 1 time in 5."""
    if len(new_df) < min_rows:
        raise ValueError(
            f"Need at least {min_rows} logged requests, found {len(new_df)}"
        )
    threshold = alpha / len(FEATURES)
    report = {}
    for col in FEATURES:
        result = ks_2samp(train_df[col], new_df[col])
        report[col] = {
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "drift": bool(result.pvalue < threshold),
        }
    return report


def load_requests(db_path="requests.db"):
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query("SELECT AT, V, AP, RH FROM requests", conn)


if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "requests.db"
    train = pd.read_csv("models/train_features.csv")
    new = load_requests(db_path)
    print(f"Logged requests: {len(new)}\n")
    report = check_drift(train, new)
    for col, r in report.items():
        flag = "DRIFT" if r["drift"] else "ok"
        print(f"{col:3s}  KS={r['statistic']:.3f}  p={r['p_value']:.4g}  {flag}")