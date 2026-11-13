import pandas as pd

FEATURES = ["AT", "V", "AP", "RH"]
TARGET = "PE"

# Physically impossible values. Wider than the training data on purpose.
VALID_RANGES = {
    "AT": (-50, 60),
    "V": (0, 150),
    "AP": (800, 1100),
    "RH": (0, 105),
}


def select_columns(df, columns):
    """Keep only the needed columns, in a fixed order."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df[columns].copy()


def to_numeric(df):
    """Turn text like '14.96' into numbers. Bad text becomes NaN."""
    return df.apply(pd.to_numeric, errors="coerce")


def check_no_missing(df):
    """Stop if any cell is empty or was not a number."""
    bad = df.isnull().sum()
    bad = bad[bad > 0]
    if len(bad) > 0:
        raise ValueError(f"Missing or non-numeric values: {bad.to_dict()}")
    return df


def check_ranges(df):
    """Stop if any value is physically impossible."""
    for col, (low, high) in VALID_RANGES.items():
        if col in df.columns and not df[col].between(low, high).all():
            raise ValueError(f"{col} has values outside {low} to {high}")
    return df


def preprocess_features(df):
    """The whole pandas pipeline for the 4 input columns."""
    return (
        df.pipe(select_columns, FEATURES)
          .pipe(to_numeric)
          .pipe(check_no_missing)
          .pipe(check_ranges)
    )


def load_training_data(path="power_plant.csv"):
    """Read the CSV and return clean inputs (X) and target (y)."""
    raw = pd.read_csv(path)
    X = preprocess_features(raw)
    y = pd.to_numeric(raw[TARGET], errors="coerce")
    if y.isnull().any():
        raise ValueError("Target column PE has missing or non-numeric values")
    return X, y