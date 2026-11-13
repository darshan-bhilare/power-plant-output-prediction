import pandas as pd
import pytest

from preprocess import preprocess_features


def good_df():
    return pd.DataFrame({"AT": [14.96], "V": [41.76], "AP": [1024.07], "RH": [73.17]})


def test_clean_data_passes_through():
    out = preprocess_features(good_df())
    assert list(out.columns) == ["AT", "V", "AP", "RH"]
    assert len(out) == 1


def test_text_numbers_are_converted():
    df = good_df().astype(str)
    out = preprocess_features(df)
    assert out["AT"].iloc[0] == pytest.approx(14.96)


def test_missing_column_raises():
    with pytest.raises(ValueError):
        preprocess_features(good_df().drop(columns=["RH"]))


def test_missing_value_raises():
    df = good_df()
    df.loc[0, "AT"] = None
    with pytest.raises(ValueError):
        preprocess_features(df)


def test_non_numeric_text_raises():
    df = good_df().astype(object)
    df.loc[0, "V"] = "hello"
    with pytest.raises(ValueError):
        preprocess_features(df)


def test_impossible_value_raises():
    df = good_df()
    df.loc[0, "RH"] = 500
    with pytest.raises(ValueError):
        preprocess_features(df)