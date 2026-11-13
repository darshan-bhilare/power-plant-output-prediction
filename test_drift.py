import numpy as np
import pandas as pd
import pytest

from drift import check_drift


def make_data(rng, at_shift=0.0, n=500):
    return pd.DataFrame({
        "AT": rng.normal(20 + at_shift, 7, n),
        "V": rng.normal(54, 12, n),
        "AP": rng.normal(1013, 6, n),
        "RH": rng.normal(73, 14, n),
    })


def test_same_distribution_is_not_flagged():
    rng = np.random.default_rng(0)
    report = check_drift(make_data(rng), make_data(rng), alpha=0.001)
    assert not any(r["drift"] for r in report.values())


def test_shifted_temperature_is_flagged():
    rng = np.random.default_rng(0)
    report = check_drift(make_data(rng), make_data(rng, at_shift=8))
    assert report["AT"]["drift"]


def test_too_few_rows_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        check_drift(make_data(rng), make_data(rng, n=5))