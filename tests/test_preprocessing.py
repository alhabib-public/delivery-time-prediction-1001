import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from delivery.preprocessing import (ALL_FEATURES, NOMINAL_FEATURES, DeliveryCleaner,
                                    DeliveryFeatureEngineer, haversine_km, make_feature_pipeline,
                                    time_to_minutes)


def test_cleaner_types_and_nan_literals(toy_raw):
    c = DeliveryCleaner().fit(toy_raw)
    out = c.transform(toy_raw)
    assert list(out.columns) == list(toy_raw.columns)
    assert out["Delivery_person_Age"].dtype == float and np.isnan(out["Delivery_person_Age"][1])
    assert np.isnan(out["Delivery_person_Ratings"][1])          # rating 6 -> NaN
    assert np.isnan(out["Restaurant_latitude"][2])              # 0.0 -> NaN
    assert out["Restaurant_latitude"][3] == pytest.approx(26.902908)  # negative -> abs
    assert out["Weatherconditions"].tolist()[:2] == ["Sunny", "Stormy"]
    assert pd.isna(out["Weatherconditions"][2])
    assert out["Road_traffic_density"][0] == "High" and pd.isna(out["Road_traffic_density"][2])
    assert pd.api.types.is_datetime64_any_dtype(out["Order_Date"])
    assert out["Time_Orderd"][0] == 11 * 60 + 30 and np.isnan(out["Time_Orderd"][2])
    assert not out.isin(["NaN", "conditions NaN"]).any().any()
    assert c.get_feature_names_out().tolist() == list(toy_raw.columns)


def test_cleaner_is_idempotent(toy_raw):
    once = DeliveryCleaner().fit_transform(toy_raw)
    twice = DeliveryCleaner().fit_transform(once)
    pd.testing.assert_frame_equal(once, twice)


def test_time_to_minutes():
    s = pd.Series(["00:00:00", "23:59:00", "12:30", "NaN", "25:00:00", None])
    out = time_to_minutes(s).tolist()
    assert out[0] == 0 and out[1] == 23 * 60 + 59 and out[2] == 750
    assert all(np.isnan(v) for v in out[3:])


def test_haversine_known_distance():
    # Bangalore MG Road -> Bangalore airport is ~30 km
    d = haversine_km(12.9752, 77.6035, 13.1989, 77.7068)
    assert 26 < d < 30
    assert haversine_km(0, 0, 0, 0) == 0
    assert np.isnan(haversine_km(np.nan, 0, 0, 0))


def test_feature_engineer_outputs(toy_raw):
    f = DeliveryFeatureEngineer().fit_transform(DeliveryCleaner().fit_transform(toy_raw))
    assert list(f.columns) == ALL_FEATURES
    assert f["prep_time_min"].tolist()[:2] == [15.0, 10.0]      # midnight wrap on row 1
    assert np.isnan(f["prep_time_min"][2])
    assert f["driver_city"].tolist() == ["INDO", "BANG", "GOA", "JAP"]
    assert f["coords_missing"][2] == 1 and f["coords_missing"][0] == 0
    assert f["distance_km"][0] == pytest.approx(3.02, abs=0.05)
    assert f["weekday"][0] == 5 and f["is_weekend"][0] == 1     # 19-03-2022 is a Saturday
    assert f["order_hour"][3] == 19


@pytest.mark.parametrize("kind", ["tree", "dense"])
def test_feature_pipeline_no_object_columns(toy_raw, kind):
    pipe = make_feature_pipeline(kind)
    Z = pipe.fit_transform(toy_raw)
    assert isinstance(Z, pd.DataFrame)
    assert all(pd.api.types.is_numeric_dtype(Z[c]) for c in Z.columns)
    if kind == "dense":
        assert np.isfinite(Z.to_numpy()).all()
        assert Z.shape[1] > len(ALL_FEATURES)          # one-hot expanded
    else:
        assert list(Z.columns[-len(NOMINAL_FEATURES):]) == NOMINAL_FEATURES
        assert (Z[NOMINAL_FEATURES].to_numpy() >= -1).all()
    # unseen category at transform time must not crash
    other = toy_raw.copy()
    other.loc[0, "Weatherconditions"] = "conditions Hail"
    Z2 = pipe.transform(other)
    assert Z2.shape == Z.shape
    clone(pipe)


def test_feature_pipeline_on_real_rows(raw_sample):
    X, _ = raw_sample
    Z = make_feature_pipeline("tree").fit_transform(X)
    assert len(Z) == len(X)
    assert Z["distance_km"].dropna().between(0, 100).all()
