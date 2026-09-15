"""sklearn transformers for the delivery dataset.

Three stages, all stateless with respect to the training data so they are safe
inside cross-validation and behave identically on ``test.csv``:

``DeliveryCleaner``          raw strings -> typed columns (NaN literals, whitespace,
                              "(min)"/"conditions" prefixes, impossible values)
``DeliveryFeatureEngineer``  typed columns -> model features (haversine distance,
                              time-of-day, prep time, calendar, missing flags, city code)
``make_preprocessor(kind)``  ColumnTransformer that encodes/imputes/scales for either
                              tree models (``"tree"``) or linear/neural models (``"dense"``)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

# ---------------------------------------------------------------------------
# column groups (raw names as they appear in the CSV, after header strip)
# ---------------------------------------------------------------------------
RAW_NUMERIC = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
    "Vehicle_condition",
    "multiple_deliveries",
]
RAW_CATEGORICAL = [
    "Weatherconditions",
    "Road_traffic_density",
    "Type_of_order",
    "Type_of_vehicle",
    "Festival",
    "City",
]
RAW_TIME = ["Time_Orderd", "Time_Order_picked"]
RAW_DATE = "Order_Date"
RAW_ID = ["ID", "Delivery_person_ID"]

TRAFFIC_LEVELS = ["Low", "Medium", "High", "Jam"]

# engineered feature groups consumed by make_preprocessor
NUMERIC_FEATURES = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Vehicle_condition",
    "multiple_deliveries",
    "Restaurant_latitude",
    "Restaurant_longitude",
    "distance_km",
    "order_hour",
    "order_minute_of_day",
    "picked_minute_of_day",
    "prep_time_min",
    "weekday",
    "is_weekend",
    "day_of_month",
    "month",
    "age_missing",
    "ratings_missing",
    "order_time_missing",
    "coords_missing",
]
ORDINAL_FEATURES = ["Road_traffic_density"]
NOMINAL_FEATURES = [
    "Weatherconditions",
    "Type_of_order",
    "Type_of_vehicle",
    "Festival",
    "City",
    "driver_city",
]
ALL_FEATURES = NUMERIC_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _strip(s: pd.Series) -> pd.Series:
    """Strip whitespace and turn the literal ``"NaN"``/empty strings into real NaN."""
    s = s.astype("string").str.strip()
    s = s.mask(s.isin(["NaN", "nan", "", "None", "conditions NaN"]))
    return s.astype(object).where(s.notna(), np.nan)


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(_strip(s), errors="coerce").astype(float)


def time_to_minutes(s: pd.Series) -> pd.Series:
    """``"19:45:00"`` -> ``1185.0`` (minutes since midnight); NaN stays NaN."""
    s = _strip(s)
    parts = s.astype("string").str.extract(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$")
    hours = pd.to_numeric(parts[0], errors="coerce")
    minutes = pd.to_numeric(parts[1], errors="coerce")
    out = (hours * 60 + minutes).astype(float)
    out[(hours >= 24) | (minutes >= 60)] = np.nan
    return out


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km (NaN propagates)."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(a, dtype=float)) for a in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))


class _FrameTransformer(TransformerMixin, BaseEstimator):
    """Base for DataFrame-in / DataFrame-out transformers with sklearn plumbing."""

    def _check_frame(self, X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"{type(self).__name__} expects a pandas DataFrame, got {type(X).__name__}")
        return X

    def fit(self, X, y=None):
        X = self._check_frame(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.feature_names_out_ = np.asarray(self._output_columns(X), dtype=object)
        return self

    def _output_columns(self, X: pd.DataFrame) -> list[str]:  # pragma: no cover - abstract
        raise NotImplementedError

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.string = True
        tags.input_tags.allow_nan = True
        tags.no_validation = True
        return tags


# ---------------------------------------------------------------------------
# stage 1: cleaning
# ---------------------------------------------------------------------------
class DeliveryCleaner(_FrameTransformer):
    """Turn the raw string frame into typed columns.

    Parameters
    ----------
    max_rating : float
        Ratings above this are treated as noise -> NaN (the data has a few ``6``).
    min_coord_abs : float
        Coordinates with ``abs(value) < min_coord_abs`` are placeholders (``0.01``,
        ``0``) -> NaN. Negative coordinates are sign-flipped (all deliveries are in
        India, which has positive lat/lon).
    """

    def __init__(self, max_rating: float = 5.0, min_coord_abs: float = 1.0):
        self.max_rating = max_rating
        self.min_coord_abs = min_coord_abs

    def _output_columns(self, X):
        return list(X.columns)

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        X = self._check_frame(X)
        out = pd.DataFrame(index=X.index)
        for c in RAW_ID:
            if c in X:
                out[c] = _strip(X[c])
        for c in RAW_NUMERIC:
            if c in X:
                out[c] = _to_num(X[c])
        if "Delivery_person_Ratings" in out:
            r = out["Delivery_person_Ratings"]
            out["Delivery_person_Ratings"] = r.where((r <= self.max_rating) & (r >= 0), np.nan)
        for c in ("Restaurant_latitude", "Restaurant_longitude",
                  "Delivery_location_latitude", "Delivery_location_longitude"):
            if c in out:
                v = out[c].abs()
                out[c] = v.where(v >= self.min_coord_abs, np.nan)
        for c in RAW_CATEGORICAL:
            if c in X:
                s = _strip(X[c])
                if c == "Weatherconditions":
                    s = s.astype("string").str.replace(r"^conditions\s+", "", regex=True)
                    s = s.astype(object).where(s.notna(), np.nan)
                out[c] = s
        if RAW_DATE in X:
            if pd.api.types.is_datetime64_any_dtype(X[RAW_DATE]):      # already cleaned
                out[RAW_DATE] = X[RAW_DATE]
            else:
                out[RAW_DATE] = pd.to_datetime(_strip(X[RAW_DATE]), format="%d-%m-%Y", errors="coerce")
        for c in RAW_TIME:
            if c in X:
                out[c] = X[c].astype(float) if pd.api.types.is_numeric_dtype(X[c]) else time_to_minutes(X[c])
        # keep any extra columns untouched (e.g. a target passed through by mistake)
        for c in X.columns:
            if c not in out:
                out[c] = X[c]
        return out[list(X.columns)]


# ---------------------------------------------------------------------------
# stage 2: feature engineering
# ---------------------------------------------------------------------------
class DeliveryFeatureEngineer(_FrameTransformer):
    """Derive model features from the cleaned frame (expects ``DeliveryCleaner`` output).

    Output columns are exactly ``ALL_FEATURES``. ``drop_ids`` keeps the two ID
    columns out of the feature matrix (they are never used by the models).
    """

    def __init__(self, drop_ids: bool = True):
        self.drop_ids = drop_ids

    def _output_columns(self, X):
        cols = list(ALL_FEATURES)
        if not self.drop_ids:
            cols = [c for c in RAW_ID if c in X] + cols
        return cols

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        X = self._check_frame(X)
        f = pd.DataFrame(index=X.index)

        if not self.drop_ids:
            for c in RAW_ID:
                if c in X:
                    f[c] = X[c]

        for c in ("Delivery_person_Age", "Delivery_person_Ratings", "Vehicle_condition",
                  "multiple_deliveries", "Restaurant_latitude", "Restaurant_longitude"):
            f[c] = X[c].astype(float)

        f["distance_km"] = haversine_km(
            X["Restaurant_latitude"], X["Restaurant_longitude"],
            X["Delivery_location_latitude"], X["Delivery_location_longitude"],
        )

        ordered = X["Time_Orderd"].astype(float)
        picked = X["Time_Order_picked"].astype(float)
        f["order_hour"] = np.floor(ordered / 60)
        f["order_minute_of_day"] = ordered
        f["picked_minute_of_day"] = picked
        prep = picked - ordered
        prep = prep.where(prep >= 0, prep + 24 * 60)      # pickup after midnight wrap
        f["prep_time_min"] = prep

        d = pd.to_datetime(X["Order_Date"])
        f["weekday"] = d.dt.weekday.astype(float)
        f["is_weekend"] = (d.dt.weekday >= 5).astype(float)
        f["day_of_month"] = d.dt.day.astype(float)
        f["month"] = d.dt.month.astype(float)

        f["age_missing"] = X["Delivery_person_Age"].isna().astype(float)
        f["ratings_missing"] = X["Delivery_person_Ratings"].isna().astype(float)
        f["order_time_missing"] = ordered.isna().astype(float)
        f["coords_missing"] = f["distance_km"].isna().astype(float)

        f["Road_traffic_density"] = X["Road_traffic_density"]
        for c in ("Weatherconditions", "Type_of_order", "Type_of_vehicle", "Festival", "City"):
            f[c] = X[c]
        driver = X["Delivery_person_ID"].astype("string")
        city = driver.str.extract(r"^([A-Za-z]+?)RES", expand=False)
        f["driver_city"] = city.astype(object).where(city.notna(), np.nan)

        return f[self._output_columns(X)]


# ---------------------------------------------------------------------------
# stage 3: encoding
# ---------------------------------------------------------------------------
def make_preprocessor(kind: str = "tree") -> ColumnTransformer:
    """Encode ``ALL_FEATURES`` for a model family.

    ``"tree"``   numeric passthrough (NaN kept: HistGradientBoosting handles it natively),
                 traffic -> ordinal 0..3 (NaN kept), nominal -> integer codes with ``-1``
                 for unknown/missing (HGB treats negatives as missing). Output is a pandas
                 DataFrame so HGB can be told ``categorical_features=NOMINAL_FEATURES``.
    ``"dense"``  median-impute + standardise numeric, ordinal traffic imputed to -1 and
                 scaled, nominal one-hot (unknown -> all zeros). For Ridge / the MLP.
    """
    if kind == "tree":
        ct = ColumnTransformer(
            [
                ("num", "passthrough", NUMERIC_FEATURES),
                ("ord", OrdinalEncoder(categories=[TRAFFIC_LEVELS],
                                       handle_unknown="use_encoded_value", unknown_value=np.nan,
                                       encoded_missing_value=np.nan), ORDINAL_FEATURES),
                ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                       encoded_missing_value=-1), NOMINAL_FEATURES),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        ct.set_output(transform="pandas")
        return ct
    if kind == "dense":
        num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        ordn = Pipeline([
            ("enc", OrdinalEncoder(categories=[TRAFFIC_LEVELS], handle_unknown="use_encoded_value",
                                   unknown_value=-1, encoded_missing_value=-1)),
            ("scale", StandardScaler()),
        ])
        cat = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        ct = ColumnTransformer(
            [("num", num, NUMERIC_FEATURES), ("ord", ordn, ORDINAL_FEATURES), ("cat", cat, NOMINAL_FEATURES)],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        ct.set_output(transform="pandas")
        return ct
    raise ValueError(f"unknown preprocessor kind {kind!r}; use 'tree' or 'dense'")


def make_feature_pipeline(kind: str = "tree") -> Pipeline:
    """cleaner -> feature engineer -> encoder (no model)."""
    return Pipeline([
        ("clean", DeliveryCleaner()),
        ("features", DeliveryFeatureEngineer()),
        ("encode", make_preprocessor(kind)),
    ])
