import numpy as np
import pandas as pd
import pytest

from delivery.data import data_dir, load_train


def _has_data() -> bool:
    try:
        data_dir()
        return True
    except FileNotFoundError:
        return False


requires_data = pytest.mark.skipif(not _has_data(), reason="dataset/ not mounted")


@pytest.fixture(scope="session")
def raw_sample():
    """First 500 rows of the real training data (raw strings) + parsed target."""
    if not _has_data():
        pytest.skip("dataset/ not mounted")
    X, y = load_train()
    return X.head(500).reset_index(drop=True), y.head(500).reset_index(drop=True)


@pytest.fixture
def toy_raw():
    """A tiny hand-written raw frame that exercises the noisy cases."""
    return pd.DataFrame({
        "ID": ["0x1 ", "0x2 ", "0x3 ", "0x4 "],
        "Delivery_person_ID": ["INDORES13DEL02 ", "BANGRES18DEL02 ", "GOARES11DEL02 ", "JAPRES12DEL03 "],
        "Delivery_person_Age": ["37", "NaN ", "23", "38"],
        "Delivery_person_Ratings": ["4.9", "6", "NaN ", "4.7"],
        "Restaurant_latitude": ["22.745049", "12.913041", "0.0", "-26.902908"],
        "Restaurant_longitude": ["75.892471", "77.683237", "0.0", "-75.792934"],
        "Delivery_location_latitude": ["22.765049", "13.043041", "0.01", "26.932908"],
        "Delivery_location_longitude": ["75.912471", "77.813237", "0.01", "75.822934"],
        "Order_Date": ["19-03-2022", "25-03-2022", "17-02-2022", "15-03-2022"],
        "Time_Orderd": ["11:30:00", "23:55:00", "NaN ", "19:45:00"],
        "Time_Order_picked": ["11:45:00", "00:05:00", "10:50:00", "19:50:00"],
        "Weatherconditions": ["conditions Sunny", "conditions Stormy", "conditions NaN", "conditions Fog"],
        "Road_traffic_density": ["High ", "Jam ", "NaN ", "Low "],
        "Vehicle_condition": ["2", "2", "0", "1"],
        "Type_of_order": ["Snack ", "Snack ", "Drinks ", "Meal "],
        "Type_of_vehicle": ["motorcycle ", "scooter ", "motorcycle ", "scooter "],
        "multiple_deliveries": ["0", "1", "NaN ", "0"],
        "Festival": ["No ", "No ", "NaN ", "Yes "],
        "City": ["Urban ", "Metropolitian ", "NaN ", "Semi-Urban "],
    })


@pytest.fixture
def toy_y():
    return np.array([24.0, 33.0, 26.0, 21.0])


@pytest.fixture(scope="module")
def raw_sample_module(raw_sample):
    """Module-scoped alias of ``raw_sample`` (session fixture) for the pipeline tests."""
    return raw_sample
