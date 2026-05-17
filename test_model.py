# =========================================================
# PYTEST UNIT TESTING
# =========================================================

import pytest
import joblib
import numpy as np

# =========================================================
# FIXTURE
# Load model once
# =========================================================

@pytest.fixture
def model():
    return joblib.load("wine_model.pkl")


# =========================================================
# TEST 1
# Check model loaded
# =========================================================

def test_model_loaded(model):
    assert model is not None


# =========================================================
# TEST 2
# Check prediction shape
# =========================================================

def test_prediction_shape(model):

    sample_data = np.array([
        [13.2, 2.77, 2.51, 18.5, 96.6, 1.04, 3.92, 0.32, 1.45, 4.8, 0.89, 2.12, 490]
    ])

    prediction = model.predict(sample_data)

    assert len(prediction) == 1


# =========================================================
# TEST 3
# Known prediction
# =========================================================

def test_known_prediction(model):

    sample_data = np.array([
        [13.2, 2.77, 2.51, 18.5, 96.6, 1.04, 3.92, 0.32, 1.45, 4.8, 0.89, 2.12, 490]
    ])

    prediction = model.predict(sample_data)

    # Expected class (this may vary, but for testing)
    assert prediction[0] in [0, 1, 2]


# =========================================================
# TEST 4
# Probabilities sum to 1
# =========================================================

def test_probability_sum(model):

    sample_data = np.array([
        [13.2, 2.77, 2.51, 18.5, 96.6, 1.04, 3.92, 0.32, 1.45, 4.8, 0.89, 2.12, 490]
    ])

    probabilities = model.predict_proba(sample_data)

    total_prob = probabilities[0].sum()

    assert total_prob == pytest.approx(1.0, abs=1e-6)


# =========================================================
# TEST 5
# Parameterized Testing
# =========================================================

@pytest.mark.parametrize(
    "sample_data",
    [
        [[13.2, 2.77, 2.51, 18.5, 96.6, 1.04, 3.92, 0.32, 1.45, 4.8, 0.89, 2.12, 490]],
        [[12.37, 0.94, 1.36, 10.6, 88, 1.98, 0.57, 0.28, 0.42, 1.95, 1.05, 1.82, 520]],
        [[14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065]],
    ]
)

def test_multiple_predictions(model, sample_data):

    prediction = model.predict(sample_data)

    assert len(prediction) == 1
    assert prediction[0] in [0, 1, 2]