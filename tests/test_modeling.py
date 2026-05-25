import pandas as pd
import pytest
from src.modeling import InsuranceModeler

from src.modeling import (
    prepare_model_data
)


def test_prepare_model_data():

    df = pd.DataFrame({

        "Gender": ["Male"],

        "Province": ["Gauteng"],

        "VehicleType": ["Sedan"],

        "make": ["Toyota"],

        "CustomValueEstimate": [100000],

        "kilowatts": [120],

        "Cylinders": [4],

        "TotalClaims": [0]
    })

    X, y_clf, y_reg, features = (
        prepare_model_data(df)
    )

    assert X.shape[0] == 1


def sample_data():
    """Create a dummy dataset based on the user's columns."""
    return pd.DataFrame({
        'RegistrationYear': [2010, 2015, 2020, 2018],
        'SumInsured': [100000, 150000, 200000, 120000],
        'TotalClaims': [5000, 0, 12000, 0],
        'Gender': ['Male', 'Female', 'Male', 'Female'],
        'TotalPremium': [500, 600, 800, 550],
        'CalculatedPremiumPerTerm': [450, 550, 750, 500]
    })

def test_preprocessing_filters_claims(sample_data):
    modeler = InsuranceModeler(sample_data)
    X, y = modeler.preprocess_for_task4()
    # Should only have 2 rows where TotalClaims > 0
    assert len(X) == 2
    assert 'VehicleAge' in X.columns

def test_model_training_output(sample_data):
    modeler = InsuranceModeler(sample_data)
    X, y = modeler.preprocess_for_task4()
    # Using small data for test, might throw warnings but should complete
    results, _, _ = modeler.train_regression_models(X, y)
    assert "XGBoost" in results
    assert results["XGBoost"]["rmse"] >= 0