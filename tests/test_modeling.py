import pandas as pd

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