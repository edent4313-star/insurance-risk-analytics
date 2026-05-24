import pandas as pd
import pytest
from src.preprocessing import clean_insurance_data
from src.preprocessing import (
    calculate_loss_ratio
)


def test_loss_ratio():

    df = pd.DataFrame({
        "TotalClaims": [100],
        "TotalPremium": [200]
    })

    df = calculate_loss_ratio(df)

    assert df["LossRatio"][0] == 0.5




def test_clean_insurance_data():
    data = {
        'TotalPremium': [100, 105, 110, 100000], # 100k is an outlier
        'TotalClaims':[0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    cleaned = clean_insurance_data(df)
    
    # Assert outlier was removed
    assert 100000 not in cleaned['TotalPremium'].values
    assert len(cleaned) == 3