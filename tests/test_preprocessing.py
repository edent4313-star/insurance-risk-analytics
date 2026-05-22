import pandas as pd

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