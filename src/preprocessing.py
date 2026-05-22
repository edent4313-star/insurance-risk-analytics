import numpy as np


def calculate_loss_ratio(df):

    df["LossRatio"] = (
        df["TotalClaims"] /
        df["TotalPremium"].replace(0, np.nan)
    )

    return df


def calculate_margin(df):

    df["Margin"] = (
        df["TotalPremium"] -
        df["TotalClaims"]
    )

    return df


def missing_value_summary(df):

    return (
        df.isnull()
        .sum()
        .sort_values(ascending=False)
    )