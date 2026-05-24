import pandas as pd

from src.preprocessing import prepare_policy_level_data
from src.hypothesis_tests import chi_square_claim_frequency_test, welch_ttest_numeric


def _toy_df():
    return pd.DataFrame({
        "UnderwrittenCoverID": [1, 1, 2, 2, 3, 3, 4, 4],
        "Province": ["Gauteng", "Gauteng", "Western Cape", "Western Cape", "Gauteng", "Gauteng", "Western Cape", "Western Cape"],
        "PostalCode": [1000, 1000, 2000, 2000, 1000, 1000, 2000, 2000],
        "VehicleType": ["Passenger Vehicle"] * 8,
        "CoverType": ["Own Damage"] * 8,
        "Gender": ["Female", "Female", "Male", "Male", "Female", "Female", "Male", "Male"],
        "TotalPremium": [100, 120, 100, 100, 80, 80, 90, 90],
        "TotalClaims": [0, 10, 0, 0, 0, 0, 20, 30],
        "TransactionMonth": pd.to_datetime(["2015-01-01"] * 8),
    })


def test_prepare_policy_level_data_adds_claimed_and_margin():
    policy = prepare_policy_level_data(_toy_df())
    assert "Claimed" in policy.columns
    assert "Margin" in policy.columns
    assert len(policy) == 4


def test_chi_square_claim_frequency_test_returns_p_value():
    policy = prepare_policy_level_data(_toy_df())
    res = chi_square_claim_frequency_test(
        policy,
        group_col="Province",
        group_a="Gauteng",
        group_b="Western Cape",
        kpi_col="Claimed",
    )
    assert 0.0 <= res["p_value"] <= 1.0


def test_welch_ttest_numeric_returns_p_value():
    policy = prepare_policy_level_data(_toy_df())
    res = welch_ttest_numeric(
        policy,
        group_col="Province",
        group_a="Gauteng",
        group_b="Western Cape",
        value_col="Margin",
        positive_only=False,
    )
    assert 0.0 <= res["p_value"] <= 1.0