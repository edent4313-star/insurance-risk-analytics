import numpy as np
import pandas as pd

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

def normalize_gender(df):
    """Normalize common gender labels to Female/Male/Unknown."""
    out = df.copy()
    if "Gender" not in out.columns:
        return out

    def _map_gender(value):
        if pd.isna(value):
            return "Unknown"
        s = str(value).strip().lower()
        if s in {"f", "female", "woman", "women"}:
            return "Female"
        if s in {"m", "male", "man", "men"}:
            return "Male"
        if s in {"", "not specified", "unknown", "nan"}:
            return "Unknown"
        return str(value).strip().title()

    out["Gender"] = out["Gender"].map(_map_gender)
    return out


def prepare_policy_level_data(df, policy_id_col="UnderwrittenCoverID"):
    """Aggregate monthly insurance rows into one policy/coverage record.

    - TotalPremium and TotalClaims are summed across months.
    - Claimed is 1 if any month has a positive claim.
    - Margin = TotalPremium - TotalClaims.
    """
    out = df.copy()

    for col in ["TotalPremium", "TotalClaims", "PostalCode", "RegistrationYear", "Cylinders", "cubiccapacity", "kilowatts", "NumberOfDoors", "CustomValueEstimate", "SumInsured"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "TotalPremium" not in out.columns or "TotalClaims" not in out.columns:
        raise KeyError("The dataset must contain TotalPremium and TotalClaims.")

    out["Claimed"] = (out["TotalClaims"].fillna(0) > 0).astype(int)
    out["Margin"] = out["TotalPremium"].fillna(0) - out["TotalClaims"].fillna(0)

    sum_cols = [c for c in ["TotalPremium", "TotalClaims", "Margin"] if c in out.columns]
    first_cols = [c for c in out.columns if c not in set(sum_cols + ["Claimed"])]

    agg = {c: "first" for c in first_cols}
    agg["TotalPremium"] = "sum"
    agg["TotalClaims"] = "sum"
    agg["Claimed"] = "max"
    agg["Margin"] = "sum"

    policy = (
        out.groupby(policy_id_col, as_index=False)
        .agg(agg)
    )

    # Recalculate after aggregation to keep margin consistent
    policy["Claimed"] = (policy["TotalClaims"].fillna(0) > 0).astype(int)
    policy["Margin"] = policy["TotalPremium"].fillna(0) - policy["TotalClaims"].fillna(0)
    return policy