import numpy as np
import pandas as pd
from scipy import stats

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

import pandas as pd
import numpy as np

def clean_insurance_data(df):
    """
    Cleans the insurance dataset by dropping missing values and 
    handling potential outliers in TotalPremium.
    """
    df_clean = df.copy()
    
    # Drop rows where critical columns are missing
    df_clean = df_clean.dropna(subset=['TotalPremium', 'TotalClaims'])
    
    # Remove extreme outliers in TotalPremium using IQR method
    Q1 = df_clean['TotalPremium'].quantile(0.25)
    Q3 = df_clean['TotalPremium'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df_clean = df_clean[(df_clean['TotalPremium'] >= lower_bound) & 
                        (df_clean['TotalPremium'] <= upper_bound)]
    
    return df_clean

def get_group_statistics(df, group_col, target_col):
    """
    Returns a dictionary of statistics for each category in the grouping column.
    """
    stats = {}
    for group in df[group_col].unique():
        subset = df[df[group_col] == group][target_col]
        stats[group] = {
            'count': len(subset),
            'mean': subset.mean(),
            'median': subset.median(),
            'std': subset.std()
        }
    return stats

def perform_t_test(df, group_col, target_col, group1, group2):
    data1 = df[df[group_col] == group1][target_col]
    data2 = df[df[group_col] == group2][target_col]
    
    # --- ADD THIS CHECK ---
    print(f"DEBUG: {group1} count: {len(data1)}, {group2} count: {len(data2)}")
    
    if len(data1) < 2 or len(data2) < 2:
        print("ERROR: Not enough data in one of the groups to perform t-test.")
        return np.nan, np.nan, data1, data2
    
    # Check if variance is zero
    if data1.var() == 0 or data2.var() == 0:
        print("ERROR: One group has zero variance (all values are the same).")
        return np.nan, np.nan, data1, data2
    # ----------------------
    
    t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=False)
    return t_stat, p_value, data1, data2


def prepare_data_for_analysis(df):
    """Cleans data and adds log-transformed columns for better statistical testing."""
    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=['TotalPremium', 'CoverCategory'])
    
    # Adding a log-transformed column here is good practice 
    # because it is a permanent feature of your dataset
    df_clean['LogPremium'] = np.log1p(df_clean['TotalPremium'])
    
    return df_clean

def get_region_statistics(df, region_col, target_col):
    
    summary = df.groupby(region_col)[target_col].agg(['count', 'mean', 'median', 'std'])
    summary.columns =['n', 'Mean', 'Median', 'Std Dev']
    return summary.round(2)

from scipy import stats

def perform_anova(df, group_col, target_col):
    """
    Groups data by category and performs a one-way ANOVA.
    """
    # Create a list of groups
    groups = [group[target_col] for name, group in df.groupby(group_col)]
    
    # Perform ANOVA
    f_stat, p_value = stats.f_oneway(*groups)
    
    return f_stat, p_value