import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

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
def clean_and_engineer_data(df: pd.DataFrame) -> pd.DataFrame:
    processed_df = df.copy()
    processed_df.columns = processed_df.columns.str.strip()

    # SAFE CLEANING: Guard and preserve numeric columns
    if 'TotalClaims' in processed_df.columns:
        # Only run string adjustments if the column was read as text/object
        if processed_df['TotalClaims'].dtype == 'object':
            processed_df['TotalClaims'] = (
                processed_df['TotalClaims']
                .astype(str)
                .str.replace(r'[^\d\.]', '', regex=True)
            )
            processed_df['TotalClaims'] = pd.to_numeric(processed_df['TotalClaims'], errors='coerce')
        
        # Take absolute value for negative accounting indicators and fill true NaNs
        processed_df['TotalClaims'] = processed_df['TotalClaims'].abs().fillna(0.0)

    # Re-establish feature targets
    processed_df['ClaimOccurred'] = (processed_df['TotalClaims'] > 0).astype(int)
    return processed_df

def prepare_and_split_data(df: pd.DataFrame, task_type: str = "severity_regression"):
    """
    Slices population subsets, separates splits 80:20, and applies 
    leakage-free ordinal transformations onto split sets with strict NaN checks.
    """
    features = [
        'VehicleAge', 'CustomValueEstimate', 'SumInsured', 'ExcessSelected',
        'AlarmImmobiliser', 'TrackingDevice', 'NewVehicle', 'Gender', 
        'MaritalStatus', 'VehicleType', 'Product', 'CoverCategory'
    ]
    features = [f for f in features if f in df.columns]

    if task_type == "severity_regression":
        # Severity targets: Filter strictly for active historical claim rows
        working_df = df[df['TotalClaims'] > 0].copy()
        target_col = 'TotalClaims'
    else:
        # Full portfolio split for the Frequency model
        working_df = df.copy()
        target_col = 'ClaimOccurred'

    X = working_df[features].copy()
    y = working_df[target_col]

    # Clean text columns before splitting to avoid object formatting issues
    cat_cols = X.select_dtypes(include=['object']).columns
    for col in cat_cols:
        X[col] = X[col].astype(str).fillna("Missing")

    # Split dataset BEFORE fitting encoders to stop Data Leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Ordinal transform categorical string text values safely
    if len(cat_cols) > 0:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
        X_test[cat_cols] = encoder.transform(X_test[cat_cols])

    # GUARANTEE NO NaNs RE-EMERGE IN TRAINING MATRICES
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    return X_train, X_test, y_train, y_test