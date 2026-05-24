from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind 



def matched_segment(df, group_col, group_a, group_b, match_cols=None):
    """Return a two-group subset with an optional dominant shared segment.

    This keeps the comparison focused on the target group variable while
    reducing confounding from other attributes.
    """
    sub = df[df[group_col].isin([group_a, group_b])].copy()
    if not match_cols:
        return sub

    for col in match_cols:
        if col not in sub.columns:
            continue

        a_vals = set(sub.loc[sub[group_col] == group_a, col].dropna().astype(str).unique())
        b_vals = set(sub.loc[sub[group_col] == group_b, col].dropna().astype(str).unique())
        common = sorted(a_vals.intersection(b_vals))
        if not common:
            continue

        common_df = sub[sub[col].astype(str).isin(common)].copy()
        if common_df.empty:
            continue

        mode_val = common_df[col].dropna().astype(str).mode()
        if not mode_val.empty:
            sub = sub[sub[col].astype(str) == mode_val.iloc[0]].copy()

    return sub


def chi_square_claim_frequency_test(df, group_col, group_a, group_b, kpi_col="Claimed"):
    """Chi-squared test for claim frequency between two groups."""
    subset = df[df[group_col].isin([group_a, group_b])].copy()
    if subset.empty:
        raise ValueError("No rows available for the requested groups.")

    contingency = pd.crosstab(subset[group_col], subset[kpi_col])
    # Ensure both claim states are present
    for col in [0, 1]:
        if col not in contingency.columns:
            contingency[col] = 0
    contingency = contingency[[0, 1]]

    chi2, p_value, dof, expected = chi2_contingency(contingency)
    return {
        "test": "chi-squared",
        "chi2": chi2,
        "p_value": p_value,
        "dof": dof,
        "contingency": contingency,
        "expected": expected,
        "group_a": group_a,
        "group_b": group_b,
        "kpi": kpi_col,
    }


def welch_ttest_numeric(df, group_col, group_a, group_b, value_col, positive_only=False):
    """Welch's t-test for numerical KPI between two groups."""
    subset = df[df[group_col].isin([group_a, group_b])].copy()
    if subset.empty:
        raise ValueError("No rows available for the requested groups.")

    a = subset.loc[subset[group_col] == group_a, value_col].dropna()
    b = subset.loc[subset[group_col] == group_b, value_col].dropna()

    if positive_only:
        a = a[a > 0]
        b = b[b > 0]

    if len(a) < 2 or len(b) < 2:
        raise ValueError("Each group must have at least two observations for Welch's t-test.")

    stat, p_value = ttest_ind(a, b, equal_var=False, nan_policy="omit")
    return {
        "test": "Welch t-test",
        "t_stat": stat,
        "p_value": p_value,
        "group_a": group_a,
        "group_b": group_b,
        "kpi": value_col,
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
    }


def decision_from_pvalue(p_value, alpha=0.05):
    return "Reject H₀" if p_value < alpha else "Fail to reject H₀"


def build_result_row(hypothesis, kpi, test_name, p_value, alpha=0.05):
    return {
        "Hypothesis": hypothesis,
        "KPI": kpi,
        "Test Used": test_name,
        "P-Value": float(p_value),
        "Decision": decision_from_pvalue(p_value, alpha=alpha),
    }