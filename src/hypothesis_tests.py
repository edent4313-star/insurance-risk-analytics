from scipy.stats import ttest_ind


def province_claim_test(
    df,
    province1,
    province2
):

    group1 = df[
        df["Province"] == province1
    ]["TotalClaims"]

    group2 = df[
        df["Province"] == province2
    ]["TotalClaims"]

    stat, p_value = ttest_ind(
        group1,
        group2,
        nan_policy="omit"
    )

    return stat, p_value