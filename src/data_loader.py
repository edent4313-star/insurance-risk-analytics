import pandas as pd


def load_data(path):

    df = pd.read_csv(
    path,
    sep="|",
    low_memory=False
)

    df["TransactionMonth"] = pd.to_datetime(
        df["TransactionMonth"],
        errors="coerce"
    )

    return df