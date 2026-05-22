import pandas as pd


def load_data(path):

    df = pd.read_csv(path, sep="|")

    df["TransactionMonth"] = pd.to_datetime(
        df["TransactionMonth"],
        errors="coerce"
    )

    return df