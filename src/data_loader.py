import pandas as pd
from seaborn.objects import Path


'''def load_data(path):

    df = pd.read_csv(
    path,
    sep="|",
    low_memory=False
)

    if "TransactionMonth" in df.columns:
        df["TransactionMonth"] = pd.to_datetime(df["TransactionMonth"], errors="coerce")
    return df'''

def load_data(path):
    
    df = pd.read_csv(
        path,
        sep=",", 
        low_memory=False
    )

    if "TransactionMonth" in df.columns:
        df["TransactionMonth"] = pd.to_datetime(df["TransactionMonth"], errors="coerce")
    return df

   

