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

# src/preprocessing.py
import pandas as pd

def load_data(path):
    """
    Loads insurance data from a CSV file and converts date columns.
    Includes error handling for file paths.
    """
    try:
        df = pd.read_csv(
            path,
            sep=",", 
            low_memory=False
        )

        # Convert date columns if they exist
        if "TransactionMonth" in df.columns:
            df["TransactionMonth"] = pd.to_datetime(df["TransactionMonth"], errors="coerce")
            
        print(f"Successfully loaded data from {path}")
        return df

    except FileNotFoundError:
        print(f"Error: The file at {path} was not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
   

