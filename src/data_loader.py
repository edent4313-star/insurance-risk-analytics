import pandas as pd
from seaborn.objects import Path


'''def load_data(path):

    df = pd.read_csv(
    path,
    sep=",",
    low_memory=False
)

    if "TransactionMonth" in df.columns:
        df["TransactionMonth"] = pd.to_datetime(df["TransactionMonth"], errors="coerce")
    return df

'''
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
   
'''
import os
import pandas as pd

file_path = "../data/MachineLearningRating_v3.csv"

# 1. Peek at the first line to auto-detect the separator
with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
    first_line = f.readline()

if ";" in first_line:
    sep = ";"
elif "\t" in first_line:
    sep = "\t"
else:
    sep = ","

print(f"Detected separator: '{sep}'")

# 2. Load the dataset with the detected separator
df = pd.read_csv(file_path, sep=sep, low_memory=False, encoding="utf-8")

print(f"Successfully loaded data from {file_path}")
print(f"Actual Rows Loaded: {len(df):,}")
print(f"Actual Columns Loaded: {len(df.columns)}")

# 3. Clean trailing whitespace in column names right away
df.columns = df.columns.str.strip()'''
