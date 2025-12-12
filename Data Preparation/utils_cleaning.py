import pandas as pd
import numpy as np

# -----------------------------------------------------------
# 1. Remove columns that are entirely NA
# -----------------------------------------------------------
def drop_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(axis=1, how="all")


# -----------------------------------------------------------
# 2. Replace invalid string tokens with NaN
# -----------------------------------------------------------
def replace_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    invalid_tokens = ["?", "NA", "N/A", "--", " ", ""]
    return df.replace(invalid_tokens, np.nan)


# -----------------------------------------------------------
# 3. Convert features to numeric when possible
#    (no warning anymore, and safe for all datasets)
# -----------------------------------------------------------
def convert_numeric_features(df: pd.DataFrame, target: str = None) -> pd.DataFrame:
    df2 = df.copy()
    for col in df2.columns:
        if col == target:
            continue

        # convert using pandas without warnings
        try:
            df2[col] = pd.to_numeric(df2[col], errors="coerce")
        except Exception:
            pass  # keep original if completely non-numeric

    return df2


# -----------------------------------------------------------
# 4. Fix Wine Quality datasets (red/white)
#    Works for both ";" and "," separators
# -----------------------------------------------------------
def fix_winequality(path: str) -> pd.DataFrame:
    """
    Loads and cleans winequality CSVs (red & white).
    Automatically detects separator and converts columns.
    """

    # Try semicolon first
    try:
        df = pd.read_csv(path, sep=";")
        if df.shape[1] > 1:
            df.columns = [c.strip().replace('"', "") for c in df.columns]
            df = df.apply(pd.to_numeric, errors="ignore")
            return df
    except Exception:
        pass

    # Try comma
    try:
        df = pd.read_csv(path, sep=",")
        if df.shape[1] > 1:
            df.columns = [c.strip().replace('"', "") for c in df.columns]
            df = df.apply(pd.to_numeric, errors="ignore")
            return df
    except Exception:
        pass

    # If file is malformed (single column)
    raw = pd.read_csv(path, header=None)
    lines = raw.iloc[:, 0]

    header = lines.iloc[0].replace('"', "").split(";")
    data = [row.replace('"', "").split(";") for row in lines.iloc[1:]]

    df = pd.DataFrame(data, columns=header)

    # convert numeric columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    return df
