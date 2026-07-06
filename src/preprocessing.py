"""
preprocessing.py
----------------
Builds dataset version v2 (cleaned) out of v1 (raw).

Steps (mirrors the "Make v2 Dataset" section of the EDA notebook):
    1. Drop non-predictive / constant / leakage columns.
    2. Fix dtypes (Total Charges: string -> numeric) and handle missing values.
    3. Encode categorical columns (binary Yes/No -> 0/1, multi-category -> LabelEncoder).
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Columns identified during EDA as non-predictive, constant, or leaking the target
DROP_COLS = [
    "CustomerID",       # unique identifier, not predictive
    "Count",            # constant (=1 for all rows)
    "Country",          # constant ('United States')
    "State",            # constant ('California')
    "Zip Code",         # very high cardinality, redundant with Lat/Long
    "City",             # very high cardinality, redundant with Lat/Long
    "Lat Long",         # redundant with Latitude + Longitude
    "Churn Label",      # text duplicate of Churn Value (the target)
    "Churn Score",      # model-generated score -> leakage
    "CLTV",             # post-churn metric -> leakage
    "Churn Reason",     # only known AFTER a customer has already churned -> leakage
]


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """'Total Charges' arrives as text in the raw file; coerce it to numeric."""
    df = df.copy()
    if df["Total Charges"].dtype == object:
        df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    return df


def drop_non_predictive_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Only 'Total Charges' has nulls (~0.16% of rows, all customers with 0 tenure
    who have not been billed yet). Dropping them is safe given the negligible size.
    """
    before = len(df)
    df = df.dropna(subset=["Total Charges"])
    dropped = before - len(df)
    if dropped:
        print(f"[preprocessing] Dropped {dropped} rows with missing 'Total Charges'")
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Yes/No columns -> 0/1. Remaining multi-category text columns -> LabelEncoder."""
    df = df.copy()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    binary_map = {"Yes": 1, "No": 0}
    binary_cols = [c for c in cat_cols if set(df[c].unique()).issubset({"Yes", "No"})]
    for col in binary_cols:
        df[col] = df[col].map(binary_map)

    multi_cat_cols = [c for c in cat_cols if c not in binary_cols]
    encoders = {}
    for col in multi_cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df


def build_v2(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Full v1 -> v2 pipeline."""
    df = fix_dtypes(df_raw)
    df = drop_non_predictive_columns(df)
    df = handle_missing_values(df)
    df = encode_categorical(df)

    non_numeric = df.select_dtypes(exclude=["int64", "float64", "int32"]).columns.tolist()
    if non_numeric:
        raise RuntimeError(f"v2 still has non-numeric columns: {non_numeric}")

    print(f"[preprocessing] v2 built: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df
