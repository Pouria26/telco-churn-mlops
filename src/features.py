"""
features.py
-----------
Builds dataset version v3 (cleaned + engineered features) starting
from the RAW (v1) dataframe, following the "Make v3 Dataset" section
of the EDA notebook. Kept separate from preprocessing.py because
feature engineering is a distinct concern from basic cleaning, even
though both eventually produce a fully-numeric, model-ready table.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

LEAKAGE_COLS = ["Churn Label", "Churn Score", "Churn Reason"]  # CLTV / Churn Value kept for now, dropped later
NON_PREDICTIVE_COLS = ["CustomerID", "Count", "Country", "State", "Lat Long"]

SERVICE_COLS = [
    "Phone Service", "Multiple Lines", "Internet Service", "Online Security",
    "Online Backup", "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies",
]
SECURITY_COLS = ["Online Security", "Online Backup", "Device Protection", "Tech Support"]
ENTERTAINMENT_COLS = ["Streaming TV", "Streaming Movies"]


def remove_non_predictive(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in LEAKAGE_COLS + NON_PREDICTIVE_COLS if c in df.columns]
    df = df.drop(columns=drop_cols)
    if "CLTV" in df.columns:
        df = df.drop(columns=["CLTV"])
    return df


def add_payment_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Uses_Auto_Payment"] = df["Payment Method"].isin(
        ["Bank transfer (automatic)", "Credit card (automatic)"]
    ).astype(int)
    return df


def add_service_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    service_binary = pd.DataFrame(index=df.index)
    for col in SERVICE_COLS:
        if col in df.columns:
            service_binary[col] = (df[col] == "Yes").astype(int)

    df["Service_Count"] = service_binary.sum(axis=1)

    available_security = [c for c in SECURITY_COLS if c in service_binary.columns]
    df["Security_Service_Count"] = service_binary[available_security].sum(axis=1)

    available_entertainment = [c for c in ENTERTAINMENT_COLS if c in service_binary.columns]
    df["Entertainment_Service_Count"] = service_binary[available_entertainment].sum(axis=1)
    return df


def add_loyalty_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Is_New_Customer"] = (df["Tenure Months"] <= 12).astype(int)
    df["Is_Long_Term_Customer"] = (df["Tenure Months"] >= 48).astype(int)
    df["Tenure_Group"] = pd.cut(
        df["Tenure Months"], bins=[0, 12, 24, 48, 72],
        labels=["New", "Growing", "Established", "Loyal"], include_lowest=True,
    )
    return df


def add_value_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Monthly_Spending_Level"] = pd.qcut(
        df["Monthly Charges"], q=4, labels=["Low", "Medium", "High", "Premium"]
    )
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MonthToMonth_HighCharge"] = (
        (df["Contract"] == "Month-to-month") & (df["Monthly Charges"] > df["Monthly Charges"].median())
    ).astype(int)
    return df


def add_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce Lat/Long/City/Zip into low-cardinality region features, then drop the raw location columns."""
    df = df.copy()
    if {"Latitude", "Longitude"}.issubset(df.columns):
        df["Geo_Region"] = df["Latitude"].round(0).astype(str) + "_" + df["Longitude"].round(0).astype(str)
    if "City" in df.columns:
        city_counts = df["City"].value_counts()
        df["City_Customer_Density"] = df["City"].map(city_counts)

    location_cols = ["Zip Code", "City", "Latitude", "Longitude"]
    df = df.drop(columns=[c for c in location_cols if c in df.columns])
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    binary_map = {"Yes": 1, "No": 0}
    binary_cols = [c for c in cat_cols if set(df[c].astype(str).unique()).issubset({"Yes", "No"})]
    for col in binary_cols:
        df[col] = df[col].map(binary_map)

    multi_cat_cols = [c for c in cat_cols if c not in binary_cols]
    for col in multi_cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    return df


def build_v3(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Full v1 -> v3 pipeline: cleaning + feature engineering combined."""
    df = df_raw.copy()
    if df["Total Charges"].dtype == object:
        df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    df = df.dropna(subset=["Total Charges"])

    df = remove_non_predictive(df)
    df = add_payment_features(df)
    df = add_service_features(df)
    df = add_loyalty_features(df)
    df = add_value_features(df)
    df = add_interaction_features(df)
    df = add_geo_features(df)
    df = encode_categorical(df)

    non_numeric = df.select_dtypes(exclude=["int64", "float64", "int32"]).columns.tolist()
    if non_numeric:
        raise RuntimeError(f"v3 still has non-numeric columns: {non_numeric}")

    print(f"[features] v3 built: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df
