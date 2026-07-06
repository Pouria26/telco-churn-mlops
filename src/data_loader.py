"""
data_loader.py
--------------
Responsible ONLY for reading / writing the three dataset versions
(v1 = raw, v2 = cleaned, v3 = cleaned + engineered features).

No cleaning or feature-engineering logic lives here on purpose:
that logic belongs to preprocessing.py and features.py so every
module has a single, clear responsibility (Single Responsibility
Principle) and can be unit-tested independently.
"""

import os
import pandas as pd

# Root of the project (one level above src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

RAW_FILE_NAME = "Telco_customer_churn.xlsx"
V2_FILE_NAME = "telco_customer_churn_v2.xlsx"
V3_FILE_NAME = "telco_customer_churn_v3.xlsx"

VERSION_FILE_MAP = {
    "v1": RAW_FILE_NAME,
    "v2": V2_FILE_NAME,
    "v3": V3_FILE_NAME,
}


def _version_path(version: str) -> str:
    if version not in VERSION_FILE_MAP:
        raise ValueError(f"Unknown dataset version '{version}'. Use one of {list(VERSION_FILE_MAP)}")
    return os.path.join(DATA_DIR, version, VERSION_FILE_MAP[version])


def load_raw_data() -> pd.DataFrame:
    """Load the untouched v1 dataset exactly as delivered by IBM."""
    path = _version_path("v1")
    df = pd.read_excel(path)
    print(f"[data_loader] Loaded RAW data: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def load_version(version: str) -> pd.DataFrame:
    """Load an already-built dataset version ('v1', 'v2' or 'v3')."""
    path = _version_path(version)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Build it first (see preprocessing.py / features.py) "
            f"or run `python run_pipeline.py --version {version}`."
        )
    df = pd.read_excel(path)
    print(f"[data_loader] Loaded {version}: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def save_version(df: pd.DataFrame, version: str) -> str:
    """Persist a dataframe as a given dataset version and return the saved path."""
    path = _version_path(version)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_excel(path, index=False)
    print(f"[data_loader] Saved {version}: {df.shape[0]:,} rows x {df.shape[1]} columns -> {path}")
    return path


def version_exists(version: str) -> bool:
    return os.path.exists(_version_path(version))
