"""
run_pipeline.py
---------------
Single, reproducible entrypoint for the whole MLOps pipeline.
Supports automated data building, specific model training, and CLTV regression tracking.

Author: Machine Learning Course Project (Dr. Bahaghighat)
"""

import argparse
import sys
import pandas as pd

from src import data_loader
from src import preprocessing
from src import features
from src import train
from src import evaluate
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Define available dataset versions according to the project requirements
ALL_VERSIONS = ["v2", "v3"]


def load_data(version: str, rebuild: bool = False) -> pd.DataFrame:
    """
    Loads a dataset version from disk, building it from raw data if missing or forced.
    Ensures that for CLTV tasks on v2, the target column is dynamically recovered.
    """
    if not rebuild and data_loader.version_exists(version):
        df = data_loader.load_version(version)
        return df

    print(f"[run_pipeline] Building '{version}' from raw data...")
    df_raw = data_loader.load_raw_data()

    if version == "v2":
        df = preprocessing.build_v2(df_raw)
    elif version == "v3":
        df = features.build_v3(df_raw)
    else:
        raise ValueError(f"Unsupported version '{version}'")

    data_loader.save_version(df, version)
    return df


def handle_cltv_target(df: pd.DataFrame, version: str) -> pd.DataFrame:
    """
    Ensures the CLTV column exists in the dataset. If it's missing (like in v2/v3),
    it safely recovers it from raw v1 data by removing the same 11 rows where
    'Total Charges' was missing, perfectly aligning the sequences without needing CustomerID.
    """
    df = df.copy()
    if "CLTV" in df.columns:
        return df

    print(f"[run_pipeline] ⚠️ 'CLTV' column not found in {version}. Recovering from raw v1 data...")
    try:
        # 1. Load the untouched raw v1 dataset
        df_raw = data_loader.load_raw_data()
        
        # 2. Replicate the row filtering logic (dropping the 11 empty Total Charges rows)
        if df_raw["Total Charges"].dtype == object:
            df_raw["Total Charges"] = pd.to_numeric(df_raw["Total Charges"], errors="coerce")
        df_raw_clean = df_raw.dropna(subset=["Total Charges"])
        
        # 3. Check if we can align safely by matching lengths
        if len(df) == len(df_raw_clean):
            print(f"[run_pipeline] ✅ Length matches perfectly ({len(df)} rows). Aligning CLTV target column...")
            df["CLTV"] = df_raw_clean["CLTV"].values
            return df
        else:
            # Fallback alignment in case data shape differs
            print(f"[run_pipeline] ⚠️ Length mismatch (DF: {len(df)}, Raw Cleaned: {len(df_raw_clean)}). Using index fallback.")
            df["CLTV"] = df_raw_clean["CLTV"].reset_index(drop=True).iloc[:len(df)].values
            return df
            
    except Exception as e:
        print(f"[run_pipeline] ❌ Error recovering CLTV column inside run_pipeline: {e}")
        sys.exit(1)


def main():
    # Setup custom formatter to make the help description look clean and professional
    parser = argparse.ArgumentParser(
        description="🚀 End-to-End MLOps Pipeline for Telco Customer Churn & CLTV Prediction 🚀",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples of Usage:
------------------
1. Run all models on all available dataset versions (v2 and v3):
   python run_pipeline.py

2. Run all models ONLY on dataset version v3:
   python run_pipeline.py --version v3

3. Force rebuild v2 and v3 datasets from raw v1 Excel, then run pipeline:
   python run_pipeline.py --rebuild-data

4. Train ONLY a specific model (e.g., Random Forest) on version v3:
   python run_pipeline.py --version v3 --model "Random Forest"

5. Run the CLTV Regression Pipeline on version v3:
   python run_pipeline.py --mode cltv --version v3

6. Run the CLTV Regression Pipeline on version v2 (automatically restores CLTV column):
   python run_pipeline.py --mode cltv --version v2
        """
    )

    # Core Pipeline Arguments
    parser.add_argument(
        "--version", 
        choices=ALL_VERSIONS, 
        default=None,
        help="Specify a single dataset version to run. If not set, runs all versions sequentially."
    )
    
    parser.add_argument(
        "--rebuild-data", 
        action="store_true",
        help="Force the preprocessing and feature engineering steps to rebuild v2/v3 from raw v1."
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Filter pipeline to train ONLY a specific model. \nChoices: 'Logistic Regression', 'Random Forest', 'XGBoost', 'CatBoost', 'Gradient Boosting', 'AdaBoost', 'LightGBM'"
    )

    parser.add_argument(
        "--mode",
        choices=["churn", "cltv"],
        default="churn",
        help="Execution objective:\n'churn' -> Classification pipeline to predict customer drops (Default).\n'cltv'  -> Regression pipeline to predict Customer Lifetime Value."
    )

    args = parser.parse_args()

    # Determine which versions to loop over
    versions = [args.version] if args.version else ALL_VERSIONS
    all_results = []

    print("\n" + "=" * 60)
    print(f"Starting pipeline execution in [{args.mode.upper()}] mode.")
    print("=" * 60)

    # Execute Pipeline based on selected mode
    if args.mode == "cltv":
        # Import dynamically or safely invoke your new CLTV module logic
        try:
            from src import cltv
        except ImportError:
            print("[run_pipeline] ❌ Error: 'src.cltv' module not found. Please create 'src/cltv.py' first.")
            return

        for version in versions:
            df = load_data(version, rebuild=args.rebuild_data)
            # Address data requirements specific to CLTV
            df = handle_cltv_target(df, version)
            
            # Execute the regression pipeline
            cltv.train_cltv_regression(df, version=version)
            
    else:
        # Standard Churn Classification Mode (Includes your original logic)
        for version in versions:
            df = load_data(version, rebuild=args.rebuild_data)
            
            # Run training. Modified to pass 'args.model' directly down to train.py
            # If args.model is provided, train.py handles filtering the dictionary internally.
            result = train.train_dataset_version(version, df, target_model=args.model)
            all_results.append(result)

        # Build and print the leaderboard if we ran the full classification loop
        if all_results and not args.model:
            leaderboard = evaluate.build_leaderboard(
                [{"dataset_version": r["dataset_version"], "model": r["model"], "test_f1": r["test_f1"]}
                 for r in all_results]
            )
            print("\n" + "=" * 60)
            print("FINAL LEADERBOARD ACROSS ALL RUNS:")
            print("=" * 60)
            print(leaderboard.to_string())
            print("=" * 60 + "\n")


if __name__ == "__main__":
    main()