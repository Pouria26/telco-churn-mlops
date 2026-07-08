"""
run_pipeline.py
---------------
Single, reproducible entrypoint for the whole MLOps pipeline.
Supports Churn classification and CLTV value classification (Low/High).

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


def main():
    parser = argparse.ArgumentParser(
        description="End-to-End MLOps Pipeline for Telco Customer Churn & CLTV Prediction",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples of Usage:
------------------
1. Run Churn classification on all dataset versions (v2 and v3):
   python run_pipeline.py

2. Run Churn classification ONLY on v3:
   python run_pipeline.py --version v3

3. Force rebuild v2/v3 datasets from raw Excel, then run:
   python run_pipeline.py --rebuild-data

4. Train ONLY a specific model (e.g., Random Forest) on v3:
   python run_pipeline.py --version v3 --model "Random Forest"

5. Run CLTV classification (Low/High Value) on all versions:
   python run_pipeline.py --mode cltv_class

6. Run CLTV classification on v3 only:
   python run_pipeline.py --mode cltv_class --version v3

7. Run CLTV classification with a specific model:
   python run_pipeline.py --mode cltv_class --model "XGBoost"
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
        help="Train ONLY a specific model.\n"
             "Choices: 'Logistic Regression', 'Random Forest', 'XGBoost', 'CatBoost',\n"
             "         'Gradient Boosting', 'AdaBoost', 'LightGBM', 'Voting Ensemble'"
    )

    parser.add_argument(
        "--mode",
        choices=["churn", "cltv_class"],
        default="churn",
        help="Execution mode (default: churn):\n"
             "  churn      -> Churn classification (predict if customer will leave)\n"
             "  cltv_class -> CLTV classification (predict Low/High value customer)"
    )

    args = parser.parse_args()

    # Determine which versions to loop over
    versions = [args.version] if args.version else ALL_VERSIONS
    all_results = []

    print("\n" + "=" * 60)
    print(f"Starting pipeline execution in [{args.mode.upper()}] mode.")
    print("=" * 60)

    # Execute Pipeline based on selected mode
    if args.mode == "cltv_class":
        try:
            from src import cltv_class
        except ImportError:
            print("[run_pipeline] Error: 'src.cltv_class' module not found.")
            return
        for version in versions:
            df = load_data(version, rebuild=args.rebuild_data)
            cltv_class.train_dataset_version(version, df, target_model=args.model)

    else:
        # Standard Churn Classification Mode
        for version in versions:
            df = load_data(version, rebuild=args.rebuild_data)
            result = train.train_dataset_version(version, df, target_model=args.model)
            all_results.append(result)

        # Build and print the leaderboard if we ran the full classification loop
        if all_results and not args.model:
            rows = []
            for r in all_results:
                rows.append({
                    "dataset_version": r.get("dataset_version", "?"),
                    "model": r.get("model", "?"),
                    "cv_f1": r.get("cv_f1", 0),
                    "cv_thr_f1": r.get("cv_thr_f1", 0),
                    "test_f1": r.get("test_f1", 0),
                    "test_acc": r.get("test_accuracy", 0),
                })
            import pandas as pd
            leaderboard = pd.DataFrame(rows)
            print("\n" + "=" * 70)
            print("FINAL LEADERBOARD")
            print("=" * 70)
            print(leaderboard.to_string(index=False))
            print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
