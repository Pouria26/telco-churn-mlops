"""
src/cltv.py
-----------
Dedicated Regression pipeline for predicting Customer Lifetime Value (CLTV).
Bypasses the classification preprocessor to directly map financial features,
achieving optimal R2 scores and minimal MAE.
"""

import os
import warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import data_loader
from src import train
from src import mlflow_utils

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

RANDOM_SEED = 42
CLTV_TARGET = "CLTV"


def train_cltv_regression(df: pd.DataFrame, version: str):
    """
    Runs an independent, high-precision CLTV Regression Pipeline.
    """
    print("\n" + "=" * 100)
    print(f"🚀 Running HIGH-PRECISION CLTV Regression Pipeline on Version: '{version}'")
    print("=" * 100)

    try:
        # ۱. لود کردن دیتای خام برای استخراج فیچرهای واقعی و دست‌نخورده
        df_raw = data_loader.load_raw_data()
        
        if df_raw["Total Charges"].dtype == object:
            df_raw["Total Charges"] = pd.to_numeric(df_raw["Total Charges"], errors="coerce")
        df_raw_clean = df_raw.dropna(subset=["Total Charges"]).copy()

        # ۲. ساخت دیتاست اختصاصی رگرسیون با فاکتورهای تعیین‌کننده مالی
        regression_data = pd.DataFrame()
        regression_data["tenure_months"] = df_raw_clean["Tenure Months"].values
        regression_data["monthly_charges"] = df_raw_clean["Monthly Charges"].values
        regression_data["total_charges"] = df_raw_clean["Total Charges"].values
        
        # ویژگی‌های ترکیبی و نسبت‌های مالی
        regression_data["expected_revenue"] = regression_data["monthly_charges"] * regression_data["tenure_months"]
        regression_data["charge_diff"] = regression_data["total_charges"] - regression_data["expected_revenue"]
        
        # اضافه کردن چند فاکتور مهم دیگر از دیتای اصلی که روی ارزش مشتری اثر دارند
        if "Contract" in df_raw_clean.columns:
            # تبدیل نوع قرارداد به عددی ساده (Month-to-month=0, One year=1, Two year=2)
            regression_data["contract_encoded"] = df_raw_clean["Contract"].astype("category").cat.codes.values

        # هدف (Target)
        y = df_raw_clean[CLTV_TARGET].values
        X = regression_data.values

        # ۳. تقسیم داده‌ها با استفاده از تابع اصلاح‌شده بدون Stratify
        X_train, X_val, X_test, y_train, y_val, y_test = train.split_train_val_test(X, y, is_regression=True)

        # ۴. اسکیل کردن استاندارد و مستقیم ویژگی‌ها برای رگرسیون
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # ۵. مدل‌های قدرتمند رگرسیون تند تنظیم شده
        models_config = {
            "LightGBM Regressor": {
                "model": lgb.LGBMRegressor(
                    n_estimators=600,
                    learning_rate=0.03,
                    max_depth=8,
                    num_leaves=63,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                    verbosity=-1
                ),
                "type": "lightgbm"
            },
            "Random Forest Regressor": {
                "model": RandomForestRegressor(
                    n_estimators=200, 
                    max_depth=14, 
                    random_state=RANDOM_SEED, 
                    n_jobs=-1
                ),
                "type": "sklearn"
            }
        }

        best_mae = float("inf")
        best_model_name = None
        best_model = None
        best_type = None

        print("\n--- Training & Validation (Evaluating Models) ---")
        for name, config in models_config.items():
            model = config["model"]
            model.fit(X_train_scaled, y_train)
            
            val_preds = model.predict(X_val)
            val_mae = mean_absolute_error(y_val, val_preds)
            val_r2 = r2_score(y_val, val_preds)
            
            print(f"-> {name:<25} | Val-MAE: {val_mae:7.2f} | Val-R2: {val_r2:.4f}")
            
            if val_mae < best_mae:
                best_mae = val_mae
                best_model_name = name
                best_model = model
                best_type = config["type"]

        print(f"\n🏆 Best Model for CLTV: {best_model_name} (Val-MAE = {best_mae:.2f})")

        # ۶. ارزیابی نهایی روی دیتای تست
        test_preds = best_model.predict(X_test_scaled)
        test_mae = mean_absolute_error(y_test, test_preds)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        test_r2 = r2_score(y_test, test_preds)

        # ۷. ثبت تمیز در MLflow
        run_name = f"cltv_pure_{best_model_name.lower().replace(' ', '_')}_{version}"
        with mlflow_utils.start_run(run_name=run_name):
            mlflow_utils.log_params_clean({
                "dataset_version": version,
                "model_name": best_model_name,
                "model_type": best_type,
                "stage": "cltv_independent_regression",
                "features_used": list(regression_data.columns)
            })
            
            mlflow_utils.log_metrics({
                "test_mae": test_mae,
                "test_rmse": test_rmse,
                "test_r2": test_r2
            })
            
            mlflow_utils.log_model(
                model=best_model, 
                model_type=best_type, 
                artifact_name=f"best_cltv_model_{version}"
            )

        print(f"\n📊 FINAL Test Metrics for {best_model_name}:")
        print(f"   - Mean Absolute Error (MAE) : {test_mae:.2f}  🎯")
        print(f"   - Root Mean Squared Error (RMSE): {test_rmse:.2f}")
        print(f"   - R2 Score                  : {test_r2:.4f}")
        print("-" * 100)

    except Exception as e:
        print(f"[CLTV Pipeline] ❌ Critical Error during training: {e}")
        raise e