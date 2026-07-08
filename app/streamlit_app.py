"""
app/streamlit_app.py
--------------------
Streamlit application for Telco Customer Churn & CLTV Prediction.
Single page: one input form, predicts both Churn and CLTV.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import yaml
import json
import joblib
from sklearn.preprocessing import LabelEncoder

from src import mlflow_utils, data_loader

st.set_page_config(page_title="Telco Churn & CLTV Prediction", page_icon="📊", layout="wide")
st.title("📊 Telco Customer Churn & CLTV Prediction")

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

RAW_INPUT_FIELDS = [
    "Gender", "Senior Citizen", "Partner", "Dependents",
    "Tenure Months", "Phone Service", "Multiple Lines",
    "Internet Service", "Online Security", "Online Backup",
    "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies",
    "Contract", "Paperless Billing", "Payment Method",
    "Monthly Charges", "Total Charges",
]

BINARY_FIELDS = ["Gender", "Senior Citizen", "Partner", "Dependents", "Phone Service", "Paperless Billing"]
MULTICAT_FIELDS = ["Multiple Lines", "Internet Service", "Online Security", "Online Backup",
                   "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies",
                   "Contract", "Payment Method"]

BINARY_OPTIONS = {
    "Gender": ["Male", "Female"],
    "Senior Citizen": ["No", "Yes"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "Phone Service": ["No", "Yes"],
    "Paperless Billing": ["No", "Yes"],
}

MULTICAT_OPTIONS = {
    "Multiple Lines": ["No", "Yes", "No phone service"],
    "Internet Service": ["DSL", "Fiber optic", "No"],
    "Online Security": ["No", "Yes", "No internet service"],
    "Online Backup": ["No", "Yes", "No internet service"],
    "Device Protection": ["No", "Yes", "No internet service"],
    "Tech Support": ["No", "Yes", "No internet service"],
    "Streaming TV": ["No", "Yes", "No internet service"],
    "Streaming Movies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "Payment Method": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
}


@st.cache_data
def load_numeric_ranges():
    try:
        df = data_loader.load_raw_data()
        if df["Total Charges"].dtype == object:
            df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
        return {
            col: {"min": float(df[col].min()), "max": float(df[col].max()), "default": float(df[col].median())}
            for col in ["Tenure Months", "Monthly Charges", "Total Charges"]
        }
    except Exception:
        return {
            "Tenure Months": {"min": 0.0, "max": 72.0, "default": 29.0},
            "Monthly Charges": {"min": 18.25, "max": 118.75, "default": 70.35},
            "Total Charges": {"min": 18.80, "max": 8684.80, "default": 1397.50},
        }


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


@st.cache_resource
def load_churn_model():
    model_path = os.path.join(MODELS_DIR, "churn_v3.joblib")
    info_path = os.path.join(MODELS_DIR, "churn_v3_info.json")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODELS_DIR, "churn_v2.joblib")
        info_path = os.path.join(MODELS_DIR, "churn_v2_info.json")
    if not os.path.exists(model_path):
        return None, None
    model = joblib.load(model_path)
    info = json.load(open(info_path)) if os.path.exists(info_path) else {}
    return model, info


@st.cache_resource
def load_cltv_model():
    model_path = os.path.join(MODELS_DIR, "cltv_class_v3.joblib")
    info_path = os.path.join(MODELS_DIR, "cltv_class_v3_info.json")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODELS_DIR, "cltv_class_v2.joblib")
        info_path = os.path.join(MODELS_DIR, "cltv_class_v2_info.json")
    if not os.path.exists(model_path):
        return None, None
    model = joblib.load(model_path)
    info = json.load(open(info_path)) if os.path.exists(info_path) else {}
    return model, info


def encode_input(values: dict) -> pd.DataFrame:
    """Convert raw user input to V3-style features for both models."""
    row = {}
    BINARY_MAP = {"Male": 1, "Female": 0, "Yes": 1, "No": 0}
    for field in BINARY_FIELDS:
        row[field] = BINARY_MAP.get(values.get(field, "No"), 0)
    for field in ["Tenure Months", "Monthly Charges", "Total Charges"]:
        row[field] = values.get(field, 0)

    LABEL_MAPS = {
        "Multiple Lines": {"No": 0, "No phone service": 1, "Yes": 2},
        "Internet Service": {"DSL": 0, "Fiber optic": 1, "No": 2},
        "Online Security": {"No": 0, "No internet service": 1, "Yes": 2},
        "Online Backup": {"No": 0, "No internet service": 1, "Yes": 2},
        "Device Protection": {"No": 0, "No internet service": 1, "Yes": 2},
        "Tech Support": {"No": 0, "No internet service": 1, "Yes": 2},
        "Streaming TV": {"No": 0, "No internet service": 1, "Yes": 2},
        "Streaming Movies": {"No": 0, "No internet service": 1, "Yes": 2},
        "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2},
        "Payment Method": {"Bank transfer (automatic)": 0, "Credit card (automatic)": 1,
                           "Electronic check": 2, "Mailed check": 3},
    }
    for field, mapping in LABEL_MAPS.items():
        row[field] = mapping.get(values.get(field, ""), 0)

    df = pd.DataFrame([row])
    # Cross features for churn model
    df["Contract_x_Internet"] = df["Contract"].astype(str) + "_" + df["Internet Service"].astype(str)
    df["Contract_x_Payment"] = df["Contract"].astype(str) + "_" + df["Payment Method"].astype(str)
    df["Charge_per_Month"] = df["Total Charges"] / (df["Tenure Months"] + 1)
    df["Monthly_x_Tenure"] = df["Monthly Charges"] * df["Tenure Months"]

    # V3 engineered features
    df["Uses_Auto_Payment"] = df["Payment Method"].isin([0, 1]).astype(int)
    df["Service_Count"] = 0
    df["Security_Service_Count"] = 0
    df["Entertainment_Service_Count"] = 0
    df["Is_New_Customer"] = (df["Tenure Months"] <= 12).astype(int)
    df["Is_Long_Term_Customer"] = (df["Tenure Months"] >= 48).astype(int)
    df["Tenure_Group"] = pd.cut(df["Tenure Months"], bins=[0, 12, 24, 48, 72], labels=[0, 1, 2, 3], include_lowest=True).astype(int)
    df["Monthly_Spending_Level"] = pd.cut(df["Monthly Charges"], bins=[0, 35, 55, 80, 200], labels=[0, 1, 2, 3], include_lowest=True).astype(int)
    df["MonthToMonth_HighCharge"] = ((df["Contract"] == 0) & (df["Monthly Charges"] > 65)).astype(int)
    df["Total_to_Tenure"] = df["Total Charges"] / (df["Tenure Months"] + 1)
    df["Charge_Ratio"] = df["Monthly Charges"] / (df["Total_to_Tenure"] + 1)
    df["Monthly_vs_Avg"] = df["Monthly Charges"] / (df["Total_to_Tenure"] + 1)
    df["High_Risk_Contract"] = ((df["Contract"] == 0) & (df["Internet Service"] == 1) & (df["Monthly Charges"] > 65)).astype(int)
    df["Low_Engagement"] = 1
    df["Digital_Services_Count"] = 0
    df["Has_Premium_Internet"] = (df["Internet Service"] == 1).astype(int)
    df["Contract_Fiber"] = ((df["Contract"] == 0) & (df["Internet Service"] == 1)).astype(int)
    df["Contract_DSL"] = ((df["Contract"] == 0) & (df["Internet Service"] == 0)).astype(int)
    df["ElectronicCheck_HighCharge"] = ((df["Payment Method"] == 2) & (df["Monthly Charges"] > 65)).astype(int)
    df["Senior_MonthToMonth"] = ((df["Senior Citizen"] == 1) & (df["Contract"] == 0)).astype(int)
    df["Single_No_Dependents"] = ((df["Partner"] == 0) & (df["Dependents"] == 0)).astype(int)
    df["NewCustomer_NoSecurity"] = ((df["Tenure Months"] <= 12) & (df["Online Security"] == 0)).astype(int)
    df["Fiber_NoTechSupport"] = ((df["Internet Service"] == 1) & (df["Tech Support"] == 0)).astype(int)
    df["Charge_per_Service"] = df["Monthly Charges"] / 2
    df["LongTenure_AutoPayment"] = ((df["Tenure Months"] >= 24) & df["Uses_Auto_Payment"]).astype(int)
    df["Geo_Region"] = 0
    df["City_Customer_Density"] = 100

    return df


def render_input_form():
    values = {}
    ranges = load_numeric_ranges()
    st.sidebar.header("Customer Information")
    for field in RAW_INPUT_FIELDS:
        if field in BINARY_FIELDS:
            values[field] = st.sidebar.selectbox(field, BINARY_OPTIONS[field], key=f"input_{field}")
        elif field in MULTICAT_FIELDS:
            values[field] = st.sidebar.selectbox(field, MULTICAT_OPTIONS[field], key=f"input_{field}")
        elif field in ranges:
            r = ranges[field]
            values[field] = st.sidebar.slider(field, r["min"], r["max"], r["default"],
                                               step=round((r["max"] - r["min"]) / 100, 2), key=f"input_{field}")
    return values


def main():
    churn_model, churn_info = load_churn_model()
    cltv_model, cltv_info = load_cltv_model()

    if churn_model is None and cltv_model is None:
        st.error("No models available. Run `python run_pipeline.py` first.")
        return

    # Sidebar: single input form
    values = render_input_form()

    # Main area: model info + predict
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Model")
        if churn_model:
            test = churn_info.get("test", {})
            c1, c2, c3 = st.columns(3)
            c1.metric("Model", churn_info.get("model_name", "Unknown"))
            c2.metric("Test F1", f"{test.get('f1', 0):.4f}")
            c3.metric("Test Accuracy", f"{test.get('accuracy', 0):.4f}")
        else:
            st.warning("Churn model not available.")

    with col2:
        st.subheader("CLTV Model")
        if cltv_model:
            test = cltv_info.get("test", {})
            c1, c2, c3 = st.columns(3)
            c1.metric("Model", cltv_info.get("model_name", "Unknown"))
            c2.metric("Test Accuracy", f"{test.get('accuracy', 0):.4f}")
            c3.metric("Test F1", f"{test.get('f1', 0):.4f}")
        else:
            st.warning("CLTV model not available.")

    if st.button("🔍 Predict", key="predict_btn"):
        input_df = encode_input(values)

        # Churn prediction
        if churn_model:
            st.markdown("---")
            st.subheader("🔴 Churn Prediction")
            try:
                prob = churn_model.predict_proba(input_df)[:, 1]
                thr = churn_info.get("best_threshold", 0.5)
                pred_val = 1 if prob[0] >= thr else 0
                c1, c2, c3 = st.columns(3)
                c1.metric("Prediction", f"{'🔴 Will Churn' if pred_val == 1 else '🟢 Will Stay'}")
                c2.metric("Churn Probability", f"{prob[0]:.2%}")
                c3.metric("Risk Level", "High" if prob[0] > 0.5 else "Medium" if prob[0] > 0.3 else "Low")
                st.progress(prob[0])
            except Exception as e:
                st.error(f"Churn prediction failed: {e}")

        # CLTV prediction
        if cltv_model:
            st.markdown("---")
            st.subheader("💰 CLTV Prediction — Customer Value Classification")
            st.markdown("Predict if customer is **Low Value** (< $4,000) or **High Value** (>= $4,000).")
            try:
                cltv_pred = cltv_model.predict(input_df)[0]
                cltv_proba = cltv_model.predict_proba(input_df)[0]
                cltv_label = "High Value" if cltv_pred == 1 else "Low Value"
                cltv_conf = cltv_proba[cltv_pred]

                c1, c2 = st.columns(2)
                c1.metric("Customer Value", f"{'🟢 High Value' if cltv_pred == 1 else '🔵 Low Value'}")
                c2.metric("Confidence", f"{cltv_conf:.2%}")
                st.progress(cltv_conf)
                st.markdown("**Definition:** High Value = CLTV >= $4,000 | Low Value = CLTV < $4,000")
            except Exception as e:
                st.error(f"CLTV prediction failed: {e}")


main()

st.markdown("---")
st.markdown("*Telco Customer Churn Prediction — MLOps Pipeline*")
