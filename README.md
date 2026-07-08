# Telco Customer Churn & CLTV Prediction

> End-to-end MLOps pipeline for predicting customer churn and classifying customer lifetime value (CLTV) using machine learning, with a Streamlit web app and Docker deployment.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-24.0-2496ED?logo=docker&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.10-0194E2)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E)

---

## Demo

https://github.com/user-attachments/assets/6420a19e-64c9-4eb1-bb47-26a888b56c30

Alternative: If the video above doesn't render, you can also view it directly:

[Watch Demo Video](assets/demo.mp4)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
- [Docker Deployment](#docker-deployment)
- [Model Performance](#model-performance)
- [How It Works](#how-it-works)
- [Screenshots](#screenshots)
- [License](#license)

---

## Overview

This project addresses two critical business problems in the telecom industry:

1. **Customer Churn Prediction** — Will this customer leave?
2. **CLTV Classification** — Is this customer High Value (>= $4,000) or Low Value (< $4,000)?

The system trains 8 machine learning models, selects the best one using F1-score optimization, and deploys a interactive web application where users can input customer information and get instant predictions for both tasks.

---

## Features

- **8 ML Models** — Logistic Regression, Random Forest, XGBoost, CatBoost, Gradient Boosting, AdaBoost, LightGBM, and Voting Ensemble
- **20+ Engineered Features** — Service counts, loyalty indicators, risk factors, charge ratios, and interaction features
- **Threshold Optimization** — Custom function that balances Precision and Recall for optimal F1-score
- **10-Fold Cross Validation** — Robust model evaluation with stratified splits
- **MLflow Tracking** — Experiment logging, metrics, and model versioning
- **Streamlit Web App** — Single-page UI with one form, predicts both Churn and CLTV simultaneously
- **Docker Deployment** — Containerized app for easy sharing and reproducibility
- **Pipeline Automation** — One command to train all models on all dataset versions

---

## Project Structure

```
telco-churn-docker-app/
├── app/
│   └── streamlit_app.py          # Streamlit web application
├── src/
│   ├── train.py                  # Churn classification pipeline
│   ├── cltv_class.py             # CLTV classification pipeline
│   ├── preprocessing.py          # V2 dataset builder
│   ├── features.py               # V3 dataset builder (engineered features)
│   ├── data_loader.py            # Data loading utilities
│   ├── evaluate.py               # Metrics and visualization
│   └── mlflow_utils.py           # MLflow tracking helpers
├── models/                       # Trained model files (.joblib)
├── data/
│   ├── v1/                       # Raw data
│   ├── v2/                       # Cleaned data
│   └── v3/                       # Cleaned + engineered features
├── assets/                       # Images and videos for README
├── config.yaml                   # Hyperparameters and settings
├── run_pipeline.py               # Main pipeline entry point
├── Dockerfile                    # Docker configuration
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Technologies

| Category | Technology | Purpose |
|----------|------------|---------|
| Language | Python 3.11 | Core programming |
| ML Framework | scikit-learn | Model training and evaluation |
| Gradient Boosting | XGBoost, CatBoost, LightGBM | Advanced ensemble models |
| Experiment Tracking | MLflow | Metrics logging and model versioning |
| Web App | Streamlit | Interactive user interface |
| Containerization | Docker | Deployment and sharing |
| Data Processing | pandas, NumPy | Data manipulation |
| Visualization | matplotlib, seaborn | Charts and confusion matrices |
| Data Storage | openpyxl | Excel file handling |

---

## Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager
- Docker (optional, for containerized deployment)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/telco-churn-docker-app.git
   cd telco-churn-docker-app
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Train Churn Models

Train all 8 models on both V2 and V3 datasets:

```bash
python run_pipeline.py
```

Train on a specific dataset version:

```bash
python run_pipeline.py --version v3
```

Train a specific model:

```bash
python run_pipeline.py --model "Random Forest"
```

### 2. Train CLTV Classification Models

Train CLTV classifiers (Low/High Value):

```bash
python run_pipeline.py --mode cltv_class
```

Train on specific version:

```bash
python run_pipeline.py --mode cltv_class --version v3
```

### 3. Run the Web App

```bash
streamlit run app/streamlit_app.py
```

Open your browser and go to `http://localhost:8501`

### 4. Full Pipeline Commands

| Command | Description |
|---------|-------------|
| `python run_pipeline.py` | Train all churn models on v2 and v3 |
| `python run_pipeline.py --version v3` | Train churn models on v3 only |
| `python run_pipeline.py --mode cltv_class` | Train CLTV classifiers |
| `python run_pipeline.py --rebuild-data` | Rebuild datasets from raw data |
| `python run_pipeline.py --model "XGBoost"` | Train only XGBoost |
| `streamlit run app/streamlit_app.py` | Launch web app |

---

## Docker Deployment

### Build the Image

```bash
docker build -t telco-churn-app .
```

### Run the Container

```bash
docker run -p 8501:8501 telco-churn-app
```

Open `http://localhost:8501` in your browser.

### Save and Share

Save the image to share with others:

```bash
# Save image to file
docker save telco-churn-app -o telco-churn-app.tar

# Others can load and run it
docker load -i telco-churn-app.tar
docker run -p 8501:8501 telco-churn-app
```

---

## Model Performance

### Churn Prediction (V3 Dataset)

| Metric | Train | Test |
|--------|-------|------|
| Accuracy | 0.799 | 0.770 |
| Precision | 0.599 | 0.549 |
| Recall | 0.737 | 0.739 |
| F1 Score | 0.661 | 0.630 |
| ROC AUC | 0.869 | 0.849 |

### CLTV Classification (V3 Dataset)

| Metric | Train | Test |
|--------|-------|------|
| Accuracy | 0.680 | 0.677 |
| Precision | 0.761 | 0.760 |
| Recall | 0.755 | 0.750 |
| F1 Score | 0.758 | 0.755 |


## How It Works

### Data Pipeline

```
Raw Data (v1) ──► Preprocessing ──► V2 Dataset (cleaned)
                    │
                    └──► Feature Engineering ──► V3 Dataset (cleaned + 20 features)
```

### Training Pipeline

```
Dataset (V2/V3) ──► Train/Test Split (85%/15%)
    │
    ├──► 10-Fold Cross Validation on 8 models
    │
    ├──► Select Best Model (F1-score)
    │
    ├──► Optimize Threshold (Precision-Recall balance)
    │
    ├──► Fit Final Model
    │
    └──► Save Model + Info JSON
```

### Prediction Pipeline

```
User Input ──► Feature Encoding ──► V3 Features
    │
    ├──► Churn Model ──► Churn Probability + Risk Level
    │
    └──► CLTV Model ──► High/Low Value + Confidence
```


## Configuration

Edit `config.yaml` to customize:

```yaml
globals:
  random_seed: 42
  target_col: "Churn Value"
  cltv_target_col: "CLTV"

models:
  random_forest:
    n_estimators: 1000
    max_depth: 10
  xgboost:
    n_estimators: 800
    learning_rate: 0.03
```

---

## Author

**Pouria Rostami**
- GitHub: [@Pouria26](https://github.com/Pouria26)

---

## Acknowledgments

- Dr. Mehdi Bahaghighat (Supervisor)
- Imam Khomeini International University, Qazvin
- IBM Telco Customer Churn Dataset

---

## License

This project is for educational purposes as a final course project.

---

## Support

If you found this project helpful, please give it a star on GitHub!

---

> Built with passion for machine learning and MLOps 🚀 — From data to deployment in one pipeline 🎯
> If you found this project helpful, please consider giving it a ⭐️
