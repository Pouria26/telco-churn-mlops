FROM python:3.11-slim

WORKDIR /app

# System deps needed by catboost/xgboost wheels
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bring in the code, the already-trained MLflow store, and the data folders
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/
COPY config.yaml ./
COPY run_pipeline.py serve.py ./
COPY data/ ./data/

EXPOSE 8000
EXPOSE 8501

# FastAPI on 8000, Streamlit on 8501
# Default: Streamlit UI
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
