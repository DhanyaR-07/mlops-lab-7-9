FROM python:3.11-slim
WORKDIR /app
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt
COPY src/ ./src/
COPY models/ ./models/
ENV MODEL_PATH=/app/models/model.joblib
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
