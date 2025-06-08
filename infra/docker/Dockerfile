FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/api /app/api
COPY src/model /app/model

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8000
ENV MODEL_URI="models:/fare_recommendation_model/latest"

# Expose port
EXPOSE 8000

# Set working directory to API folder
WORKDIR /app/api

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]