FROM python:3.10-slim

WORKDIR /app

# Copy FastAPI app
COPY app/app.py .

# Copy model
COPY models ./models

# Copy dependencies
COPY requirements.txt ./requirements.txt

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose port and start server
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
