FROM python:3.11-slim

WORKDIR /app

# Install system deps for common Python packages
RUN apt-get update && apt-get install -y build-essential git curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy only what's needed
COPY backend /app/backend
COPY KG /app/KG
COPY Graph /app/Graph
COPY streamlit_app.py /app/streamlit_app.py

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
