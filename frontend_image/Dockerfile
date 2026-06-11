FROM python:3.10-slim

# Prevent Python buffering
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_CONFIG_DIR=/app/.streamlit

# Set working directory
WORKDIR /app

RUN pip install --upgrade pip

# Copy requirements first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Cloud Run port
EXPOSE 8080

# Start Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
