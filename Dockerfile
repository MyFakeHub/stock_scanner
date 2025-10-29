FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY stock_scanner.py .

# Create logs directory
RUN mkdir -p /app/logs

# Create health check file
RUN touch /tmp/scanner_running

# Set environment variables with defaults
ENV TELEGRAM_BOT_TOKEN=""
ENV TELEGRAM_CHAT_ID=""
ENV PRICE_LIMIT=10.0
ENV SCAN_INTERVAL=300
ENV PYTHONUNBUFFERED=1

# Run the scanner
CMD ["python", "-u", "stock_scanner.py"]
