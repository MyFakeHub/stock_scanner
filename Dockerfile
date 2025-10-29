FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY stock_scanner.py .

RUN mkdir -p /app/logs
RUN touch /tmp/scanner_running

ENV TELEGRAM_BOT_TOKEN=""
ENV TELEGRAM_CHAT_ID=""
ENV PRICE_LIMIT=10.0
ENV SCAN_INTERVAL=300
ENV PYTHONUNBUFFERED=1
ENV TZ=Europe/Berlin

HEALTHCHECK --interval=5m --timeout=10s --retries=3 --start-period=30s \
  CMD python -c "import os; exit(0 if os.path.exists('/tmp/scanner_running') else 1)"

CMD ["python", "-u", "stock_scanner.py"]
