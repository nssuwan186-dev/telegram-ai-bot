FROM python:3.9-slim-buster

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-tha \
    tesseract-ocr-eng \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["uvicorn", "telegram_bot_app:app", "--host", "0.0.0.0", "--port", "8080"]