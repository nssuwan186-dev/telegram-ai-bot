# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run the Python script when the container launches
# Note: Cloud Run expects a web server for HTTP requests.
# For a Telegram bot, you'll typically use webhooks.
# If your bot uses long-polling, you might need a different setup or a small web server to keep the container alive.
# For now, we'll assume it's a long-polling bot that needs to run continuously.
# If you switch to webhooks, you'll need a web framework (like Flask/FastAPI) and an HTTP server.
CMD ["python", "telegram_bot_app.py"]