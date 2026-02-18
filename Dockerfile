# Use an official Python runtime as a parent image (pinned to bookworm for stability)
# Upgraded to 3.11 for better long-term support
FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Disable Paddle's network check which can hang or spam logs
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=1

# Set work directory
WORKDIR /app

# Install system dependencies
# ffmpeg: for faster-whisper
# libgomp1, libgl1, libglib2.0-0: for opencv/paddle (updated for Debian 12)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . /app/

# Command to run the bot
CMD ["python", "bot.py"]
