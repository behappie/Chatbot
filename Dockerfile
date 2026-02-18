# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
# ffmpeg: for faster-whisper
# libgomp1, libgl1-mesa-glx, libglib2.0-0: for opencv/paddle
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download PaddleOCR model cache during build (optional/advanced, but good for speed)
# For now, we let it download on first run to keep Dockerfile simple.

# Copy the rest of the application
COPY . /app/

# Command to run the bot
CMD ["python", "bot.py"]
