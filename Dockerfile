# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    LOG_FILE=/tmp/app.log \
    DISABLE_FILE_LOGGING=false

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # For document processing
    tesseract-ocr \
    tesseract-ocr-eng \
    # For image processing
    libgl1-mesa-glx \
    libglib2.0-0 \
    # For compilation
    gcc \
    g++ \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create non-root user for security first
RUN groupadd -r ragapp && useradd -r -g ragapp ragapp

# Copy application code
COPY . .

# Make start script executable
RUN chmod +x start.sh

# Create directories for logs and temporary files with proper ownership
RUN mkdir -p logs temp downloads cache && \
    chown -R ragapp:ragapp /app && \
    chmod -R 755 /app && \
    chmod -R 777 logs temp downloads cache

USER ragapp

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application
CMD ["./start.sh"]
