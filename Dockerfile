# Use a slim Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies that might be needed
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- Install CPU-only PyTorch FIRST from the official index ---
# Use --extra-index-url (NOT --index-url) so pip can resolve build deps (flit_core) from PyPI.
RUN pip install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install the rest
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Pre-download the SentenceTransformer model during build ---
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copy the rest of the application code
COPY . .

# Render automatically assigns a PORT environment variable (default 10000)
# The EXPOSE instruction is for documentation purposes.
EXPOSE 10000

# Run the application using Gunicorn, binding to Render's assigned PORT.
# Using sh -c allows the shell to expand the $PORT variable at runtime.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 4 app:app"]
