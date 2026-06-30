#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install Playwright browsers and system dependencies in production
playwright install chromium --with-deps

# Start the Flask app using gunicorn (install gunicorn if it's not in requirements.txt)
python3 app.py
