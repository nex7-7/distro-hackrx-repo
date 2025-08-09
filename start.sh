#!/bin/bash
set -e

# Run initialization
python init_container.py

# Start the application
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
