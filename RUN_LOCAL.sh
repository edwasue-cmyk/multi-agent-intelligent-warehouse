#!/bin/bash
set -euo pipefail

# Warehouse Operational Assistant - Local API Runner
# Automatically finds a free port and starts the FastAPI application

echo "🚀 Starting Warehouse Operational Assistant API..."

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Use port 8002 for consistency
PORT=${PORT:-8002}

echo "📡 Starting API on port $PORT"
echo "🌐 API will be available at: http://localhost:$PORT"
echo "📚 API documentation: http://localhost:$PORT/docs"
echo "🔍 OpenAPI schema: http://localhost:$PORT/openapi.json"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the FastAPI application
uvicorn chain_server.app:app --host 0.0.0.0 --port $PORT --reload