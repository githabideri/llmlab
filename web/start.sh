#!/bin/bash
# Start the LLMlab Benchmark Web UI

# Change to the web directory
cd "$(dirname "$0")"

# Activate the virtual environment
source venv/bin/activate

# Set defaults if not configured
export LLAMA_SERVER_URL="${LLAMA_SERVER_URL:-http://192.168.0.27:8080}"
export PORT="${PORT:-8000}"
export HOST="${HOST:-0.0.0.0}"

# Start the server
echo "Starting LLMlab Benchmark UI..."
echo "  Server URL: $LLAMA_SERVER_URL"
echo "  Listening on: http://$HOST:$PORT"
echo ""
echo "Press Ctrl+C to stop"
echo ""

exec python app.py
