#!/bin/bash
# Start backend only for testing

echo "Starting backend API server..."
echo ""

cd "$(dirname "$0")"

# Set UV link mode
export UV_LINK_MODE=copy

# Start backend
cd backend
echo "Backend will be available at: http://localhost:8000"
echo "API docs: http://localhost:8000/docs"
echo ""
uv run --no-project python main.py