#!/bin/bash
# Development mode: Run backend and frontend separately

echo "🔧 Starting in DEVELOPMENT mode..."
echo ""

# Check if in app directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Must run from app/ directory"
    exit 1
fi

# Install frontend dependencies if needed
if [ ! -d "frontend/node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    cd frontend && npm install && cd ..
fi

# Start backend in background
echo "📡 Starting backend API server..."
cd backend
uv run --no-project python main.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 3

# Start frontend
echo "⚛️  Starting React dev server..."
cd frontend
npm start

# Cleanup on exit
trap "kill $BACKEND_PID 2>/dev/null" EXIT
