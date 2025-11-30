#!/bin/bash
# Production mode: Build frontend and serve via FastAPI

echo "🚀 Starting in PRODUCTION mode..."
echo ""

# Check if in app directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Must run from app/ directory"
    exit 1
fi

# Install backend dependencies
echo "📦 Installing backend dependencies..."
uv sync

# Install frontend dependencies and build
echo "📦 Installing frontend dependencies..."
cd frontend
npm install

echo "🔨 Building frontend..."
npm run build

if [ ! -d "build" ]; then
    echo "❌ Frontend build failed!"
    exit 1
fi

cd ..

# Start server
echo "🚀 Starting production server..."
echo "📍 Dashboard will be available at: http://localhost:8000"
echo ""
cd backend
uv run --no-project python main.py
