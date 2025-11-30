#!/bin/bash
# Development mode: Run backend and frontend separately

echo "🔧 Starting in DEVELOPMENT mode..."
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if in app directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Must run from app/ directory"
    exit 1
fi

# Fix npm permissions if needed
if [ ! -w "$HOME/.npm" ]; then
    echo "🔧 Fixing npm permissions..."
    sudo chown -R $(whoami) "$HOME/.npm"
fi

# Install frontend dependencies if needed
if [ ! -d "frontend/node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    (cd frontend && npm install)
fi

# Install backend dependencies
echo "📦 Syncing backend dependencies..."
uv sync

# Start backend in background
echo "📡 Starting backend API server..."
(cd backend && uv run --no-project python main.py) &
BACKEND_PID=$!

# Wait for backend to start
echo "⏳ Waiting for backend to start..."
sleep 3

# Check if backend is running#!/bin/bash
# Development mode: Run backend and frontend separately

echo "🔧 Starting in DEVELOPMENT mode..."
echo ""

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if in app directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Must run from app/ directory"
    exit 1
fi

# Set UV link mode for ExFAT compatibility
export UV_LINK_MODE=copy

# Install frontend dependencies if needed
if [ ! -d "frontend/node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    (cd frontend && npm install)
    if [ $? -ne 0 ]; then
        echo "❌ npm install failed. Try running: zsh setup.sh"
        exit 1
    fi
fi

# Install backend dependencies
echo "📦 Syncing backend dependencies..."
uv sync

if [ $? -ne 0 ]; then
    echo "❌ uv sync failed"
    exit 1
fi

# Start backend in background
echo "📡 Starting backend API server..."
(cd backend && uv run --no-project python main.py) &
BACKEND_PID=$!

# Wait for backend to start
echo "⏳ Waiting for backend to start..."
sleep 3

# Check if backend is running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ Backend failed to start!"
    exit 1
fi

echo "✅ Backend running on http://localhost:8000"
echo ""

# Start frontend
echo "⚛️  Starting React dev server..."
cd frontend
npm start

# Cleanup on exit
trap "echo ''; echo '🛑 Shutting down...'; kill $BACKEND_PID 2>/dev/null" EXIT

if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ Backend failed to start!"
    exit 1
fi

echo "✅ Backend running on http://localhost:8000"
echo ""

# Start frontend
echo "⚛️  Starting React dev server..."
cd frontend
npm start

# Cleanup on exit
trap "echo ''; echo '🛑 Shutting down...'; kill $BACKEND_PID 2>/dev/null" EXIT
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
