# Use official Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# 1. Install system dependencies and Node.js (for frontend build)
RUN apt-get update && apt-get install -y \
    nodejs \
    npm \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 3. Copy Python project files and install backend dependencies with uv
COPY app/pyproject.toml app/uv.lock ./app/
WORKDIR /app/app
RUN uv sync --frozen --no-dev

# 4. Copy and build frontend
COPY app/frontend/package*.json ./frontend/
WORKDIR /app/app/frontend
RUN npm install
COPY app/frontend/ ./
RUN npm run build

# 5. Go back to /app and copy backend code and file_database
WORKDIR /app
COPY app/backend/ ./app/backend/
COPY app/file_database/ ./app/file_database/

# 6. Expose port (Render uses PORT env var, default 8000)
ENV PORT=8000
EXPOSE 8000

# 7. Set working directory to backend folder
WORKDIR /app/app/backend

# 8. Start command using uv run with correct module path
# Since we're in /app/app/backend, we can import main:app directly
CMD ["uv", "run", "--directory", "/app/app", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
