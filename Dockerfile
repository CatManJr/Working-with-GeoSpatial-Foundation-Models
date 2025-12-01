# Use official Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# 1. Install system dependencies and Node.js (for frontend build)
RUN apt-get update && apt-get install -y nodejs npm && rm -rf /var/lib/apt/lists/*

# 2. Copy and install backend dependencies
COPY app/backend/requirements.txt ./app/backend/
RUN pip install --no-cache-dir -r app/backend/requirements.txt

# 3. Copy and build frontend
COPY app/frontend/package*.json ./app/frontend/
WORKDIR /app/app/frontend
RUN npm install
COPY app/frontend/ ./
RUN npm run build

# 4. Copy remaining code (including backend code and file_database)
WORKDIR /app
COPY . .

# 5. Expose port (Render uses PORT env var, default 8000)
ENV PORT=8000
EXPOSE 8000

# 6. Start command
# Note: We run the backend, which now also serves the frontend
CMD ["uvicorn", "app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
