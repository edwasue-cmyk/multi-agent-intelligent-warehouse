#!/bin/bash
set -e

echo "🔧 Setting up development environment..."

# Ensure we're in the workspace directory
cd /workspace

# Install Python dependencies if not already installed
if [ -f "requirements.txt" ]; then
    echo "📦 Installing Python dependencies..."
    pip install -q -r requirements.txt
    pip install -q -r requirements.docker.txt
fi

# Install frontend dependencies if they exist
if [ -d "src/ui/web" ] && [ -f "src/ui/web/package.json" ]; then
    echo "📦 Installing Node.js dependencies..."
    cd src/ui/web
    npm install --silent
    cd /workspace
fi

# Create .env file from .env.example if it doesn't exist
if [ -f ".env.example" ] && [ ! -f ".env" ]; then
    echo "📝 Creating .env file from .env.example..."
    cp .env.example .env
    echo "⚠️  Please update .env with your actual API keys and configuration"
fi

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
until pg_isready -h localhost -p 5432 -U warehouse_user > /dev/null 2>&1; do
    echo "  Waiting for PostgreSQL..."
    sleep 2
done
echo "✅ PostgreSQL is ready"

until redis-cli -h localhost ping > /dev/null 2>&1; do
    echo "  Waiting for Redis..."
    sleep 2
done
echo "✅ Redis is ready"

# Note: Milvus can take longer to start, check if port is open
timeout 30 bash -c 'until nc -z localhost 19530; do sleep 2; done' && echo "✅ Milvus is ready" || echo "⚠️  Milvus may not be ready yet"

# Run database migrations or setup if needed
echo "🗄️  Setting up database..."
# Add your database initialization commands here if needed
# python -m src.api.db.init_db

echo ""
echo "✅ Development environment ready!"
echo ""
echo "📚 Quick Start Commands:"
echo "  - Start backend:  python -m uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000"
echo "  - Start frontend: cd src/ui/web && npm start"
echo "  - Run tests:      pytest tests/"
echo "  - Format code:    black src/ tests/"
echo "  - Check types:    mypy src/"
echo ""
echo "🔗 Service URLs (once running):"
echo "  - Backend API:    http://localhost:8000"
echo "  - API Docs:       http://localhost:8000/docs"
echo "  - Frontend:       http://localhost:3000"
echo "  - PostgreSQL:     localhost:5432"
echo "  - Milvus:         localhost:19530"
echo "  - Redis:          localhost:6379"
echo ""
