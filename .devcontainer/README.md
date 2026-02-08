# Dev Container Configuration

This directory contains the development container configuration for the Multi-Agent Intelligent Warehouse project.

## What is a Dev Container?

A dev container is a fully-featured development environment that runs inside a Docker container. It provides:
- Consistent development environment across all team members
- Pre-configured tools and extensions
- Isolated services (PostgreSQL, Milvus, Redis)
- GitHub Copilot integration out of the box

## Quick Start

### Prerequisites

1. **Visual Studio Code** with the **Dev Containers** extension installed
2. **Docker Desktop** running on your machine
3. **GitHub Copilot** subscription (for AI assistance)

### Opening the Dev Container

1. Open this repository in VS Code
2. When prompted, click **"Reopen in Container"** 
   - Or use Command Palette (Ctrl+Shift+P / Cmd+Shift+P) → **"Dev Containers: Reopen in Container"**
3. Wait for the container to build (first time may take 5-10 minutes)
4. The environment will be fully configured with all dependencies installed

### What Gets Installed Automatically

**System Tools:**
- Python 3.11
- Node.js 18
- Git, Docker CLI
- PostgreSQL client
- Build tools

**Python Packages:**
- All dependencies from `requirements.txt` and `requirements.docker.txt`
- Development tools: black, flake8, mypy, pytest, ipython, jupyter

**VS Code Extensions:**
- GitHub Copilot & Copilot Chat
- Python support (Pylance, Black, isort, mypy)
- TypeScript/JavaScript support
- Docker tools
- Database tools (SQLTools)
- Git tools (GitLens)
- And more...

**Services:**
- PostgreSQL 14 (port 5432)
- Milvus v2.3.0 (port 19530)
- Redis 7 (port 6379)

## Configuration Files

- **`devcontainer.json`** - Main configuration file
- **`Dockerfile`** - Custom dev container image
- **`docker-compose.yml`** - Multi-service setup (app, database, Milvus, Redis)
- **`post-create.sh`** - Setup script that runs after container creation

## Using the Dev Container

### Starting the Backend

```bash
python -m uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

Access at: http://localhost:8000
API Docs: http://localhost:8000/docs

### Starting the Frontend

```bash
cd src/ui/web
npm start
```

Access at: http://localhost:3000

### Running Tests

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/unit/test_equipment.py

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Formatting

```bash
# Format Python code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/
```

### Database Access

```bash
# Connect to PostgreSQL
psql postgresql://warehouse_user:warehouse_pass@localhost:5432/warehouse_dev

# Check Redis
redis-cli -h localhost ping
```

## GitHub Copilot in Dev Container

GitHub Copilot is automatically installed and configured in the dev container. It has access to:

1. **Project Context** - `.github/copilot-instructions.md` provides codebase patterns
2. **All Extensions** - Language support for Python and TypeScript
3. **Workspace Settings** - Pre-configured formatters and linters

### Using Copilot Effectively

- **Inline Suggestions**: Type comments describing what you want, Copilot suggests code
- **Copilot Chat**: Use the chat panel to ask questions about the codebase
- **Context-Aware**: Copilot understands the multi-agent architecture and NVIDIA Blueprint patterns

Example prompts for Copilot Chat:
- "How do I create a new agent tool for equipment management?"
- "Show me how to add a new FastAPI endpoint with authentication"
- "Explain the hybrid retrieval system in this codebase"

## Troubleshooting

### Container Won't Start

1. Ensure Docker Desktop is running
2. Check Docker has enough resources (4GB+ RAM recommended)
3. Try rebuilding: Command Palette → "Dev Containers: Rebuild Container"

### Services Not Available

If PostgreSQL, Milvus, or Redis aren't responding:

```bash
# Check service status
docker-compose -f .devcontainer/docker-compose.yml ps

# Restart services
docker-compose -f .devcontainer/docker-compose.yml restart
```

### Python Package Issues

```bash
# Reinstall packages
pip install -r requirements.txt --force-reinstall
```

### Node Modules Issues

```bash
cd src/ui/web
rm -rf node_modules package-lock.json
npm install
```

## Customizing the Dev Container

### Adding Extensions

Edit `.devcontainer/devcontainer.json` and add extension IDs to the `extensions` array:

```json
"extensions": [
  "github.copilot",
  "your-new-extension-id"
]
```

### Modifying Services

Edit `.devcontainer/docker-compose.yml` to add or modify services.

### Changing Python Version

Edit `.devcontainer/Dockerfile` and change the base image:

```dockerfile
FROM mcr.microsoft.com/devcontainers/python:3.12
```

## Environment Variables

Set your NVIDIA API key and other secrets:

1. Copy `.env.example` to `.env`
2. Update values in `.env`
3. The dev container will use these automatically

## Port Forwarding

The following ports are automatically forwarded:

| Port  | Service            | URL                      |
|-------|-------------------|--------------------------|
| 8000  | FastAPI Backend   | http://localhost:8000    |
| 3000  | React Frontend    | http://localhost:3000    |
| 5432  | PostgreSQL        | localhost:5432           |
| 19530 | Milvus            | localhost:19530          |
| 6379  | Redis             | localhost:6379           |
| 9090  | Prometheus        | http://localhost:9090    |
| 3001  | Grafana           | http://localhost:3001    |

## Benefits of Using Dev Containers

✅ **Consistent Environment** - Everyone has the same setup
✅ **Quick Onboarding** - New developers up and running in minutes
✅ **Isolated Dependencies** - No conflicts with your local machine
✅ **Pre-configured Tools** - All linters, formatters, and extensions ready
✅ **GitHub Copilot Ready** - AI assistance configured out of the box
✅ **Service Integration** - Database and cache services automatically available

## Resources

- [VS Code Dev Containers Documentation](https://code.visualstudio.com/docs/devcontainers/containers)
- [GitHub Copilot Documentation](https://docs.github.com/en/copilot)
- [Project README](../README.md)
- [Contributing Guidelines](../CONTRIBUTING.md)
