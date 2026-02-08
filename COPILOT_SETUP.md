# GitHub Copilot Setup Guide

This guide will help you set up GitHub Copilot for development on the Multi-Agent Intelligent Warehouse project.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [GitHub Copilot Features](#github-copilot-features)
- [VS Code Setup](#vs-code-setup)
- [Dev Container Setup](#dev-container-setup)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### 1. GitHub Copilot Subscription

You need an active GitHub Copilot subscription:
- **Individual**: $10/month or $100/year
- **Business**: $19/user/month
- **Free for Students/Teachers**: Through GitHub Education

Sign up at: https://github.com/features/copilot

### 2. Development Tools

- **Visual Studio Code** (v1.85 or later)
- **Git** (v2.30 or later)
- **Docker Desktop** (for dev container setup)

## Quick Start

### Option 1: Dev Container (Recommended)

The fastest way to get started with everything pre-configured:

1. **Open in VS Code**
   ```bash
   code multi-agent-intelligent-warehouse
   ```

2. **Install Dev Containers Extension**
   - Install from: https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers

3. **Reopen in Container**
   - VS Code will prompt: "Folder contains a Dev Container configuration file. Reopen folder to develop in a container?"
   - Click **"Reopen in Container"**
   - Wait 5-10 minutes for first build

4. **You're Ready!**
   - GitHub Copilot is automatically installed
   - All dependencies are configured
   - Services (PostgreSQL, Milvus, Redis) are running

### Option 2: Local Setup

If you prefer working on your local machine:

1. **Install VS Code Extensions**

   Open VS Code and install:
   - [GitHub Copilot](https://marketplace.visualstudio.com/items?itemName=GitHub.copilot)
   - [GitHub Copilot Chat](https://marketplace.visualstudio.com/items?itemName=GitHub.copilot-chat)
   - [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
   - [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance)

   Or install all recommended extensions:
   ```bash
   # Open VS Code in project directory
   code .
   # Press Ctrl+Shift+P (Cmd+Shift+P on Mac)
   # Type: "Extensions: Show Recommended Extensions"
   # Click "Install All"
   ```

2. **Sign in to GitHub Copilot**
   - Click the GitHub Copilot icon in the status bar
   - Sign in with your GitHub account
   - Authorize VS Code

3. **Verify Setup**
   - Open a Python file (e.g., `src/api/app.py`)
   - Start typing a comment like `# Function to calculate average`
   - Copilot should suggest code completions

## GitHub Copilot Features

### 1. Code Completions

Copilot suggests code as you type:

```python
# Type this comment:
# Create a FastAPI endpoint to get equipment by ID

# Copilot will suggest something like:
@router.get("/equipment/{equipment_id}")
async def get_equipment(equipment_id: str, db=Depends(get_db_session)):
    equipment = db.query(Equipment).filter_by(id=equipment_id).first()
    if not equipment:
        raise HTTPException(status_code=404, detail="Equipment not found")
    return equipment
```

### 2. Copilot Chat

Ask questions about the codebase:

- **Open Chat**: Click the chat icon in the sidebar or press `Ctrl+Shift+I` (Cmd+Shift+I on Mac)

**Example Questions:**
- "How does the multi-agent system work?"
- "Show me how to add a new agent tool"
- "Explain the hybrid retrieval system"
- "How do I implement authentication in a new endpoint?"
- "What's the pattern for creating service classes?"

### 3. Context-Aware Suggestions

Copilot understands the project through:

- **`.github/copilot-instructions.md`** - Codebase patterns and conventions
- **Open files** - Current context
- **File structure** - Project organization
- **Comments** - Your intent

### 4. Test Generation

Generate tests automatically:

```python
# Select a function, then in Copilot Chat:
# "Generate pytest tests for this function"

# Or type a comment above a function:
# Write comprehensive tests for this function including edge cases
```

### 5. Code Explanation

Get explanations for complex code:

```python
# Select code, then in Copilot Chat:
# "Explain what this code does"
# "How can I improve this function?"
# "Are there any security issues here?"
```

## VS Code Setup

### Configuration Files

The project includes pre-configured VS Code settings:

- **`.vscode/settings.json`** - Editor, linting, formatting settings
- **`.vscode/extensions.json`** - Recommended extensions

These are automatically applied when you open the project.

### Key Features Enabled

✅ **Python Formatting** - Black auto-format on save
✅ **Import Sorting** - isort auto-organize on save
✅ **Type Checking** - mypy enabled
✅ **Linting** - flake8 configured
✅ **Testing** - pytest integration
✅ **Copilot** - Enabled for all file types

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+I` / `Cmd+I` | Open Copilot inline chat |
| `Ctrl+Shift+I` / `Cmd+Shift+I` | Open Copilot chat panel |
| `Tab` | Accept Copilot suggestion |
| `Alt+]` / `Option+]` | Next suggestion |
| `Alt+[` / `Option+[` | Previous suggestion |
| `Esc` | Dismiss suggestion |

## Dev Container Setup

### What's Included

The dev container provides a complete development environment:

**Pre-installed Tools:**
- Python 3.11 with all project dependencies
- Node.js 18 for frontend development
- Git, Docker CLI, PostgreSQL client
- Black, flake8, mypy, pytest
- Jupyter for notebooks

**Running Services:**
- PostgreSQL 14 (localhost:5432)
- Milvus v2.3.0 (localhost:19530)
- Redis 7 (localhost:6379)

**VS Code Extensions:**
- GitHub Copilot & Copilot Chat
- All Python development tools
- TypeScript/JavaScript support
- Database tools
- Git tools

### Using the Dev Container

1. **Start Development**
   ```bash
   # Backend
   python -m uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

   # Frontend
   cd src/ui/web && npm start
   ```

2. **Run Tests**
   ```bash
   pytest tests/
   ```

3. **Format Code**
   ```bash
   black src/ tests/
   ```

4. **Access Services**
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Frontend: http://localhost:3000

### Rebuilding the Container

If you need to rebuild:

```
Ctrl+Shift+P / Cmd+Shift+P → "Dev Containers: Rebuild Container"
```

## Best Practices

### 1. Write Clear Comments

Copilot works best with descriptive comments:

```python
# ✅ GOOD - Specific and clear
# Create a FastAPI endpoint that retrieves equipment status by ID
# with JWT authentication and returns JSON response

# ❌ BAD - Too vague
# Get equipment
```

### 2. Use Type Hints

Copilot generates better suggestions with type hints:

```python
# ✅ GOOD
def get_equipment(equipment_id: str, db: Session) -> Optional[Equipment]:

# ❌ BAD
def get_equipment(equipment_id, db):
```

### 3. Leverage Context

Keep relevant files open so Copilot understands context:

- Models (e.g., `equipment.py`)
- Related services
- Similar implementations

### 4. Review Suggestions

Always review Copilot's suggestions:

- **Security**: Check for SQL injection, XSS, auth bypasses
- **Error Handling**: Ensure proper exception handling
- **Testing**: Verify edge cases are covered
- **Style**: Confirm it matches project conventions

### 5. Use Copilot Chat for Complex Tasks

For multi-step changes, use Copilot Chat to:

- Plan the approach
- Generate multiple related files
- Get architecture advice
- Debug issues

### 6. Provide Feedback

Help improve Copilot:

- 👍 Good suggestions - Accept them
- 👎 Bad suggestions - Dismiss or modify
- Report issues through GitHub

## Troubleshooting

### Copilot Not Working

1. **Check Extension Status**
   - Look for Copilot icon in status bar
   - Should show "Copilot" (not "Copilot: Disabled")

2. **Sign In**
   - Click Copilot icon → "Sign in to GitHub"
   - Authorize VS Code

3. **Check Subscription**
   - Visit: https://github.com/settings/copilot
   - Ensure subscription is active

4. **Reload VS Code**
   ```
   Ctrl+Shift+P → "Developer: Reload Window"
   ```

### No Suggestions Appearing

1. **Check File Type Support**
   - Copilot may be disabled for certain file types
   - Check: `.vscode/settings.json` → `github.copilot.enable`

2. **Network Issues**
   - Copilot requires internet connection
   - Check firewall/proxy settings

3. **Clear Cache**
   - Close VS Code
   - Delete: `~/.vscode/extensions/github.copilot-*`
   - Restart VS Code and reinstall extension

### Slow Suggestions

1. **Check System Resources**
   - Close unnecessary applications
   - Ensure sufficient RAM available

2. **Reduce Context**
   - Close unused editor tabs
   - Smaller files process faster

3. **Check Network**
   - Slow connection affects Copilot
   - Try wired connection if on Wi-Fi

### Dev Container Issues

See [.devcontainer/README.md](.devcontainer/README.md) for detailed troubleshooting.

## Getting Help

### Documentation

- **GitHub Copilot Docs**: https://docs.github.com/en/copilot
- **VS Code Copilot Guide**: https://code.visualstudio.com/docs/editor/artificial-intelligence
- **Project Contributing Guide**: [CONTRIBUTING.md](CONTRIBUTING.md)

### Support

- **GitHub Copilot Support**: https://support.github.com/
- **Project Issues**: https://github.com/edwasue-cmyk/multi-agent-intelligent-warehouse/issues
- **VS Code Issues**: https://github.com/microsoft/vscode/issues

## Advanced Topics

### Custom Prompts

Create custom prompts for common tasks:

```python
# In .vscode/settings.json, you can configure custom snippets
# that work well with Copilot
```

### Team Collaboration

Share Copilot best practices with your team:

1. Document common patterns in `.github/copilot-instructions.md`
2. Create code review guidelines for AI-generated code
3. Share useful prompts and examples

### Integration with CI/CD

The project includes GitHub Actions that work alongside Copilot:

- **CodeQL** - Security scanning
- **SonarQube** - Code quality
- **Tests** - Automated testing

Always run these checks before committing AI-generated code.

## Summary

✅ **GitHub Copilot is an AI pair programmer** that suggests code as you type
✅ **Two setup options**: Dev Container (recommended) or local setup
✅ **Context-aware**: Uses `.github/copilot-instructions.md` and project structure
✅ **Features**: Code completion, chat, test generation, explanations
✅ **Best practices**: Clear comments, type hints, review suggestions
✅ **Always review AI suggestions** for security and correctness

Happy coding with GitHub Copilot! 🚀
