FROM mcr.microsoft.com/devcontainers/python:3.11

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    postgresql-client \
    build-essential \
    curl \
    git \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 18.x
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g npm@latest

# Set up Python virtual environment
WORKDIR /workspace

# Copy requirements files
COPY requirements.txt requirements.docker.txt ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install -r requirements.docker.txt

# Install development tools
RUN pip install \
    black \
    flake8 \
    mypy \
    pytest \
    pytest-cov \
    pytest-asyncio \
    ipython \
    jupyter

# Create a non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Configure git to be safe with the workspace directory
RUN git config --system --add safe.directory /workspace

# Set the default shell to bash
ENV SHELL=/bin/bash

# Set Python path
ENV PYTHONPATH=/workspace

# Keep container running
CMD ["sleep", "infinity"]
