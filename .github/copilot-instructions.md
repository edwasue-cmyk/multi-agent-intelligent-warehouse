# GitHub Copilot Instructions for Multi-Agent Intelligent Warehouse

This file provides context and guidelines for GitHub Copilot to generate high-quality, consistent code for this project.

## Project Overview

This is a production-grade Multi-Agent Intelligent Warehouse system built on NVIDIA AI Blueprints, featuring:
- **Multi-Agent AI System** with LangGraph orchestration
- **FastAPI Backend** (Python 3.11+) with 5 specialized agents
- **React Frontend** (TypeScript/React 19+) with Material-UI
- **Hybrid RAG Stack** (PostgreSQL + Milvus vector database)
- **NVIDIA NIMs** for LLM inference and embeddings
- **Enterprise Security** with JWT authentication and NeMo Guardrails

## Code Organization

```
src/
├── api/                    # FastAPI backend application
│   ├── agents/            # Multi-agent implementations (Equipment, Operations, Safety, Forecasting, Document)
│   ├── routers/           # FastAPI route handlers
│   ├── services/          # Business logic layer
│   ├── graphs/            # LangGraph orchestration workflows
│   └── app.py             # Main FastAPI entry point
├── adapters/              # Third-party system integrations (WMS, ERP, IoT, RFID)
├── retrieval/             # RAG pipeline (vector + structured data)
├── memory/                # Conversation state management
└── ui/web/                # React TypeScript frontend
```

## Python Backend Patterns

### 1. FastAPI Route Handlers

**Pattern:** Create routers in `src/api/routers/` following RESTful conventions.

```python
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Optional
from pydantic import BaseModel

from src.api.services.auth_service import verify_token
from src.api.services.database import get_db_session
from src.api.utils.error_handler import handle_api_error

router = APIRouter(prefix="/api/v1/resource", tags=["Resource"])

class ResourceRequest(BaseModel):
    """Request model with validation."""
    name: str
    description: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "Example Resource",
                "description": "Optional description"
            }
        }

@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_resource(
    request: ResourceRequest,
    token: dict = Depends(verify_token),
    db=Depends(get_db_session)
):
    """Create a new resource with proper authentication and error handling."""
    try:
        # Validate permissions
        if token["role"] not in ["admin", "manager"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        
        # Business logic here
        result = await create_resource_logic(request, db)
        return {"status": "success", "data": result}
        
    except Exception as e:
        return handle_api_error(e, "create_resource")
```

**Key Points:**
- Always use dependency injection for authentication and database sessions
- Include request/response models with Pydantic validation
- Use proper HTTP status codes
- Wrap endpoints in try-except with `handle_api_error`
- Check role-based permissions when needed
- Add OpenAPI tags and docstrings

### 2. Agent Tool Implementation

**Pattern:** Agent tools follow the structure in `src/api/agents/*/action_tools.py`.

```python
from typing import Dict, Any, Optional
from langchain.tools import BaseTool
from pydantic import Field

from src.api.services.database import get_db_session
from src.api.services.monitoring.metrics import track_tool_execution

class EquipmentStatusTool(BaseTool):
    """Tool for retrieving equipment status and telemetry."""
    
    name: str = "equipment_status"
    description: str = """
    Get current status and telemetry data for warehouse equipment.
    
    Args:
        equipment_id: The unique identifier for the equipment
        include_telemetry: Whether to include sensor data (default: True)
    
    Returns:
        Equipment status including location, battery level, and operational state.
    """
    
    equipment_id: str = Field(..., description="Equipment ID to query")
    include_telemetry: bool = Field(default=True, description="Include telemetry data")
    
    @track_tool_execution("equipment_status")
    def _run(self, equipment_id: str, include_telemetry: bool = True) -> Dict[str, Any]:
        """Synchronous execution."""
        db = get_db_session()
        try:
            # Query equipment data
            equipment = db.query(Equipment).filter_by(id=equipment_id).first()
            if not equipment:
                return {"error": f"Equipment {equipment_id} not found"}
            
            result = {
                "equipment_id": equipment.id,
                "status": equipment.status,
                "location": equipment.location,
                "battery_level": equipment.battery_level
            }
            
            if include_telemetry:
                result["telemetry"] = self._get_telemetry(equipment_id, db)
            
            return result
            
        finally:
            db.close()
    
    async def _arun(self, equipment_id: str, include_telemetry: bool = True):
        """Async execution - delegates to sync for now."""
        return self._run(equipment_id, include_telemetry)
```

**Key Points:**
- Inherit from `BaseTool` (LangChain)
- Provide clear `name` and `description` for LLM context
- Use Pydantic `Field` for parameter descriptions
- Implement both `_run` (sync) and `_arun` (async) methods
- Track metrics with `@track_tool_execution` decorator
- Handle database sessions properly (get, try/finally, close)
- Return structured dictionaries with clear error messages

### 3. Service Layer Pattern

**Pattern:** Business logic lives in `src/api/services/`.

```python
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from src.api.services.database import get_db_session
from src.api.services.llm.nvidia_client import NVIDIAClient
from src.api.services.monitoring.metrics import record_operation_time

logger = logging.getLogger(__name__)

class ForecastingService:
    """Service for demand forecasting and inventory predictions."""
    
    def __init__(self):
        self.llm_client = NVIDIAClient()
    
    @record_operation_time("forecast_demand")
    async def forecast_demand(
        self,
        sku: str,
        forecast_days: int = 30,
        include_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Generate demand forecast for a given SKU.
        
        Args:
            sku: Stock Keeping Unit identifier
            forecast_days: Number of days to forecast (default: 30)
            include_confidence: Include confidence intervals (default: True)
        
        Returns:
            Forecast data with predictions and optional confidence intervals
        """
        try:
            db = get_db_session()
            
            # Get historical data
            historical = self._get_historical_demand(sku, db, days=90)
            
            if not historical:
                logger.warning(f"No historical data found for SKU: {sku}")
                return {"error": "Insufficient historical data"}
            
            # Generate forecast using ML models
            forecast = self._generate_forecast(historical, forecast_days)
            
            # Add confidence intervals if requested
            if include_confidence:
                forecast["confidence_intervals"] = self._calculate_confidence(forecast)
            
            # Store forecast in database
            self._save_forecast(sku, forecast, db)
            
            return {
                "sku": sku,
                "forecast": forecast,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Forecast generation failed for {sku}: {str(e)}")
            raise
        finally:
            db.close()
```

**Key Points:**
- Service classes encapsulate business logic
- Use dependency injection for external services (LLM clients, databases)
- Track operation times with `@record_operation_time` decorator
- Always use logging for errors and warnings
- Handle exceptions at service layer
- Clean up resources in `finally` blocks
- Return structured dictionaries with clear keys

### 4. Security Best Practices

**Always implement these security patterns:**

```python
# 1. Input Validation with Pydantic
from pydantic import BaseModel, validator, Field

class SafeRequest(BaseModel):
    query: str = Field(..., max_length=1000, description="User query")
    
    @validator('query')
    def validate_query(cls, v):
        # Sanitize input
        if any(dangerous in v.lower() for dangerous in ['<script>', 'javascript:', 'onerror=']):
            raise ValueError("Invalid input detected")
        return v.strip()

# 2. JWT Authentication
from src.api.services.auth_service import verify_token

@router.get("/protected")
async def protected_route(token: dict = Depends(verify_token)):
    # Token automatically validated, contains: user_id, username, role
    if token["role"] not in ["admin", "manager"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")

# 3. Rate Limiting (implement in middleware)
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@router.post("/search")
@limiter.limit("10/minute")
async def search_endpoint(request: Request):
    # Rate limited to 10 requests per minute
    pass

# 4. SQL Injection Prevention (use ORM)
# ✅ CORRECT - Use SQLAlchemy ORM
equipment = db.query(Equipment).filter_by(id=equipment_id).first()

# ❌ WRONG - Never use raw SQL with user input
# db.execute(f"SELECT * FROM equipment WHERE id = '{equipment_id}'")

# 5. Guardrails Integration (for LLM safety)
from src.api.services.guardrails_service import GuardrailsService

guardrails = GuardrailsService()

# Check user input
safe, message = await guardrails.validate_input(user_query)
if not safe:
    raise HTTPException(status_code=400, detail=message)
```

### 5. Database Patterns

**Use SQLAlchemy ORM consistently:**

```python
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime

from src.api.services.database import Base

class Equipment(Base):
    """Equipment model with proper relationships and validation."""
    
    __tablename__ = "equipment"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    status = Column(String, default="available")
    location = Column(String)
    battery_level = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    assignments = relationship("Assignment", back_populates="equipment")
    telemetry = relationship("Telemetry", back_populates="equipment")
    
    def to_dict(self):
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "location": self.location,
            "battery_level": self.battery_level
        }
```

## React Frontend Patterns

### 1. Component Structure

**Pattern:** Use TypeScript with proper typing and Material-UI components.

```typescript
import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  CircularProgress,
  Alert
} from '@mui/material';
import { useQuery, useMutation } from '@tanstack/react-query';

import { api } from '../../services/api';
import { Equipment } from '../../types/equipment';

interface EquipmentCardProps {
  equipmentId: string;
  onUpdate?: (equipment: Equipment) => void;
}

export const EquipmentCard: React.FC<EquipmentCardProps> = ({
  equipmentId,
  onUpdate
}) => {
  const {
    data: equipment,
    isLoading,
    error
  } = useQuery({
    queryKey: ['equipment', equipmentId],
    queryFn: () => api.getEquipment(equipmentId),
    refetchInterval: 30000 // Refresh every 30 seconds
  });

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" p={4}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error">
        Failed to load equipment: {error.message}
      </Alert>
    );
  }

  return (
    <Card>
      <CardContent>
        <Typography variant="h6">{equipment?.name}</Typography>
        <Typography color="textSecondary">
          Status: {equipment?.status}
        </Typography>
        <Typography>
          Battery: {equipment?.battery_level}%
        </Typography>
      </CardContent>
    </Card>
  );
};
```

**Key Points:**
- Use TypeScript interfaces for props
- Use TanStack Query for data fetching
- Implement loading and error states
- Use Material-UI components consistently
- Export components as named exports
- Follow functional component patterns with hooks

### 2. API Service Pattern

```typescript
// src/ui/web/src/services/api.ts
import axios, { AxiosInstance } from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

class APIService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add auth token interceptor
    this.client.interceptors.request.use((config) => {
      const token = localStorage.getItem('auth_token');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    });
  }

  async getEquipment(id: string) {
    const response = await this.client.get(`/api/v1/equipment/${id}`);
    return response.data;
  }

  async chat(message: string) {
    const response = await this.client.post('/api/v1/chat', {
      message,
      session_id: this.getSessionId(),
    });
    return response.data;
  }

  private getSessionId(): string {
    let sessionId = sessionStorage.getItem('session_id');
    if (!sessionId) {
      sessionId = crypto.randomUUID();
      sessionStorage.setItem('session_id', sessionId);
    }
    return sessionId;
  }
}

export const api = new APIService();
```

## Testing Patterns

### Python Unit Tests

```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from src.api.app import app

client = TestClient(app)

@pytest.fixture
def mock_db():
    """Mock database session."""
    with patch('src.api.services.database.get_db_session') as mock:
        yield mock.return_value

@pytest.fixture
def auth_headers():
    """Valid authentication headers."""
    return {"Authorization": "Bearer valid-test-token"}

def test_get_equipment_success(mock_db, auth_headers):
    """Test successful equipment retrieval."""
    response = client.get("/api/v1/equipment/EQ001", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["equipment_id"] == "EQ001"

def test_get_equipment_not_found(mock_db, auth_headers):
    """Test equipment not found error handling."""
    response = client.get("/api/v1/equipment/INVALID", headers=auth_headers)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
```

### TypeScript Unit Tests

```typescript
import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { EquipmentCard } from './EquipmentCard';
import { api } from '../../services/api';

jest.mock('../../services/api');

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: false },
  },
});

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <QueryClientProvider client={queryClient}>
    {children}
  </QueryClientProvider>
);

test('displays equipment data', async () => {
  const mockEquipment = {
    id: 'EQ001',
    name: 'Forklift A',
    status: 'active',
    battery_level: 85,
  };

  (api.getEquipment as jest.Mock).mockResolvedValue(mockEquipment);

  render(<EquipmentCard equipmentId="EQ001" />, { wrapper });

  await waitFor(() => {
    expect(screen.getByText('Forklift A')).toBeInTheDocument();
    expect(screen.getByText(/Status: active/)).toBeInTheDocument();
    expect(screen.getByText(/Battery: 85%/)).toBeInTheDocument();
  });
});
```

## Environment Variables

Always use environment variables for configuration. Reference `.env.example`:

```python
import os
from dotenv import load_dotenv

load_dotenv()

# Required
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY environment variable is required")

# Optional with defaults
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/warehouse")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
```

## Documentation Standards

- **Docstrings:** Use Google-style docstrings for Python
- **Type Hints:** Always include type hints in Python
- **Comments:** Explain "why" not "what"
- **API Docs:** FastAPI generates OpenAPI docs automatically
- **README Updates:** Update relevant docs when adding features

## Common Pitfalls to Avoid

1. **Don't bypass authentication** - Always use `Depends(verify_token)`
2. **Don't use raw SQL** - Use SQLAlchemy ORM to prevent injection
3. **Don't ignore errors** - Always implement proper error handling
4. **Don't skip input validation** - Use Pydantic models with validators
5. **Don't hardcode credentials** - Use environment variables
6. **Don't forget database cleanup** - Use try/finally or context managers
7. **Don't skip metrics tracking** - Use provided decorators for monitoring
8. **Don't ignore NeMo Guardrails** - Validate LLM inputs/outputs for safety

## Monitoring and Observability

Always track operations with Prometheus metrics:

```python
from src.api.services.monitoring.metrics import (
    track_tool_execution,
    record_operation_time,
    increment_counter,
    observe_histogram
)

@track_tool_execution("my_tool")
def my_tool_function():
    """Automatically tracked in metrics."""
    pass

@record_operation_time("forecast_generation")
async def generate_forecast():
    """Records execution time in histogram."""
    pass

# Manual metrics
increment_counter("custom_events", {"event_type": "user_login"})
observe_histogram("query_latency", 0.234, {"query_type": "vector_search"})
```

## Dependencies and Imports

**Python imports order:**
1. Standard library imports
2. Third-party imports
3. Local application imports

```python
# Standard library
import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Third-party
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

# Local
from src.api.services.auth_service import verify_token
from src.api.services.database import get_db_session
from src.api.utils.error_handler import handle_api_error
```

## Additional Resources

- **Architecture Docs:** `/docs/architecture/` - ADRs and system design
- **Security Docs:** `/docs/security/` - CVE mitigations and best practices
- **API Reference:** Start server and visit http://localhost:8000/docs
- **Test Guides:** `/tests/README.md` - Testing strategies and examples
- **CONTRIBUTING.md** - PR workflow and code review process

## Summary

When generating code for this project:
- Follow FastAPI and React best practices
- Implement security-first (JWT, validation, guardrails)
- Use type hints and Pydantic models
- Track metrics for observability
- Handle errors gracefully
- Write tests for new functionality
- Keep code consistent with existing patterns
