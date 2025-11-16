"""
MCP Adapters for Warehouse Operational Assistant

This package contains MCP-enabled adapters for various external systems
including ERP, WMS, IoT, RFID, Time Attendance, and Forecasting systems.
"""

from .erp_adapter import MCPERPAdapter
from .forecasting_adapter import (
    ForecastingMCPAdapter,
    ForecastingAdapterConfig,
    get_forecasting_adapter,
)

__all__ = [
    "MCPERPAdapter",
    "ForecastingMCPAdapter",
    "ForecastingAdapterConfig",
    "get_forecasting_adapter",
]

__version__ = "1.0.0"
