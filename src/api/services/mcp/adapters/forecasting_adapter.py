"""
MCP Adapter for Forecasting Action Tools

This adapter wraps the ForecastingActionTools class to make it compatible
with the MCP (Model Context Protocol) system for tool discovery and execution.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field

from src.api.services.mcp.base import (
    MCPAdapter,
    AdapterConfig,
    AdapterType,
    MCPTool,
    MCPToolType,
)
from src.api.services.mcp.client import MCPConnectionType
from src.api.agents.forecasting.forecasting_action_tools import (
    get_forecasting_action_tools,
)
from src.api.services.mcp.parameter_validator import get_parameter_validator

logger = logging.getLogger(__name__)


class ForecastingAdapterConfig(AdapterConfig):
    """Configuration for Forecasting MCP Adapter."""

    adapter_type: AdapterType = field(default=AdapterType.FORECASTING)
    name: str = field(default="forecasting_tools")
    endpoint: str = field(default="local://forecasting_tools")
    connection_type: MCPConnectionType = field(default=MCPConnectionType.STDIO)
    description: str = field(default="Demand forecasting and prediction tools")
    version: str = field(default="1.0.0")
    enabled: bool = field(default=True)
    timeout_seconds: int = field(default=30)
    retry_attempts: int = field(default=3)
    batch_size: int = field(default=100)


class ForecastingMCPAdapter(MCPAdapter):
    """MCP Adapter for Forecasting Action Tools."""

    def __init__(self, config: ForecastingAdapterConfig = None, mcp_client: Optional[Any] = None):
        super().__init__(config or ForecastingAdapterConfig(), mcp_client)
        self.forecasting_tools = None

    async def initialize(self) -> bool:
        """Initialize the adapter."""
        try:
            self.forecasting_tools = await get_forecasting_action_tools()
            await self._register_tools()
            logger.info(
                f"Forecasting MCP Adapter initialized successfully with {len(self.tools)} tools"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Forecasting MCP Adapter: {e}")
            return False

    async def connect(self) -> bool:
        """Connect to the forecasting tools service."""
        try:
            if self.forecasting_tools:
                self.connected = True
                logger.info("Forecasting MCP Adapter connected")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to connect Forecasting MCP Adapter: {e}")
            return False

    async def disconnect(self) -> bool:
        """Disconnect from the forecasting tools service."""
        try:
            self.connected = False
            logger.info("Forecasting MCP Adapter disconnected")
            return True
        except Exception as e:
            logger.error(f"Failed to disconnect Forecasting MCP Adapter: {e}")
            return False

    async def execute_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool with parameter validation."""
        try:
            # Get the tool definition
            if tool_name not in self.tools:
                return {
                    "error": f"Tool '{tool_name}' not found",
                    "available_tools": list(self.tools.keys()),
                }

            tool_def = self.tools[tool_name]

            # Validate parameters
            validator = await get_parameter_validator()
            validation_result = await validator.validate_tool_parameters(
                tool_name, tool_def.parameters, arguments
            )

            if not validation_result.is_valid:
                return {
                    "error": "Parameter validation failed",
                    "validation_summary": validator.get_validation_summary(
                        validation_result
                    ),
                    "issues": [
                        {
                            "parameter": issue.parameter,
                            "level": issue.level.value,
                            "message": issue.message,
                            "suggestion": issue.suggestion,
                        }
                        for issue in validation_result.errors
                    ],
                    "suggestions": validator.get_improvement_suggestions(
                        validation_result
                    ),
                }

            # Use validated arguments
            validated_args = validation_result.validated_arguments

            # Execute the tool using the base class method
            result = await super().execute_tool(tool_name, validated_args)

            # Add validation warnings if any
            if validation_result.warnings:
                if isinstance(result, dict):
                    result["validation_warnings"] = [
                        {
                            "parameter": warning.parameter,
                            "message": warning.message,
                            "suggestion": warning.suggestion,
                        }
                        for warning in validation_result.warnings
                    ]

            return result

        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {"error": str(e)}

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the adapter."""
        try:
            if self.forecasting_tools:
                return {
                    "status": "healthy",
                    "timestamp": datetime.utcnow().isoformat(),
                    "tools_count": len(self.tools),
                    "connected": self.connected,
                }
            else:
                return {
                    "status": "unhealthy",
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": "Forecasting tools not initialized",
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
            }

    async def _register_tools(self) -> None:
        """Register forecasting tools as MCP tools."""
        if not self.forecasting_tools:
            logger.warning("Forecasting tools not available for registration")
            return

        logger.info("Starting tool registration for Forecasting MCP Adapter")

        # Register get_forecast tool
        self.tools["get_forecast"] = MCPTool(
            name="get_forecast",
            description="Get demand forecast for a specific SKU",
            tool_type=MCPToolType.FUNCTION,
            parameters={
                "type": "object",
                "properties": {
                    "sku": {
                        "type": "string",
                        "description": "Stock Keeping Unit identifier",
                    },
                    "horizon_days": {
                        "type": "integer",
                        "description": "Number of days to forecast (default: 30)",
                        "default": 30,
                    },
                },
                "required": ["sku"],
            },
            handler=self._handle_get_forecast,
        )

        # Register get_batch_forecast tool
        self.tools["get_batch_forecast"] = MCPTool(
            name="get_batch_forecast",
            description="Get demand forecasts for multiple SKUs",
            tool_type=MCPToolType.FUNCTION,
            parameters={
                "type": "object",
                "properties": {
                    "skus": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of Stock Keeping Unit identifiers",
                    },
                    "horizon_days": {
                        "type": "integer",
                        "description": "Number of days to forecast (default: 30)",
                        "default": 30,
                    },
                },
                "required": ["skus"],
            },
            handler=self._handle_get_batch_forecast,
        )

        # Register get_reorder_recommendations tool
        self.tools["get_reorder_recommendations"] = MCPTool(
            name="get_reorder_recommendations",
            description="Get automated reorder recommendations based on forecasts",
            tool_type=MCPToolType.FUNCTION,
            parameters={
                "type": "object",
                "properties": {},
            },
            handler=self._handle_get_reorder_recommendations,
        )

        # Register get_model_performance tool
        self.tools["get_model_performance"] = MCPTool(
            name="get_model_performance",
            description="Get model performance metrics for all forecasting models",
            tool_type=MCPToolType.FUNCTION,
            parameters={
                "type": "object",
                "properties": {},
            },
            handler=self._handle_get_model_performance,
        )

        # Register get_forecast_dashboard tool
        self.tools["get_forecast_dashboard"] = MCPTool(
            name="get_forecast_dashboard",
            description="Get comprehensive forecasting dashboard data",
            tool_type=MCPToolType.FUNCTION,
            parameters={
                "type": "object",
                "properties": {},
            },
            handler=self._handle_get_forecast_dashboard,
        )

        # Register get_business_intelligence tool
        self.tools["get_business_intelligence"] = MCPTool(
            name="get_business_intelligence",
            description="Get business intelligence summary for forecasting",
            tool_type=MCPToolType.FUNCTION,
            parameters={
                "type": "object",
                "properties": {},
            },
            handler=self._handle_get_business_intelligence,
        )

        logger.info(
            f"Registered {len(self.tools)} forecasting tools: {list(self.tools.keys())}"
        )

    async def _handle_get_forecast(
        self, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle get_forecast tool execution."""
        try:
            result = await self.forecasting_tools.get_forecast(
                sku=arguments["sku"],
                horizon_days=arguments.get("horizon_days", 30),
            )
            return result
        except Exception as e:
            logger.error(f"Error executing get_forecast: {e}")
            return {"error": str(e)}

    async def _handle_get_batch_forecast(
        self, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle get_batch_forecast tool execution."""
        try:
            result = await self.forecasting_tools.get_batch_forecast(
                skus=arguments["skus"],
                horizon_days=arguments.get("horizon_days", 30),
            )
            return result
        except Exception as e:
            logger.error(f"Error executing get_batch_forecast: {e}")
            return {"error": str(e)}

    async def _handle_get_reorder_recommendations(
        self, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle get_reorder_recommendations tool execution."""
        try:
            result = await self.forecasting_tools.get_reorder_recommendations()
            return {"recommendations": result}
        except Exception as e:
            logger.error(f"Error executing get_reorder_recommendations: {e}")
            return {"error": str(e)}

    async def _handle_get_model_performance(
        self, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle get_model_performance tool execution."""
        try:
            result = await self.forecasting_tools.get_model_performance()
            return {"model_metrics": result}
        except Exception as e:
            logger.error(f"Error executing get_model_performance: {e}")
            return {"error": str(e)}

    async def _handle_get_forecast_dashboard(
        self, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle get_forecast_dashboard tool execution."""
        try:
            result = await self.forecasting_tools.get_forecast_dashboard()
            return result
        except Exception as e:
            logger.error(f"Error executing get_forecast_dashboard: {e}")
            return {"error": str(e)}

    async def _handle_get_business_intelligence(
        self, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle get_business_intelligence tool execution."""
        try:
            result = await self.forecasting_tools.get_business_intelligence()
            return result
        except Exception as e:
            logger.error(f"Error executing get_business_intelligence: {e}")
            return {"error": str(e)}


# Global instance
_forecasting_adapter: Optional[ForecastingMCPAdapter] = None


async def get_forecasting_adapter() -> ForecastingMCPAdapter:
    """Get the global forecasting adapter instance."""
    global _forecasting_adapter
    if _forecasting_adapter is None:
        config = ForecastingAdapterConfig()
        _forecasting_adapter = ForecastingMCPAdapter(config)
        await _forecasting_adapter.initialize()
    return _forecasting_adapter

