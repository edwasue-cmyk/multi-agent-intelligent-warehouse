"""
MCP-Enabled Forecasting Agent

This agent integrates with the Model Context Protocol (MCP) system to provide
dynamic tool discovery and execution for demand forecasting operations.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json
from datetime import datetime

from src.api.services.llm.nim_client import get_nim_client, LLMResponse
from src.retrieval.hybrid_retriever import get_hybrid_retriever, SearchContext
from src.memory.memory_manager import get_memory_manager
from src.api.services.mcp.tool_discovery import (
    ToolDiscoveryService,
    DiscoveredTool,
    ToolCategory,
)
from src.api.services.mcp.base import MCPManager
from .forecasting_action_tools import get_forecasting_action_tools

logger = logging.getLogger(__name__)


@dataclass
class MCPForecastingQuery:
    """MCP-enabled forecasting query."""

    intent: str
    entities: Dict[str, Any]
    context: Dict[str, Any]
    user_query: str
    mcp_tools: List[str] = None
    tool_execution_plan: List[Dict[str, Any]] = None


@dataclass
class MCPForecastingResponse:
    """MCP-enabled forecasting response."""

    response_type: str
    data: Dict[str, Any]
    natural_language: str
    recommendations: List[str]
    confidence: float
    actions_taken: List[Dict[str, Any]]
    mcp_tools_used: List[str] = None
    tool_execution_results: Dict[str, Any] = None


class ForecastingAgent:
    """
    MCP-enabled Forecasting Agent.

    This agent integrates with the Model Context Protocol (MCP) system to provide:
    - Dynamic tool discovery and execution for forecasting operations
    - MCP-based tool binding and routing
    - Enhanced tool selection and validation
    - Comprehensive error handling and fallback mechanisms
    """

    def __init__(self):
        self.nim_client = None
        self.hybrid_retriever = None
        self.forecasting_tools = None
        self.mcp_manager = None
        self.tool_discovery = None
        self.conversation_context = {}
        self.mcp_tools_cache = {}
        self.tool_execution_history = []

    async def initialize(self) -> None:
        """Initialize the agent with required services including MCP."""
        try:
            self.nim_client = await get_nim_client()
            self.hybrid_retriever = await get_hybrid_retriever()
            self.forecasting_tools = await get_forecasting_action_tools()

            # Initialize MCP components
            self.mcp_manager = MCPManager()
            self.tool_discovery = ToolDiscoveryService()

            # Start tool discovery
            await self.tool_discovery.start_discovery()

            # Register MCP sources
            await self._register_mcp_sources()

            logger.info("MCP-enabled Forecasting Agent initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MCP Forecasting Agent: {e}")
            raise

    async def _register_mcp_sources(self) -> None:
        """Register MCP sources for tool discovery."""
        try:
            # Import and register the forecasting MCP adapter (if it exists)
            try:
                from src.api.services.mcp.adapters.forecasting_adapter import (
                    get_forecasting_adapter,
                )

                forecasting_adapter = await get_forecasting_adapter()
                await self.tool_discovery.register_discovery_source(
                    "forecasting_tools", forecasting_adapter, "mcp_adapter"
                )
            except ImportError:
                logger.info("Forecasting MCP adapter not found, using direct tools")

            logger.info("MCP sources registered successfully")
        except Exception as e:
            logger.error(f"Failed to register MCP sources: {e}")

    async def process_query(
        self,
        query: str,
        session_id: str = "default",
        context: Optional[Dict[str, Any]] = None,
        mcp_results: Optional[Any] = None,
    ) -> MCPForecastingResponse:
        """
        Process a forecasting query with MCP integration.

        Args:
            query: User's forecasting query
            session_id: Session identifier for context
            context: Additional context
            mcp_results: Optional MCP execution results from planner graph

        Returns:
            MCPForecastingResponse with MCP tool execution results
        """
        try:
            # Initialize if needed
            if (
                not self.nim_client
                or not self.hybrid_retriever
                or not self.tool_discovery
            ):
                await self.initialize()

            # Parse query to extract intent and entities
            parsed_query = await self._parse_query(query, context)

            # Discover available tools
            available_tools = await self._discover_tools(parsed_query)

            # Execute tools based on query intent
            tool_results = await self._execute_forecasting_tools(
                parsed_query, available_tools
            )

            # Generate natural language response
            response = await self._generate_response(
                query, parsed_query, tool_results, context
            )

            return response

        except Exception as e:
            logger.error(f"Error processing forecasting query: {e}")
            return MCPForecastingResponse(
                response_type="error",
                data={"error": str(e)},
                natural_language=f"I encountered an error processing your forecasting query: {str(e)}",
                recommendations=[],
                confidence=0.0,
                actions_taken=[],
            )

    async def _parse_query(
        self, query: str, context: Optional[Dict[str, Any]]
    ) -> MCPForecastingQuery:
        """Parse the user query to extract intent and entities."""
        try:
            # Use LLM to extract intent and entities
            parse_prompt = [
                {
                    "role": "system",
                    "content": """You are a demand forecasting expert. Parse warehouse forecasting queries and extract intent, entities, and context.

Return JSON format:
{
    "intent": "forecast",
    "entities": {"sku": "SKU001", "horizon_days": 30}
}

Intent options: forecast, reorder_recommendation, model_performance, dashboard, business_intelligence

Examples:
- "What's the forecast for SKU FRI001?" → {"intent": "forecast", "entities": {"sku": "FRI001"}}
- "Show me reorder recommendations" → {"intent": "reorder_recommendation", "entities": {}}
- "What's the model performance?" → {"intent": "model_performance", "entities": {}}

Return only valid JSON.""",
                },
                {
                    "role": "user",
                    "content": f'Query: "{query}"\nContext: {context or {}}',
                },
            ]

            llm_response = await self.nim_client.generate_response(parse_prompt)
            parsed = json.loads(llm_response.content)

            return MCPForecastingQuery(
                intent=parsed.get("intent", "forecast"),
                entities=parsed.get("entities", {}),
                context=context or {},
                user_query=query,
            )

        except Exception as e:
            logger.warning(f"Failed to parse query with LLM, using simple extraction: {e}")
            # Simple fallback parsing
            query_lower = query.lower()
            entities = {}
            
            # Extract SKU if mentioned
            import re
            sku_match = re.search(r'\b([A-Z]{3}\d{3})\b', query)
            if sku_match:
                entities["sku"] = sku_match.group(1)
            
            # Extract horizon days
            days_match = re.search(r'(\d+)\s*days?', query)
            if days_match:
                entities["horizon_days"] = int(days_match.group(1))
            
            # Determine intent
            if "reorder" in query_lower or "recommendation" in query_lower:
                intent = "reorder_recommendation"
            elif "model" in query_lower or "performance" in query_lower:
                intent = "model_performance"
            elif "dashboard" in query_lower or "summary" in query_lower:
                intent = "dashboard"
            elif "business intelligence" in query_lower or "bi" in query_lower:
                intent = "business_intelligence"
            else:
                intent = "forecast"

            return MCPForecastingQuery(
                intent=intent,
                entities=entities,
                context=context or {},
                user_query=query,
            )

    async def _discover_tools(
        self, query: MCPForecastingQuery
    ) -> List[DiscoveredTool]:
        """Discover available forecasting tools."""
        try:
            # Get tools from MCP discovery by category
            discovered_tools = await self.tool_discovery.get_tools_by_category(
                ToolCategory.FORECASTING
            )
            
            # Also search by query keywords
            if query.user_query:
                keyword_tools = await self.tool_discovery.search_tools(query.user_query)
                discovered_tools.extend(keyword_tools)

            # Add direct tools if MCP doesn't have them
            if not discovered_tools:
                discovered_tools = [
                    DiscoveredTool(
                        name="get_forecast",
                        description="Get demand forecast for a specific SKU",
                        category=ToolCategory.FORECASTING,
                        parameters={"sku": "string", "horizon_days": "integer"},
                    ),
                    DiscoveredTool(
                        name="get_batch_forecast",
                        description="Get demand forecasts for multiple SKUs",
                        category=ToolCategory.FORECASTING,
                        parameters={"skus": "list", "horizon_days": "integer"},
                    ),
                    DiscoveredTool(
                        name="get_reorder_recommendations",
                        description="Get automated reorder recommendations",
                        category=ToolCategory.FORECASTING,
                        parameters={},
                    ),
                    DiscoveredTool(
                        name="get_model_performance",
                        description="Get model performance metrics",
                        category=ToolCategory.FORECASTING,
                        parameters={},
                    ),
                    DiscoveredTool(
                        name="get_forecast_dashboard",
                        description="Get comprehensive forecasting dashboard",
                        category=ToolCategory.FORECASTING,
                        parameters={},
                    ),
                ]

            return discovered_tools

        except Exception as e:
            logger.error(f"Failed to discover tools: {e}")
            return []

    async def _execute_forecasting_tools(
        self, query: MCPForecastingQuery, tools: List[DiscoveredTool]
    ) -> Dict[str, Any]:
        """Execute forecasting tools based on query intent."""
        tool_results = {}
        actions_taken = []

        try:
            intent = query.intent
            entities = query.entities

            if intent == "forecast":
                # Single SKU forecast
                sku = entities.get("sku")
                if sku:
                    forecast = await self.forecasting_tools.get_forecast(
                        sku, entities.get("horizon_days", 30)
                    )
                    tool_results["forecast"] = forecast
                    actions_taken.append(
                        {
                            "action": "get_forecast",
                            "sku": sku,
                            "horizon_days": entities.get("horizon_days", 30),
                        }
                    )
                else:
                    # Batch forecast for multiple SKUs or all
                    skus = entities.get("skus", [])
                    if not skus:
                        # Get all SKUs from inventory
                        from src.retrieval.structured.sql_retriever import SQLRetriever
                        sql_retriever = SQLRetriever()
                        sku_results = await sql_retriever.fetch_all(
                            "SELECT DISTINCT sku FROM inventory_items ORDER BY sku LIMIT 10"
                        )
                        skus = [row["sku"] for row in sku_results]

                    forecast = await self.forecasting_tools.get_batch_forecast(
                        skus, entities.get("horizon_days", 30)
                    )
                    tool_results["batch_forecast"] = forecast
                    actions_taken.append(
                        {
                            "action": "get_batch_forecast",
                            "skus": skus,
                            "horizon_days": entities.get("horizon_days", 30),
                        }
                    )

            elif intent == "reorder_recommendation":
                recommendations = await self.forecasting_tools.get_reorder_recommendations()
                tool_results["reorder_recommendations"] = recommendations
                actions_taken.append({"action": "get_reorder_recommendations"})

            elif intent == "model_performance":
                performance = await self.forecasting_tools.get_model_performance()
                tool_results["model_performance"] = performance
                actions_taken.append({"action": "get_model_performance"})

            elif intent == "dashboard":
                dashboard = await self.forecasting_tools.get_forecast_dashboard()
                tool_results["dashboard"] = dashboard
                actions_taken.append({"action": "get_forecast_dashboard"})

            elif intent == "business_intelligence":
                bi = await self.forecasting_tools.get_business_intelligence()
                tool_results["business_intelligence"] = bi
                actions_taken.append({"action": "get_business_intelligence"})

            else:
                # Default: get dashboard
                dashboard = await self.forecasting_tools.get_forecast_dashboard()
                tool_results["dashboard"] = dashboard
                actions_taken.append({"action": "get_forecast_dashboard"})

        except Exception as e:
            logger.error(f"Error executing forecasting tools: {e}")
            tool_results["error"] = str(e)

        return tool_results

    async def _generate_response(
        self,
        original_query: str,
        parsed_query: MCPForecastingQuery,
        tool_results: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> MCPForecastingResponse:
        """Generate natural language response from tool results."""
        try:
            # Format tool results for LLM
            results_summary = json.dumps(tool_results, default=str, indent=2)

            response_prompt = [
                {
                    "role": "system",
                    "content": """You are a demand forecasting assistant. Generate clear, helpful responses based on forecasting data.
                    
Your responses should:
1. Directly answer the user's query
2. Include key numbers and insights from the data
3. Provide actionable recommendations if applicable
4. Be concise but informative
5. Use natural, conversational language""",
                },
                {
                    "role": "user",
                    "content": f"""User Query: {original_query}
Query Intent: {parsed_query.intent}

Forecasting Results:
{results_summary}

Generate a natural language response:""",
                },
            ]

            llm_response = await self.nim_client.generate_response(response_prompt)
            natural_language = llm_response.content

            # Extract recommendations
            recommendations = []
            if "reorder_recommendations" in tool_results:
                for rec in tool_results["reorder_recommendations"]:
                    if rec.get("urgency_level") in ["CRITICAL", "HIGH"]:
                        recommendations.append(
                            f"Reorder {rec['sku']}: {rec['recommended_order_quantity']} units ({rec['urgency_level']})"
                        )

            # Calculate confidence based on data availability
            confidence = 0.8 if tool_results and "error" not in tool_results else 0.3

            return MCPForecastingResponse(
                response_type=parsed_query.intent,
                data=tool_results,
                natural_language=natural_language,
                recommendations=recommendations,
                confidence=confidence,
                actions_taken=parsed_query.tool_execution_plan or [],
                mcp_tools_used=[tool.name for tool in await self._discover_tools(parsed_query)],
                tool_execution_results=tool_results,
            )

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return MCPForecastingResponse(
                response_type="error",
                data={"error": str(e)},
                natural_language=f"I encountered an error: {str(e)}",
                recommendations=[],
                confidence=0.0,
                actions_taken=[],
            )


# Global instance
_forecasting_agent: Optional[ForecastingAgent] = None


async def get_forecasting_agent() -> ForecastingAgent:
    """Get or create the global forecasting agent instance."""
    global _forecasting_agent
    if _forecasting_agent is None:
        _forecasting_agent = ForecastingAgent()
        await _forecasting_agent.initialize()
    return _forecasting_agent

