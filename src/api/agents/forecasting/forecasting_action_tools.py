"""
Forecasting Agent Action Tools

Provides action tools for demand forecasting that use the forecasting service API.
These tools wrap the existing forecasting system endpoints as MCP-compatible tools.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import httpx
import os

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Forecast result from the forecasting service."""

    sku: str
    predictions: List[float]
    confidence_intervals: List[tuple]
    forecast_date: str
    horizon_days: int
    model_metrics: Dict[str, Any]
    recent_average_demand: float


@dataclass
class ReorderRecommendation:
    """Reorder recommendation from the forecasting service."""

    sku: str
    current_stock: int
    recommended_order_quantity: int
    urgency_level: str
    reason: str
    confidence_score: float
    estimated_arrival_date: str


@dataclass
class ModelPerformance:
    """Model performance metrics."""

    model_name: str
    accuracy_score: float
    mape: float
    last_training_date: str
    prediction_count: int
    drift_score: float
    status: str


class ForecastingActionTools:
    """
    Action tools for demand forecasting.
    
    These tools call the existing forecasting service API endpoints,
    making the forecasting system available as tools for the agent.
    """

    def __init__(self):
        self.api_base_url = os.getenv("API_BASE_URL", "http://localhost:8001")
        self.forecasting_service = None  # Will be initialized if available

    async def initialize(self) -> None:
        """Initialize the action tools."""
        try:
            # Try to import and use the forecasting service directly if available
            try:
                from src.api.routers.advanced_forecasting import AdvancedForecastingService
                self.forecasting_service = AdvancedForecastingService()
                await self.forecasting_service.initialize()
                logger.info("✅ Forecasting action tools initialized with direct service")
            except Exception as e:
                logger.warning(f"Could not initialize direct service, will use API calls: {e}")
                self.forecasting_service = None

        except Exception as e:
            logger.error(f"Failed to initialize forecasting action tools: {e}")
            raise

    async def get_forecast(
        self, sku: str, horizon_days: int = 30
    ) -> Dict[str, Any]:
        """
        Get demand forecast for a specific SKU.
        
        Args:
            sku: Stock Keeping Unit identifier
            horizon_days: Number of days to forecast (default: 30)
            
        Returns:
            Dictionary containing forecast predictions, confidence intervals, and metrics
        """
        try:
            # Use direct service if available
            if self.forecasting_service:
                forecast = await self.forecasting_service.get_real_time_forecast(
                    sku, horizon_days
                )
                return forecast

            # Fallback to API call
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.api_base_url}/api/v1/forecasting/real-time",
                    json={"sku": sku, "horizon_days": horizon_days},
                )
                response.raise_for_status()
                return response.json()

        except Exception as e:
            logger.error(f"Failed to get forecast for {sku}: {e}")
            raise

    async def get_batch_forecast(
        self, skus: List[str], horizon_days: int = 30
    ) -> Dict[str, Any]:
        """
        Get demand forecasts for multiple SKUs.
        
        Args:
            skus: List of Stock Keeping Unit identifiers
            horizon_days: Number of days to forecast (default: 30)
            
        Returns:
            Dictionary mapping SKU to forecast results
        """
        try:
            # Use direct service if available
            if self.forecasting_service:
                results = {}
                for sku in skus:
                    try:
                        forecast = await self.forecasting_service.get_real_time_forecast(
                            sku, horizon_days
                        )
                        results[sku] = forecast
                    except Exception as e:
                        logger.warning(f"Failed to forecast {sku}: {e}")
                        continue
                return results

            # Fallback to API call
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.api_base_url}/api/v1/forecasting/batch-forecast",
                    json={"skus": skus, "horizon_days": horizon_days},
                )
                response.raise_for_status()
                return response.json()

        except Exception as e:
            logger.error(f"Failed to get batch forecast: {e}")
            raise

    async def get_reorder_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get automated reorder recommendations based on forecasts.
        
        Returns:
            List of reorder recommendations with urgency levels
        """
        try:
            # Use direct service if available
            if self.forecasting_service:
                recommendations = await self.forecasting_service.generate_reorder_recommendations()
                # Convert Pydantic models to dicts (Pydantic v2 uses model_dump(), v1 uses dict())
                result = []
                for rec in recommendations:
                    if hasattr(rec, 'model_dump'):
                        result.append(rec.model_dump())
                    elif hasattr(rec, 'dict'):
                        result.append(rec.dict())
                    elif isinstance(rec, dict):
                        result.append(rec)
                    else:
                        # Fallback: convert to dict manually
                        result.append({
                            'sku': getattr(rec, 'sku', ''),
                            'current_stock': getattr(rec, 'current_stock', 0),
                            'recommended_order_quantity': getattr(rec, 'recommended_order_quantity', 0),
                            'urgency_level': getattr(rec, 'urgency_level', ''),
                            'reason': getattr(rec, 'reason', ''),
                            'confidence_score': getattr(rec, 'confidence_score', 0.0),
                            'estimated_arrival_date': getattr(rec, 'estimated_arrival_date', ''),
                        })
                return result

            # Fallback to API call
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.api_base_url}/api/v1/forecasting/reorder-recommendations"
                )
                response.raise_for_status()
                data = response.json()
                # Handle both list and dict responses
                if isinstance(data, dict) and "recommendations" in data:
                    return data["recommendations"]
                return data if isinstance(data, list) else []

        except Exception as e:
            logger.error(f"Failed to get reorder recommendations: {e}")
            raise

    async def get_model_performance(self) -> List[Dict[str, Any]]:
        """
        Get model performance metrics for all forecasting models.
        
        Returns:
            List of model performance metrics
        """
        try:
            # Use direct service if available
            if self.forecasting_service:
                metrics = await self.forecasting_service.get_model_performance_metrics()
                # Convert Pydantic models to dicts (Pydantic v2 uses model_dump(), v1 uses dict())
                result = []
                for m in metrics:
                    if hasattr(m, 'model_dump'):
                        result.append(m.model_dump())
                    elif hasattr(m, 'dict'):
                        result.append(m.dict())
                    elif isinstance(m, dict):
                        result.append(m)
                    else:
                        # Fallback: convert to dict manually
                        result.append({
                            'model_name': getattr(m, 'model_name', ''),
                            'accuracy_score': getattr(m, 'accuracy_score', 0.0),
                            'mape': getattr(m, 'mape', 0.0),
                            'last_training_date': getattr(m, 'last_training_date', ''),
                            'prediction_count': getattr(m, 'prediction_count', 0),
                            'drift_score': getattr(m, 'drift_score', 0.0),
                            'status': getattr(m, 'status', ''),
                        })
                return result

            # Fallback to API call
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.api_base_url}/api/v1/forecasting/model-performance"
                )
                response.raise_for_status()
                data = response.json()
                # Handle both list and dict responses
                if isinstance(data, dict) and "model_metrics" in data:
                    return data["model_metrics"]
                return data if isinstance(data, list) else []

        except Exception as e:
            logger.error(f"Failed to get model performance: {e}")
            raise

    async def get_forecast_dashboard(self) -> Dict[str, Any]:
        """
        Get comprehensive forecasting dashboard data.
        
        Returns:
            Dictionary containing forecast summary, model performance, and recommendations
        """
        try:
            # Use direct service if available
            if self.forecasting_service:
                dashboard = await self.forecasting_service.get_enhanced_business_intelligence()
                return dashboard

            # Fallback to API call
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.api_base_url}/api/v1/forecasting/dashboard"
                )
                response.raise_for_status()
                return response.json()

        except Exception as e:
            logger.error(f"Failed to get forecast dashboard: {e}")
            raise

    async def get_business_intelligence(self) -> Dict[str, Any]:
        """
        Get business intelligence summary for forecasting.
        
        Returns:
            Dictionary containing business intelligence metrics and insights
        """
        try:
            # Use direct service if available
            if self.forecasting_service:
                bi = await self.forecasting_service.get_enhanced_business_intelligence()
                return bi

            # Fallback to API call
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.api_base_url}/api/v1/forecasting/business-intelligence/enhanced"
                )
                response.raise_for_status()
                return response.json()

        except Exception as e:
            logger.error(f"Failed to get business intelligence: {e}")
            raise


# Global instance
_forecasting_action_tools: Optional[ForecastingActionTools] = None


async def get_forecasting_action_tools() -> ForecastingActionTools:
    """Get or create the global forecasting action tools instance."""
    global _forecasting_action_tools
    if _forecasting_action_tools is None:
        _forecasting_action_tools = ForecastingActionTools()
        await _forecasting_action_tools.initialize()
    return _forecasting_action_tools

