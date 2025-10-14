from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging
from chain_server.graphs.mcp_integrated_planner_graph import get_mcp_planner_graph
from chain_server.services.guardrails.guardrails_service import guardrails_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Chat"])

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    reply: str
    route: str
    intent: str
    session_id: str
    context: Optional[Dict[str, Any]] = None
    structured_data: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[str]] = None
    confidence: Optional[float] = None
    actions_taken: Optional[List[Dict[str, Any]]] = None

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Process warehouse operational queries through the multi-agent planner with guardrails.
    
    This endpoint routes user messages to appropriate specialized agents
    (Inventory, Operations, Safety) based on intent classification and
    returns synthesized responses. All inputs and outputs are checked for
    safety, compliance, and security violations.
    """
    try:
        # Check input safety with guardrails
        input_safety = await guardrails_service.check_input_safety(req.message, req.context)
        if not input_safety.is_safe:
            logger.warning(f"Input safety violation: {input_safety.violations}")
            return ChatResponse(
                reply=guardrails_service.get_safety_response(input_safety.violations),
                route="guardrails",
                intent="safety_violation",
                session_id=req.session_id or "default",
                context={"safety_violations": input_safety.violations},
                confidence=input_safety.confidence
            )
        
        # Process the query through the MCP planner graph with error handling
        try:
            mcp_planner = await get_mcp_planner_graph()
            result = await mcp_planner.process_warehouse_query(
                message=req.message,
                session_id=req.session_id or "default",
                context=req.context
            )
        except Exception as query_error:
            logger.error(f"Query processing error: {query_error}")
            # Return a more helpful fallback response
            error_type = type(query_error).__name__
            error_message = str(query_error)
            
            # Provide specific error messages based on error type
            if "timeout" in error_message.lower():
                user_message = "The request timed out. Please try again with a simpler question."
            elif "connection" in error_message.lower():
                user_message = "I'm having trouble connecting to the processing service. Please try again in a moment."
            elif "validation" in error_message.lower():
                user_message = "There was an issue with your request format. Please try rephrasing your question."
            else:
                user_message = "I encountered an error processing your query. Please try rephrasing your question or contact support if the issue persists."
            
            return ChatResponse(
                reply=user_message,
                route="error",
                intent="error",
                session_id=req.session_id or "default",
                context={
                    "error": error_message,
                    "error_type": error_type,
                    "suggestions": [
                        "Try rephrasing your question",
                        "Check if the equipment ID or task name is correct",
                        "Contact support if the issue persists"
                    ]
                },
                confidence=0.0,
                recommendations=[
                    "Try rephrasing your question",
                    "Check if the equipment ID or task name is correct", 
                    "Contact support if the issue persists"
                ]
            )
        
        # Check output safety with guardrails
        output_safety = await guardrails_service.check_output_safety(result["response"], req.context)
        if not output_safety.is_safe:
            logger.warning(f"Output safety violation: {output_safety.violations}")
            return ChatResponse(
                reply=guardrails_service.get_safety_response(output_safety.violations),
                route="guardrails",
                intent="safety_violation",
                session_id=req.session_id or "default",
                context={"safety_violations": output_safety.violations},
                confidence=output_safety.confidence
            )
        
        # Extract structured response if available
        structured_response = result.get("structured_response", {})
        
        return ChatResponse(
            reply=result["response"],
            route=result["route"],
            intent=result["intent"],
            session_id=result["session_id"],
            context=result.get("context"),
            structured_data=structured_response.get("data"),
            recommendations=structured_response.get("recommendations"),
            confidence=structured_response.get("confidence"),
            actions_taken=structured_response.get("actions_taken")
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        # Return a user-friendly error response with helpful suggestions
        return ChatResponse(
            reply="I'm sorry, I encountered an unexpected error. Please try again or contact support if the issue persists.",
            route="error",
            intent="error",
            session_id=req.session_id or "default",
            context={
                "error": str(e),
                "error_type": type(e).__name__,
                "suggestions": [
                    "Try refreshing the page",
                    "Check your internet connection",
                    "Contact support if the issue persists"
                ]
            },
            confidence=0.0,
            recommendations=[
                "Try refreshing the page",
                "Check your internet connection", 
                "Contact support if the issue persists"
            ]
        )
