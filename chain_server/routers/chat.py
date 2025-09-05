from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging
from chain_server.graphs.planner_graph import process_warehouse_query
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
        
        # Process the query through the planner graph with error handling
        try:
            result = await process_warehouse_query(
                message=req.message,
                session_id=req.session_id or "default",
                context=req.context
            )
        except Exception as query_error:
            logger.error(f"Query processing error: {query_error}")
            # Return a fallback response instead of failing
            return ChatResponse(
                reply=f"I encountered an error processing your query: {str(query_error)}. Please try rephrasing your question or contact support if the issue persists.",
                route="error",
                intent="error",
                session_id=req.session_id or "default",
                context={"error": str(query_error)},
                confidence=0.0
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
        structured_response = result.get("context", {}).get("structured_response", {})
        
        return ChatResponse(
            reply=result["response"],
            route=result["route"],
            intent=result["intent"],
            session_id=result["session_id"],
            context=result.get("context"),
            structured_data=structured_response.get("structured_data"),
            recommendations=structured_response.get("recommendations"),
            confidence=structured_response.get("confidence"),
            actions_taken=structured_response.get("actions_taken")
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        # Return a user-friendly error response instead of raising an exception
        return ChatResponse(
            reply="I'm sorry, I encountered an unexpected error. Please try again or contact support if the issue persists.",
            route="error",
            intent="error",
            session_id=req.session_id or "default",
            context={"error": str(e)},
            confidence=0.0
        )
