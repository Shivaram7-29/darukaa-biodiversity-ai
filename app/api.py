"""
FastAPI backend for the Darukaa.Earth Biodiversity Intelligence Chatbot.
"""

import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.conversation import ConversationManager


# --- Global state ---
conversation_manager: ConversationManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the conversation manager on startup."""
    global conversation_manager
    print("[API] Initializing conversation manager and knowledge base...")
    conversation_manager = ConversationManager()
    info = conversation_manager.get_retriever_info()
    print(f"[API] Knowledge base ready: {info['count']} documents indexed.")
    yield
    print("[API] Shutting down.")


app = FastAPI(
    title="Darukaa.Earth Biodiversity Intelligence API",
    description="AI Environmental Scientist assistant for biodiversity and environment questions.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response Models ---

class EnvironmentalData(BaseModel):
    """Structured environmental input parameters."""
    soil_organic_carbon: float | None = Field(None, description="Soil organic carbon in percent")
    soil_ph: float | None = Field(None, description="Soil pH value")
    soil_moisture: float | None = Field(None, description="Soil moisture as percent of field capacity")
    rainfall: float | None = Field(None, description="Annual rainfall in mm")
    temperature: float | None = Field(None, description="Temperature in Celsius")
    land_use: str | None = Field(None, description="Land use type, e.g. 'rice monoculture'")
    crop_type: str | None = Field(None, description="Crop type, e.g. 'wheat'")
    biodiversity_index: float | None = Field(None, description="Biodiversity Shannon index")
    pollution_level: str | None = Field(None, description="Pollution level: low/moderate/high/severe")
    deforestation_rate: float | None = Field(None, description="Deforestation rate in percent per year")
    region: str | None = Field(None, description="Geographic region")
    latitude: float | None = Field(None, description="Latitude coordinate (bonus)")
    longitude: float | None = Field(None, description="Longitude coordinate (bonus)")
    additional_notes: str | None = Field(None, description="Any additional context")


class ChatRequest(BaseModel):
    """Chat request with optional environmental data."""
    message: str = Field(..., description="The user's question or message")
    conversation_id: str | None = Field(None, description="Conversation ID for context continuity")
    environmental_data: EnvironmentalData | None = Field(None, description="Optional structured environmental inputs")


class RecommendationItem(BaseModel):
    """Structured recommendation item with evidence-grounded quantitative estimates."""
    action: str
    scientific_basis: str
    scientific_reasoning: str | None = None  # synonym for hackathon challenge
    metrics_affected: list[str] = []
    impacted_environmental_metrics: list[str] = []  # synonym for hackathon challenge
    time_horizon: str | None = None
    measurable_improvement: str | None = None  # evidence-backed quantitative estimate
    confidence: str | None = None
    evidence: str | None = None
    scientific_reference: str | None = None  # synonym for hackathon challenge


class ChatResponse(BaseModel):
    """Structured chat response."""
    response: str
    conversation_id: str
    concerns: list[str] = []
    clarifying_questions: list[str] = []
    sources: list[str] = []
    assessment: dict | None = None
    recommendations: list[RecommendationItem] = []
    confidence: str | None = None


class HealthResponse(BaseModel):
    status: str
    knowledge_base: dict


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and knowledge base status."""
    if conversation_manager is None:
        raise HTTPException(status_code=503, detail="Service not yet initialized")
    info = conversation_manager.get_retriever_info()
    return HealthResponse(
        status="healthy",
        knowledge_base=info,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint. Accepts a question and optional environmental data,
    returns a scientifically grounded response with sources and assessment.
    """
    if conversation_manager is None:
        raise HTTPException(status_code=503, detail="Service not yet initialized")

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Generate conversation ID if not provided
    conv_id = request.conversation_id or str(uuid.uuid4())

    # Convert environmental data to dict (if provided)
    env_data = None
    if request.environmental_data:
        env_data = request.environmental_data.model_dump(exclude_none=True)

    try:
        result = conversation_manager.process_message(
            conversation_id=conv_id,
            user_message=request.message,
            environmental_data=env_data,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

    return ChatResponse(
        response=result["response"],
        conversation_id=conv_id,
        concerns=result["concerns"],
        clarifying_questions=result["clarifying_questions"],
        sources=result["sources"],
        assessment=result["assessment"],
        recommendations=result.get("recommendations", []),
        confidence=result.get("confidence"),
    )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint. Progressively streams text chunks as they are generated by Gemini,
    followed by the final complete structured payload (assessment, recommendations, sources, etc.).
    """
    if conversation_manager is None:
        raise HTTPException(status_code=503, detail="Service not yet initialized")

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    conv_id = request.conversation_id or str(uuid.uuid4())

    env_data = None
    if request.environmental_data:
        env_data = request.environmental_data.model_dump(exclude_none=True)

    def event_generator():
        try:
            for item in conversation_manager.process_message_stream(
                conversation_id=conv_id,
                user_message=request.message,
                environmental_data=env_data,
            ):
                if item.get("type") == "final" and "data" in item:
                    item["data"]["conversation_id"] = conv_id
                yield json.dumps(item) + "\n"
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str.upper():
                clean_err = "Gemini API quota exceeded (free-tier limit reached). Please verify your Google API key or try again in a few moments."
            else:
                clean_err = f"Error during streaming: {err_str}"
            yield json.dumps({"type": "error", "error": clean_err}) + "\n"

    return StreamingResponse(
        event_generator(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/chat/{conversation_id}/clear")
async def clear_conversation(conversation_id: str):
    """Clear a conversation's history."""
    if conversation_manager is None:
        raise HTTPException(status_code=503, detail="Service not yet initialized")
    conversation_manager.clear_conversation(conversation_id)
    return {"status": "cleared", "conversation_id": conversation_id}


@app.post("/admin/rebuild-index")
async def rebuild_index():
    """Rebuild the knowledge base vector index."""
    if conversation_manager is None:
        raise HTTPException(status_code=503, detail="Service not yet initialized")
    try:
        conversation_manager.rebuild_knowledge_index()
        info = conversation_manager.get_retriever_info()
        return {"status": "rebuilt", "knowledge_base": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error rebuilding index: {str(e)}")
