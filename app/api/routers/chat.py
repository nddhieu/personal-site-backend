import json
import logging
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel  # kept for compatibility, not directly used
import asyncio
import httpx
import os

from app.services.llm_service import LLMService
from app.services.gemini_service import GeminiService
from app.services.process_chat_service import process_request

logger = logging.getLogger(__name__)

CHAT_MODEL_BACKEND = os.getenv("CHAT_MODEL_BACKEND", "gemini")

router = APIRouter(prefix="/api", tags=["chat"])

from app.schemas.message import ChatRequest, ChatResponse

_llm = LLMService()
_gemini = GeminiService()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    try:
        if CHAT_MODEL_BACKEND == "gemini":
            # messages = [
            #     {"role": "system", "content": "You are a helpful assistant focused on financial markets."},
            #     {"role": "user", "content": req.text}
            # ]

            return await process_request(req)
            # response_text = _gemini.chat(messages)
            # backend = "gemini"
        else:
            response_text = _llm.chat([{"role": "user", "content": req.text}])
            backend = "mistral"
        return ChatResponse(response=response_text, backend=backend)
    except Exception as e:
        logger.exception("Chat endpoint failed")
        raise HTTPException(status_code=500, detail=str(e))



