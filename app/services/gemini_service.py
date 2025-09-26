import os
import logging
from typing import List, Dict, Any, Optional

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover
    genai = None

logger = logging.getLogger(__name__)

LOG_LLM_PROMPTS = os.getenv("LOG_LLM_PROMPTS", "false").lower() == "true"

GEMINI_MODEL_NAME = "gemini-2.5-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Optional: retry once if the model stops due to MAX_TOKENS without emitting text
GEMINI_RETRY_ON_MAX_TOKENS = os.getenv("GEMINI_RETRY_ON_MAX_TOKENS", "false").lower() == "true"
# Hard cap for retries to avoid runaway outputs
GEMINI_MAX_TOKENS_HARD_CAP = int(os.getenv("GEMINI_MAX_TOKENS_HARD_CAP", "2048"))

class GeminiService:
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.model_name = GEMINI_MODEL_NAME
        self.api_key = (api_key or GEMINI_API_KEY)
        self.client_ready = False
        self._init_client()

    def _init_client(self) -> None:
        if genai is None:
            logger.warning("google-generativeai package not installed. Install 'google-generativeai' to use Gemini backend.")
            return
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not set. Set it in environment to enable Gemini calls.")
            return
        try:
            genai.configure(api_key=self.api_key)
            self.client_ready = True
            logger.info("Gemini client configured")
        except Exception as e:
            logger.error(f"Failed to configure Gemini client: {e}")

    def count_tokens(self, text: str) -> int:
        """
        Use Gemini's native token counting. Returns -1 if unavailable (for fallback).
        """
        if not self.client_ready or genai is None:
            return -1
        try:
            model = genai.GenerativeModel(self.model_name)
            res = model.count_tokens(text or "")
            # Support different attribute casings if any
            total = getattr(res, "total_tokens", None) or getattr(res, "totalTokens", None)
            return int(total) if total is not None else -1
        except Exception as e:
            logger.debug(f"Gemini count_tokens failed: {e}")
            return -1

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 1512) -> str:
        if not self.client_ready:
            return "Gemini API is not configured. Set GEMINI_API_KEY and restart."
        try:
            system_parts = [m["content"] for m in messages if m.get("role") == "system"]
            user_messages = [m["content"] for m in messages if m.get("role") == "user"]
            system_prompt = "\n\n".join(system_parts) if system_parts else None
            prompt = "\n\n".join(user_messages)

            if LOG_LLM_PROMPTS:
                logger.debug(f"Gemini request | model={self.model_name} system_prompt_len={len(system_prompt) if system_prompt else 0} prompt_len={len(prompt)} temp={temperature} max_tokens={max_tokens}")
            else:
                logger.debug(f"Gemini request | model={self.model_name} messages_count={len(messages)} temp={temperature} max_tokens={max_tokens}")

            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            logger.debug(f"Gemini prompt={prompt}")
            model = genai.GenerativeModel(self.model_name, system_instruction=system_prompt)
            resp = model.generate_content(prompt, generation_config=generation_config)
            # Avoid direct access to resp.text which can raise when no valid Part exists (e.g., safety blocked)
            try:
                logger.debug(f"model response ={resp}")
                text_attr = getattr(resp, "text", None)
                if isinstance(text_attr, str) and text_attr.strip():
                    text = text_attr
                    logger.debug(f"Gemini response preview={text[:200]}")
                    return text
            except Exception as e_text:
                logger.debug(f"Gemini .text accessor unavailable: {e_text}")
            # Try candidates -> content.parts flow
            cand = getattr(resp, "candidates", None)
            if cand:
                try:
                    first = cand[0]
                    # If the candidate was blocked or truncated, finish_reason will indicate it
                    finish_reason = getattr(first, "finish_reason", None) or getattr(first, "finishReason", None)
                    # Extract content parts
                    content = getattr(first, "content", None)
                    parts = getattr(content, "parts", None)
                    if parts:
                        out = []
                        for p in parts:
                            # Gemini parts can be text, inline_data, or function_call; take only text
                            t = getattr(p, "text", None)
                            if isinstance(t, str) and t:
                                out.append(t)
                        text = "".join(out).strip()
                        if text:
                            logger.debug(f"Gemini response preview={text[:200]}")
                            return text
                    # Fall back to safety/metadata inspection
                    safety = getattr(first, "safety_ratings", None) or getattr(first, "safetyRatings", None)
                    if safety or (isinstance(finish_reason, str) and finish_reason.upper() == "SAFETY"):
                        logger.warning(f"Gemini response blocked by safety. finish_reason={finish_reason} ratings={safety}")
                        return "Sorry, I can’t respond to that request due to safety policies. Please try rephrasing."
                except Exception as e_cand:
                    logger.debug(f"Gemini candidates parsing failed: {e_cand}")
            # Streaming and non-standard responses may put output under resp.prompt_feedback / safety
            try:
                pf = getattr(resp, "prompt_feedback", None) or getattr(resp, "promptFeedback", None)
                if pf and getattr(pf, "block_reason", None):
                    logger.warning(f"Gemini prompt blocked. reason={getattr(pf,'block_reason',None)} safety={getattr(pf,'safety_ratings',None)}")
                    return "Sorry, I can’t respond to that request due to safety policies. Please try rephrasing."
            except Exception:
                pass
            logger.debug("Gemini response empty; returning default message")
            return "I'm sorry, I didn't understand that. Can you rephrase?"
        except Exception as e:
            logger.error(f"Gemini chat failed: {e}")
            return "Sorry, something went wrong. Please contact the administrator."