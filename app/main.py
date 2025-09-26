import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Configure logging as early as possible
# Configure logging early using environment only (no separate helper)
level_name = os.getenv("LOG_LEVEL", "INFO").upper()
level = getattr(logging, level_name, logging.INFO)
root = logging.getLogger()
root.setLevel(level)
if not root.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)
# Align common framework loggers to the chosen level without adding handlers
for lname in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi", "starlette"):
    logging.getLogger(lname).setLevel(level)
logging.getLogger(__name__).info(f"Logging configured | level={logging.getLevelName(level)}")

app = FastAPI(title="Investor Chatbot")

# CORS config
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "").strip()
ALLOW_ORIGIN_REGEX = os.getenv("ALLOW_ORIGIN_REGEX", "https?://(localhost|127\\.0\\.0\\.1)(:\\d+)?")

cors_kwargs = {
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}
if ALLOW_ORIGINS:
    origins = [o.strip() for o in ALLOW_ORIGINS.split(",") if o.strip()]
    cors_kwargs["allow_origins"] = origins
else:
    cors_kwargs["allow_origin_regex"] = ALLOW_ORIGIN_REGEX

app.add_middleware(CORSMiddleware, **cors_kwargs)

# Include API routers
try:
    from app.api.routers.chat import router as chat_router
    app.include_router(chat_router)
except Exception as e:
    logging.getLogger(__name__).warning(f"API router not mounted: {e}")

# Health endpoint
@app.get("/health")
async def health():
    logging.getLogger(__name__).debug("Health check endpoint hit")
    return {"status": "ok"}
