import os
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_MODELS_ENDPOINT = os.getenv("GITHUB_MODELS_ENDPOINT", "https://models.github.ai/inference")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")

if not GITHUB_TOKEN:
    raise RuntimeError("GITHUB_TOKEN not set in environment")

UPLOAD_DIR = "uploads"
CHROMA_DIR = "./chroma_db"

# Security settings
MAX_UPLOAD_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_UPLOAD_FILES = 20  # Max files per session
SESSION_CLEANUP_HOURS = 24  # Clean up sessions older than 24 hours


def get_allowed_origins() -> list[str]:
    """Get allowed CORS origins - hardened for production"""
    configured_origins = os.getenv("FRONTEND_ORIGINS")

    if configured_origins:
        # Whitelist specific configured origins only
        origins = [
            origin.strip()
            for origin in configured_origins.split(",")
            if origin.strip()
        ]
        # Only allow https in production (optional http for dev)
        return origins

    # Development origins only
    return [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]
