import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from api.intel_routes import router as intel_router
from settings import UPLOAD_DIR, get_allowed_origins
from security import rate_limit_middleware


os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="MindMatrix Backend")

# Add rate limiting middleware FIRST (before other middleware)
app.middleware("http")(rate_limit_middleware)

# CORS configuration - hardened for production
allowed_origins = get_allowed_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Only these specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # Restrict HTTP methods
    allow_headers=["Content-Type", "Authorization"],  # Only necessary headers
)

app.include_router(router)
app.include_router(intel_router)
