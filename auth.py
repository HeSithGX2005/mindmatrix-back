"""Authentication and authorization utilities"""
from typing import Optional
import re
from fastapi import HTTPException, Header, Depends
from supabase import Client
import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")

from supabase import create_client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


async def verify_jwt_token(authorization: Optional[str] = Header(None)) -> dict:
    """Verify JWT token from Authorization header and return user data"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    try:
        token = authorization.replace("Bearer ", "").strip()
        if not token:
            raise HTTPException(status_code=401, detail="Invalid token format")
        
        # Verify token with Supabase
        response = supabase.auth.get_user(token)
        if not response or not response.user:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        
        return {
            "user_id": response.user.id,
            "email": response.user.email,
            "token": token
        }
    except HTTPException:
        raise
    except Exception as e:
        # Don't expose internal error details
        raise HTTPException(status_code=401, detail="Invalid token")


async def verify_session_ownership(user_data: dict = Depends(verify_jwt_token), session_id: str = None) -> dict:
    """Verify that user owns the session"""
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")

    # Basic format guard to avoid accepting arbitrary strings.
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,500}", session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id format")
    
    try:
        # Check if session belongs to user
        response = supabase.table("projects").select("user_id").eq("session_id", session_id).limit(1).execute()
        
        if not response.data or len(response.data) == 0:
            # Allow pre-persist sessions generated client-side before project row exists.
            # Expected pattern from frontend: session_<user_id>_<timestamp>
            expected_prefix = f"session_{user_data['user_id']}_"
            if session_id.startswith(expected_prefix):
                return {**user_data, "session_id": session_id}

            # Backward-compatible support for dash variant if used by older clients.
            alt_prefix = f"session-{user_data['user_id']}-"
            if session_id.startswith(alt_prefix):
                return {**user_data, "session_id": session_id}

            raise HTTPException(status_code=404, detail="Session not found")
        
        session_owner_id = response.data[0]["user_id"]
        
        if session_owner_id != user_data["user_id"]:
            raise HTTPException(status_code=403, detail="Unauthorized access to this session")
        
        return {**user_data, "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        # Log error server-side but don't expose to client
        print(f"[Auth] Session verification error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
