import os
from fastapi import APIRouter, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
from supabase import create_client, Client
from typing import Optional
from uuid import UUID
from auth import verify_jwt_token

router = APIRouter()

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


@router.get("/intel/questions")
async def get_questions(user_data: dict = Depends(verify_jwt_token)):
    """Get all intel board questions (authenticated endpoint)"""
    try:
        response = supabase.table("intel_board_questions").select("*").order("created_at", desc=True).execute()
        return response.data
    except Exception as e:
        print(f"[Intel] Error fetching questions: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch questions")


@router.get("/intel/questions/{question_id}")
async def get_question(
    question_id: str,
    user_data: dict = Depends(verify_jwt_token)
):
    """Get specific question with its answers (authenticated endpoint)"""
    try:
        question = supabase.table("intel_board_questions").select("*").eq("id", question_id).single().execute()
        answers = supabase.table("intel_board_answers").select("*").eq("question_id", question_id).order("created_at", desc=False).execute()
        
        return {
            "question": question.data,
            "answers": answers.data
        }
    except Exception as e:
        print(f"[Intel] Error fetching question {question_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch question")


@router.post("/intel/questions")
async def create_question(
    data: dict,
    user_data: dict = Depends(verify_jwt_token)
):
    """Create a new intel board question (authenticated)"""
    try:
        if "content" not in data or not data.get("content", "").strip():
            raise HTTPException(status_code=400, detail="content is required and cannot be empty")
        
        if len(data.get("content", "")) > 10000:
            raise HTTPException(status_code=400, detail="content too long (max 10000 characters)")
        
        question_data = {
            "user_id": user_data["user_id"],
            "content": data["content"].strip(),
            "x_position": int(data.get("x_position", 0)) if data.get("x_position") else 0,
            "y_position": int(data.get("y_position", 0)) if data.get("y_position") else 0,
            "question_color": data.get("question_color", "question-blue"),
        }
        
        response = supabase.table("intel_board_questions").insert(question_data).execute()
        return response.data[0] if response.data else None
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Intel] Error creating question for user {user_data['user_id']}: {e}")
        raise HTTPException(status_code=500, detail="Failed to create question")


@router.put("/intel/questions/{question_id}")
async def update_question(
    question_id: str,
    data: dict,
    user_data: dict = Depends(verify_jwt_token)
):
    """Update question position (only by owner)"""
    try:
        # Verify ownership
        question = supabase.table("intel_board_questions").select("user_id").eq("id", question_id).single().execute()
        if not question.data or question.data.get("user_id") != user_data["user_id"]:
            raise HTTPException(status_code=403, detail="Unauthorized to update this question")
        
        update_data = {}
        if "x_position" in data:
            update_data["x_position"] = int(data["x_position"])
        if "y_position" in data:
            update_data["y_position"] = int(data["y_position"])
        if "content" in data and data["content"].strip():
            if len(data["content"]) > 10000:
                raise HTTPException(status_code=400, detail="content too long")
            update_data["content"] = data["content"].strip()
        if "question_color" in data:
            update_data["question_color"] = data["question_color"]
        
        response = supabase.table("intel_board_questions").update(update_data).eq("id", question_id).execute()
        return response.data[0] if response.data else None
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Intel] Error updating question {question_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update question")


@router.delete("/intel/questions/{question_id}")
async def delete_question(
    question_id: str,
    user_data: dict = Depends(verify_jwt_token)
):
    """Delete a question (only by owner)"""
    try:
        # Verify ownership
        question = supabase.table("intel_board_questions").select("user_id").eq("id", question_id).single().execute()
        if not question.data or question.data.get("user_id") != user_data["user_id"]:
            raise HTTPException(status_code=403, detail="Unauthorized to delete this question")
        
        response = supabase.table("intel_board_questions").delete().eq("id", question_id).execute()
        return {"success": True, "message": "Question deleted"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Intel] Error deleting question {question_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete question")


@router.post("/intel/answers")
async def create_answer(
    data: dict,
    user_data: dict = Depends(verify_jwt_token)
):
    """Create an answer to a question (authenticated)"""
    try:
        if "question_id" not in data or "content" not in data:
            raise HTTPException(status_code=400, detail="question_id and content are required")
        
        if not data.get("content", "").strip():
            raise HTTPException(status_code=400, detail="content cannot be empty")
        
        if len(data.get("content", "")) > 5000:
            raise HTTPException(status_code=400, detail="content too long (max 5000 characters)")
        
        # Verify question exists
        question = supabase.table("intel_board_questions").select("id").eq("id", data["question_id"]).single().execute()
        if not question.data:
            raise HTTPException(status_code=404, detail="Question not found")
        
        answer_data = {
            "question_id": data["question_id"],
            "user_id": user_data["user_id"],
            "content": data["content"].strip(),
            "is_accepted": False,
        }
        
        response = supabase.table("intel_board_answers").insert(answer_data).execute()
        return response.data[0] if response.data else None
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Intel] Error creating answer for user {user_data['user_id']}: {e}")
        raise HTTPException(status_code=500, detail="Failed to create answer")


@router.put("/intel/answers/{answer_id}/accept")
async def accept_answer(
    answer_id: str,
    user_data: dict = Depends(verify_jwt_token)
):
    """Accept an answer (mark as correct) and award XP to answerer"""
    try:
        # Get the answer with its question
        answer_response = supabase.table("intel_board_answers").select("*, intel_board_questions(user_id)").eq("id", answer_id).single().execute()
        answer = answer_response.data
        
        if not answer:
            raise HTTPException(status_code=404, detail="Answer not found")
        
        # Verify that user owns the question
        question_owner_id = answer.get("intel_board_questions", {}).get("user_id")
        if question_owner_id != user_data["user_id"]:
            raise HTTPException(status_code=403, detail="Only question owner can accept answers")
        
        # Mark answer as accepted
        supabase.table("intel_board_answers").update({"is_accepted": True}).eq("id", answer_id).execute()
        
        # Award XP to the answerer (25 XP for helpful answer)
        answerer_id = answer["user_id"]
        try:
            xp_result = supabase.rpc("award_xp", {
                "p_user_id": answerer_id,
                "p_amount": 25,
                "p_reason": "Helpful answer on Intel Board",
                "p_reference_id": answer_id
            }).execute()
            return {
                "success": True,
                "message": "Answer accepted and XP awarded",
                "xp_awarded": 25
            }
        except Exception as xp_error:
            print(f"[Intel] XP award error: {xp_error}")
            # If XP award fails, still return success for accepting answer
            return {
                "success": True,
                "message": "Answer accepted (XP award pending)",
                "xp_awarded": 0
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/intel/answers/{answer_id}")
async def delete_answer(answer_id: str):
    """Delete an answer"""
    try:
        response = supabase.table("intel_board_answers").delete().eq("id", answer_id).execute()
        return {"success": True, "message": "Answer deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
