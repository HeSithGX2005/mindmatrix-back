import os
from datetime import datetime, timezone
import uuid
import re
from typing import Any

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request
from supabase import create_client, Client

from dependencies import chat_sessions, collection
from settings import UPLOAD_DIR, MAX_UPLOAD_FILE_SIZE
from auth import verify_jwt_token, verify_session_ownership
from schemas import (
    UploadAndChunkRequest,
    ChatRequest,
    QuizGenerationRequest,
    QuizIdentityRequest,
    QuizSubmissionRequest,
    TopicsExtractionRequest,
)
from security import get_client_ip
from services.document_service import extract_text_by_file_type, chunk_text, embed_text
from services.retrieval_service import (
    get_session_documents,
    get_session_documents_with_metadata,
    retrieve_relevant_chunks,
)
from services.tutor_service import (
    answer_from_chunks_with_history,
    extract_topics_from_chunks,
    generate_quiz_from_chunks,
    grade_quiz_submission,
)


router = APIRouter()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client | None = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None
UNKNOWN_MATERIAL_RESPONSE = "i don't know based on the provided material."
_CONTINUE_INTENTS = {
    "yes",
    "continue",
    "go on",
    "next",
    "more",
    "proceed",
    "go ahead",
    "keep going",
    "carry on",
    "ok",
    "okay",
    "sure",
}
_FULL_LESSON_INTENTS = {
    "full pdf",
    "explain everything",
    "teach everything",
    "start lesson",
    "start teaching",
}


def _is_unknown_material_answer(answer: str) -> bool:
    return UNKNOWN_MATERIAL_RESPONSE in (answer or "").strip().lower()


def _normalize_intent_text(text: str) -> str:
    lowered = (text or "").lower().strip()
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip(" .,!?:;\t\n\r")


def _is_continue_intent(question: str) -> bool:
    normalized = _normalize_intent_text(question)
    if not normalized:
        return False

    if normalized in _CONTINUE_INTENTS:
        return True

    if len(normalized) <= 28 and re.fullmatch(
        r"(?:please\s+)?(?:yes|continue|go on|next|more|proceed|go ahead|keep going|carry on|ok|okay|sure)(?:\s+please)?",
        normalized,
    ):
        return True

    return False


def _is_full_lesson_intent(question: str) -> bool:
    normalized = _normalize_intent_text(question)
    if not normalized:
        return False

    if normalized in _FULL_LESSON_INTENTS:
        return True

    return any(phrase in normalized for phrase in _FULL_LESSON_INTENTS)


def _build_history_text(history: list[dict[str, Any]], max_messages: int = 4) -> str:
    history_text = ""
    for msg in history[-max_messages:]:
        role = str(msg.get("role", "user")).upper()
        content = str(msg.get("content", ""))
        history_text += f"{role}: {content}\\n"
    return history_text


def _fallback_answer_from_chunks(chunks: list[str]) -> str:
    if not chunks:
        return "I could not find enough detail in the uploaded material yet. Try asking about a specific topic or section."

    def _clean_preview(chunk_text: str, max_len: int = 240) -> str:
        cleaned = " ".join((chunk_text or "").split())
        cleaned = re.sub(r"https?://\\S+|www\\.\\S+", "", cleaned)
        cleaned = re.sub(r"\\b\\S+@\\S+\\b", "", cleaned)
        cleaned = re.sub(r"\\s{2,}", " ", cleaned).strip(" -|•")
        if len(cleaned) > max_len:
            return f"{cleaned[:max_len].rstrip()}..."
        return cleaned

    previews: list[str] = []
    for chunk in chunks[:3]:
        cleaned = _clean_preview(chunk)
        if not cleaned:
            continue

        sentences = re.split(r"(?<=[.!?])\\s+", cleaned)
        selected = ""
        for sentence in sentences:
            candidate = sentence.strip()
            if len(candidate) >= 45 and any(ch.isalpha() for ch in candidate):
                selected = candidate
                break

        previews.append(selected or cleaned)
        if len(previews) >= 2:
            break

    if not previews:
        return "I found your uploaded material, but it appears to contain very little readable text. Try uploading a clearer version."

    lines = [
        "Here is a simpler explanation from your uploaded material:",
    ]
    for index, preview in enumerate(previews, start=1):
        lines.append(f"{index}. {preview}")

    lines.append("If you want, say continue and I will teach the next section in the same style.")
    return "\n".join(lines)


def _to_public_quiz_questions(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": q.get("id"),
            "question_type": q.get("question_type", "single"),
            "question": q.get("question"),
            "options": q.get("options"),
            "chunk_index": q.get("chunk_index"),
        }
        for q in questions
    ]


def _load_project_info(session_id: str) -> dict[str, Any] | None:
    if not supabase:
        return None

    try:
        response = (
            supabase
            .table("projects")
            .select("id, user_id")
            .eq("session_id", session_id)
            .limit(1)
            .execute()
        )
        rows = response.data or []
        return rows[0] if rows else None
    except Exception as exc:
        print(f"[quiz-persistence] Could not load project info for session {session_id}: {exc}")
        return None


def _persist_quiz_record(
    session_id: str,
    quiz_id: str,
    scope: str,
    difficulty: str,
    selected_start_index: int,
    selected_end_index: int,
    questions: list[dict[str, Any]],
) -> None:
    if not supabase:
        return

    try:
        project = _load_project_info(session_id)
        now_iso = datetime.now(timezone.utc).isoformat()
        payload = {
            "quiz_id": quiz_id,
            "session_id": session_id,
            "project_id": project.get("id") if project else None,
            "user_id": project.get("user_id") if project else None,
            "scope": scope,
            "difficulty": difficulty,
            "question_count": len(questions),
            "start_index": selected_start_index,
            "end_index": selected_end_index,
            "questions": questions,
            "attempts": 0,
            "is_finished": False,
            "last_score": None,
            "last_submitted_at": None,
            "last_result": None,
            "created_at": now_iso,
            "updated_at": now_iso,
        }
        supabase.table("project_quizzes").upsert(payload).execute()
    except Exception as exc:
        print(f"[quiz-persistence] Could not persist quiz {quiz_id}: {exc}")


def _get_persisted_quiz(session_id: str, quiz_id: str) -> dict[str, Any] | None:
    if not supabase:
        return None

    try:
        response = (
            supabase
            .table("project_quizzes")
            .select("*")
            .eq("session_id", session_id)
            .eq("quiz_id", quiz_id)
            .limit(1)
            .execute()
        )
        rows = response.data or []
        return rows[0] if rows else None
    except Exception as exc:
        print(f"[quiz-persistence] Could not fetch quiz {quiz_id}: {exc}")
        return None


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.post("/upload-and-chunk")
async def upload_and_chunk(
    request: Request,
    file: UploadFile = File(...),
    session_id: str = Form(...),
    user_data: dict = Depends(verify_jwt_token)
):
    """Upload and chunk PDF with security checks"""
    try:
        # Validate session ownership
        session_check = await verify_session_ownership(user_data, session_id)
        
        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        file_ext = os.path.splitext(file.filename.lower())[1]
        allowed_extensions = {".pdf", ".docx", ".pptx", ".txt"}
        if file_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail="Allowed file types: PDF, DOCX, PPTX, TXT")
        
        # Validate file size
        file_size = 0
        file_content = await file.read()
        file_size = len(file_content)
        
        if file_size > MAX_UPLOAD_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: 50MB. Received: {file_size / 1024 / 1024:.1f}MB"
            )
        
        if file_size == 0:
            raise HTTPException(status_code=400, detail="File is empty")
        
        # Save file temporarily
        file_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
        
        try:
            with open(file_path, "wb") as f:
                f.write(file_content)
        except Exception as e:
            print(f"[Upload] File write error: {e}")
            raise HTTPException(status_code=500, detail="Failed to save file")
        
        try:
            extracted_text = extract_text_by_file_type(file_path, file_ext)
            if not extracted_text:
                raise HTTPException(
                    status_code=400,
                    detail="Could not extract text from file. Ensure it's not corrupted, empty, or image-only."
                )
            
            chunks = chunk_text(extracted_text)
            stored_count = 0
            
            for idx, chunk in enumerate(chunks):
                embedding = embed_text(chunk)
                
                if not embedding:
                    continue
                
                collection.add(
                    documents=[chunk],
                    embeddings=[embedding],
                    metadatas=[{
                        "session_id": session_id,
                        "filename": file.filename,
                        "chunk_id": idx
                    }],
                    ids=[f"{file_id}_{idx}"]
                )
                
                stored_count += 1
            
            return {
                "message": "Chunking + embedding + storage successful",
                "filename": file.filename,
                "total_chunks": len(chunks),
                "stored_chunks": stored_count
            }
        finally:
            # Always clean up temp file
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"[Upload] Cleanup error: {e}")
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Upload] Unexpected error for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="File processing failed")


@router.post("/teach")
async def teach(
    question: str = Form(...),
    session_id: str = Form(...),
    start_index: int | None = Form(None),
    user_data: dict = Depends(verify_jwt_token)
):
    """Teach endpoint with authentication and session ownership verification"""
    try:
        # Verify session ownership
        session_check = await verify_session_ownership(user_data, session_id)
        
        # Validate question input
        if not question or not question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        if len(question) > 5000:
            raise HTTPException(status_code=400, detail="Question too long (max 5000 characters)")
        
        # Validate start_index
        if start_index is not None and start_index < 0:
            raise HTTPException(status_code=400, detail="start_index must be >= 0")
        
        if session_id not in chat_sessions:
            chat_sessions[session_id] = {
                "history": [],
                "mode": "qa",
                "current_index": 0,
                "quizzes": {},
            }

        session = chat_sessions[session_id]

        session_documents = get_session_documents(session_id)

        if not session_documents:
            raise HTTPException(
                status_code=404,
                detail="No uploaded material found for this session."
            )
        
        total_chunks = len(session_documents)
        is_continue = _is_continue_intent(question)
        is_full = _is_full_lesson_intent(question)

        if is_full:
            session["mode"] = "teach"
            session["current_index"] = min(max(start_index or 0, 0), total_chunks)
            session["history"] = []

            start = session["current_index"]
            end = min(start + 3, total_chunks)
            batch = session_documents[start:end]
            session["current_index"] = end

            history_text = _build_history_text(session["history"])

            answer = answer_from_chunks_with_history(
                batch,
                question,
                history_text
            )

            is_unknown_answer = _is_unknown_material_answer(answer)
            if is_unknown_answer:
                answer = _fallback_answer_from_chunks(batch)

            if end < total_chunks:
                answer += "\n\nWould you like me to continue to the next section?"
            else:
                answer += "\n\nYou've reached the end of this material. Ask a question or start over when you're ready."
                session["mode"] = "qa"

            session["history"].append({"role": "user", "content": question})
            session["history"].append({"role": "assistant", "content": answer})

            return {
                "status": "ok",
                "mode": session["mode"],
                "answer": answer,
                "next_index": end,
                "total_chunks": total_chunks,
                "is_complete": end >= total_chunks
            }

        if (session["mode"] == "teach" or start_index is not None) and is_continue:
            if start_index is not None:
                session["mode"] = "teach"
                session["current_index"] = min(max(start_index, 0), total_chunks)

            start = min(session["current_index"], total_chunks)

            if start >= total_chunks:
                session["mode"] = "qa"
                answer = "You've already finished this material. Ask a question or start over when you're ready."

                session["history"].append({"role": "user", "content": question})
                session["history"].append({"role": "assistant", "content": answer})

                return {
                    "status": "ready",
                    "mode": "qa",
                    "answer": answer,
                    "next_index": total_chunks,
                    "total_chunks": total_chunks,
                    "is_complete": True
                }

            end = min(start + 3, total_chunks)
            batch = session_documents[start:end]
            session["current_index"] = end

            history_text = _build_history_text(session["history"])

            answer = answer_from_chunks_with_history(
                batch,
                question,
                history_text
            )

            is_unknown_answer = _is_unknown_material_answer(answer)
            if is_unknown_answer:
                answer = _fallback_answer_from_chunks(batch)

            if end < total_chunks:
                answer += "\n\nWould you like me to continue to the next section?"
            else:
                answer += "\n\nYou've reached the end of this material. Ask a question or start over when you're ready."
                session["mode"] = "qa"

            session["history"].append({"role": "user", "content": question})
            session["history"].append({"role": "assistant", "content": answer})

            return {
                "status": "ready",
                "mode": session["mode"],
                "answer": answer,
                "next_index": end,
                "total_chunks": total_chunks,
                "is_complete": end >= total_chunks
            }

        if session["mode"] == "teach" and not is_continue:
            session["mode"] = "qa"

        retrieved_chunks = retrieve_relevant_chunks(
            question,
            session_id=session_id,
            k=3
        )

        if not retrieved_chunks:
            # Fallback to first material chunks so users can still learn when similarity retrieval misses.
            retrieved_chunks = session_documents[:3]

        history_text = _build_history_text(session["history"])

        answer = answer_from_chunks_with_history(
            retrieved_chunks,
            question,
            history_text
        )

        if _is_unknown_material_answer(answer):
            answer = _fallback_answer_from_chunks(retrieved_chunks)

        session["history"].append({"role": "user", "content": question})
        session["history"].append({"role": "assistant", "content": answer})

        return {
            "status": "ready",
            "mode": "qa",
            "answer": answer
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Teach] Error for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to process question")


@router.post("/quiz/generate")
async def generate_quiz(
    data: QuizGenerationRequest,
    user_data: dict = Depends(verify_jwt_token)
):
    """Generate a customized quiz from either selected parts or full project chunks."""
    try:
        # Verify session ownership
        session_check = await verify_session_ownership(user_data, data.session_id)
        
        session_id = data.session_id
        scope = data.scope.lower()
        difficulty = data.difficulty.lower()
        question_count = data.question_count

        safe_question_count = max(5, min(40, question_count))

        chunks = get_session_documents_with_metadata(session_id)
        if not chunks:
            raise HTTPException(status_code=404, detail="No uploaded material found")

        selected_chunks = chunks
        selected_start_index = 0
        selected_end_index = len(chunks) - 1
        selected_topic_ids = data.topic_ids or []

        if scope == "parts":
            if isinstance(selected_topic_ids, list) and len(selected_topic_ids) > 0:
                topics = extract_topics_from_chunks(chunks)
                topic_by_id = {topic.get("id"): topic for topic in topics}

                topic_ranges = []
                for topic_id in selected_topic_ids:
                    topic = topic_by_id.get(topic_id)
                    if not topic:
                        continue
                    topic_ranges.append((
                        int(topic.get("start_index", 0)),
                        int(topic.get("end_index", 0)),
                    ))

                selected_indexes = []
                for start_index, end_index in topic_ranges:
                    safe_start = max(0, min(start_index, len(chunks) - 1))
                    safe_end = max(safe_start, min(end_index, len(chunks) - 1))
                    selected_indexes.extend(list(range(safe_start, safe_end + 1)))

                deduped_indexes = sorted(list(set(selected_indexes)))
                selected_chunks = [chunks[idx] for idx in deduped_indexes]

                if deduped_indexes:
                    selected_start_index = deduped_indexes[0]
                    selected_end_index = deduped_indexes[-1]
            else:
                start_index = int(data.start_index) if data.start_index is not None else 0
                end_index = int(data.end_index) if data.end_index is not None else start_index + 4

                if end_index < start_index:
                    start_index, end_index = end_index, start_index

                selected_start_index = max(0, start_index)
                selected_end_index = min(end_index, len(chunks) - 1)

                selected_chunks = chunks[selected_start_index:selected_end_index + 1]

        if not selected_chunks:
            raise HTTPException(status_code=400, detail="No chunks available for the selected quiz scope")

        questions = generate_quiz_from_chunks(
            selected_chunks,
            safe_question_count,
            difficulty=difficulty
        )

        if not questions:
            raise HTTPException(status_code=500, detail="Failed to generate quiz questions")

        quiz_id = str(uuid.uuid4())

        if session_id not in chat_sessions:
            chat_sessions[session_id] = {
                "history": [],
                "mode": "qa",
                "current_index": 0,
                "quizzes": {},
            }

        quizzes = chat_sessions[session_id].setdefault("quizzes", {})
        quizzes[quiz_id] = {
            "questions": questions,
            "difficulty": difficulty,
            "scope": scope,
            "start_index": selected_start_index,
            "end_index": selected_end_index,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "attempts": 0,
            "is_finished": False,
            "last_score": None,
            "last_submitted_at": None,
            "last_result": None,
        }

        _persist_quiz_record(
            session_id=session_id,
            quiz_id=quiz_id,
            scope=scope,
            difficulty=difficulty,
            selected_start_index=selected_start_index,
            selected_end_index=selected_end_index,
            questions=questions,
        )

        public_questions = _to_public_quiz_questions(questions)

        return {
            "quiz_id": quiz_id,
            "session_id": session_id,
            "scope": scope,
            "difficulty": difficulty,
            "question_count": len(public_questions),
            "selected_range": {
                "start_index": selected_start_index,
                "end_index": selected_end_index,
            },
            "questions": public_questions,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quiz/history")
async def quiz_history(
    data: TopicsExtractionRequest,
    user_data: dict = Depends(verify_jwt_token)
):
    """Return quiz history for a learning project session."""
    try:
        # Verify session ownership
        session_check = await verify_session_ownership(user_data, data.session_id)
        
        session_id = data.session_id

        history: list[dict[str, Any]] = []

        if supabase:
            try:
                response = (
                    supabase
                    .table("project_quizzes")
                    .select("quiz_id, session_id, scope, difficulty, question_count, start_index, end_index, created_at, attempts, is_finished, last_score, last_submitted_at")
                    .eq("session_id", session_id)
                    .order("created_at", desc=True)
                    .execute()
                )
                rows = response.data or []
                history = [
                    {
                        "quiz_id": row.get("quiz_id"),
                        "session_id": row.get("session_id"),
                        "scope": row.get("scope", "all"),
                        "difficulty": row.get("difficulty", "medium"),
                        "question_count": int(row.get("question_count", 0)),
                        "selected_range": {
                            "start_index": int(row.get("start_index", 0)),
                            "end_index": int(row.get("end_index", 0)),
                        },
                        "created_at": row.get("created_at"),
                        "attempts": int(row.get("attempts", 0)),
                        "is_finished": bool(row.get("is_finished", False)),
                        "last_score": row.get("last_score"),
                        "last_submitted_at": row.get("last_submitted_at"),
                    }
                    for row in rows
                ]
            except Exception as exc:
                print(f"[quiz-persistence] Could not load persisted history for session {session_id}: {exc}")

        if not history:
            session = chat_sessions.get(session_id) or {}
            quizzes = session.get("quizzes") or {}
            for quiz_id, quiz_data in quizzes.items():
                questions = quiz_data.get("questions") or []
                history.append({
                    "quiz_id": quiz_id,
                    "session_id": session_id,
                    "scope": quiz_data.get("scope", "all"),
                    "difficulty": quiz_data.get("difficulty", "medium"),
                    "question_count": len(questions),
                    "selected_range": {
                        "start_index": int(quiz_data.get("start_index", 0)),
                        "end_index": int(quiz_data.get("end_index", 0)),
                    },
                    "created_at": quiz_data.get("created_at"),
                    "attempts": int(quiz_data.get("attempts", 0)),
                    "is_finished": bool(quiz_data.get("is_finished", False)),
                    "last_score": quiz_data.get("last_score"),
                    "last_submitted_at": quiz_data.get("last_submitted_at"),
                })

            history.sort(key=lambda item: item.get("created_at") or "", reverse=True)

        return {
            "session_id": session_id,
            "quizzes": history,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quiz/run")
async def quiz_run(
    data: QuizIdentityRequest,
    user_data: dict = Depends(verify_jwt_token)
):
    """Return a previously generated quiz so learners can retake it."""
    try:
        # Verify session ownership
        session_check = await verify_session_ownership(user_data, data.session_id)
        
        session_id = data.session_id
        quiz_id = data.quiz_id

        quiz_data = _get_persisted_quiz(session_id, quiz_id)
        if not quiz_data:
            session = chat_sessions.get(session_id) or {}
            quizzes = session.get("quizzes") or {}
            quiz_data = quizzes.get(quiz_id)

        if not quiz_data:
            raise HTTPException(status_code=404, detail="Quiz not found")

        questions = quiz_data.get("questions") or []
        public_questions = _to_public_quiz_questions(questions)

        return {
            "quiz_id": quiz_id,
            "session_id": session_id,
            "scope": quiz_data.get("scope", "all"),
            "difficulty": quiz_data.get("difficulty", "medium"),
            "question_count": len(public_questions),
            "selected_range": {
                "start_index": int(quiz_data.get("start_index", 0)),
                "end_index": int(quiz_data.get("end_index", 0)),
            },
            "is_finished": bool(quiz_data.get("is_finished", False)),
            "last_score": quiz_data.get("last_score"),
            "last_submitted_at": quiz_data.get("last_submitted_at"),
            "questions": public_questions,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quiz/result")
async def quiz_result(
    data: QuizIdentityRequest,
    user_data: dict = Depends(verify_jwt_token)
):
    """Return the latest submitted result for a quiz."""
    try:
        await verify_session_ownership(user_data, data.session_id)

        session_id = data.session_id
        quiz_id = data.quiz_id

        quiz_data = _get_persisted_quiz(session_id, quiz_id)
        if not quiz_data:
            session = chat_sessions.get(session_id) or {}
            quizzes = session.get("quizzes") or {}
            quiz_data = quizzes.get(quiz_id)

        if not quiz_data:
            raise HTTPException(status_code=404, detail="Quiz not found")

        last_result = quiz_data.get("last_result")
        if not isinstance(last_result, dict):
            raise HTTPException(status_code=404, detail="No submitted result found for this quiz")

        return {
            "quiz_id": quiz_id,
            "session_id": session_id,
            "is_finished": bool(quiz_data.get("is_finished", False)),
            "attempts": int(quiz_data.get("attempts", 0)),
            "last_score": quiz_data.get("last_score"),
            "last_submitted_at": quiz_data.get("last_submitted_at"),
            "result": last_result,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quiz/delete")
async def delete_quiz(
    data: QuizIdentityRequest,
    user_data: dict = Depends(verify_jwt_token)
):
    """Delete a previously generated quiz from a session."""
    try:
        # Verify session ownership
        session_check = await verify_session_ownership(user_data, data.session_id)
        
        session_id = data.session_id
        quiz_id = data.quiz_id

        deleted_any = False

        if supabase:
            try:
                response = (
                    supabase
                    .table("project_quizzes")
                    .delete()
                    .eq("session_id", session_id)
                    .eq("quiz_id", quiz_id)
                    .execute()
                )
                deleted_any = bool(response.data)
            except Exception as exc:
                print(f"[quiz-persistence] Could not delete persisted quiz {quiz_id}: {exc}")

        session = chat_sessions.get(session_id) or {}
        quizzes = session.get("quizzes") or {}
        if quiz_id in quizzes:
            del quizzes[quiz_id]
            deleted_any = True

        if not deleted_any:
            raise HTTPException(status_code=404, detail="Quiz not found")

        return {
            "status": "ok",
            "session_id": session_id,
            "quiz_id": quiz_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quiz/topics")
async def quiz_topics(
    data: TopicsExtractionRequest,
    user_data: dict = Depends(verify_jwt_token)
):
    """Return student-friendly topic options for quiz customization."""
    try:
        # Verify session ownership
        session_check = await verify_session_ownership(user_data, data.session_id)
        
        session_id = data.session_id

        chunks = get_session_documents_with_metadata(session_id)
        if not chunks:
            raise HTTPException(status_code=404, detail="No uploaded material found")

        topics = extract_topics_from_chunks(chunks)

        return {
            "session_id": session_id,
            "topics": topics,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quiz/submit")
async def submit_quiz(
    data: QuizSubmissionRequest,
    user_data: dict = Depends(verify_jwt_token)
):
    """Submit quiz answers and return score plus learn-again links for wrong answers."""
    try:
        # Verify session ownership
        session_check = await verify_session_ownership(user_data, data.session_id)
        
        session_id = data.session_id
        quiz_id = data.quiz_id
        answers = data.answers

        if not isinstance(answers, dict):
            raise HTTPException(status_code=400, detail="answers must be an object")

        quiz_data = _get_persisted_quiz(session_id, quiz_id)
        session = chat_sessions.get(session_id) or {}
        quizzes = session.get("quizzes") or {}
        memory_quiz_data = quizzes.get(quiz_id)
        if not quiz_data:
            quiz_data = memory_quiz_data

        if not quiz_data:
            raise HTTPException(status_code=404, detail="Quiz not found. Generate quiz again.")

        questions = quiz_data.get("questions") or []
        results = grade_quiz_submission(questions, answers)

        next_attempts = int(quiz_data.get("attempts", 0)) + 1
        submitted_at = datetime.now(timezone.utc).isoformat()

        if memory_quiz_data:
            memory_quiz_data["attempts"] = next_attempts
            memory_quiz_data["is_finished"] = True
            memory_quiz_data["last_score"] = results.get("score")
            memory_quiz_data["last_submitted_at"] = submitted_at
            memory_quiz_data["last_result"] = results

        if supabase:
            try:
                supabase.table("project_quizzes").update({
                    "attempts": next_attempts,
                    "is_finished": True,
                    "last_score": results.get("score"),
                    "last_submitted_at": submitted_at,
                    "last_result": results,
                    "updated_at": submitted_at,
                }).eq("session_id", session_id).eq("quiz_id", quiz_id).execute()
            except Exception as exc:
                print(f"[quiz-persistence] Could not update attempts for quiz {quiz_id}: {exc}")

        return {
            "quiz_id": quiz_id,
            "session_id": session_id,
            **results,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
