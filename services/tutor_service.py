from dependencies import llm_client
from settings import OPENAI_MODEL
import json
import random
import re
import time

from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError


def _create_chat_completion(messages: list[dict], temperature: int = 1, retries: int = 2):
    """Create a chat completion with small retry/backoff for transient provider errors."""
    for attempt in range(retries + 1):
        try:
            return llm_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=temperature,
            )
        except (RateLimitError, APIConnectionError, APITimeoutError, APIError):
            if attempt >= retries:
                raise
            time.sleep(1.2 * (attempt + 1))


def _extract_json_object(raw_content: str) -> dict:
    """Extract JSON object from model output that may contain extra text."""
    if not raw_content:
        return {}

    try:
        return json.loads(raw_content)
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", raw_content)
    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def _clean_text_preview(text: str, max_len: int = 220) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= max_len:
        return cleaned
    return f"{cleaned[:max_len].rstrip()}..."


def _first_sentence(text: str) -> str:
    normalized = " ".join((text or "").split())
    if not normalized:
        return ""

    sentence_match = re.split(r"(?<=[.!?])\s+", normalized, maxsplit=1)
    sentence = sentence_match[0] if sentence_match else normalized
    return _clean_text_preview(sentence, max_len=180)


def _extract_json_array(raw_content: str) -> list:
    if not raw_content:
        return []

    try:
        parsed = json.loads(raw_content)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    match = re.search(r"\[[\s\S]*\]", raw_content)
    if not match:
        return []

    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def extract_topics_from_chunks(chunks: list[dict], max_topics: int = 8) -> list[dict]:
    """Extract user-friendly topic list from uploaded material chunks."""
    if not chunks:
        return []

    safe_max_topics = max(3, min(12, max_topics))
    selected_chunks = chunks[: min(len(chunks), 18)]

    context_lines = []
    for chunk in selected_chunks:
        context_lines.append(
            f"[chunk:{chunk.get('chunk_index', 0)}] {_clean_text_preview(chunk.get('text', ''), max_len=420)}"
        )

    context_text = "\n\n".join(context_lines)

    prompt = f"""
From these study chunks, identify {safe_max_topics} main learning topics.

Rules:
- Output only a JSON array.
- Each item: {{"title": "...", "start_chunk": 0, "end_chunk": 2, "summary": "..."}}
- Titles must be short and student-friendly.
- start_chunk/end_chunk must match chunk indices present in input.
- Cover different parts of the material.

Chunks:
{context_text}
"""

    try:
        response = _create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You extract concise educational topics and return strict JSON arrays."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=1,
        )

        raw_output = response.choices[0].message.content or ""
        parsed_topics = _extract_json_array(raw_output)

        normalized = []
        for idx, topic in enumerate(parsed_topics[:safe_max_topics]):
            if not isinstance(topic, dict):
                continue

            title = str(topic.get("title", "")).strip()
            summary = str(topic.get("summary", "")).strip()

            try:
                start_chunk = int(topic.get("start_chunk", 0))
            except Exception:
                start_chunk = 0

            try:
                end_chunk = int(topic.get("end_chunk", start_chunk))
            except Exception:
                end_chunk = start_chunk

            if not title:
                continue

            if end_chunk < start_chunk:
                start_chunk, end_chunk = end_chunk, start_chunk

            normalized.append({
                "id": f"topic_{idx + 1}",
                "title": title,
                "summary": summary or "Review this part to strengthen your understanding.",
                "start_index": max(0, start_chunk),
                "end_index": max(0, end_chunk),
            })

        if normalized:
            return normalized
    except Exception:
        pass

    # Fallback: split full material into simple ranges.
    chunk_count = len(chunks)
    topic_count = min(safe_max_topics, max(3, min(6, chunk_count)))
    window = max(1, chunk_count // topic_count)

    fallback_topics = []
    cursor = 0
    for idx in range(topic_count):
        start_index = cursor
        if idx == topic_count - 1:
            end_index = chunk_count - 1
        else:
            end_index = min(chunk_count - 1, cursor + window - 1)

        preview_text = chunks[start_index].get("text", "") if start_index < chunk_count else ""
        fallback_topics.append({
            "id": f"topic_{idx + 1}",
            "title": f"Topic {idx + 1}",
            "summary": _first_sentence(preview_text) or "Core concept from your uploaded material.",
            "start_index": start_index,
            "end_index": end_index,
        })
        cursor = end_index + 1
        if cursor >= chunk_count:
            break

    return fallback_topics


def _fallback_quiz_from_chunks(
    chunks: list[dict],
    question_count: int
) -> list[dict]:
    if not chunks:
        return []

    rng = random.Random()
    questions: list[dict] = []

    sampled_chunks = chunks.copy()
    rng.shuffle(sampled_chunks)

    while len(sampled_chunks) < question_count:
        sampled_chunks.extend(chunks)

    sampled_chunks = sampled_chunks[:question_count]

    all_statements = []
    for chunk in chunks:
        statement = _first_sentence(chunk.get("text", ""))
        if statement:
            all_statements.append({
                "chunk_index": chunk.get("chunk_index", 0),
                "statement": statement,
            })

    for idx, chunk in enumerate(sampled_chunks):
        correct_statement = _first_sentence(chunk.get("text", ""))
        if not correct_statement:
            correct_statement = _clean_text_preview(chunk.get("text", ""), max_len=140)

        distractor_pool = [
            item["statement"]
            for item in all_statements
            if item["statement"] != correct_statement
        ]
        rng.shuffle(distractor_pool)
        distractors = distractor_pool[:3]

        while len(distractors) < 3:
            distractors.append("This statement is not directly supported in the uploaded material.")

        options = [correct_statement, *distractors]
        rng.shuffle(options)
        correct_option = options.index(correct_statement)

        questions.append({
            "id": f"q_{idx + 1}",
            "question_type": "single",
            "question": "Which statement is directly supported by the learning material?",
            "options": options,
            "correct_option": correct_option,
            "explanation": "The correct option is taken directly from one of your project chunks.",
            "chunk_index": chunk.get("chunk_index", 0),
            "chunk_preview": _clean_text_preview(chunk.get("text", "")),
        })

    return questions


def generate_quiz_from_chunks(
    chunks: list[dict],
    question_count: int,
    difficulty: str = "medium"
) -> list[dict]:
    """Generate mixed-type quiz questions (single, multiple, text) from selected material chunks."""
    if not chunks:
        return []

    safe_question_count = max(5, min(40, question_count))
    selected_chunks = chunks[:]
    if len(selected_chunks) > 20:
        selected_chunks = selected_chunks[:20]

    context_sections = []
    for chunk in selected_chunks:
        chunk_index = chunk.get("chunk_index", 0)
        chunk_text = _clean_text_preview(chunk.get("text", ""), max_len=1200)
        context_sections.append(f"[chunk:{chunk_index}] {chunk_text}")

    context = "\n\n".join(context_sections)

    prompt = f"""
Create exactly {safe_question_count} quiz questions from the provided chunks.
Difficulty: {difficulty}.

Rules:
- Mix question types across the set:
    - "single": one correct option (radio)
    - "multiple": multiple correct options (checkboxes)
    - "text": short typed answer
- For "single" and "multiple", provide 4 options.
- For "multiple", provide at least 2 correct options.
- For "text", options must be an empty array and provide acceptable_answers.
- Use only facts present in the chunks.
- For each question include the chunk index it is based on.
- Output only valid JSON (no markdown).

Required JSON schema:
{{
  "questions": [
    {{
            "question_type": "single",
      "question": "...",
      "options": ["...", "...", "...", "..."],
      "correct_option": 0,
            "correct_options": [],
            "acceptable_answers": [],
      "explanation": "...",
      "chunk_index": 0
    }}
  ]
}}

Chunks:
{context}
"""

    try:
        response = _create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You generate high-quality educational quizzes and always return strict JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=1,
        )

        raw_output = response.choices[0].message.content or ""
        parsed = _extract_json_object(raw_output)
        parsed_questions = parsed.get("questions") if isinstance(parsed, dict) else None

        if not isinstance(parsed_questions, list) or len(parsed_questions) == 0:
            return _fallback_quiz_from_chunks(selected_chunks, safe_question_count)

        normalized_questions = []

        for idx, question in enumerate(parsed_questions[:safe_question_count]):
            if not isinstance(question, dict):
                continue

            raw_question_type = str(question.get("question_type", "")).strip().lower()
            question_text = str(question.get("question", "")).strip()
            options = question.get("options")
            correct_option = question.get("correct_option")
            correct_options = question.get("correct_options")
            acceptable_answers = question.get("acceptable_answers")
            explanation = str(question.get("explanation", "")).strip()
            chunk_index = question.get("chunk_index", 0)

            inferred_correct_options: list[int] = []
            if isinstance(correct_options, list):
                for value in correct_options:
                    try:
                        normalized_value = int(value)
                    except Exception:
                        continue
                    if 0 <= normalized_value <= 3:
                        inferred_correct_options.append(normalized_value)
                inferred_correct_options = sorted(list(set(inferred_correct_options)))

            inferred_text_answers: list[str] = []
            if isinstance(acceptable_answers, list):
                inferred_text_answers = [
                    str(item).strip().lower()
                    for item in acceptable_answers
                    if str(item).strip()
                ]
                inferred_text_answers = sorted(list(set(inferred_text_answers)))

            if raw_question_type in {"single", "multiple", "text"}:
                question_type = raw_question_type
            elif len(inferred_text_answers) > 0:
                question_type = "text"
            elif len(inferred_correct_options) > 1:
                question_type = "multiple"
            else:
                question_type = "single"

            if not question_text:
                continue

            normalized_options: list[str] = []
            normalized_correct = None
            normalized_correct_options: list[int] = []
            normalized_acceptable_answers: list[str] = []

            if question_type in {"single", "multiple"}:
                if not isinstance(options, list) or len(options) != 4:
                    continue

                normalized_options = [str(option).strip() for option in options]
                if any(not option for option in normalized_options):
                    continue

                if question_type == "single":
                    try:
                        normalized_correct = int(correct_option)
                    except Exception:
                        if len(inferred_correct_options) > 1:
                            question_type = "multiple"
                            normalized_correct_options = inferred_correct_options
                        else:
                            continue

                    if question_type == "single":
                        if normalized_correct < 0 or normalized_correct > 3:
                            if len(inferred_correct_options) > 1:
                                question_type = "multiple"
                                normalized_correct_options = inferred_correct_options
                                normalized_correct = None
                            else:
                                continue

                if question_type == "multiple":
                    normalized_correct_options = inferred_correct_options
                    if len(normalized_correct_options) < 2:
                        continue
            else:
                normalized_acceptable_answers = inferred_text_answers
                if len(normalized_acceptable_answers) == 0:
                    continue

            try:
                normalized_chunk_index = int(chunk_index)
            except Exception:
                normalized_chunk_index = 0

            source_chunk = next(
                (chunk for chunk in selected_chunks if chunk.get("chunk_index") == normalized_chunk_index),
                None
            )

            normalized_questions.append({
                "id": f"q_{idx + 1}",
                "question_type": question_type,
                "question": question_text,
                "options": normalized_options,
                "correct_option": normalized_correct,
                "correct_options": normalized_correct_options,
                "acceptable_answers": normalized_acceptable_answers,
                "explanation": explanation or "Review the referenced chunk for the key concept.",
                "chunk_index": normalized_chunk_index,
                "chunk_preview": _clean_text_preview((source_chunk or {}).get("text", "")),
            })

        if len(normalized_questions) < safe_question_count:
            fallback_questions = _fallback_quiz_from_chunks(
                selected_chunks,
                safe_question_count - len(normalized_questions)
            )
            normalized_questions.extend(fallback_questions)

        return normalized_questions[:safe_question_count]
    except Exception:
        return _fallback_quiz_from_chunks(selected_chunks, safe_question_count)


def grade_quiz_submission(
    questions: list[dict],
    answers: dict
) -> dict:
    """Grade submitted quiz answers and include learn-again details for wrong answers."""
    if not questions:
        return {
            "score": 0,
            "correct_count": 0,
            "wrong_count": 0,
            "total_questions": 0,
            "questions": [],
            "wrong_answers": [],
        }

    result_rows = []
    wrong_answers = []
    correct_count = 0

    for question in questions:
        question_id = question.get("id")
        question_type = question.get("question_type", "single")
        correct_option = question.get("correct_option")
        correct_options = question.get("correct_options") or []
        acceptable_answers = question.get("acceptable_answers") or []
        user_answer = answers.get(question_id)

        normalized_user_answer = user_answer
        is_correct = False

        if question_type == "single":
            try:
                user_answer_int = int(user_answer)
            except Exception:
                user_answer_int = -1
            normalized_user_answer = user_answer_int
            is_correct = user_answer_int == correct_option
        elif question_type == "multiple":
            if isinstance(user_answer, list):
                normalized_values = []
                for value in user_answer:
                    try:
                        normalized_value = int(value)
                    except Exception:
                        continue
                    if 0 <= normalized_value <= 3:
                        normalized_values.append(normalized_value)
                normalized_values = sorted(list(set(normalized_values)))
            else:
                normalized_values = []
            normalized_user_answer = normalized_values
            normalized_correct_values = sorted(list(set([
                int(value)
                for value in correct_options
                if isinstance(value, int) or (isinstance(value, str) and value.isdigit())
            ])))
            is_correct = normalized_values == normalized_correct_values
        else:
            normalized_text = str(user_answer or "").strip().lower()
            normalized_user_answer = normalized_text
            is_correct = any(
                candidate in normalized_text or normalized_text in candidate
                for candidate in [str(item).strip().lower() for item in acceptable_answers if str(item).strip()]
            ) and len(normalized_text) > 0

        if is_correct:
            correct_count += 1

        row = {
            "id": question_id,
            "question_type": question_type,
            "question": question.get("question", ""),
            "options": question.get("options", []),
            "user_answer": normalized_user_answer,
            "correct_option": correct_option,
            "correct_options": correct_options,
            "acceptable_answers": acceptable_answers,
            "is_correct": is_correct,
            "explanation": question.get("explanation", ""),
            "chunk_index": question.get("chunk_index", 0),
            "chunk_preview": question.get("chunk_preview", ""),
        }
        result_rows.append(row)

        if not is_correct:
            wrong_answers.append({
                "id": question_id,
                "question_type": question_type,
                "question": question.get("question", ""),
                "your_answer": normalized_user_answer,
                "correct_option": correct_option,
                "correct_options": correct_options,
                "acceptable_answers": acceptable_answers,
                "correct_text": (
                    question.get("options", [""])[correct_option]
                    if isinstance(correct_option, int)
                    and 0 <= correct_option < len(question.get("options", []))
                    else ""
                ),
                "chunk_index": question.get("chunk_index", 0),
                "chunk_preview": question.get("chunk_preview", ""),
                "learn_again": {
                    "chunk_index": question.get("chunk_index", 0),
                    "label": "Learn this part again"
                }
            })

    total_questions = len(questions)
    wrong_count = total_questions - correct_count
    score = round((correct_count / total_questions) * 100) if total_questions > 0 else 0

    return {
        "score": score,
        "correct_count": correct_count,
        "wrong_count": wrong_count,
        "total_questions": total_questions,
        "questions": result_rows,
        "wrong_answers": wrong_answers,
    }


def answer_from_chunks_with_history(
    chunks: list[str],
    question: str,
    history: str
) -> str:
    if not chunks:
        return "I don't know based on the provided material."

    context = "\n\n".join(chunks)

    prompt = f"""
You are an AI tutor.

Use the lecture context as the SOURCE OF TRUTH.
Use the conversation history only to understand how to respond.

If the answer is not in the lecture context, say:
"I don't know based on the provided material."

--- LECTURE CONTEXT ---
{context}

--- CONVERSATION HISTORY ---
{history}

Current question:
{question}
"""

    try:
        response = _create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert AI tutor who explains concepts clearly step-by-step."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=1,
        )
        return response.choices[0].message.content
    except RateLimitError:
        return (
            "I hit a temporary rate limit from the model provider. "
            "Please wait a few seconds and ask again."
        )
    except (APIConnectionError, APITimeoutError, APIError):
        return (
            "I could not reach the model provider right now. "
            "Please retry in a moment."
        )
