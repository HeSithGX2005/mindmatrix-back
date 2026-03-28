from typing import Optional

from dependencies import collection
from services.document_service import embed_text


def get_session_documents(session_id: str) -> list[str]:
    results = collection.get(
        where={"session_id": session_id},
        include=["documents", "metadatas"]
    )

    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []

    if not documents:
        return []

    ordered = sorted(
        zip(documents, metadatas),
        key=lambda item: (
            item[1].get("filename", ""),
            item[1].get("chunk_id", 0)
        )
    )

    return [document for document, _ in ordered]


def get_session_documents_with_metadata(session_id: str) -> list[dict]:
    """Return ordered session chunks with chunk index metadata."""
    results = collection.get(
        where={"session_id": session_id},
        include=["documents", "metadatas"]
    )

    documents = results.get("documents") or []
    metadatas = results.get("metadatas") or []

    if not documents:
        return []

    ordered = sorted(
        zip(documents, metadatas),
        key=lambda item: (
            item[1].get("filename", ""),
            item[1].get("chunk_id", 0)
        )
    )

    normalized = []
    for ordered_index, (document, metadata) in enumerate(ordered):
        normalized.append({
            "text": document,
            "filename": metadata.get("filename", ""),
            "chunk_index": ordered_index,
            "source_chunk_id": metadata.get("chunk_id", 0),
        })

    return normalized


def retrieve_relevant_chunks(
    question: str,
    session_id: Optional[str] = None,
    k: int = 3
) -> list[str]:
    """
    Retrieve top-k relevant chunks from ChromaDB using vector similarity.
    """
    question_embedding = embed_text(question)

    if not question_embedding:
        return []

    query_kwargs = {
        "query_embeddings": [question_embedding],
        "n_results": k
    }

    if session_id:
        query_kwargs["where"] = {"session_id": session_id}

    results = collection.query(**query_kwargs)

    docs = results.get("documents")

    if not docs or not docs[0]:
        return []

    return docs[0]
