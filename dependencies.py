from typing import Dict
from typing import Optional

import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from settings import CHROMA_DIR, GITHUB_MODELS_ENDPOINT, GITHUB_TOKEN


llm_client = OpenAI(api_key=GITHUB_TOKEN, base_url=GITHUB_MODELS_ENDPOINT)
_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
	"""Lazily initialize the embedding model to keep app startup fast."""
	global _embedding_model
	if _embedding_model is None:
		_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
	return _embedding_model

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(name="mindmatrix_chunks")

chat_sessions: Dict[str, dict] = {}
