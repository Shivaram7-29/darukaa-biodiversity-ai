"""
Remote embedding module using Google Gemini's embedding API via google-genai SDK.
Zero local ML model weights are loaded into RAM (avoiding PyTorch and ONNX memory overhead),
enabling effortless deployment on constrained platforms like Render's 512 MB free tier.
"""

import os
from chromadb.utils.embedding_functions import GoogleGenaiEmbeddingFunction
from src.config import EMBEDDING_MODEL, EMBEDDING_DIMENSION, GOOGLE_API_KEY

# Singleton embedding function instance
_embedding_function = None


def get_embedding_function() -> GoogleGenaiEmbeddingFunction:
    """Return the singleton ChromaDB Google GenAI embedding function."""
    global _embedding_function
    if _embedding_function is None:
        api_key = GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable is required.")

        _embedding_function = GoogleGenaiEmbeddingFunction(
            model_name=EMBEDDING_MODEL,
            dimension=EMBEDDING_DIMENSION,
            api_key_env_var="GOOGLE_API_KEY" if (GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY")) else "GEMINI_API_KEY",
        )
    return _embedding_function


def create_embedding(text: str) -> list[float]:
    """
    Convert text into a numerical vector using Google's remote Gemini embedding API.
    """
    ef = get_embedding_function()
    vec = ef([text])[0]
    return vec.tolist() if hasattr(vec, "tolist") else list(vec)


def create_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Convert a batch of texts into numerical vectors using Google's remote Gemini embedding API.
    """
    if not texts:
        return []
    ef = get_embedding_function()
    vecs = ef(texts)
    return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vecs]