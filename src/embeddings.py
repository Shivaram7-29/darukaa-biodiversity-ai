"""
Lightweight embedding module using ChromaDB's built-in ONNX embedding function.
Runs all-MiniLM-L6-v2 directly via onnxruntime without loading PyTorch or sentence-transformers,
substantially reducing memory footprint for low-memory environments (e.g. Render 512 MB).
"""

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

# Singleton embedding function instance
_embedding_function = None


def get_embedding_function() -> DefaultEmbeddingFunction:
    """Return the singleton ChromaDB default ONNX embedding function."""
    global _embedding_function
    if _embedding_function is None:
        _embedding_function = DefaultEmbeddingFunction()
    return _embedding_function


def create_embedding(text: str) -> list[float]:
    """
    Convert text into a numerical vector using Chroma's lightweight default ONNX embedding function.
    """
    ef = get_embedding_function()
    vec = ef([text])[0]
    return vec.tolist() if hasattr(vec, "tolist") else list(vec)


def create_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Convert a batch of texts into numerical vectors using Chroma's lightweight default ONNX embedding function.
    """
    if not texts:
        return []
    ef = get_embedding_function()
    vecs = ef(texts)
    return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vecs]