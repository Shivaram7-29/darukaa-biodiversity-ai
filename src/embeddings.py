from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL

# Load the embedding model (singleton)
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def create_embedding(text: str) -> list[float]:
    """
    Convert text into a numerical vector using sentence-transformers.
    """
    model = _get_model()
    return model.encode(text).tolist()


def create_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Convert a batch of texts into numerical vectors.
    """
    model = _get_model()
    return model.encode(texts).tolist()