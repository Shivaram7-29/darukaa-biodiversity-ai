"""
RAG retriever: manages ChromaDB vector store and performs similarity search
for relevant environmental knowledge.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
from src.config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR
from src.embeddings import create_embedding, create_embeddings_batch, get_embedding_function
from src.knowledge_base import load_knowledge_base, prepare_documents_for_vectordb


class EnvironmentalRetriever:
    """Manages the ChromaDB vector store for environmental knowledge retrieval."""

    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._ef = get_embedding_function()
        self._collection = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._ef,
        )
        # Ingest knowledge if the collection is empty
        if self._collection.count() == 0:
            self._ingest_knowledge()

    def _ingest_knowledge(self):
        """Load knowledge base and ingest into ChromaDB."""
        knowledge = load_knowledge_base()
        documents, metadatas, ids = prepare_documents_for_vectordb(knowledge)
        embeddings = create_embeddings_batch(documents)

        # ChromaDB has batch size limits; add in chunks
        batch_size = 40
        for i in range(0, len(documents), batch_size):
            end = min(i + batch_size, len(documents))
            self._collection.add(
                documents=documents[i:end],
                metadatas=metadatas[i:end],
                ids=ids[i:end],
                embeddings=embeddings[i:end],
            )

        print(f"[Retriever] Ingested {len(documents)} documents into ChromaDB.")

    def retrieve(self, query: str, n_results: int = 8, filter_dict: dict | None = None) -> list[dict]:
        """
        Retrieve the most relevant knowledge documents for a query.

        Args:
            query: The user's question or topic.
            n_results: Number of results to return.
            filter_dict: Optional ChromaDB metadata filter.

        Returns:
            List of dicts with keys: document, metadata, distance.
        """
        query_embedding = create_embedding(query)

        search_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, self._collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if filter_dict:
            search_kwargs["where"] = filter_dict

        results = self._collection.query(**search_kwargs)

        retrieved = []
        for i in range(len(results["documents"][0])):
            retrieved.append({
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })

        return retrieved

    def retrieve_by_variables(self, variables: list[str], n_results: int = 8) -> list[dict]:
        """
        Retrieve knowledge related to specific environmental variables.
        Combines text query with variable names for better retrieval.
        """
        query = f"Environmental factors: {', '.join(variables)}. " \
                f"How do these variables interact and what recommendations exist?"
        return self.retrieve(query, n_results=n_results)

    def get_collection_info(self) -> dict:
        """Return information about the vector store collection."""
        return {
            "name": self._collection.name,
            "count": self._collection.count(),
        }

    def rebuild_index(self):
        """Delete and rebuild the vector store from knowledge base."""
        self._client.delete_collection(CHROMA_COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._ef,
        )
        self._ingest_knowledge()
