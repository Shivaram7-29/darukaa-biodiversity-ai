"""
Knowledge base loader: reads the JSON knowledge file and prepares documents
for ingestion into ChromaDB.
"""

import json
from pathlib import Path
from src.config import KNOWLEDGE_BASE_PATH


def load_knowledge_base() -> list[dict]:
    """Load the environmental knowledge base from JSON."""
    kb_path = Path(KNOWLEDGE_BASE_PATH)
    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge base not found at {kb_path}")

    with open(kb_path, "r", encoding="utf-8") as f:
        knowledge = json.load(f)

    return knowledge


def prepare_documents_for_vectordb(knowledge: list[dict]) -> tuple[list[str], list[dict], list[str]]:
    """
    Convert knowledge entries into documents suitable for vector database ingestion.

    Returns:
        tuple of (documents, metadatas, ids)
        - documents: list of text strings to embed
        - metadatas: list of metadata dicts for each document
        - ids: list of unique IDs for each document
    """
    documents = []
    metadatas = []
    ids = []

    for entry in knowledge:
        entry_id = entry.get("id", entry["topic"].replace(" ", "_"))

        # -- Main knowledge document --
        main_doc = _build_main_document(entry)
        documents.append(main_doc)
        metadatas.append({
            "id": entry_id,
            "topic": entry["topic"],
            "category": entry.get("category", "general"),
            "doc_type": "knowledge",
            "variables": ", ".join(entry.get("environmental_variables", [])),
            "source": entry.get("source", ""),
        })
        ids.append(f"{entry_id}_knowledge")

        # -- Individual recommendation documents --
        for i, rec in enumerate(entry.get("recommended_actions", [])):
            if isinstance(rec, dict):
                rec_doc = _build_recommendation_document(entry, rec)
                documents.append(rec_doc)
                metadatas.append({
                    "id": entry_id,
                    "topic": entry["topic"],
                    "category": entry.get("category", "general"),
                    "doc_type": "recommendation",
                    "action": rec.get("action", ""),
                    "confidence": rec.get("confidence", ""),
                    "source": entry.get("source", ""),
                })
                ids.append(f"{entry_id}_rec_{i}")
            elif isinstance(rec, str):
                # Handle legacy format (plain string recommendations)
                rec_doc = f"Topic: {entry['topic']}\nRecommendation: {rec}"
                documents.append(rec_doc)
                metadatas.append({
                    "id": entry_id,
                    "topic": entry["topic"],
                    "category": entry.get("category", "general"),
                    "doc_type": "recommendation",
                    "action": rec,
                    "source": entry.get("source", ""),
                })
                ids.append(f"{entry_id}_rec_{i}")

        # -- Interaction documents (how variables relate) --
        interactions = entry.get("interactions", {})
        if interactions:
            interaction_doc = _build_interaction_document(entry, interactions)
            documents.append(interaction_doc)
            metadatas.append({
                "id": entry_id,
                "topic": entry["topic"],
                "category": entry.get("category", "general"),
                "doc_type": "interaction",
                "variables": ", ".join(interactions.keys()),
                "source": entry.get("source", ""),
            })
            ids.append(f"{entry_id}_interactions")

    return documents, metadatas, ids


def _build_main_document(entry: dict) -> str:
    """Build a comprehensive text document from a knowledge entry."""
    parts = [
        f"Topic: {entry['topic']}",
        f"Category: {entry.get('category', 'general')}",
        f"Environmental Variables: {', '.join(entry.get('environmental_variables', []))}",
        f"Knowledge: {entry['knowledge']}",
    ]

    thresholds = entry.get("thresholds", {})
    if thresholds:
        threshold_lines = [f"  {k}: {v}" for k, v in thresholds.items()]
        parts.append("Thresholds:\n" + "\n".join(threshold_lines))

    if entry.get("source"):
        parts.append(f"Source: {entry['source']}")

    return "\n".join(parts)


def _build_recommendation_document(entry: dict, rec: dict) -> str:
    """Build a text document from a recommendation."""
    parts = [
        f"Topic: {entry['topic']}",
        f"Action: {rec.get('action', '')}",
        f"Scientific Basis: {rec.get('scientific_basis', '')}",
        f"Metrics Affected: {', '.join(rec.get('metrics_affected', []))}",
        f"Time Horizon: {rec.get('time_horizon', '')}",
        f"Measurable Improvement: {rec.get('measurable_improvement', '')}",
        f"Confidence: {rec.get('confidence', '')}",
        f"Source: {entry.get('source', '')}",
    ]
    return "\n".join(parts)


def _build_interaction_document(entry: dict, interactions: dict) -> str:
    """Build a text document describing variable interactions."""
    parts = [
        f"Topic: {entry['topic']}",
        "Environmental Variable Interactions:",
    ]
    for var, description in interactions.items():
        parts.append(f"  {var}: {description}")

    return "\n".join(parts)
