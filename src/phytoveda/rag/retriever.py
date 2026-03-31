"""Semantic retrieval over indexed Ayurvedic texts for RAG pipeline.

Queries the ChromaDB vector store built by the indexer. Constructs
structured queries from model outputs (species + pathology + Dosha)
and returns ranked passages with relevance scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings

from phytoveda.vrikshayurveda.mapper import DoshaAssessment


@dataclass
class RetrievalResult:
    """A retrieved passage with relevance score and source metadata."""

    text: str
    source: str
    chapter: str
    score: float  # Lower distance = more relevant for ChromaDB


def build_query(
    species_name: str,
    pathology_label: str,
    dosha: DoshaAssessment | None = None,
) -> str:
    """Construct a natural language query from model outputs.

    Combines species identity, pathology diagnosis, and Dosha classification
    into a single query string optimized for semantic retrieval.
    """
    parts = [
        f"Medicinal plant {species_name}",
        f"diagnosed with {pathology_label}",
    ]

    if dosha:
        parts.append(f"Ayurvedic classification {dosha.dosha.value}")
        if dosha.classical_symptoms:
            parts.append(f"symptoms include {dosha.classical_symptoms[0]}")

    parts.append("traditional Ayurvedic treatment Rasa Guna Virya Vipaka medicinal properties")

    return " ".join(parts)


class AyurvedicRetriever:
    """Semantic search over vector-indexed Ayurvedic knowledge base.

    Connects to the ChromaDB collection built by AyurvedicTextIndexer
    and retrieves the most relevant passages for a given query.

    Usage:
        retriever = AyurvedicRetriever(persist_dir="data/chromadb")
        results = retriever.retrieve("Neem bacterial spot Pitta treatment")
    """

    def __init__(
        self,
        persist_dir: str | Path = "data/chromadb",
        collection_name: str = "ayurvedic_texts",
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self._collection: chromadb.Collection | None = None

    def _get_collection(self) -> chromadb.Collection:
        """Lazily connect to ChromaDB collection."""
        if self._collection is None:
            client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = client.get_collection(self.collection_name)
        return self._collection

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        source_filter: str | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve most relevant passages for a given query.

        Args:
            query: Natural language query string.
            top_k: Number of results to return.
            source_filter: Optional filter to restrict results to a specific
                source text (e.g., "Charaka Samhita").

        Returns:
            List of RetrievalResult sorted by relevance (best first).
        """
        collection = self._get_collection()

        where_filter = {"source": source_filter} if source_filter else None

        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter,
        )

        retrieval_results: list[RetrievalResult] = []

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances, strict=True):
            retrieval_results.append(
                RetrievalResult(
                    text=doc,
                    source=meta.get("source", "Unknown"),
                    chapter=meta.get("chapter", "Unknown"),
                    score=dist,
                )
            )

        return retrieval_results

    def retrieve_for_diagnosis(
        self,
        species_name: str,
        pathology_label: str,
        dosha: DoshaAssessment | None = None,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Retrieve passages relevant to a specific diagnosis.

        Convenience method that builds a query from model outputs and retrieves.
        """
        query = build_query(species_name, pathology_label, dosha)
        return self.retrieve(query, top_k=top_k)
