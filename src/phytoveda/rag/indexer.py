"""Index Ayurvedic texts for RAG retrieval.

Loads plain-text source files, chunks them into semantically coherent passages,
generates embeddings via sentence-transformers, and stores them in a ChromaDB
persistent collection.

Knowledge base:
    1. Charaka Samhita — foundational Ayurvedic medical text
    2. Susruta Samhita — classical surgical/medical text
    3. Ayurvedic Pharmacopoeia of India (API) — pharmacological reference
    4. Vrikshayurveda — plant science texts (Surapala, Varahamihira)

Text file convention:
    texts_dir/
        charaka_samhita/
            chapter_01.txt
            chapter_02.txt
        susruta_samhita/
            ...
        pharmacopoeia/
            ...
        vrikshayurveda/
            ...

    Or flat layout: texts_dir/*.txt (source name derived from filename).
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
from chromadb.config import Settings


@dataclass
class TextChunk:
    """A semantically coherent passage from an Ayurvedic text."""

    text: str
    source: str
    chapter: str
    chunk_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    page: int | None = None


AYURVEDIC_SOURCES = [
    "Charaka Samhita",
    "Susruta Samhita",
    "Ayurvedic Pharmacopoeia of India",
    "Vrikshayurveda",
]

# Map directory names to canonical source names
_DIR_TO_SOURCE: dict[str, str] = {
    "charaka_samhita": "Charaka Samhita",
    "charaka": "Charaka Samhita",
    "susruta_samhita": "Susruta Samhita",
    "susruta": "Susruta Samhita",
    "pharmacopoeia": "Ayurvedic Pharmacopoeia of India",
    "api": "Ayurvedic Pharmacopoeia of India",
    "vrikshayurveda": "Vrikshayurveda",
}


def _normalize_source_name(name: str) -> str:
    """Map a directory or file name to a canonical source name."""
    key = name.lower().replace(" ", "_").replace("-", "_")
    return _DIR_TO_SOURCE.get(key, name)


def chunk_text(
    text: str,
    source: str,
    chapter: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[TextChunk]:
    """Split text into overlapping chunks at paragraph boundaries.

    Strategy: split on double newlines (paragraph boundaries) first,
    then merge small paragraphs into chunks up to chunk_size tokens (approx words).
    Overlap is achieved by repeating the last chunk_overlap words of the previous chunk.
    """
    # Split into paragraphs
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    if not paragraphs:
        return []

    chunks: list[TextChunk] = []
    current_words: list[str] = []
    overlap_buffer: list[str] = []

    for para in paragraphs:
        words = para.split()

        if len(current_words) + len(words) > chunk_size and current_words:
            # Emit current chunk
            chunk_text_str = " ".join(current_words)
            chunks.append(TextChunk(text=chunk_text_str, source=source, chapter=chapter))

            # Keep overlap for next chunk
            overlap_buffer = current_words[-chunk_overlap:] if chunk_overlap > 0 else []
            current_words = list(overlap_buffer) + words
        else:
            current_words.extend(words)

    # Emit final chunk
    if current_words:
        chunk_text_str = " ".join(current_words)
        chunks.append(TextChunk(text=chunk_text_str, source=source, chapter=chapter))

    return chunks


class AyurvedicTextIndexer:
    """Chunk, embed, and index Ayurvedic texts into a ChromaDB vector store.

    Usage:
        indexer = AyurvedicTextIndexer("data/texts")
        chunks = indexer.load_and_chunk()
        indexer.build_index(persist_dir="data/chromadb")
    """

    def __init__(
        self,
        texts_dir: str | Path,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        collection_name: str = "ayurvedic_texts",
    ) -> None:
        self.texts_dir = Path(texts_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name
        self.chunks: list[TextChunk] = []

    def load_and_chunk(self) -> list[TextChunk]:
        """Load all text files from texts_dir and split into chunks.

        Supports hierarchical layout (source_dir/chapter.txt) and
        flat layout (source_chapter.txt).
        """
        self.chunks = []

        if not self.texts_dir.exists():
            return self.chunks

        # Hierarchical: texts_dir/<source>/<chapter>.txt
        for source_dir in sorted(self.texts_dir.iterdir()):
            if source_dir.is_dir() and not source_dir.name.startswith("."):
                source_name = _normalize_source_name(source_dir.name)
                for txt_file in sorted(source_dir.glob("*.txt")):
                    chapter = txt_file.stem
                    text = txt_file.read_text(encoding="utf-8")
                    self.chunks.extend(
                        chunk_text(text, source_name, chapter, self.chunk_size, self.chunk_overlap)
                    )

        # Flat: texts_dir/*.txt
        for txt_file in sorted(self.texts_dir.glob("*.txt")):
            source_name = _normalize_source_name(txt_file.stem)
            text = txt_file.read_text(encoding="utf-8")
            self.chunks.extend(
                chunk_text(text, source_name, txt_file.stem, self.chunk_size, self.chunk_overlap)
            )

        return self.chunks

    def build_index(
        self,
        persist_dir: str | Path = "data/chromadb",
        embedding_function: object | None = None,
    ) -> chromadb.Collection:
        """Build a ChromaDB collection from loaded text chunks.

        Args:
            persist_dir: Directory for persistent ChromaDB storage.
            embedding_function: Optional ChromaDB-compatible embedding function.
                If None, uses ChromaDB's default (all-MiniLM-L6-v2).

        Returns:
            The populated ChromaDB collection.
        """
        if not self.chunks:
            self.load_and_chunk()

        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        # Delete existing collection if rebuilding
        import contextlib

        with contextlib.suppress(Exception):
            client.delete_collection(self.collection_name)

        kwargs: dict = {"name": self.collection_name}
        if embedding_function is not None:
            kwargs["embedding_function"] = embedding_function

        collection = client.get_or_create_collection(**kwargs)

        if not self.chunks:
            return collection

        # Batch insert (ChromaDB recommends batches <= 5000)
        batch_size = 5000
        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i : i + batch_size]
            collection.add(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[{"source": c.source, "chapter": c.chapter} for c in batch],
            )

        return collection
