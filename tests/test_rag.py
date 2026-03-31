"""Tests for the RAG pipeline: indexer, retriever, and report generator.

Uses temporary directories and ChromaDB ephemeral storage — no external
APIs or large text files required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phytoveda.rag.indexer import (
    AyurvedicTextIndexer,
    TextChunk,
    _normalize_source_name,
    chunk_text,
)
from phytoveda.rag.report_generator import (
    BotanicalReport,
    ReportGenerator,
    build_report_prompt,
)
from phytoveda.rag.retriever import (
    AyurvedicRetriever,
    RetrievalResult,
    build_query,
)
from phytoveda.vrikshayurveda.mapper import Dosha, DoshaAssessment


# ─── Helpers ────────────────────────────────────────────────────────────────


def _make_dosha(dosha: Dosha = Dosha.PITTA) -> DoshaAssessment:
    return DoshaAssessment(
        dosha=dosha,
        confidence=0.85,
        cv_features=["RGB shifts"],
        classical_symptoms=["Yellowing of leaves"],
        treatments=["Yashtimadhu decoction"],
        contraindications=["Avoid heating"],
    )


def _write_texts(tmp_path: Path) -> Path:
    """Create sample Ayurvedic text files in hierarchical layout."""
    texts_dir = tmp_path / "texts"

    # Charaka Samhita chapters
    charaka_dir = texts_dir / "charaka_samhita"
    charaka_dir.mkdir(parents=True)
    (charaka_dir / "chapter_01.txt").write_text(
        "The Rasa of Neem (Azadirachta indica) is Tikta (bitter) and Kashaya (astringent).\n\n"
        "Its Guna includes Laghu (light) and Ruksha (dry). The Virya is Sheeta (cooling).\n\n"
        "Vipaka is Katu (pungent post-digestive effect). Neem pacifies Pitta and Kapha doshas.\n\n"
        "It is used in the treatment of skin diseases, fever, and inflammatory conditions."
    )
    (charaka_dir / "chapter_02.txt").write_text(
        "Haritaki (Terminalia chebula) is considered the king of medicines in Ayurveda.\n\n"
        "Its Rasa encompasses five tastes except Lavana (salty). Virya is Ushna (hot).\n\n"
        "Haritaki promotes longevity and is one of the three fruits in Triphala formulation."
    )

    # Vrikshayurveda
    vrik_dir = texts_dir / "vrikshayurveda"
    vrik_dir.mkdir(parents=True)
    (vrik_dir / "plant_pathology.txt").write_text(
        "According to Surapala's Vrikshayurveda, plant diseases arise from Tridosha imbalance.\n\n"
        "Vataja Vyadhi manifests as desiccation and geometric deformation of leaves.\n\n"
        "Pittaja Vyadhi shows as yellowing chlorosis and necrotic burn lesions.\n\n"
        "Kaphaja Vyadhi presents as powdery deposits and hypertrophic growth.\n\n"
        "Treatment follows the principle of balancing the aggravated Dosha."
    )

    return texts_dir


# ─── Text Chunking Tests ───────────────────────────────────────────────────


class TestChunkText:
    def test_basic_chunking(self) -> None:
        text = "Paragraph one about Neem.\n\nParagraph two about Tulsi.\n\nParagraph three about Haritaki."
        chunks = chunk_text(text, source="Test", chapter="ch1", chunk_size=10, chunk_overlap=0)
        assert len(chunks) >= 1
        assert all(isinstance(c, TextChunk) for c in chunks)
        assert all(c.source == "Test" for c in chunks)

    def test_overlap_preserves_context(self) -> None:
        # Create text with enough paragraphs to trigger multiple chunks
        paragraphs = [f"Word{i} " * 20 for i in range(5)]
        text = "\n\n".join(paragraphs)
        chunks = chunk_text(text, "src", "ch", chunk_size=30, chunk_overlap=5)
        if len(chunks) > 1:
            # Last words of chunk N should appear at start of chunk N+1
            last_words = chunks[0].text.split()[-5:]
            next_start = chunks[1].text.split()[:5]
            assert last_words == next_start

    def test_empty_text(self) -> None:
        assert chunk_text("", "src", "ch") == []
        assert chunk_text("   \n\n   ", "src", "ch") == []

    def test_single_paragraph(self) -> None:
        chunks = chunk_text("A single paragraph.", "src", "ch")
        assert len(chunks) == 1
        assert chunks[0].text == "A single paragraph."

    def test_chunk_ids_unique(self) -> None:
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = chunk_text(text, "src", "ch", chunk_size=5, chunk_overlap=0)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))


class TestNormalizeSourceName:
    def test_known_sources(self) -> None:
        assert _normalize_source_name("charaka_samhita") == "Charaka Samhita"
        assert _normalize_source_name("susruta_samhita") == "Susruta Samhita"
        assert _normalize_source_name("pharmacopoeia") == "Ayurvedic Pharmacopoeia of India"
        assert _normalize_source_name("vrikshayurveda") == "Vrikshayurveda"

    def test_unknown_passthrough(self) -> None:
        assert _normalize_source_name("custom_text") == "custom_text"


# ─── Indexer Tests ──────────────────────────────────────────────────────────


class TestAyurvedicTextIndexer:
    def test_load_and_chunk_hierarchical(self, tmp_path: Path) -> None:
        texts_dir = _write_texts(tmp_path)
        indexer = AyurvedicTextIndexer(texts_dir, chunk_size=100)
        chunks = indexer.load_and_chunk()
        assert len(chunks) > 0
        sources = {c.source for c in chunks}
        assert "Charaka Samhita" in sources
        assert "Vrikshayurveda" in sources

    def test_load_flat_layout(self, tmp_path: Path) -> None:
        texts_dir = tmp_path / "flat_texts"
        texts_dir.mkdir()
        (texts_dir / "vrikshayurveda.txt").write_text(
            "Vata causes desiccation.\n\nPitta causes yellowing."
        )
        indexer = AyurvedicTextIndexer(texts_dir)
        chunks = indexer.load_and_chunk()
        assert len(chunks) >= 1
        assert chunks[0].source == "Vrikshayurveda"

    def test_empty_directory(self, tmp_path: Path) -> None:
        texts_dir = tmp_path / "empty"
        texts_dir.mkdir()
        indexer = AyurvedicTextIndexer(texts_dir)
        assert indexer.load_and_chunk() == []

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        indexer = AyurvedicTextIndexer(tmp_path / "nope")
        assert indexer.load_and_chunk() == []

    def test_build_index(self, tmp_path: Path) -> None:
        texts_dir = _write_texts(tmp_path)
        indexer = AyurvedicTextIndexer(texts_dir, chunk_size=50)
        indexer.load_and_chunk()

        persist_dir = tmp_path / "chromadb"
        collection = indexer.build_index(persist_dir=persist_dir)
        assert collection.count() > 0
        assert collection.count() == len(indexer.chunks)

    def test_rebuild_index_replaces(self, tmp_path: Path) -> None:
        texts_dir = _write_texts(tmp_path)
        indexer = AyurvedicTextIndexer(texts_dir, chunk_size=50)
        indexer.load_and_chunk()
        persist_dir = tmp_path / "chromadb"

        col1 = indexer.build_index(persist_dir=persist_dir)
        count1 = col1.count()

        col2 = indexer.build_index(persist_dir=persist_dir)
        assert col2.count() == count1  # Rebuilt, not appended


# ─── Retriever Tests ────────────────────────────────────────────────────────


class TestBuildQuery:
    def test_basic_query(self) -> None:
        q = build_query("Azadirachta indica", "Bacterial Spot")
        assert "Azadirachta indica" in q
        assert "Bacterial Spot" in q

    def test_query_with_dosha(self) -> None:
        dosha = _make_dosha(Dosha.PITTA)
        q = build_query("Neem", "Yellow Leaf Disease", dosha)
        assert "Pittaja Vyadhi" in q
        assert "Rasa" in q


class TestAyurvedicRetriever:
    def _build_index(self, tmp_path: Path) -> Path:
        """Build a test ChromaDB index and return persist_dir."""
        texts_dir = _write_texts(tmp_path)
        indexer = AyurvedicTextIndexer(texts_dir, chunk_size=50)
        indexer.load_and_chunk()
        persist_dir = tmp_path / "chromadb"
        indexer.build_index(persist_dir=persist_dir)
        return persist_dir

    def test_retrieve_basic(self, tmp_path: Path) -> None:
        persist_dir = self._build_index(tmp_path)
        retriever = AyurvedicRetriever(persist_dir=persist_dir)
        results = retriever.retrieve("Neem medicinal properties", top_k=3)
        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert all(r.source for r in results)

    def test_retrieve_relevance(self, tmp_path: Path) -> None:
        persist_dir = self._build_index(tmp_path)
        retriever = AyurvedicRetriever(persist_dir=persist_dir)
        results = retriever.retrieve("Neem Azadirachta indica Tikta bitter", top_k=3)
        # The Charaka chapter about Neem should be top result
        assert any("Neem" in r.text or "Azadirachta" in r.text for r in results)

    def test_retrieve_with_source_filter(self, tmp_path: Path) -> None:
        persist_dir = self._build_index(tmp_path)
        retriever = AyurvedicRetriever(persist_dir=persist_dir)
        results = retriever.retrieve("plant disease", top_k=10, source_filter="Vrikshayurveda")
        assert all(r.source == "Vrikshayurveda" for r in results)

    def test_retrieve_for_diagnosis(self, tmp_path: Path) -> None:
        persist_dir = self._build_index(tmp_path)
        retriever = AyurvedicRetriever(persist_dir=persist_dir)
        dosha = _make_dosha(Dosha.PITTA)
        results = retriever.retrieve_for_diagnosis(
            "Azadirachta indica", "Yellow Leaf Disease", dosha, top_k=3
        )
        assert len(results) > 0

    def test_empty_results(self, tmp_path: Path) -> None:
        """Query against an index should always return something (even if low relevance)."""
        persist_dir = self._build_index(tmp_path)
        retriever = AyurvedicRetriever(persist_dir=persist_dir)
        results = retriever.retrieve("completely unrelated topic about cars", top_k=2)
        # ChromaDB returns results even for unrelated queries (just low relevance)
        assert isinstance(results, list)


# ─── Report Generator Tests ────────────────────────────────────────────────


class TestBuildReportPrompt:
    def test_prompt_structure(self) -> None:
        dosha = _make_dosha()
        contexts = [
            RetrievalResult("Neem is Tikta and Kashaya.", "Charaka Samhita", "ch1", 0.1)
        ]
        prompt = build_report_prompt("Azadirachta indica", "Bacterial Spot", dosha, contexts, 0.92)
        assert "Azadirachta indica" in prompt
        assert "Bacterial Spot" in prompt
        assert "92.0%" in prompt
        assert "Charaka Samhita" in prompt
        assert "Neem is Tikta" in prompt

    def test_prompt_no_contexts(self) -> None:
        dosha = _make_dosha()
        prompt = build_report_prompt("Neem", "Healthy", dosha, [])
        assert "No retrieved contexts available" in prompt


class TestReportGenerator:
    def test_offline_report_healthy(self) -> None:
        gen = ReportGenerator()
        dosha = _make_dosha(Dosha.HEALTHY)
        dosha.treatments = ["No intervention required"]
        report = gen.generate_offline("Azadirachta indica", "Healthy", dosha)
        assert isinstance(report, BotanicalReport)
        assert report.species_name == "Azadirachta indica"
        assert "Suitable" in report.procurement_quality
        assert report.pathology_diagnosis == "Healthy"

    def test_offline_report_diseased(self) -> None:
        gen = ReportGenerator()
        dosha = _make_dosha(Dosha.PITTA)
        report = gen.generate_offline("Azadirachta indica", "Bacterial Spot", dosha)
        assert "Conditional" in report.procurement_quality
        assert "Bacterial Spot" in report.pathology_diagnosis
        assert len(report.treatments) > 0

    def test_report_to_json(self) -> None:
        gen = ReportGenerator()
        dosha = _make_dosha()
        report = gen.generate_offline("Neem", "Yellow Leaf Disease", dosha)
        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert parsed["species_name"] == "Neem"
        assert parsed["pathology_diagnosis"] == "Yellow Leaf Disease"

    def test_report_to_markdown(self) -> None:
        gen = ReportGenerator()
        dosha = _make_dosha()
        report = gen.generate_offline("Neem", "Powdery Mildew", dosha)
        md = report.to_markdown()
        assert "# Botanical Report: Neem" in md
        assert "Powdery Mildew" in md
        assert "Procurement Quality" in md

    def test_parse_response(self) -> None:
        gen = ReportGenerator()
        dosha = _make_dosha()
        mock_response = json.dumps({
            "species_family": "Meliaceae",
            "sanskrit_name": "Nimba",
            "common_names": ["Neem", "Margosa"],
            "rasa": "Tikta, Kashaya",
            "guna": "Laghu, Ruksha",
            "virya": "Sheeta",
            "vipaka": "Katu",
            "health_status": "Affected by bacterial infection",
            "dosha_assessment": "Pittaja Vyadhi — thermal metabolic distress",
            "procurement_quality": "Conditional — requires treatment",
            "full_report_text": "Detailed analysis of the specimen...",
        })
        report = gen._parse_response("Neem", "Bacterial Spot", dosha, mock_response)
        assert report.species_family == "Meliaceae"
        assert report.sanskrit_name == "Nimba"
        assert report.rasa == "Tikta, Kashaya"
        assert report.virya == "Sheeta"

    def test_parse_response_invalid_json(self) -> None:
        gen = ReportGenerator()
        dosha = _make_dosha()
        with pytest.raises(RuntimeError, match="Failed to parse"):
            gen._parse_response("Neem", "Healthy", dosha, "not valid json {{{")
