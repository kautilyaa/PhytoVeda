"""Tests for the traceability module: event logging, GPS, hash chain."""

from __future__ import annotations

from pathlib import Path

import pytest

from phytoveda.traceability.events import (
    EventLedger,
    GeoLocation,
    IdentificationEvent,
)


class TestGeoLocation:
    def test_round_trip(self) -> None:
        loc = GeoLocation(latitude=12.9716, longitude=77.5946, altitude=920.0)
        d = loc.to_dict()
        restored = GeoLocation.from_dict(d)
        assert restored.latitude == 12.9716
        assert restored.longitude == 77.5946
        assert restored.altitude == 920.0

    def test_optional_fields(self) -> None:
        loc = GeoLocation(latitude=0.0, longitude=0.0)
        assert loc.altitude is None
        assert loc.accuracy_meters is None


class TestIdentificationEvent:
    def test_compute_hash_deterministic(self) -> None:
        event = IdentificationEvent(
            event_id="test1", timestamp="2026-03-30T12:00:00",
            species_name="Neem", pathology_label="Healthy",
            species_confidence=0.95, pathology_confidence=0.90,
            model_version="v0.1",
        )
        h1 = event._compute_hash()
        h2 = event._compute_hash()
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_hash_changes_with_data(self) -> None:
        e1 = IdentificationEvent(
            event_id="test1", timestamp="2026-03-30T12:00:00",
            species_name="Neem", pathology_label="Healthy",
            species_confidence=0.95, pathology_confidence=0.90,
            model_version="v0.1",
        )
        e2 = IdentificationEvent(
            event_id="test1", timestamp="2026-03-30T12:00:00",
            species_name="Tulsi", pathology_label="Healthy",  # Changed species
            species_confidence=0.95, pathology_confidence=0.90,
            model_version="v0.1",
        )
        assert e1._compute_hash() != e2._compute_hash()

    def test_round_trip(self) -> None:
        event = IdentificationEvent(
            event_id="abc123", timestamp="2026-03-30T12:00:00",
            species_name="Neem", pathology_label="Bacterial Spot",
            species_confidence=0.9, pathology_confidence=0.8,
            model_version="v0.1", dosha="Pittaja Vyadhi",
            location=GeoLocation(12.97, 77.59),
        )
        d = event.to_dict()
        restored = IdentificationEvent.from_dict(d)
        assert restored.event_id == "abc123"
        assert restored.species_name == "Neem"
        assert restored.location is not None
        assert restored.location.latitude == 12.97


class TestEventLedger:
    def test_record_event(self, tmp_path: Path) -> None:
        ledger = EventLedger(persist_path=tmp_path / "events.jsonl")
        event = ledger.record(species="Neem", pathology="Healthy")
        assert event.event_id
        assert event.event_hash
        assert event.species_name == "Neem"
        assert ledger.event_count == 1

    def test_record_with_gps(self, tmp_path: Path) -> None:
        ledger = EventLedger(persist_path=tmp_path / "events.jsonl")
        event = ledger.record(
            species="Neem", pathology="Healthy",
            latitude=12.9716, longitude=77.5946, altitude=920.0,
        )
        assert event.location is not None
        assert event.location.latitude == 12.9716

    def test_hash_chain(self, tmp_path: Path) -> None:
        ledger = EventLedger(persist_path=tmp_path / "events.jsonl")
        e1 = ledger.record(species="Neem", pathology="Healthy")
        e2 = ledger.record(species="Tulsi", pathology="Healthy")
        e3 = ledger.record(species="Haritaki", pathology="Bacterial Spot")

        assert e1.previous_hash == ""
        assert e2.previous_hash == e1.event_hash
        assert e3.previous_hash == e2.event_hash

    def test_verify_chain_valid(self, tmp_path: Path) -> None:
        ledger = EventLedger(persist_path=tmp_path / "events.jsonl")
        ledger.record(species="Neem", pathology="Healthy")
        ledger.record(species="Tulsi", pathology="Healthy")
        ledger.record(species="Haritaki", pathology="Bacterial Spot")
        assert ledger.verify_chain() is True

    def test_verify_chain_tampered(self, tmp_path: Path) -> None:
        ledger = EventLedger(persist_path=tmp_path / "events.jsonl")
        ledger.record(species="Neem", pathology="Healthy")
        ledger.record(species="Tulsi", pathology="Healthy")

        # Tamper with an event
        ledger._events[0].species_name = "TAMPERED"
        assert ledger.verify_chain() is False

    def test_verify_empty_chain(self, tmp_path: Path) -> None:
        ledger = EventLedger(persist_path=tmp_path / "events.jsonl")
        assert ledger.verify_chain() is True  # Empty chain is valid

    def test_persistence(self, tmp_path: Path) -> None:
        path = tmp_path / "events.jsonl"
        ledger1 = EventLedger(persist_path=path)
        ledger1.record(species="Neem", pathology="Healthy")
        ledger1.record(species="Tulsi", pathology="Powdery Mildew")

        # New instance loads from file
        ledger2 = EventLedger(persist_path=path)
        assert ledger2.event_count == 2
        assert ledger2.verify_chain() is True

    def test_get_events_by_species(self, tmp_path: Path) -> None:
        ledger = EventLedger(persist_path=tmp_path / "events.jsonl")
        ledger.record(species="Neem", pathology="Healthy")
        ledger.record(species="Tulsi", pathology="Healthy")
        ledger.record(species="Neem", pathology="Bacterial Spot")

        neem_events = ledger.get_events_by_species("Neem")
        assert len(neem_events) == 2

    def test_get_events_by_batch(self, tmp_path: Path) -> None:
        ledger = EventLedger(persist_path=tmp_path / "events.jsonl")
        ledger.record(species="Neem", pathology="Healthy", batch_id="BATCH-001")
        ledger.record(species="Tulsi", pathology="Healthy", batch_id="BATCH-001")
        ledger.record(species="Haritaki", pathology="Healthy", batch_id="BATCH-002")

        batch1 = ledger.get_events_by_batch("BATCH-001")
        assert len(batch1) == 2

    def test_get_events_in_region(self, tmp_path: Path) -> None:
        ledger = EventLedger(persist_path=tmp_path / "events.jsonl")
        # Bangalore area
        ledger.record(species="Neem", pathology="Healthy", latitude=12.97, longitude=77.59)
        # Delhi area
        ledger.record(species="Tulsi", pathology="Healthy", latitude=28.61, longitude=77.21)
        # No GPS
        ledger.record(species="Haritaki", pathology="Healthy")

        # Bounding box around Bangalore
        results = ledger.get_events_in_region(12.0, 13.5, 77.0, 78.0)
        assert len(results) == 1
        assert results[0].species_name == "Neem"

    def test_biodiversity_summary(self, tmp_path: Path) -> None:
        ledger = EventLedger(persist_path=tmp_path / "events.jsonl")
        ledger.record(species="Neem", pathology="Healthy")
        ledger.record(species="Neem", pathology="Bacterial Spot")
        ledger.record(species="Tulsi", pathology="Healthy")

        summary = ledger.biodiversity_summary()
        assert summary["Neem"] == 2
        assert summary["Tulsi"] == 1

    def test_record_all_fields(self, tmp_path: Path) -> None:
        ledger = EventLedger(persist_path=tmp_path / "events.jsonl")
        event = ledger.record(
            species="Santalum album", pathology="Healthy",
            species_confidence=0.92, pathology_confidence=0.88,
            model_version="v0.2",
            latitude=12.97, longitude=77.59,
            dosha="Swastha",
            conservation_alert="WARNING: Vulnerable species",
            operator_id="tech-042",
            batch_id="BATCH-007",
        )
        assert event.operator_id == "tech-042"
        assert event.conservation_alert == "WARNING: Vulnerable species"
        assert event.model_version == "v0.2"
