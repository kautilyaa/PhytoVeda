"""Identification event logging with GPS geo-tagging and cryptographic hashing.

Each identification event is recorded as an immutable IdentificationEvent with:
    - Species + pathology prediction results
    - GPS coordinates (latitude, longitude, altitude)
    - Cryptographic SHA-256 hash for tamper-proof supply chain audit trail
    - Blockchain-ready: hash chain links each event to the previous

Usage:
    ledger = EventLedger(persist_path="data/events.jsonl")
    event = ledger.record(
        species="Azadirachta indica", pathology="Healthy",
        confidence=0.95, latitude=12.9716, longitude=77.5946,
    )
    print(event.event_hash)  # SHA-256 hash for blockchain submission
    ledger.verify_chain()    # Verify integrity of entire audit trail
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass
class GeoLocation:
    """GPS coordinates for a harvest/identification site."""

    latitude: float
    longitude: float
    altitude: float | None = None
    accuracy_meters: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> GeoLocation:
        return cls(**data)


@dataclass
class IdentificationEvent:
    """A single plant identification event in the supply chain audit trail.

    Each event is cryptographically hashed and linked to the previous event,
    forming a hash chain suitable for blockchain submission.
    """

    event_id: str
    timestamp: str
    species_name: str
    pathology_label: str
    species_confidence: float
    pathology_confidence: float
    model_version: str
    location: GeoLocation | None = None
    dosha: str = ""
    conservation_alert: str = ""
    operator_id: str = ""
    batch_id: str = ""
    previous_hash: str = ""
    event_hash: str = ""

    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of the event data (excluding event_hash itself)."""
        payload = {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "species_name": self.species_name,
            "pathology_label": self.pathology_label,
            "species_confidence": self.species_confidence,
            "pathology_confidence": self.pathology_confidence,
            "model_version": self.model_version,
            "location": self.location.to_dict() if self.location else None,
            "dosha": self.dosha,
            "conservation_alert": self.conservation_alert,
            "operator_id": self.operator_id,
            "batch_id": self.batch_id,
            "previous_hash": self.previous_hash,
        }
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.location:
            d["location"] = self.location.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> IdentificationEvent:
        loc_data = data.pop("location", None)
        location = GeoLocation.from_dict(loc_data) if loc_data else None
        return cls(location=location, **data)


class EventLedger:
    """Append-only ledger of identification events with hash chain integrity.

    Events are stored in a JSONL (JSON Lines) file. Each event's hash
    includes the previous event's hash, creating a tamper-evident chain.

    Usage:
        ledger = EventLedger(persist_path="data/events.jsonl")
        event = ledger.record(species="Neem", pathology="Healthy", confidence=0.95)
        assert ledger.verify_chain()
    """

    def __init__(self, persist_path: str | Path = "data/events.jsonl") -> None:
        self.persist_path = Path(persist_path)
        self._events: list[IdentificationEvent] = []
        self._load()

    def _load(self) -> None:
        """Load events from JSONL file."""
        if not self.persist_path.exists():
            return
        for line in self.persist_path.read_text(encoding="utf-8").strip().splitlines():
            if line.strip():
                self._events.append(IdentificationEvent.from_dict(json.loads(line)))

    def _append_to_file(self, event: IdentificationEvent) -> None:
        """Append a single event to the JSONL file."""
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        with self.persist_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")

    def record(
        self,
        species: str,
        pathology: str,
        species_confidence: float = 0.0,
        pathology_confidence: float = 0.0,
        model_version: str = "v0.1",
        latitude: float | None = None,
        longitude: float | None = None,
        altitude: float | None = None,
        dosha: str = "",
        conservation_alert: str = "",
        operator_id: str = "",
        batch_id: str = "",
    ) -> IdentificationEvent:
        """Record a new identification event.

        Args:
            species: Identified species name.
            pathology: Pathology diagnosis label.
            species_confidence: Model confidence for species prediction.
            pathology_confidence: Model confidence for pathology prediction.
            model_version: Version of the model that made the prediction.
            latitude: GPS latitude of identification site.
            longitude: GPS longitude of identification site.
            altitude: GPS altitude in meters.
            dosha: Vrikshayurveda Dosha classification.
            conservation_alert: Conservation alert message (if any).
            operator_id: ID of the operator/technician.
            batch_id: Batch ID for pharmaceutical auditing.

        Returns:
            The recorded IdentificationEvent with computed hash.
        """
        location = None
        if latitude is not None and longitude is not None:
            location = GeoLocation(
                latitude=latitude, longitude=longitude, altitude=altitude,
            )

        previous_hash = self._events[-1].event_hash if self._events else ""

        now = datetime.now(UTC)
        event_id = hashlib.sha256(
            f"{now.isoformat()}{species}{pathology}{previous_hash}".encode()
        ).hexdigest()[:16]

        event = IdentificationEvent(
            event_id=event_id,
            timestamp=now.isoformat(),
            species_name=species,
            pathology_label=pathology,
            species_confidence=round(species_confidence, 4),
            pathology_confidence=round(pathology_confidence, 4),
            model_version=model_version,
            location=location,
            dosha=dosha,
            conservation_alert=conservation_alert,
            operator_id=operator_id,
            batch_id=batch_id,
            previous_hash=previous_hash,
        )
        event.event_hash = event._compute_hash()

        self._events.append(event)
        self._append_to_file(event)
        return event

    def verify_chain(self) -> bool:
        """Verify the integrity of the entire event hash chain.

        Returns True if every event's hash is valid and chain links are correct.
        """
        for i, event in enumerate(self._events):
            # Verify hash matches content
            expected_hash = event._compute_hash()
            if event.event_hash != expected_hash:
                return False

            # Verify chain link
            if i == 0:
                if event.previous_hash != "":
                    return False
            else:
                if event.previous_hash != self._events[i - 1].event_hash:
                    return False

        return True

    def get_events_by_species(self, species: str) -> list[IdentificationEvent]:
        """Filter events by species name."""
        lower = species.lower()
        return [e for e in self._events if e.species_name.lower() == lower]

    def get_events_by_batch(self, batch_id: str) -> list[IdentificationEvent]:
        """Filter events by batch ID for pharmaceutical auditing."""
        return [e for e in self._events if e.batch_id == batch_id]

    def get_events_in_region(
        self,
        lat_min: float, lat_max: float,
        lon_min: float, lon_max: float,
    ) -> list[IdentificationEvent]:
        """Filter events within a geographic bounding box."""
        results = []
        for e in self._events:
            if e.location is None:
                continue
            if (lat_min <= e.location.latitude <= lat_max
                    and lon_min <= e.location.longitude <= lon_max):
                results.append(e)
        return results

    def biodiversity_summary(self) -> dict[str, int]:
        """Count unique species identifications (for biodiversity mapping)."""
        counts: dict[str, int] = {}
        for e in self._events:
            counts[e.species_name] = counts.get(e.species_name, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    @property
    def event_count(self) -> int:
        return len(self._events)

    @property
    def all_events(self) -> list[IdentificationEvent]:
        return list(self._events)
