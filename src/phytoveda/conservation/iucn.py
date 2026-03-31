"""IUCN Red List conservation status registry and endangered species alerts.

Cross-references identified species against conservation status data to:
    - Flag endangered/critically endangered species during identification
    - Warn against harvesting protected species
    - Log sightings for conservation mapping

IUCN Red List Categories (from least to most threatened):
    LC — Least Concern
    NT — Near Threatened
    VU — Vulnerable
    EN — Endangered
    CR — Critically Endangered
    EW — Extinct in the Wild
    EX — Extinct
    DD — Data Deficient
    NE — Not Evaluated

The built-in registry covers key Ayurvedic medicinal species known to be at risk.
Users can extend the registry with additional species data from the IUCN API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ConservationStatus(Enum):
    """IUCN Red List conservation categories."""

    NE = "Not Evaluated"
    DD = "Data Deficient"
    LC = "Least Concern"
    NT = "Near Threatened"
    VU = "Vulnerable"
    EN = "Endangered"
    CR = "Critically Endangered"
    EW = "Extinct in the Wild"
    EX = "Extinct"


# Statuses that trigger harvest warnings
HARVEST_WARNING_STATUSES = {
    ConservationStatus.VU,
    ConservationStatus.EN,
    ConservationStatus.CR,
    ConservationStatus.EW,
    ConservationStatus.EX,
}

# Statuses that trigger immediate alerts
ALERT_STATUSES = {
    ConservationStatus.EN,
    ConservationStatus.CR,
    ConservationStatus.EW,
}


@dataclass
class SpeciesConservation:
    """Conservation data for a single species."""

    scientific_name: str
    common_name: str
    status: ConservationStatus
    population_trend: str = "unknown"  # "increasing", "decreasing", "stable", "unknown"
    threats: list[str] = field(default_factory=list)
    native_regions: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class ConservationAlert:
    """Alert generated when a protected species is identified."""

    species_name: str
    status: ConservationStatus
    severity: str  # "warning", "critical", "harvest_prohibited"
    message: str
    harvest_allowed: bool
    population_trend: str

    def to_dict(self) -> dict:
        return {
            "species_name": self.species_name,
            "status": self.status.value,
            "severity": self.severity,
            "message": self.message,
            "harvest_allowed": self.harvest_allowed,
            "population_trend": self.population_trend,
        }


# ─── Built-in Registry of Ayurvedic Medicinal Species ─────────────────────
# Conservation data for key species referenced in the PhytoVeda pipeline.
# Sources: IUCN Red List, Botanical Survey of India, published literature.

BUILTIN_SPECIES: list[SpeciesConservation] = [
    # ── Critically Endangered / Endangered ──
    SpeciesConservation(
        "Santalum album", "Sandalwood",
        ConservationStatus.VU,
        population_trend="decreasing",
        threats=["Overexploitation", "Illegal logging", "Spike disease"],
        native_regions=["India", "Indonesia", "Australia"],
        notes="Chandana — one of the most valuable and overexploited Ayurvedic woods",
    ),
    SpeciesConservation(
        "Saraca asoca", "Ashoka Tree",
        ConservationStatus.VU,
        population_trend="decreasing",
        threats=["Overexploitation for bark", "Habitat loss", "Substitution pressure"],
        native_regions=["India", "Sri Lanka", "Myanmar"],
        notes="Sacred Ayurvedic tree; bark used in Ashokarishta formulation",
    ),
    SpeciesConservation(
        "Rauvolfia serpentina", "Sarpagandha",
        ConservationStatus.EN,
        population_trend="decreasing",
        threats=["Overharvesting for reserpine", "Habitat destruction"],
        native_regions=["India", "Bangladesh", "Sri Lanka"],
        notes="Source of reserpine; listed in CITES Appendix II",
    ),
    SpeciesConservation(
        "Aquilaria malaccensis", "Agarwood",
        ConservationStatus.CR,
        population_trend="decreasing",
        threats=["Illegal harvesting", "Habitat loss", "Over-collection for oud oil"],
        native_regions=["India", "Southeast Asia"],
        notes="Agaru — critically endangered; CITES Appendix II listed",
    ),
    SpeciesConservation(
        "Nardostachys jatamansi", "Spikenard",
        ConservationStatus.CR,
        population_trend="decreasing",
        threats=["Overharvesting from wild", "Habitat degradation in Himalayas"],
        native_regions=["India", "Nepal", "China"],
        notes="Jatamansi — Himalayan herb used in neurological Ayurvedic formulations",
    ),
    SpeciesConservation(
        "Piper longum", "Long Pepper",
        ConservationStatus.NT,
        population_trend="decreasing",
        threats=["Overharvesting", "Habitat loss in Western Ghats"],
        native_regions=["India", "Sri Lanka", "Southeast Asia"],
        notes="Pippali — one of Trikatu ingredients; wild populations declining",
    ),
    SpeciesConservation(
        "Glycyrrhiza glabra", "Licorice",
        ConservationStatus.NT,
        population_trend="stable",
        threats=["Overharvesting", "Land use changes"],
        native_regions=["India", "Central Asia", "Mediterranean"],
        notes="Yashtimadhu — key Pitta-pacifying herb; cultivated but wild stocks declining",
    ),
    # ── Least Concern (common Ayurvedic species) ──
    SpeciesConservation(
        "Azadirachta indica", "Neem",
        ConservationStatus.LC,
        population_trend="stable",
        native_regions=["India", "South Asia"],
        notes="Nimba — widely cultivated; no conservation concern",
    ),
    SpeciesConservation(
        "Ocimum tenuiflorum", "Tulsi",
        ConservationStatus.LC,
        population_trend="stable",
        native_regions=["India", "Southeast Asia"],
        notes="Tulasi — sacred basil; extensively cultivated",
    ),
    SpeciesConservation(
        "Terminalia chebula", "Haritaki",
        ConservationStatus.LC,
        population_trend="stable",
        native_regions=["India", "Nepal", "Southeast Asia"],
        notes="Haritaki — king of medicines; abundant in deciduous forests",
    ),
    SpeciesConservation(
        "Moringa oleifera", "Drumstick",
        ConservationStatus.LC,
        population_trend="increasing",
        native_regions=["India", "Africa"],
        notes="Sojina — fast-growing; widely cultivated globally",
    ),
    SpeciesConservation(
        "Curcuma longa", "Turmeric",
        ConservationStatus.LC,
        population_trend="stable",
        native_regions=["India", "Southeast Asia"],
        notes="Haridra — extensively cultivated crop; no conservation concern",
    ),
    SpeciesConservation(
        "Cinnamomum camphora", "Camphor",
        ConservationStatus.LC,
        population_trend="stable",
        native_regions=["China", "Japan", "Taiwan"],
        notes="Karpura — cultivated widely; naturalized in India",
    ),
]


class ConservationRegistry:
    """Registry of species conservation status for endangered species monitoring.

    Usage:
        registry = ConservationRegistry()
        alert = registry.check("Rauvolfia serpentina")
        if alert and not alert.harvest_allowed:
            print(f"WARNING: {alert.message}")
    """

    def __init__(self) -> None:
        self._species: dict[str, SpeciesConservation] = {}
        # Register built-in species
        for sp in BUILTIN_SPECIES:
            self.register(sp)

    def register(self, species: SpeciesConservation) -> None:
        """Add or update a species in the registry."""
        # Index by multiple name variants for lookup
        for key in self._name_variants(species.scientific_name, species.common_name):
            self._species[key] = species

    def _name_variants(self, scientific: str, common: str) -> list[str]:
        """Generate lookup variants for a species."""
        variants = [
            scientific,
            scientific.lower(),
            scientific.replace(" ", "_").lower(),
            common,
            common.lower(),
            common.replace(" ", "_").lower(),
        ]
        return [v for v in variants if v]

    def lookup(self, name: str) -> SpeciesConservation | None:
        """Look up conservation data by species name (scientific or common).

        Returns None if species is not in the registry.
        """
        for variant in [name, name.strip(), name.lower(), name.replace("_", " ").lower()]:
            if variant in self._species:
                return self._species[variant]
        return None

    def check(self, species_name: str) -> ConservationAlert | None:
        """Check if a species triggers a conservation alert.

        Returns a ConservationAlert if the species is at risk (VU or above),
        None if species is safe or not in the registry.
        """
        species = self.lookup(species_name)
        if species is None:
            return None

        if species.status not in HARVEST_WARNING_STATUSES:
            return None

        # Determine severity
        if species.status in {ConservationStatus.CR, ConservationStatus.EW}:
            severity = "harvest_prohibited"
            harvest_allowed = False
            message = (
                f"CRITICAL: {species.scientific_name} ({species.common_name}) is "
                f"{species.status.value}. Harvesting is prohibited. "
                f"Population trend: {species.population_trend}."
            )
        elif species.status == ConservationStatus.EN:
            severity = "critical"
            harvest_allowed = False
            message = (
                f"ENDANGERED: {species.scientific_name} ({species.common_name}) is "
                f"{species.status.value}. Harvesting strongly discouraged. "
                f"Population trend: {species.population_trend}."
            )
        else:  # VU
            severity = "warning"
            harvest_allowed = True  # Allowed but flagged
            message = (
                f"WARNING: {species.scientific_name} ({species.common_name}) is "
                f"{species.status.value}. Sustainable harvesting practices required. "
                f"Population trend: {species.population_trend}."
            )

        return ConservationAlert(
            species_name=species.scientific_name,
            status=species.status,
            severity=severity,
            message=message,
            harvest_allowed=harvest_allowed,
            population_trend=species.population_trend,
        )

    def get_threatened_species(self) -> list[SpeciesConservation]:
        """Return all species in the registry with status VU or higher."""
        seen: set[str] = set()
        result: list[SpeciesConservation] = []
        for sp in self._species.values():
            if sp.scientific_name not in seen and sp.status in HARVEST_WARNING_STATUSES:
                seen.add(sp.scientific_name)
                result.append(sp)
        return sorted(result, key=lambda s: list(ConservationStatus).index(s.status), reverse=True)

    @property
    def species_count(self) -> int:
        """Number of unique species in the registry."""
        return len({sp.scientific_name for sp in self._species.values()})

    def summary(self) -> str:
        """Human-readable summary of the registry."""
        counts: dict[str, int] = {}
        seen: set[str] = set()
        for sp in self._species.values():
            if sp.scientific_name not in seen:
                seen.add(sp.scientific_name)
                counts[sp.status.value] = counts.get(sp.status.value, 0) + 1
        lines = [f"Conservation Registry: {len(seen)} species"]
        for status in ConservationStatus:
            if status.value in counts:
                lines.append(f"  {status.name} ({status.value}): {counts[status.value]}")
        return "\n".join(lines)
