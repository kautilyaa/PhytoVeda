"""Unified label taxonomy across all 6 datasets.

Handles:
    - Species label unification (overlapping species across datasets)
    - Pathology label mapping to unified categories
    - Label encoding/decoding
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ─── Unified Pathology Classes ───────────────────────────────────────────────

PATHOLOGY_CLASSES = [
    "Healthy",
    "Bacterial Spot",
    "Shot Hole",
    "Powdery Mildew",
    "Yellow Leaf Disease",
    "Nitrogen Deficiency",
    "Potassium Deficiency",
    "Unhealthy",  # Generic unhealthy from CIMPD
]

PATHOLOGY_TO_ID: dict[str, int] = {name: i for i, name in enumerate(PATHOLOGY_CLASSES)}
ID_TO_PATHOLOGY: dict[int, str] = {i: name for i, name in enumerate(PATHOLOGY_CLASSES)}

# Map raw dataset labels to unified pathology IDs
RAW_PATHOLOGY_MAPPING: dict[str, int] = {
    # AI-MedLeafX labels
    "Healthy": 0,
    "healthy": 0,
    "Bacterial Spot": 1,
    "bacterial_spot": 1,
    "Bacterial_Spot": 1,
    "Shot Hole": 2,
    "shot_hole": 2,
    "Shot_Hole": 2,
    "Powdery Mildew": 3,
    "powdery_mildew": 3,
    "Powdery_Mildew": 3,
    "Yellow Leaf Disease": 4,
    "yellow_leaf": 4,
    "Yellow_Leaf": 4,
    "Yellow": 4,
    "yellow": 4,
    # CIMPD labels
    "Unhealthy": 7,
    "unhealthy": 7,
    "diseased": 7,
    "Diseased": 7,
    # EarlyNSD labels
    "Nitrogen Deficiency": 5,
    "nitrogen_deficiency": 5,
    "N_deficiency": 5,
    "N-deficiency": 5,
    "Potassium Deficiency": 6,
    "potassium_deficiency": 6,
    "K_deficiency": 6,
    "K-deficiency": 6,
}


def map_pathology_label(raw_label: str | None) -> int:
    """Map a raw pathology label from any dataset to unified ID.

    Returns 0 (Healthy) for None or unrecognized labels.
    """
    if raw_label is None:
        return 0
    return RAW_PATHOLOGY_MAPPING.get(raw_label, RAW_PATHOLOGY_MAPPING.get(raw_label.strip(), 0))


# ─── Species Taxonomy ────────────────────────────────────────────────────────

@dataclass
class SpeciesInfo:
    """Botanical species metadata."""

    scientific_name: str
    common_name: str
    sanskrit_name: str = ""
    family: str = ""
    source_datasets: list[str] = field(default_factory=list)


# Key medicinal species that appear across datasets (canonical entries)
# These ensure overlapping species get a single unified ID
CANONICAL_SPECIES: list[SpeciesInfo] = [
    SpeciesInfo("Azadirachta indica", "Neem", "Nimba", "Meliaceae",
                ["herbify", "medleafx", "cimpd"]),
    SpeciesInfo("Terminalia chebula", "Haritaki", "Haritaki", "Combretaceae",
                ["medleafx"]),
    SpeciesInfo("Moringa oleifera", "Drumstick", "Sojina", "Moringaceae",
                ["medleafx", "cimpd"]),
    SpeciesInfo("Cinnamomum camphora", "Camphor", "Karpura", "Lauraceae",
                ["medleafx"]),
    SpeciesInfo("Ocimum tenuiflorum", "Tulsi", "Tulasi", "Lamiaceae",
                ["herbify", "cimpd"]),
    SpeciesInfo("Mentha arvensis", "Pudina", "Pudina", "Lamiaceae",
                ["cimpd"]),
    SpeciesInfo("Datura stramonium", "Jimsonweed", "Dhatura", "Solanaceae",
                ["cimpd"]),
    SpeciesInfo("Curcuma longa", "Turmeric", "Haridra", "Zingiberaceae",
                ["herbify"]),
    SpeciesInfo("Santalum album", "Sandalwood", "Chandana", "Santalaceae",
                ["herbify"]),
    SpeciesInfo("Glycyrrhiza glabra", "Licorice", "Yashtimadhu", "Fabaceae",
                ["herbify"]),
]


class SpeciesTaxonomy:
    """Manages unified species label mapping across all datasets.

    Builds a global species index by scanning dataset directories.
    Handles overlapping species (e.g., Neem appears in Herbify, MedLeafX, CIMPD)
    by assigning them a single canonical ID.
    """

    def __init__(self) -> None:
        self._name_to_id: dict[str, int] = {}
        self._id_to_info: dict[int, SpeciesInfo] = {}
        self._next_id = 0

        # Register canonical species first
        for species in CANONICAL_SPECIES:
            self._register(species)

    def _register(self, species: SpeciesInfo) -> int:
        """Register a species and return its unified ID."""
        # Check if already registered under any known name
        for name in self._all_names(species):
            if name in self._name_to_id:
                return self._name_to_id[name]

        # New species — assign next ID
        sid = self._next_id
        self._next_id += 1
        self._id_to_info[sid] = species

        # Register all name variants
        for name in self._all_names(species):
            self._name_to_id[name] = sid

        return sid

    def _all_names(self, species: SpeciesInfo) -> list[str]:
        """Generate all name variants for matching."""
        names = [
            species.scientific_name,
            species.scientific_name.lower(),
            species.scientific_name.replace(" ", "_"),
            species.scientific_name.replace(" ", "_").lower(),
            species.common_name,
            species.common_name.lower(),
        ]
        if species.sanskrit_name:
            names.extend([species.sanskrit_name, species.sanskrit_name.lower()])
        return [n for n in names if n]

    def get_or_register(self, name: str, dataset_source: str = "") -> int:
        """Get species ID by name, registering it if new.

        This is the primary interface for dataset loaders. Pass the raw
        folder/label name from any dataset — it will be unified.
        """
        # Try exact match first
        normalized = name.strip()
        if normalized in self._name_to_id:
            return self._name_to_id[normalized]

        # Try lowercase
        lower = normalized.lower()
        if lower in self._name_to_id:
            return self._name_to_id[lower]

        # Try with underscores replaced
        underscore = normalized.replace("_", " ")
        if underscore in self._name_to_id:
            return self._name_to_id[underscore]
        if underscore.lower() in self._name_to_id:
            return self._name_to_id[underscore.lower()]

        # Not found — register as new species
        info = SpeciesInfo(
            scientific_name=normalized,
            common_name=normalized,
            source_datasets=[dataset_source] if dataset_source else [],
        )
        return self._register(info)

    def get_id(self, name: str) -> int | None:
        """Get species ID by name, or None if not registered."""
        normalized = name.strip()
        for variant in [normalized, normalized.lower(),
                        normalized.replace("_", " "),
                        normalized.replace("_", " ").lower()]:
            if variant in self._name_to_id:
                return self._name_to_id[variant]
        return None

    def get_info(self, species_id: int) -> SpeciesInfo | None:
        """Get species metadata by ID."""
        return self._id_to_info.get(species_id)

    def get_name(self, species_id: int) -> str:
        """Get species name by ID."""
        info = self._id_to_info.get(species_id)
        return info.scientific_name if info else f"Unknown_{species_id}"

    @property
    def num_species(self) -> int:
        """Total number of registered species."""
        return self._next_id

    def summary(self) -> str:
        """Human-readable summary of the taxonomy."""
        lines = [f"Species Taxonomy: {self.num_species} species registered"]
        for sid in range(self.num_species):
            info = self._id_to_info[sid]
            sources = ", ".join(info.source_datasets) if info.source_datasets else "unknown"
            lines.append(f"  [{sid:>3}] {info.scientific_name} ({info.common_name}) [{sources}]")
        return "\n".join(lines)
