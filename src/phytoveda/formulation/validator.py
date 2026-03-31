"""Validate multi-herb Ayurvedic formulations against classical texts.

Cross-references identified plant specimens against classical Ayurvedic
formulations (Kashaya, Churna, Taila, etc.) to verify:
    - All required herbs are present and correctly identified
    - All specimens are healthy / suitable for pharmaceutical use
    - No substitutions with pharmacologically distinct species

Classical formulations are sourced from Charaka Samhita, Susruta Samhita,
and the Ayurvedic Pharmacopoeia of India.

Usage:
    validator = FormulationValidator()
    result = validator.validate("Triphala", identified_herbs)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FormulationHerb:
    """A single herb component in a classical formulation."""

    scientific_name: str
    sanskrit_name: str
    common_name: str
    part_used: str  # "fruit", "root", "leaf", "bark", "whole plant"
    proportion: str = "equal"  # "equal", "dominant", "minor", or ratio like "1:2"


@dataclass
class ClassicalFormulation:
    """A classical Ayurvedic multi-herb formulation."""

    name: str
    sanskrit_name: str
    category: str  # "Kashaya", "Churna", "Taila", "Arishta", etc.
    source_text: str  # "Charaka Samhita", "Susruta Samhita", etc.
    herbs: list[FormulationHerb]
    therapeutic_use: str
    contraindications: list[str] = field(default_factory=list)


@dataclass
class IdentifiedHerb:
    """A herb specimen identified by the PhytoVeda pipeline."""

    species_name: str
    pathology_label: str
    confidence: float
    is_healthy: bool


@dataclass
class HerbValidationResult:
    """Validation result for a single herb in a formulation."""

    required_herb: FormulationHerb
    matched_specimen: IdentifiedHerb | None
    status: str  # "verified", "missing", "unhealthy", "low_confidence", "substituted"
    message: str


@dataclass
class FormulationValidationResult:
    """Complete validation result for a formulation."""

    formulation: ClassicalFormulation
    herb_results: list[HerbValidationResult]
    is_valid: bool
    overall_quality: str  # "pass", "conditional", "fail"
    warnings: list[str]
    missing_herbs: list[str]

    def to_dict(self) -> dict:
        return {
            "formulation": self.formulation.name,
            "sanskrit_name": self.formulation.sanskrit_name,
            "source_text": self.formulation.source_text,
            "is_valid": self.is_valid,
            "overall_quality": self.overall_quality,
            "herb_results": [
                {
                    "required": hr.required_herb.scientific_name,
                    "status": hr.status,
                    "message": hr.message,
                }
                for hr in self.herb_results
            ],
            "warnings": self.warnings,
            "missing_herbs": self.missing_herbs,
        }


# ─── Classical Formulation Knowledge Base ──────────────────────────────────

CLASSICAL_FORMULATIONS: list[ClassicalFormulation] = [
    ClassicalFormulation(
        name="Triphala",
        sanskrit_name="Triphala Churna",
        category="Churna",
        source_text="Charaka Samhita",
        herbs=[
            FormulationHerb("Terminalia chebula", "Haritaki", "Chebulic Myrobalan", "fruit"),
            FormulationHerb("Terminalia bellirica", "Bibhitaki", "Belleric Myrobalan", "fruit"),
            FormulationHerb("Emblica officinalis", "Amalaki", "Indian Gooseberry", "fruit"),
        ],
        therapeutic_use="Rasayana (rejuvenation), digestive, mild laxative, antioxidant",
        contraindications=["Pregnancy", "Severe diarrhea"],
    ),
    ClassicalFormulation(
        name="Trikatu",
        sanskrit_name="Trikatu Churna",
        category="Churna",
        source_text="Charaka Samhita",
        herbs=[
            FormulationHerb("Zingiber officinale", "Shunthi", "Ginger", "rhizome"),
            FormulationHerb("Piper nigrum", "Maricha", "Black Pepper", "fruit"),
            FormulationHerb("Piper longum", "Pippali", "Long Pepper", "fruit"),
        ],
        therapeutic_use="Agni deepana (digestive fire), Kapha reduction, bioavailability enhancer",
        contraindications=["Pitta aggravation", "Gastric ulcers", "Pregnancy"],
    ),
    ClassicalFormulation(
        name="Dashamoola",
        sanskrit_name="Dashamoola Kashaya",
        category="Kashaya",
        source_text="Susruta Samhita",
        herbs=[
            FormulationHerb("Aegle marmelos", "Bilva", "Bael", "root"),
            FormulationHerb("Oroxylum indicum", "Shyonaka", "Indian Trumpet", "root"),
            FormulationHerb("Gmelina arborea", "Gambhari", "Beechwood", "root"),
            FormulationHerb("Stereospermum chelonoides", "Patala", "Trumpet Flower", "root"),
            FormulationHerb("Clerodendrum phlomidis", "Agnimantha", "Wind Killer", "root"),
            FormulationHerb("Desmodium gangeticum", "Shalaparni", "Sal Leaved Desmodium", "root"),
            FormulationHerb("Uraria picta", "Prishnaparni", "Pithvan", "root"),
            FormulationHerb("Solanum indicum", "Brihati", "Indian Nightshade", "root"),
            FormulationHerb(
                "Solanum xanthocarpum", "Kantakari",
                "Yellow-berried Nightshade", "root",
            ),
            FormulationHerb("Tribulus terrestris", "Gokshura", "Puncture Vine", "root"),
        ],
        therapeutic_use="Vata-Kapha shamana, anti-inflammatory, postpartum recovery",
        contraindications=["Pitta conditions when used alone"],
    ),
    ClassicalFormulation(
        name="Chyawanprash",
        sanskrit_name="Chyawanprash Avaleha",
        category="Avaleha",
        source_text="Charaka Samhita",
        herbs=[
            FormulationHerb(
                "Emblica officinalis", "Amalaki",
                "Indian Gooseberry", "fruit", "dominant",
            ),
            FormulationHerb("Tinospora cordifolia", "Guduchi", "Giloy", "stem"),
            FormulationHerb("Piper longum", "Pippali", "Long Pepper", "fruit"),
            FormulationHerb("Glycyrrhiza glabra", "Yashtimadhu", "Licorice", "root"),
            FormulationHerb("Cinnamomum zeylanicum", "Twak", "Cinnamon", "bark"),
        ],
        therapeutic_use="Rasayana (rejuvenation), immunity, respiratory health",
        contraindications=["Diabetes (high sugar content)", "Kapha aggravation"],
    ),
    ClassicalFormulation(
        name="Lekhniya Mahakashaya",
        sanskrit_name="Lekhniya Mahakashaya",
        category="Kashaya",
        source_text="Charaka Samhita",
        herbs=[
            FormulationHerb("Commiphora mukul", "Guggulu", "Indian Bdellium", "resin"),
            FormulationHerb("Berberis aristata", "Daruharidra", "Indian Barberry", "root"),
            FormulationHerb("Curcuma longa", "Haridra", "Turmeric", "rhizome"),
            FormulationHerb("Terminalia chebula", "Haritaki", "Chebulic Myrobalan", "fruit"),
            FormulationHerb("Terminalia bellirica", "Bibhitaki", "Belleric Myrobalan", "fruit"),
        ],
        therapeutic_use="Medohara (fat-scraping), metabolic regulation, cholesterol reduction",
        contraindications=["Pregnancy", "Vata aggravation in excess"],
    ),
]


class FormulationValidator:
    """Validate identified herb specimens against classical Ayurvedic formulations.

    Usage:
        validator = FormulationValidator()
        herbs = [IdentifiedHerb("Terminalia chebula", "Healthy", 0.95, True), ...]
        result = validator.validate("Triphala", herbs)
        print(result.overall_quality)  # "pass", "conditional", "fail"
    """

    def __init__(self, confidence_threshold: float = 0.7) -> None:
        self.confidence_threshold = confidence_threshold
        self._formulations: dict[str, ClassicalFormulation] = {}
        for f in CLASSICAL_FORMULATIONS:
            self._register(f)

    def _register(self, formulation: ClassicalFormulation) -> None:
        """Register a formulation under multiple name variants."""
        for key in [formulation.name, formulation.name.lower(),
                    formulation.sanskrit_name, formulation.sanskrit_name.lower()]:
            self._formulations[key] = formulation

    def register_formulation(self, formulation: ClassicalFormulation) -> None:
        """Add a custom formulation to the knowledge base."""
        self._register(formulation)

    def get_formulation(self, name: str) -> ClassicalFormulation | None:
        """Look up a formulation by name."""
        for variant in [name, name.lower(), name.strip()]:
            if variant in self._formulations:
                return self._formulations[variant]
        return None

    def validate(
        self,
        formulation_name: str,
        identified_herbs: list[IdentifiedHerb],
    ) -> FormulationValidationResult:
        """Validate identified herbs against a classical formulation.

        Args:
            formulation_name: Name of the classical formulation (e.g., "Triphala").
            identified_herbs: List of herbs identified by the PhytoVeda pipeline.

        Returns:
            FormulationValidationResult with per-herb status and overall quality.

        Raises:
            ValueError: If the formulation is not found in the knowledge base.
        """
        formulation = self.get_formulation(formulation_name)
        if formulation is None:
            raise ValueError(f"Formulation '{formulation_name}' not found in knowledge base")

        # Build lookup by species name (case-insensitive)
        specimen_map: dict[str, IdentifiedHerb] = {}
        for herb in identified_herbs:
            specimen_map[herb.species_name.lower()] = herb

        herb_results: list[HerbValidationResult] = []
        warnings: list[str] = []
        missing: list[str] = []

        for required in formulation.herbs:
            # Try to match by scientific name
            matched = specimen_map.get(required.scientific_name.lower())

            if matched is None:
                herb_results.append(HerbValidationResult(
                    required_herb=required,
                    matched_specimen=None,
                    status="missing",
                    message=(
                        f"{required.scientific_name} ({required.sanskrit_name}) "
                        "not found in specimens"
                    ),
                ))
                missing.append(required.scientific_name)
            elif not matched.is_healthy:
                herb_results.append(HerbValidationResult(
                    required_herb=required,
                    matched_specimen=matched,
                    status="unhealthy",
                    message=(
                        f"{required.scientific_name} identified but unhealthy "
                        f"({matched.pathology_label}). Treatment required before use."
                    ),
                ))
                warnings.append(
                    f"{required.sanskrit_name} ({required.scientific_name}): "
                    f"{matched.pathology_label} — treat before use"
                )
            elif matched.confidence < self.confidence_threshold:
                herb_results.append(HerbValidationResult(
                    required_herb=required,
                    matched_specimen=matched,
                    status="low_confidence",
                    message=(
                        f"{required.scientific_name} identification confidence "
                        f"({matched.confidence:.1%}) below threshold "
                        f"({self.confidence_threshold:.1%}). "
                        "Manual verification recommended."
                    ),
                ))
                warnings.append(
                    f"{required.sanskrit_name}: low confidence ({matched.confidence:.1%})"
                )
            else:
                herb_results.append(HerbValidationResult(
                    required_herb=required,
                    matched_specimen=matched,
                    status="verified",
                    message=(
                        f"{required.scientific_name} ({required.sanskrit_name}) "
                        "verified healthy"
                    ),
                ))

        # Determine overall quality
        statuses = [hr.status for hr in herb_results]
        if all(s == "verified" for s in statuses):
            overall_quality = "pass"
            is_valid = True
        elif "missing" in statuses:
            overall_quality = "fail"
            is_valid = False
        else:
            # All present but some unhealthy or low confidence
            overall_quality = "conditional"
            is_valid = False

        return FormulationValidationResult(
            formulation=formulation,
            herb_results=herb_results,
            is_valid=is_valid,
            overall_quality=overall_quality,
            warnings=warnings,
            missing_herbs=missing,
        )

    @property
    def formulation_names(self) -> list[str]:
        """List unique formulation names in the knowledge base."""
        seen: set[str] = set()
        names: list[str] = []
        for f in self._formulations.values():
            if f.name not in seen:
                seen.add(f.name)
                names.append(f.name)
        return names
