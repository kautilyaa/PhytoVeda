"""Vrikshayurveda Dosha mapping: pathology outputs -> Tridosha diagnosis + treatments.

Maps modern CV pathology classifications to ancient Ayurvedic diagnostic framework:
    - Vataja Vyadhi (Vata): desiccation, geometric deformations, wilting
    - Pittaja Vyadhi (Pitta): yellowing, chlorosis, necrotic lesions
    - Kaphaja Vyadhi (Kapha): powdery mildew, edema, hypertrophy
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Dosha(Enum):
    """The three Doshas from Vrikshayurveda plant pathology."""

    VATA = "Vataja Vyadhi"
    PITTA = "Pittaja Vyadhi"
    KAPHA = "Kaphaja Vyadhi"
    HEALTHY = "Swastha"  # Healthy / balanced


@dataclass
class DoshaAssessment:
    """Result of Vrikshayurveda mapping."""

    dosha: Dosha
    confidence: float
    cv_features: list[str]
    classical_symptoms: list[str]
    treatments: list[str]
    contraindications: list[str]


# Deterministic mapping: pathology label -> Dosha
PATHOLOGY_TO_DOSHA: dict[str, Dosha] = {
    "Healthy": Dosha.HEALTHY,
    "Shot Hole": Dosha.VATA,
    "Nitrogen Deficiency": Dosha.VATA,
    "Yellow Leaf Disease": Dosha.PITTA,
    "Bacterial Spot": Dosha.PITTA,
    "Powdery Mildew": Dosha.KAPHA,
    "Potassium Deficiency": Dosha.VATA,
    "Unhealthy": Dosha.PITTA,  # Default for unspecified unhealthy
}

# CV feature correlates per Dosha
DOSHA_CV_FEATURES: dict[Dosha, list[str]] = {
    Dosha.VATA: [
        "Edge deformation",
        "Texture variance / roughness",
        "Structural asymmetry",
        "Leaf margin curling",
        "Desiccation patterns",
    ],
    Dosha.PITTA: [
        "RGB/HSV colorimetric shifts",
        "Localized necrotic lesions",
        "Burn artifacts",
        "Severe chlorotic patterns",
        "Premature discoloration",
    ],
    Dosha.KAPHA: [
        "Morphological thickening",
        "Surface dullness / white artifacts",
        "Abnormal Leaf Area Index (LAI)",
        "Excessive but irregular growth",
        "Powdery surface deposits",
    ],
    Dosha.HEALTHY: [],
}

# Classical symptoms from Vrikshayurveda texts
DOSHA_CLASSICAL_SYMPTOMS: dict[Dosha, list[str]] = {
    Dosha.VATA: [
        "Plant becomes lean with severe geometrical deformations",
        "Anomalous knots and globules on trunks or leaves",
        "Rapid desiccation and drying of branches",
        "Fruits abnormally hard with reduced sap and juice",
    ],
    Dosha.PITTA: [
        "Inability to withstand solar radiation",
        "Profound yellowing (chlorosis) of leaves",
        "Premature shedding of foliage and branches",
        "Rapid rotting of fruits before ripening",
        "Systemic paleness throughout the plant",
    ],
    Dosha.KAPHA: [
        "Hypertrophic but severely deformed leaves",
        "Delayed and abnormally prolonged fruit bearing",
        "Dwarfed overall growth",
        "Loss of natural olfactory and gustatory profiles (medicinal potency ruined)",
        "Paleness without the desiccation seen in Vata",
    ],
    Dosha.HEALTHY: ["Plant exhibits balanced growth and vitality"],
}

# Traditional therapeutic interventions from Vrikshayurveda
DOSHA_TREATMENTS: dict[Dosha, list[str]] = {
    Dosha.VATA: [
        "Irrigate with Kunapajala (liquid organic fermented manure) for intense nourishment",
        "Apply animal-derived fats to restore vitality",
        "Fumigate locally with Azadirachta indica (Neem) leaves to ward off pests",
        "Provide nutrient-dense formulations to replenish sap content",
    ],
    Dosha.PITTA: [
        "Root irrigation with Glycyrrhiza glabra (Yashtimadhu/Licorice) decoction",
        "Apply Madhuca indica decoction for cooling effect",
        "Apply milk mixed with honey to the soil",
        "Spray cooling Triphala decoction on foliage",
    ],
    Dosha.KAPHA: [
        "Soil treatment with bitter Panchamoola (five root) decoction",
        "Apply mustard-based formulations for heating/drying effect",
        "Strictly reduce ambient moisture around the plant",
        "Avoid overly sweet or oily fertilizers",
        "Apply astringent, heating interventions to counter cold-damp stagnation",
    ],
    Dosha.HEALTHY: ["No intervention required — specimen suitable for pharmaceutical procurement"],
}

DOSHA_CONTRAINDICATIONS: dict[Dosha, list[str]] = {
    Dosha.VATA: ["Avoid cold or drying treatments", "Do not reduce moisture further"],
    Dosha.PITTA: ["Avoid heating interventions", "Do not expose to direct intense sunlight"],
    Dosha.KAPHA: ["Avoid sweet, oily, or cold fertilizers", "Do not increase watering"],
    Dosha.HEALTHY: [],
}


class VrikshayurvedaMapper:
    """Maps model pathology outputs to Vrikshayurveda Tridosha diagnosis."""

    def assess(
        self,
        pathology_label: str,
        confidence: float,
    ) -> DoshaAssessment:
        """Map a pathology prediction to Dosha diagnosis with treatments.

        Args:
            pathology_label: Predicted pathology class name.
            confidence: Model confidence score for this prediction.

        Returns:
            Complete Dosha assessment with CV features, symptoms, and treatments.
        """
        dosha = PATHOLOGY_TO_DOSHA.get(pathology_label, Dosha.PITTA)

        return DoshaAssessment(
            dosha=dosha,
            confidence=confidence,
            cv_features=DOSHA_CV_FEATURES[dosha],
            classical_symptoms=DOSHA_CLASSICAL_SYMPTOMS[dosha],
            treatments=DOSHA_TREATMENTS[dosha],
            contraindications=DOSHA_CONTRAINDICATIONS[dosha],
        )
