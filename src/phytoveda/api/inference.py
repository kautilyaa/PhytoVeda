"""Inference pipeline: image -> model -> Dosha -> RAG -> report.

Orchestrates the full PhytoVeda identification pipeline:
    1. Preprocess uploaded image to 512x512
    2. Run HierViTCMTL forward pass (species + pathology logits)
    3. Decode predictions via taxonomy
    4. Compute uncertainty scores for active learning
    5. Map pathology to Vrikshayurveda Dosha assessment
    6. Optionally retrieve RAG context and generate LLM report
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from phytoveda.active_learning.uncertainty import UncertaintySampler, UncertaintyScore
from phytoveda.data.taxonomy import ID_TO_PATHOLOGY, SpeciesTaxonomy
from phytoveda.models.cmtl import HierViTCMTL
from phytoveda.rag.report_generator import BotanicalReport, ReportGenerator
from phytoveda.rag.retriever import AyurvedicRetriever
from phytoveda.vrikshayurveda.mapper import DoshaAssessment, VrikshayurvedaMapper

# ImageNet normalization constants (same as training pipeline)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_inference_transform(image_size: int = 512) -> transforms.Compose:
    """Build preprocessing transform for a given image size."""
    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.LANCZOS,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


@dataclass
class PredictionResult:
    """Full prediction output from the inference pipeline."""

    species_name: str
    species_confidence: float
    top_k_species: list[tuple[str, float]]
    pathology_label: str
    pathology_confidence: float
    top_k_pathology: list[tuple[str, float]]
    dosha: DoshaAssessment
    uncertainty: UncertaintyScore
    report: BotanicalReport | None = None

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        d: dict = {
            "species_name": self.species_name,
            "species_confidence": round(self.species_confidence, 4),
            "top_k_species": [
                {"name": name, "confidence": round(conf, 4)}
                for name, conf in self.top_k_species
            ],
            "pathology_label": self.pathology_label,
            "pathology_confidence": round(self.pathology_confidence, 4),
            "top_k_pathology": [
                {"name": name, "confidence": round(conf, 4)}
                for name, conf in self.top_k_pathology
            ],
            "dosha": {
                "dosha": self.dosha.dosha.value,
                "confidence": round(self.dosha.confidence, 4),
                "cv_features": self.dosha.cv_features,
                "classical_symptoms": self.dosha.classical_symptoms,
                "treatments": self.dosha.treatments,
                "contraindications": self.dosha.contraindications,
            },
            "uncertainty": {
                "least_confidence": round(self.uncertainty.least_confidence, 4),
                "margin": round(self.uncertainty.margin, 4),
                "entropy": round(self.uncertainty.entropy, 4),
                "combined": round(self.uncertainty.combined, 4),
                "is_uncertain": self.uncertainty.is_uncertain,
            },
        }
        if self.report is not None:
            d["report"] = {
                "species_family": self.report.species_family,
                "sanskrit_name": self.report.sanskrit_name,
                "common_names": self.report.common_names,
                "rasa": self.report.rasa,
                "guna": self.report.guna,
                "virya": self.report.virya,
                "vipaka": self.report.vipaka,
                "health_status": self.report.health_status,
                "dosha_assessment": self.report.dosha_assessment,
                "procurement_quality": self.report.procurement_quality,
                "full_report_text": self.report.full_report_text,
            }
        return d


class InferencePipeline:
    """Orchestrates the full PhytoVeda inference flow.

    Usage:
        pipeline = InferencePipeline.from_checkpoint("checkpoints/best_model.pt")
        result = pipeline.predict(image)
    """

    def __init__(
        self,
        model: HierViTCMTL,
        taxonomy: SpeciesTaxonomy,
        device: torch.device | None = None,
        image_size: int = 512,
        retriever: AyurvedicRetriever | None = None,
        report_generator: ReportGenerator | None = None,
        uncertainty_sampler: UncertaintySampler | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.taxonomy = taxonomy
        self.image_size = image_size
        self.transform = build_inference_transform(image_size)
        self.mapper = VrikshayurvedaMapper()
        self.retriever = retriever
        self.report_generator = report_generator
        self.sampler = uncertainty_sampler or UncertaintySampler()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        taxonomy: SpeciesTaxonomy | None = None,
        device: torch.device | None = None,
        retriever: AyurvedicRetriever | None = None,
        report_generator: ReportGenerator | None = None,
        **model_kwargs,
    ) -> InferencePipeline:
        """Load pipeline from a saved checkpoint.

        The checkpoint must contain 'model_state_dict' and optionally 'config'.
        Model constructor kwargs can be passed directly or read from the checkpoint.

        Args:
            checkpoint_path: Path to the saved .pt checkpoint.
            taxonomy: Species taxonomy. If None, a default is created.
            device: Torch device for inference.
            retriever: Optional RAG retriever for report generation.
            report_generator: Optional LLM report generator.
            **model_kwargs: Overrides for HierViTCMTL constructor
                (num_species, num_pathologies, backbone_name, etc.).
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Extract model config from checkpoint if available
        config = checkpoint.get("config", {})
        num_species = model_kwargs.pop("num_species", config.get("num_species", 147))
        num_pathologies = model_kwargs.pop("num_pathologies", config.get("num_pathologies", 8))
        backbone_name = model_kwargs.pop(
            "backbone_name",
            config.get("backbone", "vit_huge_patch14_dinov2.lvd142m"),
        )
        image_size = model_kwargs.pop("image_size", config.get("image_size", 512))

        model = HierViTCMTL(
            num_species=num_species,
            num_pathologies=num_pathologies,
            backbone_name=backbone_name,
            pretrained=False,  # Loading from checkpoint, no need for pretrained init
            image_size=image_size,
            **model_kwargs,
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        tax = taxonomy or SpeciesTaxonomy()

        return cls(
            model=model,
            taxonomy=tax,
            device=device,
            image_size=image_size,
            retriever=retriever,
            report_generator=report_generator,
        )

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess a PIL image to model input tensor.

        Args:
            image: RGB PIL Image of any size.

        Returns:
            Tensor of shape (1, 3, image_size, image_size) on the correct device.
        """
        tensor = self.transform(image.convert("RGB"))
        return tensor.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        top_k: int = 5,
        generate_report: bool = False,
    ) -> PredictionResult:
        """Run full inference pipeline on a single image.

        Args:
            image: RGB PIL Image.
            top_k: Number of top predictions to return.
            generate_report: If True and report_generator is available,
                generate a full botanical report.

        Returns:
            PredictionResult with species, pathology, Dosha, uncertainty, and optional report.
        """
        input_tensor = self.preprocess(image)
        species_logits, pathology_logits = self.model(input_tensor)

        # Decode species predictions
        species_probs = F.softmax(species_logits, dim=1)[0]
        sp_top_vals, sp_top_ids = species_probs.topk(min(top_k, species_probs.size(0)))
        top_k_species = [
            (self.taxonomy.get_name(idx.item()), val.item())
            for val, idx in zip(sp_top_vals, sp_top_ids, strict=True)
        ]
        species_name = top_k_species[0][0]
        species_confidence = top_k_species[0][1]

        # Decode pathology predictions
        pathology_probs = F.softmax(pathology_logits, dim=1)[0]
        pa_top_vals, pa_top_ids = pathology_probs.topk(min(top_k, pathology_probs.size(0)))
        top_k_pathology = [
            (ID_TO_PATHOLOGY.get(idx.item(), f"Unknown_{idx.item()}"), val.item())
            for val, idx in zip(pa_top_vals, pa_top_ids, strict=True)
        ]
        pathology_label = top_k_pathology[0][0]
        pathology_confidence = top_k_pathology[0][1]

        # Uncertainty scoring (use species logits as primary signal)
        uncertainty = self.sampler.score(species_logits)[0]

        # Vrikshayurveda Dosha mapping
        dosha = self.mapper.assess(pathology_label, pathology_confidence)

        # Optional report generation
        report: BotanicalReport | None = None
        if generate_report and self.report_generator is not None:
            contexts = []
            if self.retriever is not None:
                contexts = self.retriever.retrieve_for_diagnosis(
                    species_name, pathology_label, dosha, top_k=5
                )
            try:
                report = self.report_generator.generate(
                    species_name, pathology_label, dosha, contexts, species_confidence
                )
            except Exception:
                # Fall back to offline report if LLM fails
                report = self.report_generator.generate_offline(
                    species_name, pathology_label, dosha
                )

        return PredictionResult(
            species_name=species_name,
            species_confidence=species_confidence,
            top_k_species=top_k_species,
            pathology_label=pathology_label,
            pathology_confidence=pathology_confidence,
            top_k_pathology=top_k_pathology,
            dosha=dosha,
            uncertainty=uncertainty,
            report=report,
        )
