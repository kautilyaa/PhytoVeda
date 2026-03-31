"""Oracle loop: LLM-based and human expert labeling for quarantined images.

Two oracle paths:
    1. LLMOracle — multimodal LLM classification (~99% of quarantine)
       Supports: Gemini, Claude, OpenAI, Llama
    2. HumanExpertQueue — JSON-based queue for the hardest ~1% of images

The oracle pipeline processes quarantined images, labels them, and feeds
results back to the QuarantineManager for retraining.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

from phytoveda.active_learning.quarantine import QuarantineEntry, QuarantineManager
from phytoveda.llm.providers import BaseLLM, get_provider


class OracleSource(Enum):
    """Source of the oracle label."""

    LLM = "llm"
    HUMAN_EXPERT = "human_expert"


@dataclass
class OracleLabel:
    """Label provided by the oracle for a quarantined image."""

    quarantine_id: str
    image_path: str
    species_label: str
    pathology_label: str
    source: OracleSource
    confidence: float
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["source"] = self.source.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> OracleLabel:
        data["source"] = OracleSource(data["source"])
        return cls(**data)


LLM_CLASSIFICATION_PROMPT = """\
You are an expert botanist specializing in medicinal plants of the Indian subcontinent.

Analyze this plant leaf image and provide:
1. **Species identification** — the most likely species (scientific name)
2. **Health/pathology assessment** — one of the following categories:
   - Healthy
   - Bacterial Spot
   - Shot Hole
   - Powdery Mildew
   - Yellow Leaf Disease
   - Nitrogen Deficiency
   - Potassium Deficiency
   - Unhealthy (if diseased but category unclear)

The AI model's top predictions were:
- Species: {top_species}
- Pathology: {top_pathology}

Respond with valid JSON:
{{
    "species": "scientific name",
    "pathology": "one of the pathology categories above",
    "confidence": 0.0 to 1.0,
    "notes": "brief reasoning"
}}
"""


class LLMOracle:
    """Automated oracle using multimodal LLM for secondary classification.

    Sends quarantined leaf images to a multimodal LLM for a second opinion.
    Supports Gemini, Claude, OpenAI, and Llama backends.

    Usage:
        oracle = LLMOracle(provider="gemini", api_key="...")
        oracle = LLMOracle(provider="claude", api_key="sk-ant-...")
        oracle = LLMOracle(provider="llama")  # Local Ollama
        label = oracle.classify(entry)
    """

    def __init__(
        self,
        provider: str = "gemini",
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        llm: BaseLLM | None = None,
    ) -> None:
        self._provider_name = provider
        self._model_name = model_name
        self._api_key = api_key
        self._base_url = base_url
        self._llm = llm

    def _get_llm(self) -> BaseLLM:
        """Lazily initialize the LLM provider."""
        if self._llm is None:
            self._llm = get_provider(
                provider=self._provider_name,
                model_name=self._model_name,
                api_key=self._api_key,
                base_url=self._base_url,
                temperature=0.1,
                max_tokens=512,
            )
        return self._llm

    def classify(self, entry: QuarantineEntry) -> OracleLabel:
        """Classify a quarantined image using the multimodal LLM.

        Args:
            entry: QuarantineEntry with image path and model predictions.

        Returns:
            OracleLabel with LLM classification.

        Raises:
            FileNotFoundError: If the image file doesn't exist.
            RuntimeError: If LLM response cannot be parsed.
        """
        import PIL.Image

        # Find the quarantined image copy
        image_path = Path(entry.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = PIL.Image.open(image_path)

        # Format top predictions for context
        top_species = ", ".join(
            f"{name} ({prob:.1%})" for name, prob in entry.top_k_species[:3]
        )
        top_pathology = ", ".join(
            f"{name} ({prob:.1%})" for name, prob in entry.top_k_pathology[:3]
        )

        prompt = LLM_CLASSIFICATION_PROMPT.format(
            top_species=top_species,
            top_pathology=top_pathology,
        )

        llm = self._get_llm()
        response_text = llm.generate_with_image(
            "You are an expert botanist specializing in medicinal plants.",
            prompt,
            img,
        )

        return self._parse_response(entry, response_text)

    def _parse_response(self, entry: QuarantineEntry, response_text: str) -> OracleLabel:
        """Parse LLM JSON response into an OracleLabel."""
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse LLM oracle response: {e}") from e

        return OracleLabel(
            quarantine_id=entry.quarantine_id,
            image_path=entry.image_path,
            species_label=data.get("species", "Unknown"),
            pathology_label=data.get("pathology", "Healthy"),
            source=OracleSource.LLM,
            confidence=float(data.get("confidence", 0.0)),
            notes=data.get("notes", ""),
        )


class HumanExpertQueue:
    """JSON-based queue for human Ayurvedic botanist labeling.

    The hardest ~1% of quarantined images that the LLM oracle cannot
    confidently classify are routed to this queue. Labels are persisted
    to a JSON file for later integration.

    Usage:
        queue = HumanExpertQueue(queue_dir="data/expert_queue")
        queue.enqueue(entry)
        # ... human labels via external tool / UI ...
        queue.submit_label(quarantine_id, "Azadirachta indica", "Bacterial Spot")
    """

    def __init__(self, queue_dir: str | Path = "data/expert_queue") -> None:
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self._pending: list[QuarantineEntry] = []
        self._labeled: list[OracleLabel] = []
        self._load()

    def _pending_path(self) -> Path:
        return self.queue_dir / "pending.json"

    def _labeled_path(self) -> Path:
        return self.queue_dir / "labeled.json"

    def _load(self) -> None:
        """Load persisted queue state."""
        pending_file = self._pending_path()
        if pending_file.exists():
            data = json.loads(pending_file.read_text(encoding="utf-8"))
            self._pending = [QuarantineEntry.from_dict(e) for e in data]

        labeled_file = self._labeled_path()
        if labeled_file.exists():
            data = json.loads(labeled_file.read_text(encoding="utf-8"))
            self._labeled = [OracleLabel.from_dict(e) for e in data]

    def _save(self) -> None:
        """Persist queue state to JSON files."""
        self._pending_path().write_text(
            json.dumps([e.to_dict() for e in self._pending], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._labeled_path().write_text(
            json.dumps([l.to_dict() for l in self._labeled], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def enqueue(self, entry: QuarantineEntry) -> None:
        """Add a quarantined image to the human expert queue."""
        # Avoid duplicates
        if any(e.quarantine_id == entry.quarantine_id for e in self._pending):
            return
        self._pending.append(entry)
        self._save()

    def submit_label(
        self,
        quarantine_id: str,
        species_label: str,
        pathology_label: str,
        notes: str = "",
    ) -> OracleLabel | None:
        """Submit a human expert label for a pending image.

        Returns the OracleLabel if found, None if quarantine_id not in queue.
        """
        for i, entry in enumerate(self._pending):
            if entry.quarantine_id == quarantine_id:
                label = OracleLabel(
                    quarantine_id=quarantine_id,
                    image_path=entry.image_path,
                    species_label=species_label,
                    pathology_label=pathology_label,
                    source=OracleSource.HUMAN_EXPERT,
                    confidence=1.0,  # Human expert = full confidence
                    notes=notes,
                )
                self._labeled.append(label)
                self._pending.pop(i)
                self._save()
                return label
        return None

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def labeled_count(self) -> int:
        return len(self._labeled)

    @property
    def all_labels(self) -> list[OracleLabel]:
        return list(self._labeled)


class OraclePipeline:
    """Orchestrates the full oracle labeling pipeline.

    Flow:
        1. Process quarantined images through LLM oracle
        2. If LLM confidence >= threshold -> accept label
        3. If LLM confidence < threshold -> route to human expert queue
        4. Apply labels back to QuarantineManager
    """

    def __init__(
        self,
        quarantine: QuarantineManager,
        llm_oracle: LLMOracle | None = None,
        expert_queue: HumanExpertQueue | None = None,
        llm_confidence_threshold: float = 0.7,
    ) -> None:
        self.quarantine = quarantine
        self.llm_oracle = llm_oracle
        self.expert_queue = expert_queue
        self.llm_confidence_threshold = llm_confidence_threshold

    def process_pending(self) -> dict[str, int]:
        """Process all pending quarantine entries through the oracle pipeline.

        Returns:
            Dict with counts: {"llm_labeled", "human_routed", "errors"}.
        """
        stats = {"llm_labeled": 0, "human_routed": 0, "errors": 0}
        pending = self.quarantine.pending()

        for entry in pending:
            if self.llm_oracle is None:
                # No LLM available — route all to human
                if self.expert_queue:
                    self.expert_queue.enqueue(entry)
                    stats["human_routed"] += 1
                continue

            try:
                label = self.llm_oracle.classify(entry)

                if label.confidence >= self.llm_confidence_threshold:
                    # LLM confident — accept label
                    self.quarantine.mark_labeled(
                        entry.quarantine_id, label.species_label, label.pathology_label
                    )
                    stats["llm_labeled"] += 1
                elif self.expert_queue:
                    # LLM unsure — route to human
                    self.expert_queue.enqueue(entry)
                    stats["human_routed"] += 1
            except Exception:
                stats["errors"] += 1
                if self.expert_queue:
                    self.expert_queue.enqueue(entry)
                    stats["human_routed"] += 1

        return stats

    def apply_human_labels(self) -> int:
        """Apply completed human expert labels back to quarantine.

        Returns:
            Number of labels applied.
        """
        if not self.expert_queue:
            return 0

        count = 0
        for label in self.expert_queue.all_labels:
            applied = self.quarantine.mark_labeled(
                label.quarantine_id, label.species_label, label.pathology_label
            )
            if applied:
                count += 1
        return count
