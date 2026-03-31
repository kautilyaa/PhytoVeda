# PhytoVeda

**Multimodal Pharmacognosy: AI-Driven Botanical Identification and Textual Mapping in Ayurveda**

PhytoVeda is a dual-head Vision Transformer system that identifies 8,000+ medicinal plant species from leaf images, diagnoses plant health and disease status, maps pathological findings to the ancient Vrikshayurveda Tridosha framework (Vata/Pitta/Kapha), and prescribes traditional therapeutic interventions via a RAG-powered LLM indexed against classical Ayurvedic texts. The system is self-sustaining through an active learning pipeline with uncertainty sampling, deployed on Google Colab Enterprise with Python 3.14+ free-threading.

## Tech Stack

- **Language**: Python 3.14+ (free-threaded build with GIL disabled, PEP 734 sub-interpreters via `concurrent.interpreters`, mimalloc-based GC)
- **Deep Learning**: PyTorch with `torch.compile`, `timm` library for ViT model zoo
- **Model Architecture**: Hierarchical Vision Transformer (HierViT) backbone, initialized from DINOv2 / ViT-Huge pretrained weights
- **Training**: Mixed precision (AMP), focal loss for disease head, cross-entropy for species head, dynamic task weighting per epoch
- **LLM / RAG**: Gemini 1.5 Pro (Vertex AI), vector store indexed against Ayurvedic texts
- **Cloud**: Google Colab Enterprise, Vertex AI Pipelines, Google Cloud Storage (GCS)
- **Data**: ~103K+ images federated from 6 datasets, standardized to 512x512 pixels
- **MLOps**: CI/CT/CD via Vertex AI Pipelines, active learning with uncertainty sampling

## Architecture

### Conditional Multi-Task Learning (CMTL) — Dual-Head Design

```
                    Input Image (512x512)
                           |
                    [Patch Embedding + Positional Encoding]
                           |
              ┌────────────────────────────┐
              │   Shared HierViT Backbone   │
              │  (DINOv2 / ViT-Huge init)   │
              │  Progressive spatial reduction│
              │  Increasing channel depth     │
              └──────────┬─────────────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
    ┌─────────┴─────────┐  ┌───────┴────────┐
    │  Species Head     │  │ Pathology Head  │
    │  MLP + Softmax    │  │ MLP + Sigmoid   │
    │  8,000+ classes   │  │ Disease classes  │
    │  Cross-Entropy    │  │ Focal Loss      │
    └───────────────────┘  └─────────────────┘
```

### Loss Function

```
L_total(t) = w_species(t) * L_species + w_disease(t) * L_disease + lambda * L_reg
```

- `L_species`: Cross-entropy loss for species identification
- `L_disease`: Focal loss for disease classification (handles class imbalance — healthy leaves vastly outnumber diseased)
- `w_species(t)`, `w_disease(t)`: Dynamic weights updated every epoch based on loss ratios / gradient magnitudes (NOT static)
- `L_reg`: Orthogonal or entropy regularization for robustness
- `lambda`: Regularization coefficient (hyperparameter)

## Datasets

| Dataset | Images | Species/Classes | Focus | Architectural Utility |
|---------|--------|-----------------|-------|-----------------------|
| **Herbify** | 6,104 | 91 species | Healthy baselines, rich metadata | Primary species classification weights |
| **Assam Medicinal Leaf Set** | 7,341 | 10 classes | Regional morphological variance (NE India) | Prevents geographic overfitting |
| **AI-MedLeafX** | 10,858 orig / 65,178 aug | 4 species | Bacterial Spot, Shot Hole, Powdery Mildew, Yellow Leaf | High-res disease mapping for Neem, Haritaki, Camphor, Moringa |
| **CIMPD** | 9,130 | 23 species | Healthy vs. Unhealthy (unconstrained) | Robust feature extraction in noisy field conditions |
| **SIMP** | 2,503 | 20 species | Herbs, shrubs, creepers, climbers, trees | Morphological diversity beyond ovate leaves |
| **EarlyNSD** | 2,700 | 3 species | Nitrogen & Potassium deficiency | Abiotic stress detection distinct from pathogenic infection |

**Total**: ~38,636 original images; ~103K+ with augmentation. All standardized to **512x512 pixels**.

**Augmentation pipeline**: Multi-angle rotations, horizontal flipping, zooming, brightness/contrast adjustment, Gaussian noise injection, color jittering (H/S/V), affine transformations.

## Key Domain Concepts

### Vrikshayurveda (Ancient Plant Science)

Classical Sanskrit compendium on plant life and health (by Surapala, Varahamihira). Categorizes plant disorders into:
- **Endogenous (Nija)**: Internal physiological imbalances of Vata, Pitta, Kapha
- **Exogenous (Aganthuja)**: External factors (insects, frost, physical trauma)

### Tridosha Mapping (Pathology Output -> Ayurvedic Diagnosis)

| Dosha | Plant Symptoms | CV Features | Traditional Treatment |
|-------|---------------|-------------|----------------------|
| **Vataja Vyadhi (Vata)** | Desiccation, geometric deformations, wilting, curling margins, knots, hard fruits with reduced sap | Edge deformation, texture variance, structural asymmetry | Kunapajala (liquid organic manure), animal-derived fats, Neem (Azadirachta indica) fumigation |
| **Pittaja Vyadhi (Pitta)** | Yellowing/chlorosis, premature shedding, rapid rotting, necrotic lesions, systemic paleness | RGB/HSV colorimetric shifts, localized necrosis, burn artifacts | Yashtimadhu (Glycyrrhiza glabra) decoction, Madhuca indica, milk + honey, cooling Triphala spray |
| **Kaphaja Vyadhi (Kapha)** | Powdery mildew, edema, hypertrophy, dwarfed growth, loss of olfactory/gustatory profiles | Morphological thickening, surface dullness, abnormal Leaf Area Index (LAI) | Panchamoola (five root) decoction, mustard, reduce moisture and oily fertilizers |

### Pharmacological Properties (Rasa-Guna-Virya-Vipaka)

Classical Ayurvedic properties of medicinal herbs that must be preserved through proper plant health:
- **Rasa**: Taste
- **Guna**: Qualities
- **Virya**: Potency
- **Vipaka**: Post-digestive effect

### Active Learning

The model self-improves by identifying uncertain predictions and routing them for labeling:
- **Least Confidence**: max(softmax_probs) < threshold
- **Margin Sampling**: prob_1st - prob_2nd < margin_threshold (e.g., Terminalia chebula vs. T. bellirica confusion)
- **Entropy**: Uniform probability distribution across all classes

Uncertain images are quarantined to GCS, labeled by LLM oracle (Gemini 1.5 Pro / GLM 5) or human Ayurvedic botanist (~1% of images), then used for incremental retraining.

## Development Environment

- **Primary**: Google Colab Enterprise with Gemini 2.5 agentic features
- **Python**: 3.14+ free-threaded build (`PYTHON_GIL=0`)
  - Primary interpreter: model inference API serving
  - Background sub-interpreters: data ingestion, validation, active learning retraining
- **Cloud Storage (GCS)**:
  - `gs://phytoveda-datasets/` — raw and processed datasets
  - `gs://phytoveda-checkpoints/` — model weights
  - `gs://phytoveda-quarantine/` — active learning uncertain images
  - `gs://phytoveda-metrics/` — evaluation artifacts
- **MLOps**: Vertex AI Pipelines for CI/CT/CD orchestration
- **Data Access**: Google Drive mounting via `drive.mount` for persistent data

## Project Structure

```
PhytoVeda/
  CLAUDE.md                     # This file
  README.md                     # Project overview
  ROADMAP.md                    # Implementation roadmap (Phases 0-9)
  instructions.md               # Comprehensive specification
  pyproject.toml                # Python 3.14+ project config
  configs/
    hiervit_cmtl.yaml           # Model and training configuration
  src/
    phytoveda/
      __init__.py
      data/                     # DataLoaders, augmentation, dataset federation
        datasets.py             # FederatedBotanicalDataset, 6 dataset loaders, splits
        augmentation.py         # Albumentations 2.0 augmentation pipeline
        preprocessing.py        # 512x512 normalization, corrupt image handling
        taxonomy.py             # SpeciesTaxonomy and PathologyMapping registries
        download.py             # Dataset download + extraction utilities
      models/                   # HierViT, CMTL heads, loss functions
        backbone.py             # HierViT from timm (DINOv2/ViT-Huge)
        heads.py                # Species (softmax/CE) + Pathology (focal loss) heads
        cmtl.py                 # HierViTCMTL combined model
        losses.py               # Focal loss, dynamic task weighting (DWA)
      training/                 # Training loop, evaluation, checkpointing
        trainer.py              # Epoch-based training with AMP, warmup+cosine LR
        evaluation.py           # F1, accuracy, top-K, confusion matrices
      vrikshayurveda/           # Dosha mapping, feature-to-symptom rules
        mapper.py               # VrikshayurvedaMapper class
      llm/                      # Unified LLM provider abstraction
        providers.py            # Gemini, Claude, OpenAI, Llama — BaseLLM + factory
      rag/                      # Text indexing, vector store, LLM integration
        indexer.py              # Chunk and embed Ayurvedic texts (ChromaDB)
        retriever.py            # Semantic search over vector DB
        report_generator.py     # Multi-LLM report synthesis + offline fallback
      active_learning/          # Uncertainty sampling, oracle loop, quarantine
        uncertainty.py          # Least confidence, margin, entropy sampling
        quarantine.py           # Local quarantine management with manifest
        oracle.py               # LLM oracle + human expert queue + pipeline
      api/                      # FastAPI inference server
        server.py               # POST /identify, GET /health, GET /model/version
        inference.py            # InferencePipeline: image → model → Dosha → report
      conservation/             # IUCN Red List monitoring
        iucn.py                 # ConservationRegistry, endangered species alerts
      traceability/             # Supply chain audit trail
        events.py               # EventLedger, GPS geo-tagging, SHA-256 hash chain
      formulation/              # Classical formulation validation
        validator.py            # FormulationValidator, 5 classical Ayurvedic recipes
      colab/                    # Google Colab integration
        drive.py                # DriveManager: SSD + Drive path strategy, sync
        environment.py          # ColabEnvironment: GPU, packages, free-threading
        training.py             # ColabTrainer, grad accum, crash ckpt, GPU monitor, auto batch
        data_cache.py           # DatasetCache: splits, taxonomy, history, download tracking
  tests/
    test_data.py                # 76 tests — data pipeline, taxonomy, augmentation
    test_models.py              # Model architecture tests
    test_training.py            # 22 tests — trainer, evaluation, checkpointing
    test_uncertainty.py         # Uncertainty sampler tests
    test_vrikshayurveda.py      # Dosha mapping tests
    test_rag.py                 # 28 tests — indexer, retriever, report generator
    test_llm_providers.py       # 24 tests — multi-LLM provider abstraction
    test_active_learning.py     # 20 tests — quarantine, oracle, pipeline
    test_api.py                 # 30 tests — inference pipeline, API endpoints
    test_conservation.py        # 17 tests — IUCN registry, alerts
    test_traceability.py        # 15 tests — GPS, hash chain, ledger
    test_formulation.py         # 17 tests — formulation validation
    test_colab.py               # 57 tests — DriveManager, ColabEnvironment
    test_colab_training.py      # 52 tests — grad accum, crash ckpt, GPU monitor, dataset cache
  notebooks/
    phytoveda_colab.ipynb       # Full pipeline Colab notebook (Drive-backed)
  data/                         # Local data cache / symlinks to GCS
```

## Conventions

- All images preprocessed to **512x512 pixels** with ImageNet normalization
- **Focal loss** for disease/pathology head; **cross-entropy** for species head
- Dynamic task weights updated **per epoch** (never static manual tuning)
- HierViT backbone preferred over flat ViT for multi-scale feature extraction
- Data loaders **must leverage free-threading** to saturate all CPU cores
- Model checkpoints saved to GCS with **F1-score gating** (deploy only if F1 improves)
- Type hints required (Python 3.14 style)
- Docstrings for all public functions
- YAML-based configuration for all hyperparameters

## Key Commands

```bash
# Training (supports CLI overrides for any config field)
python -m phytoveda.training.trainer --config configs/hiervit_cmtl.yaml
python -m phytoveda.training.trainer --config configs/hiervit_cmtl.yaml --epochs 50 --lr 1e-4 --batch-size 32

# Resume training from checkpoint
python -m phytoveda.training.trainer --config configs/hiervit_cmtl.yaml --resume checkpoints/best_model.pt

# Launch inference API server
python -m phytoveda.api.server --checkpoint checkpoints/best_model.pt
python -m phytoveda.api.server --checkpoint checkpoints/best_model.pt --chromadb-dir data/chromadb --gemini-api-key $KEY

# Run tests
pytest tests/ -v

# Lint and format
ruff check src/ && ruff format src/
```

## RAG Knowledge Base

The LLM is indexed against these classical texts for report generation:
1. **Charaka Samhita** — Foundational Ayurvedic medical text
2. **Susruta Samhita** — Classical surgical/medical text
3. **Ayurvedic Pharmacopoeia of India (API)** — Official pharmacological reference
4. **Vrikshayurveda** — Plant science texts by Surapala and Varahamihira

**Report output includes**: Authenticated botanical identity, classical medicinal properties (Rasa/Guna/Virya/Vipaka), current health status, Vrikshayurveda Dosha assessment, traditional therapeutic intervention, and pharmaceutical procurement quality assessment.
