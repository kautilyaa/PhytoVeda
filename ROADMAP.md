# PhytoVeda Implementation Roadmap

**Multimodal Pharmacognosy: AI-Driven Botanical Identification and Textual Mapping in Ayurveda**

Last Updated: 2026-03-31

**Status Legend**: `[ ]` Not started | `[~]` In progress | `[x]` Complete

---

## Phase Dependencies

```
Phase 0 (Docs) --> Phase 1 (Setup) --> Phase 2 (Data) --> Phase 3 (Model) --> Phase 4 (Training)
                                                                                     |
                                                          Phase 5 (Vrikshayurveda) <-+-> Phase 6 (RAG)
                                                                                     |
                                                                              Phase 7 (Active Learning)
                                                                                     |
                                                                              Phase 8 (API) ✅
                                                                                     |
                                                                              Phase 9 (Advanced) ✅
```

Phases 5 and 6 can be developed **in parallel** once Phase 4 is complete. Phase 7 depends on Phases 4 and 6. Phases 8 and 9 are sequential after Phase 7.

---

## Phase 0: Documentation and Planning ✅

**Goal**: Establish the project specification and development context.

- [x] Write `instructions.md` — comprehensive architectural blueprint (42KB)
- [x] Create `CLAUDE.md` — Claude Code context file with tech stack, architecture, and conventions
- [x] Create `ROADMAP.md` — this document

**Deliverable**: Complete project documentation ready for development kickoff.

---

## Phase 1: Project Setup and Infrastructure ✅

**Goal**: Establish the development environment and project skeleton.

### Tasks
- [x] Initialize Python 3.14+ project with `pyproject.toml`
  - `requires-python = ">=3.14"`
  - Configure free-threaded build flag (`PYTHON_GIL=0`)
  - Core dependencies: torch, torchvision, timm, numpy, Pillow, scikit-learn, albumentations, pyyaml, chromadb, sentence-transformers, google-generativeai, wandb, ruff, pytest
- [x] Set up Google Colab Enterprise notebook template
  - `DriveManager` class: SSD paths for datasets, Drive paths for persistent storage
  - `ColabEnvironment` class: GPU detection, package installation, env verification
  - Full pipeline notebook (`notebooks/phytoveda_colab.ipynb`) with Drive integration
  - Mount Google Drive via `drive.mount`, scaffold all directories
- [ ] Provision GCS buckets:
  - `gs://phytoveda-datasets/` — raw and processed datasets
  - `gs://phytoveda-checkpoints/` — model weights and optimizer states
  - `gs://phytoveda-quarantine/` — active learning uncertain images
  - `gs://phytoveda-metrics/` — evaluation artifacts and logs
- [x] Scaffold project directory structure (as defined in CLAUDE.md)
- [x] Configure tooling: `ruff` (lint + format), `mypy` (type checking), `pytest`
- [ ] Verify Python 3.14+ free-threading works in Colab environment

**Deliverable**: Running Python 3.14+ environment on Colab with GCS access. Empty project skeleton passes `pytest` and linting.

---

## Phase 2: Data Pipeline ✅

**Goal**: Download, validate, preprocess, augment, and federate all 6 datasets into a unified DataLoader.

### Dataset Acquisition
- [x] Build dataset download scripts (`download.py`) with platform-specific handlers (Kaggle, GitHub, Mendeley)
- [x] Download validation with 90% image count tolerance

### Unified Label Taxonomy
- [x] Build species label mapping across all 6 datasets — `SpeciesTaxonomy` class handles overlapping species (e.g., Neem appears in Herbify, MedLeafX, CIMPD → single ID)
- [x] Build pathology label taxonomy: 8 unified classes (Healthy, Bacterial Spot, Shot Hole, Powdery Mildew, Yellow Leaf Disease, Nitrogen Deficiency, Potassium Deficiency, Unhealthy)
- [x] Handle missing labels: datasets without pathology annotations default to Healthy (ID 0)

### Preprocessing Pipeline
- [x] Resize all images to **512x512 pixels** with LANCZOS interpolation
- [x] Normalize pixel values (ImageNet mean/std)
- [x] Implement corrupt/missing image detection (`validate_image`, black tensor fallback)

### Augmentation Pipeline (On-the-fly During Training)
- [x] Implement with albumentations:
  - RandomResizedCrop, HorizontalFlip, RandomRotate90, Rotate (±45°)
  - ColorJitter (brightness/contrast/saturation/hue)
  - GaussNoise, GaussianBlur, Affine transforms
  - ImageNet normalization + ToTensorV2

### Dataset Federation
- [x] Create `FederatedBotanicalDataset` class wrapping all 6 datasets
  - Returns `(image_tensor[3, 512, 512], species_id, pathology_id)`
  - Dual-label support with graceful handling of missing pathology labels
- [x] Implement **weighted sampling** (`build_weighted_sampler`) for class imbalance
- [x] **Train/validation/test splits**: 70/15/15 stratified by species+pathology distribution

### DataLoader Configuration
- [x] PyTorch DataLoader with configurable `num_workers`, `pin_memory`, `persistent_workers`
- [x] `build_dataloaders()` assembles full pipeline: load → taxonomy → split → augment → DataLoader

### Tests
- [x] 76 tests in `test_data.py`: taxonomy, pathology mapping, all 6 loaders, FederatedBotanicalDataset, build_datasets, weighted sampler, augmentation, preprocessing, download validation

**Deliverable**: `FederatedBotanicalDataset` class yielding `(image_tensor[3,512,512], species_id, pathology_id)` tuples. Verified with unit tests showing correct label distribution, augmentation diversity, and no data leakage.

---

## Phase 3: Model Architecture — Vision Transformer ✅

**Goal**: Implement the Hierarchical ViT backbone with dual CMTL classification heads.

### Shared HierViT Backbone
- [x] Load pretrained foundation model from `timm` (DINOv2 / ViT-Huge variants)
- [x] `HierViTBackbone` class: `num_classes=0` feature extractor, returns `(B, feature_dim)`

### Species Classification Head
- [x] `SpeciesHead`: MLP (Linear → GELU → Dropout → Linear) over backbone features
- [x] Loss: **Cross-Entropy** via `nn.CrossEntropyLoss`

### Pathology Classification Head
- [x] `PathologyHead`: Parallel MLP on same backbone features
- [x] Loss: **Focal Loss** with configurable gamma and per-class alpha

### Dynamic Task Weighting
- [x] `DynamicTaskWeighting` with Dynamic Weight Average (DWA) strategy
- [x] `w_species(t)` and `w_disease(t)` computed from loss ratios via softmax normalization
- [ ] Additional strategies: Uncertainty weighting, GradNorm (stretch goal)

### Combined Loss Function
- [x] `CMTLLoss`: `L_total = w_species * L_species + w_disease * L_disease + lambda * L_reg`
- [x] Entropy regularization on species predictions for robustness
- [x] All parameters configurable (gamma, alpha, lambda, temperature)

### Configuration
- [x] YAML-based model config (`configs/hiervit_cmtl.yaml`)

### Tests
- [x] 3 tests in `test_models.py`: FocalLoss shape, FocalLoss with alpha, CMTLLoss components

**Deliverable**: `HierViTCMTL` model class that accepts a batch of 512x512 images and returns `(species_logits, disease_logits)`. All parameters configurable via YAML.

---

## Phase 4: Training Pipeline ✅

**Goal**: End-to-end training loop with evaluation, checkpointing, and experiment tracking.

### Training Loop
- [x] Epoch-based training with configurable epochs
- [x] Dynamic task weight update at each epoch boundary (via `DynamicTaskWeighting`)
- [x] **Gradient clipping** (`nn.utils.clip_grad_norm_`, configurable max norm)
- [x] **Learning rate scheduling**: Linear warmup + Cosine annealing (`SequentialLR`)
- [x] **Mixed precision training**: `torch.amp.autocast` + `GradScaler` (CUDA-aware, CPU-safe)

### Evaluation Metrics
- [x] Per-task accuracy (species, pathology)
- [x] Per-task weighted **F1-score** (primary gating metric)
- [x] **Top-5 accuracy** for species head
- [x] Per-class precision, recall, F1 via `evaluate_detailed()`
- [x] Confusion matrices for both heads

### Checkpointing
- [x] `CheckpointManager` with top-K F1-gated saving (min-heap eviction)
- [x] Saves model weights, optimizer state, scheduler state, epoch, metrics
- [x] Always maintains `best_model.pt` symlink

### CLI Entry Point
- [x] `main()` in `trainer.py`: loads YAML config, builds data pipeline + model, trains
- [x] CLI args: `--config`, `--data-root`, `--epochs`, `--batch-size`, `--lr`, `--resume`
- [x] `TrainConfig.from_yaml()` maps nested YAML to flat dataclass with CLI overrides

### Tests
- [x] 22 tests in `test_training.py`: config loading (YAML/partial/overrides), CheckpointManager (save/evict/contents), evaluation (metrics/ranges/top-5/detailed), Trainer (init/epoch/full loop/warmup/no-warmup/resume)

**Deliverable**: Training script that trains HierViTCMTL on federated dataset, produces checkpoints, and logs metrics. F1-score reported for both species and pathology heads on validation set.

---

## Phase 5: Vrikshayurveda Mapping System ✅

**Goal**: Translate model pathology outputs into Tridosha diagnoses with traditional treatments.

### Dosha Classification Logic
- [x] **Vataja Vyadhi (Vata)**: Shot Hole, Nitrogen Deficiency, Potassium Deficiency
- [x] **Pittaja Vyadhi (Pitta)**: Yellow Leaf Disease, Bacterial Spot, Unhealthy
- [x] **Kaphaja Vyadhi (Kapha)**: Powdery Mildew

### Feature-to-Symptom Mapping
- [x] Deterministic `PATHOLOGY_TO_DOSHA` mapping
- [x] CV feature correlates per Dosha (`DOSHA_CV_FEATURES`)
- [x] Classical symptoms from Vrikshayurveda texts (`DOSHA_CLASSICAL_SYMPTOMS`)
- [x] Confidence scoring from pathology prediction confidence

### Treatment Knowledge Base
- [x] `DOSHA_TREATMENTS` dict with traditional interventions per Dosha
- [x] `DOSHA_CONTRAINDICATIONS` dict
- [x] `treatments.yaml` — structured primary/secondary treatments with descriptions

### Integration
- [x] `VrikshayurvedaMapper.assess()` — returns `DoshaAssessment` with dosha, confidence, CV features, symptoms, treatments, contraindications

### Tests
- [x] 5 tests in `test_vrikshayurveda.py`: Vata, Pitta, Kapha, Healthy, Nitrogen Deficiency mappings

**Deliverable**: `VrikshayurvedaMapper` class that accepts model outputs and returns structured Dosha diagnosis with traditional treatment recommendations.

---

## Phase 6: RAG and LLM Integration ✅

**Goal**: Build retrieval-augmented generation pipeline indexed against Ayurvedic texts for comprehensive report generation.

### Text Indexing
- [x] `AyurvedicTextIndexer`: load text files from hierarchical or flat directory layouts
- [x] `chunk_text()`: paragraph-based splitting with configurable chunk size and overlap
- [x] `build_index()`: ChromaDB persistent collection with source/chapter metadata
- [x] Source name normalization (directory → canonical name mapping)

### RAG Pipeline
- [x] `build_query()`: construct queries from species + pathology + Dosha classification
- [x] `AyurvedicRetriever.retrieve()`: ChromaDB vector similarity search
- [x] `retrieve_for_diagnosis()`: convenience method combining query construction + retrieval
- [x] Source filtering support (restrict to specific texts)

### LLM Report Generation
- [x] `ReportGenerator.generate()`: Gemini 1.5 Pro API with structured JSON output
- [x] `build_report_prompt()`: prompt template with RAG context and model predictions
- [x] `generate_offline()`: fallback report using only Dosha mapping (no API needed)
- [x] `BotanicalReport` dataclass with `.to_json()` and `.to_markdown()` output formats
- [x] `_parse_response()`: JSON parsing with error handling

### Report Structure
- [x] Authenticated botanical identity (species, family, Sanskrit/common names)
- [x] Classical medicinal properties (Rasa, Guna, Virya, Vipaka)
- [x] Health status and pathology diagnosis
- [x] Vrikshayurveda Dosha assessment
- [x] Traditional therapeutic intervention
- [x] Procurement quality assessment

### Multi-LLM Provider Abstraction
- [x] `BaseLLM` abstract class with `generate()` and `generate_with_image()` interface
- [x] `GeminiLLM` — google-generativeai SDK (default provider)
- [x] `ClaudeLLM` — anthropic SDK with base64 image encoding
- [x] `OpenAILLM` — openai SDK with JSON response format
- [x] `LlamaLLM` — Ollama local server / Together.ai / any OpenAI-compatible endpoint
- [x] `LLMConfig` dataclass with `resolved_model` property (default model per provider)
- [x] `get_provider()` factory: case-insensitive, supports string or enum, validates provider name
- [x] `ReportGenerator` and `LLMOracle` updated to use provider abstraction (lazy init, injectable)
- [x] Optional dependencies in `pyproject.toml`: `[claude]`, `[openai]`, `[llama]`, `[all-llms]`

### Tests
- [x] 28 tests in `test_rag.py`: chunking (basic/overlap/empty/unique IDs), source normalization, indexer (hierarchical/flat/empty/build/rebuild), query building, retriever (basic/relevance/filter/diagnosis/empty), report generator (offline/JSON/markdown/parse/invalid)
- [x] 24 tests in `test_llm_providers.py`: LLMProvider enum, LLMConfig (defaults/resolution/override), get_provider factory (all 4 providers, enum, case-insensitive, unknown, custom params), provider structure (BaseLLM, generate, generate_with_image), ReportGenerator multi-LLM integration

**Deliverable**: End-to-end pipeline that takes model outputs, queries Ayurvedic texts, and generates a comprehensive LLM-synthesized report. Supports Gemini, Claude, OpenAI, and Llama backends via a unified provider abstraction.

---

## Phase 7: Active Learning Pipeline ✅

**Goal**: Implement self-sustaining continuous learning with uncertainty sampling and oracle labeling.

### Uncertainty Sampling Module
- [x] **Least Confidence**: `1 - max(softmax_probs)` > threshold
- [x] **Margin Sampling**: `prob_1st - prob_2nd` < margin threshold
- [x] **Entropy**: `-sum(p * log(p))` > entropy threshold
- [x] **Combined scoring**: weighted combination of all three strategies
- [x] Configurable thresholds per strategy

### Quarantine Pipeline
- [x] `QuarantineManager`: copies uncertain images + JSON metadata to local quarantine directory
- [x] Persistent manifest (`manifest.json`) surviving process restarts
- [x] `mark_labeled()`: apply oracle labels back to quarantine entries
- [x] `export_for_retraining()`: extract labeled images as `(path, species, pathology)` tuples
- [x] Tracking: `total_count`, `pending_count`, `labeled_count`, `summary()`

### Oracle Loop
- [x] `LLMOracle`: Gemini 1.5 Pro multimodal classification of quarantined leaf images
- [x] `HumanExpertQueue`: JSON-persisted queue with `enqueue()` / `submit_label()` / deduplication
- [x] `OracleLabel` dataclass with source tracking (LLM vs human expert)
- [x] `OraclePipeline`: orchestrates LLM-first, human-fallback flow based on confidence threshold
- [x] `apply_human_labels()`: syncs expert labels back to quarantine manager

### Tests
- [x] 20 tests in `test_active_learning.py`: quarantine (save/copy/metadata/manifest/label/export/multi/summary), entry serialization, expert queue (enqueue/dedup/submit/persist), oracle label round-trip, pipeline (routing/apply), end-to-end uncertainty→quarantine→label→export flow
- [x] 4 tests in `test_uncertainty.py`: confident/uncertain/margin/batch scoring

**Deliverable**: Pipeline: inference → uncertainty scores → quarantine → oracle labeling → retraining data export. Self-sustaining with LLM oracle + human expert fallback.

---

## Phase 8: API and Deployment ✅

**Goal**: User-facing inference API for plant identification and diagnosis.

### Inference Pipeline
- [x] `InferencePipeline` class: image → model → Dosha → RAG → report
- [x] `from_checkpoint()`: load pipeline from saved `.pt` checkpoint with config extraction
- [x] Configurable `image_size` — transform matches model's expected input
- [x] `PredictionResult` dataclass with `.to_dict()` serialization
- [x] Top-K species and pathology predictions with confidence scores
- [x] Uncertainty scoring integrated (active learning signal in every response)
- [x] Optional report generation with Gemini LLM (falls back to offline)

### FastAPI Server
- [x] `POST /identify` — upload plant leaf image, get full diagnosis (JPEG/PNG)
  - Species identification (top-K with confidence)
  - Pathology/disease diagnosis (top-K with confidence)
  - Vrikshayurveda Tridosha assessment (Dosha, CV features, treatments)
  - Uncertainty score for active learning
  - Optional full botanical report (`generate_report=true`)
  - Inference latency tracked (`inference_time_ms`)
- [x] `GET /health` — service health check (status, model_loaded, device, uptime)
- [x] `GET /model/version` — model version, metrics, device, num_species
- [x] Input validation: file type check (415), corrupt image handling (400), no-model guard (503)

### CLI Entry Point
- [x] `python -m phytoveda.api.server --checkpoint <path>` launches uvicorn
- [x] CLI args: `--host`, `--port`, `--device`, `--chromadb-dir`, `--gemini-api-key`, `--workers`
- [x] Optional RAG retrieval (requires `--chromadb-dir` with built index)
- [x] Optional LLM reports (requires `--gemini-api-key`)

### Dependencies
- [x] `fastapi`, `uvicorn[standard]`, `python-multipart` added to `pyproject.toml`

### Tests
- [x] 30 tests in `test_api.py`: inference transform (shape/normalize), InferencePipeline (predict/fields/dosha/uncertainty/top-k/clamp/report/dict/preprocess/sizes), checkpoint loading (load/config/override), API endpoints (health/no-model/version/JPEG/PNG/top-k/unsupported/invalid/no-model/shape/report)

### Future (stretch goals)
- [x] `torch.compile` optimization for low-latency inference (Colab-safe fallback in `compile_model()`)
- [ ] Python 3.14 sub-interpreters: primary serves API, background handles retraining
- [ ] Model quantization (INT8) and ONNX export for mobile
- [ ] Batch inference for pharmaceutical auditing workflows

**Deliverable**: Deployable FastAPI inference server. Upload a leaf image, receive species + pathology + Dosha + treatments + optional LLM report.

---

## Phase 9: Advanced Features ✅

**Goal**: Enterprise and conservation features for commercial viability and ecological impact.

### Conservation Monitoring (IUCN Red List)
- [x] `ConservationStatus` enum: NE, DD, LC, NT, VU, EN, CR, EW, EX
- [x] `ConservationRegistry` with built-in data for 14 key Ayurvedic species
  - CR: Aquilaria malaccensis (Agarwood), Nardostachys jatamansi (Spikenard)
  - EN: Rauvolfia serpentina (Sarpagandha)
  - VU: Santalum album (Sandalwood), Saraca asoca (Ashoka)
  - NT: Piper longum (Pippali), Glycyrrhiza glabra (Yashtimadhu)
  - LC: Neem, Tulsi, Haritaki, Moringa, Turmeric, Camphor
- [x] `ConservationAlert` with severity levels: warning (VU), critical (EN), harvest_prohibited (CR)
- [x] `check()` returns alert with harvest_allowed flag and conservation message
- [x] `get_threatened_species()` — aggregate query for all at-risk species
- [x] Extensible: `register()` to add custom species data

### Supply Chain Traceability (GPS + Blockchain-Ready Hashing)
- [x] `IdentificationEvent` dataclass: species, pathology, confidence, GPS, Dosha, operator, batch
- [x] `GeoLocation` dataclass: latitude, longitude, altitude, accuracy
- [x] SHA-256 cryptographic hashing of every event
- [x] **Hash chain**: each event links to previous event's hash (blockchain-ready audit trail)
- [x] `EventLedger`: JSONL append-only persistent ledger
  - `record()` — log identification with GPS, operator, batch ID
  - `verify_chain()` — verify tamper-proof integrity of entire chain
  - `get_events_by_species()`, `get_events_by_batch()` — query filters
  - `get_events_in_region()` — geographic bounding box filter for biodiversity mapping
  - `biodiversity_summary()` — species frequency counts for heat maps

### Formulation Validation
- [x] `FormulationValidator` with knowledge base of 5 classical formulations:
  - Triphala (3 herbs), Trikatu (3), Dashamoola (10), Chyawanprash (5), Lekhniya Mahakashaya (5)
- [x] Each formulation: name, Sanskrit name, category, source text, herbs, therapeutic use, contraindications
- [x] `validate()` checks identified herbs against required formulation components
- [x] Per-herb status: verified, missing, unhealthy, low_confidence
- [x] Overall quality: pass (all verified), conditional (present but issues), fail (missing herbs)
- [x] `FormulationValidationResult.to_dict()` for API serialization
- [x] Extensible: `register_formulation()` for custom recipes

### Tests
- [x] 17 tests in `test_conservation.py`: status enum, registry (lookup/case/custom/alerts), severity levels (VU/EN/CR), aggregate queries
- [x] 15 tests in `test_traceability.py`: GeoLocation, event hashing (deterministic/tamper), hash chain (valid/tampered/empty), persistence, query filters (species/batch/region), biodiversity summary
- [x] 17 tests in `test_formulation.py`: knowledge base, lookup, Triphala (pass/missing/unhealthy/low-confidence), Dashamoola, custom formulation, edge cases

**Deliverable**: Enterprise-grade conservation monitoring (IUCN alerts), tamper-proof supply chain traceability (SHA-256 hash chain with GPS), and classical formulation validation against Ayurvedic texts.

---

## Test Suite Summary

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_data.py` | 76 | Taxonomy, pathology mapping, 6 dataset loaders, FederatedBotanicalDataset, splits, sampler, augmentation, preprocessing, download |
| `test_models.py` | 3 | FocalLoss, CMTLLoss |
| `test_training.py` | 22 | TrainConfig, CheckpointManager, evaluation, Trainer |
| `test_vrikshayurveda.py` | 5 | Dosha mapping (Vata/Pitta/Kapha/Healthy/N-deficiency) |
| `test_uncertainty.py` | 4 | Uncertainty scoring strategies |
| `test_rag.py` | 28 | Indexer, retriever, report generator |
| `test_llm_providers.py` | 24 | LLMProvider enum, LLMConfig, get_provider factory, provider structure, ReportGenerator multi-LLM |
| `test_active_learning.py` | 20 | Quarantine, expert queue, oracle pipeline, E2E flow |
| `test_api.py` | 30 | Inference pipeline, checkpoint loading, FastAPI endpoints |
| `test_conservation.py` | 17 | IUCN registry, conservation alerts, severity levels |
| `test_traceability.py` | 15 | GPS, event hashing, hash chain, ledger queries |
| `test_formulation.py` | 17 | Formulation validation, Triphala/Dashamoola, custom recipes |
| `test_colab.py` | 57 | DriveManager paths/scaffold/sync, ColabEnvironment, human_size |
| `test_colab_training.py` | 52 | GradAccumConfig, CrashCheckpointer, GPUMonitor, compile_model, DatasetCache, ColabTrainer |
| **Total** | **370** | |

---

## Key Technical Specifications Reference

### Loss Function
```
L_total(t) = w_species(t) * L_species + w_disease(t) * L_disease + lambda * L_reg
```
- `L_species`: Cross-Entropy loss
- `L_disease`: Focal loss (gamma=2.0, alpha=per-class weights)
- Dynamic weights updated per epoch (loss ratio / gradient magnitude method)
- `L_reg`: Entropy regularization on species predictions

### Dataset Summary
| Dataset | Original Images | Augmented | Species | Key Pathologies |
|---------|-----------------|-----------|---------|-----------------|
| Herbify | 6,104 | — | 91 | Healthy baselines |
| Assam | 7,341 | — | 10 | Regional variance |
| AI-MedLeafX | 10,858 | 65,178 | 4 | Bacterial Spot, Shot Hole, Powdery Mildew, Yellow Leaf |
| CIMPD | 9,130 | — | 23 | Healthy/Unhealthy |
| SIMP | 2,503 | — | 20 | Morphological diversity |
| EarlyNSD | 2,700 | — | 3 | N/K deficiency |
| **Total** | **~38,636** | **~103K+** | **~147 unique** | — |
