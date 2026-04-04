# PhytoVeda — Project Overview

**Multimodal Pharmacognosy: AI-Driven Botanical Identification and Textual Mapping in Ayurveda**

---

## Problem Statement

The taxonomic crisis in Ayurvedic pharmacognosy: India has 8,000+ medicinal plant species, but manual identification is error-prone, slow, and the expertise is dying — concentrated in aging rural practitioners with no successors. This creates three cascading failures:

1. **Adulteration** — Visually similar but pharmacologically inert/toxic plants enter the supply chain. A misidentified herb in a Triphala formulation doesn't just fail therapeutically — it can introduce mycotoxins.
2. **Extinction** — Unchecked wild-harvesting pushes critical species (Sandalwood, Sarpagandha, Agarwood) toward IUCN Red List status because harvesters can't distinguish endangered species from common ones.
3. **Quality degradation** — Even correctly identified plants may be diseased (Powdery Mildew, Bacterial Spot) or nutrient-deficient (N/K), which alters their secondary metabolite profiles. A Neem leaf with severe leaf blotch won't yield the curcuminoid concentrations needed for anti-inflammatory formulations.

No existing system does both — current botanical AI only identifies species. It doesn't assess plant health, diagnose disease through the Ayurvedic lens, or prescribe traditional remedies.

---

## Why

| Dimension | The Problem | The Stakes |
|-----------|------------|------------|
| **Economic** | $23B Ayurvedic industry by 2028, built on unverified raw inputs | Contaminated batches lead to product recalls, regulatory shutdowns |
| **Knowledge loss** | Deep traditional botanical expertise restricted to aging practitioners | One generation away from irreversible loss of 5,000 years of plant science |
| **Ecological** | Over-harvesting of wild medicinal plants | IUCN-listed species (Aquilaria malaccensis, Rauvolfia serpentina) heading toward extinction |
| **Medical safety** | Diseased plants with fungal toxins enter formulations | Patients receive therapeutically void or actively harmful medicine |
| **Pharmacological** | Plant health directly modulates medicinal potency | A nitrogen-deficient Curcuma longa produces inadequate curcuminoids — the drug doesn't work |

The core insight: Species ID alone is insufficient. A plant's **health status** determines whether it is pharmacologically viable. You need both heads — taxonomy AND pathology — evaluated together.

---

## What

PhytoVeda is a **dual-head Vision Transformer system** that performs 6 functions from a single leaf image:

```
                         Leaf Image (512x512)
                                |
                    +-----------+-----------+
                    |   HierViT Backbone     |
                    |   (DINOv2 pretrained)   |
                    +-----------+-----------+
                                |
                    +-----------+-----------+
                    |                       |
            +-------+-------+       +-------+-------+
            | Species Head  |       | Pathology Head|
            | 77+ classes   |       | 8 classes     |
            +-------+-------+       +-------+-------+
                    |                       |
                    v                       v
           1. Species ID            2. Disease Diagnosis
           "Azadirachta indica"     "Bacterial Spot (92%)"
                    |                       |
                    +-----------+-----------+
                                |
                    3. Vrikshayurveda Dosha Mapping
                       "Pittaja Vyadhi (Pitta)"
                                |
                    4. RAG over Classical Texts
                       (Charaka, Susruta, API, Vrikshayurveda)
                                |
                    5. LLM Report Generation
                       (Rasa/Guna/Virya/Vipaka + treatments)
                                |
                    6. Conservation + Traceability
                       (IUCN alerts, GPS hash chain, formulation validation)
```

### The 6 Outputs

| # | Output | Botanical Significance |
|---|--------|----------------------|
| 1 | **Species identification** (top-K with confidence) | Authenticates the plant — is this actually Neem or a look-alike? |
| 2 | **Pathology diagnosis** (8 classes) | Determines health: Healthy, Bacterial Spot, Shot Hole, Powdery Mildew, Yellow Leaf, N-deficiency, K-deficiency, Unhealthy |
| 3 | **Tridosha mapping** | Translates modern pathology into Vrikshayurveda framework — Vataja (structural), Pittaja (metabolic), Kaphaja (moisture) |
| 4 | **Classical text retrieval** | Retrieves relevant passages from Charaka Samhita, Susruta Samhita, Ayurvedic Pharmacopoeia |
| 5 | **Therapeutic intervention** | Prescribes traditional treatments: Kunapajala for Vata, Yashtimadhu decoction for Pitta, Panchamoola for Kapha |
| 6 | **Supply chain audit** | GPS-tagged, SHA-256 hash-chained event ledger + IUCN endangered species alerts + formulation validation |

### The Dosha Mapping (Ancient to Modern Bridge)

| Modern Pathology | Dosha | Classical Symptoms | CV Features Detected | Traditional Treatment |
|-----------------|-------|-------------------|---------------------|----------------------|
| Shot Hole, N/K deficiency | **Vata** (desiccation) | Knots, drying, hard fruits, reduced sap | Edge deformation, texture variance, structural asymmetry | Kunapajala (fermented manure), Neem fumigation, animal fats |
| Yellow Leaf, Bacterial Spot | **Pitta** (metabolic heat) | Yellowing, premature shedding, rotting, paleness | RGB/HSV color shifts, necrotic lesions, burn artifacts | Yashtimadhu decoction, milk + honey, cooling Triphala spray |
| Powdery Mildew | **Kapha** (excess moisture) | Hypertrophy, dwarfed growth, loss of taste/smell | Surface dullness, morphological thickening, abnormal LAI | Panchamoola decoction, mustard, reduce moisture |

---

## How

### Data Foundation (6 Federated Datasets)

| Dataset | Images | Species | Botanical Role |
|---------|--------|---------|---------------|
| **Herbify** | 6,104 | 91 | Healthy baselines, broad taxonomy |
| **Assam MED117** | 158,847 | 10 | NE India regional morphological variance |
| **AI-MedLeafX** | 76,036 | 4 | Disease labels: Bacterial Spot, Shot Hole, Powdery Mildew, Yellow Leaf |
| **CIMPD** | 8,592 | 23 | Field-captured Healthy vs Unhealthy, noisy backgrounds |
| **SIMPD V1** | 2,513 | 20 | Herbs, shrubs, creepers, climbers, trees — morphological diversity |
| **EarlyNSD** | 2,700 | 3 | Nitrogen and Potassium deficiency — abiotic stress |
| **Total** | **~254,792** | **77 unified** | **8 pathology classes** |

**Why these 6**: Each plugs a specific gap — Herbify for breadth, MedLeafX for disease depth, CIMPD for noise robustness, SIMP for geometric diversity, EarlyNSD for abiotic vs biotic distinction, Assam for geographic generalization.

### Model Architecture

- **Backbone**: HierViT (DINOv2 pretrained) — hierarchical multi-scale features capture both macro leaf shape and micro lesion texture
- **Species Head**: MLP + Softmax + Cross-Entropy loss
- **Pathology Head**: MLP + Sigmoid + Focal Loss (handles severe class imbalance — healthy leaves vastly outnumber diseased)
- **Dynamic Task Weighting**: Weights auto-adjust per epoch so neither head dominates training
- **Loss Function**: `L = w_species(t) * L_CE + w_disease(t) * L_focal + lambda * L_reg`

### RAG Knowledge Base

The LLM is indexed against classical Ayurvedic texts for report generation:

1. **Charaka Samhita** — Foundational Ayurvedic medical text
2. **Susruta Samhita** — Classical surgical/medical text
3. **Ayurvedic Pharmacopoeia of India (API)** — Official pharmacological reference
4. **Vrikshayurveda** — Plant science texts by Surapala and Varahamihira

**Report output includes**: Authenticated botanical identity, classical medicinal properties (Rasa/Guna/Virya/Vipaka), current health status, Vrikshayurveda Dosha assessment, traditional therapeutic intervention, and pharmaceutical procurement quality assessment.

### Self-Sustaining Active Learning Loop

```
User captures leaf
    -> Model predicts
    -> Uncertain? -> Quarantine to Drive
        -> LLM Oracle (Gemini) attempts label
            -> Confident? -> Accept
            -> Not confident? -> Route to human Ayurvedic botanist (top 1%)
        -> Labeled data -> Incremental retraining -> Deploy if F1 improves
```

Only the hardest ~1% of images need human review. The model continuously improves.

### Tech Stack

- **Language**: Python 3.14+ (free-threaded build, GIL disabled)
- **Deep Learning**: PyTorch with torch.compile, timm library for ViT model zoo
- **Model**: HierViT backbone from DINOv2 pretrained weights
- **Training**: Mixed precision (AMP), Focal Loss + Cross-Entropy, Dynamic Task Weighting
- **LLM / RAG**: Multi-provider (Gemini, Claude, OpenAI, Llama), ChromaDB vector store
- **Cloud**: Google Colab Enterprise, Google Drive persistence
- **MLOps**: F1-gated checkpointing, crash recovery, active learning pipeline

---

## When (Phase Timeline)

| Phase | Status | What It Delivered |
|-------|--------|-------------------|
| **Phase 0**: Documentation | Done | instructions.md (42KB spec), CLAUDE.md, ROADMAP.md |
| **Phase 1**: Infrastructure | Done | Colab setup, DriveManager, ColabEnvironment, project scaffold |
| **Phase 2**: Data Pipeline | Done | 6 dataset downloaders, unified taxonomy (77 species, 8 pathologies), augmentation, federated DataLoader — 76 tests |
| **Phase 3**: Model Architecture | Done | HierViTCMTL (DINOv2 backbone + dual heads), Focal Loss, Dynamic Task Weighting — 3 tests |
| **Phase 4**: Training Pipeline | Done | AMP training, warmup+cosine LR, F1-gated checkpoints, ColabTrainer with crash recovery — 22 tests |
| **Phase 5**: Vrikshayurveda | Done | Pathology to Dosha mapping, treatments, contraindications — 5 tests |
| **Phase 6**: RAG + LLM | Done | ChromaDB indexer, retriever, multi-LLM report generator (Gemini/Claude/OpenAI/Llama) — 52 tests |
| **Phase 7**: Active Learning | Done | Uncertainty sampling, quarantine, LLM oracle, human expert queue — 24 tests |
| **Phase 8**: API | Done | FastAPI server, InferencePipeline, /identify endpoint — 30 tests |
| **Phase 9**: Advanced | Done | IUCN conservation alerts, GPS hash-chain traceability, formulation validation — 49 tests |
| **Current**: Model Training | In Progress | Training ViT-Base DINOv2 on 254K images, T4 GPU |

**Total: 370 tests, all passing.** All 10 phases of code are complete.

### What Remains (Post-Training)

1. **Train to convergence** — achieve strong F1 scores on both species and pathology heads
2. **Index Ayurvedic texts** — upload Charaka/Susruta/API texts to Drive, build ChromaDB vector store
3. **End-to-end inference** — leaf image to species + disease + Dosha + full report
4. **Deploy API** — FastAPI server with ngrok tunnel from Colab
5. **Active learning loop** — start quarantining uncertain predictions, feed back into training

---

## Test Suite Summary

| Test File | Tests | Coverage |
|-----------|-------|----------|
| test_data.py | 76 | Taxonomy, pathology mapping, 6 dataset loaders, federation, splits, sampler, augmentation, preprocessing, download |
| test_models.py | 3 | FocalLoss, CMTLLoss |
| test_training.py | 22 | TrainConfig, CheckpointManager, evaluation, Trainer |
| test_vrikshayurveda.py | 5 | Dosha mapping (Vata/Pitta/Kapha/Healthy/N-deficiency) |
| test_uncertainty.py | 4 | Uncertainty scoring strategies |
| test_rag.py | 28 | Indexer, retriever, report generator |
| test_llm_providers.py | 24 | Multi-LLM provider abstraction |
| test_active_learning.py | 20 | Quarantine, expert queue, oracle pipeline, E2E flow |
| test_api.py | 30 | Inference pipeline, checkpoint loading, FastAPI endpoints |
| test_conservation.py | 17 | IUCN registry, conservation alerts |
| test_traceability.py | 15 | GPS, event hashing, hash chain, ledger queries |
| test_formulation.py | 17 | Formulation validation, classical recipes |
| test_colab.py | 57 | DriveManager, ColabEnvironment |
| test_colab_training.py | 52 | GradAccum, CrashCheckpointer, GPUMonitor, DatasetCache, ColabTrainer |
| **Total** | **370** | |
