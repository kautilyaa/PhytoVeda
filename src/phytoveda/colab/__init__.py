"""Google Colab integration — Drive-backed persistent storage + SSD acceleration.

Provides:
    - DriveManager: path strategy (SSD for data, Drive for results/checkpoints)
    - ColabEnvironment: GPU detection, package installation, environment verification
    - ColabTrainer: gradient accumulation, crash checkpointing, GPU monitoring
    - DatasetCache: cache splits/taxonomy/history to Drive for instant restart
    - GPUMonitor: real-time GPU memory tracking
    - find_max_batch_size: auto-detect largest batch that fits in GPU memory
    - compile_model: torch.compile with Colab-safe fallback

Design:
    - Datasets download to /content/ (Colab SSD) for fast I/O during training
    - Checkpoints, results, ChromaDB, quarantine, logs persist to Google Drive
    - Automatic directory scaffolding on both SSD and Drive
    - Sync utilities to copy between SSD and Drive
"""

from phytoveda.colab.data_cache import DatasetCache, compute_split_hash
from phytoveda.colab.dataset_setup import print_dataset_status, setup_datasets
from phytoveda.colab.drive import DriveManager
from phytoveda.colab.environment import ColabEnvironment
from phytoveda.colab.training import (
    ColabTrainer,
    CrashCheckpointer,
    GPUMonitor,
    GradAccumConfig,
    compile_model,
    find_max_batch_size,
)

__all__ = [
    "ColabEnvironment",
    "ColabTrainer",
    "CrashCheckpointer",
    "DatasetCache",
    "DriveManager",
    "GPUMonitor",
    "GradAccumConfig",
    "compile_model",
    "compute_split_hash",
    "find_max_batch_size",
    "print_dataset_status",
    "setup_datasets",
]
