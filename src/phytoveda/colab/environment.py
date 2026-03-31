"""Colab environment detection, GPU setup, and package installation.

Handles:
    - Detecting whether code is running inside Google Colab
    - GPU/TPU availability and CUDA memory reporting
    - Installing PhytoVeda and optional LLM dependencies
    - Verifying the Python version and free-threading status
    - Setting environment variables (PYTHON_GIL=0, CUDA flags)
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field


@dataclass
class ColabEnvironment:
    """Detect and configure the Google Colab runtime environment.

    Usage:
        from phytoveda.colab import ColabEnvironment

        env = ColabEnvironment()
        print(env.summary())

        # Install PhytoVeda with Claude + OpenAI support
        env.install_phytoveda(extras=["claude", "openai", "dev"])

        # Configure for free-threaded training
        env.configure_free_threading()
    """

    _info: dict[str, str] = field(default_factory=dict, repr=False)

    # ─── Detection ───────────────────────────────────────────────────────

    @staticmethod
    def is_colab() -> bool:
        """Check if running inside Google Colab."""
        try:
            import google.colab  # type: ignore[import-untyped]  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def python_version() -> str:
        """Return Python version string."""
        return sys.version

    @staticmethod
    def python_version_tuple() -> tuple[int, int, int]:
        """Return Python version as (major, minor, micro)."""
        return (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)

    @staticmethod
    def is_free_threaded() -> bool:
        """Check if running a free-threaded Python build (GIL disabled)."""
        # Python 3.13+ has sys._is_gil_enabled()
        if hasattr(sys, "_is_gil_enabled"):
            return not sys._is_gil_enabled()
        # Check environment variable
        return os.environ.get("PYTHON_GIL", "1") == "0"

    # ─── GPU / Accelerator ───────────────────────────────────────────────

    @staticmethod
    def has_gpu() -> bool:
        """Check if CUDA GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    @staticmethod
    def gpu_name() -> str | None:
        """Get the name of the current CUDA GPU."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
        except ImportError:
            pass
        return None

    @staticmethod
    def gpu_memory_gb() -> float | None:
        """Get total GPU memory in GB."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except ImportError:
            pass
        return None

    @staticmethod
    def gpu_memory_free_gb() -> float | None:
        """Get free GPU memory in GB."""
        try:
            import torch
            if torch.cuda.is_available():
                free, _ = torch.cuda.mem_get_info(0)
                return free / (1024 ** 3)
        except ImportError:
            pass
        return None

    @staticmethod
    def has_tpu() -> bool:
        """Check if TPU is available."""
        try:
            import torch_xla  # type: ignore[import-untyped]  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def device() -> str:
        """Get the best available device string for PyTorch."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    # ─── Installation ────────────────────────────────────────────────────

    @staticmethod
    def install_phytoveda(
        extras: list[str] | None = None,
        editable: bool = True,
        repo_path: str = "/content/drive/MyDrive/PhytoVeda",
        quiet: bool = False,
    ) -> int:
        """Install PhytoVeda package in the Colab runtime.

        Args:
            extras: Optional dependency groups (e.g., ["dev", "claude", "openai"]).
            editable: Install in editable mode (-e).
            repo_path: Path to the PhytoVeda repository.
            quiet: Suppress pip output.

        Returns:
            pip return code.
        """
        extras_str = f"[{','.join(extras)}]" if extras else ""
        flag = "-e" if editable else ""
        quiet_flag = "-q" if quiet else ""
        cmd = f"pip install {quiet_flag} {flag} '{repo_path}{extras_str}'".strip()
        # Clean up double spaces
        cmd = " ".join(cmd.split())
        result = subprocess.run(cmd, shell=True, capture_output=quiet)
        return result.returncode

    @staticmethod
    def install_packages(packages: list[str], quiet: bool = False) -> int:
        """Install additional pip packages.

        Args:
            packages: List of package specifiers.
            quiet: Suppress pip output.

        Returns:
            pip return code.
        """
        pkgs = " ".join(f"'{p}'" for p in packages)
        quiet_flag = "-q" if quiet else ""
        cmd = f"pip install {quiet_flag} {pkgs}"
        result = subprocess.run(cmd, shell=True, capture_output=quiet)
        return result.returncode

    # ─── Environment Configuration ───────────────────────────────────────

    @staticmethod
    def configure_free_threading() -> None:
        """Set environment variables for Python 3.14+ free-threaded mode."""
        os.environ["PYTHON_GIL"] = "0"

    @staticmethod
    def configure_cuda() -> None:
        """Set recommended CUDA environment variables for training."""
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0;7.5;8.0;8.6;8.9;9.0")

    @staticmethod
    def set_wandb_key(api_key: str) -> None:
        """Set WandB API key for experiment tracking."""
        os.environ["WANDB_API_KEY"] = api_key

    @staticmethod
    def set_seed(seed: int = 42) -> None:
        """Set random seeds for reproducibility across all libraries."""
        import random
        random.seed(seed)

        try:
            import numpy as np
            np.random.seed(seed)
        except ImportError:
            pass

        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            pass

    # ─── Summary ─────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Generate a human-readable environment summary."""
        gpu = self.gpu_name()
        gpu_mem = self.gpu_memory_gb()
        gpu_free = self.gpu_memory_free_gb()

        lines = [
            "PhytoVeda Environment",
            "=" * 50,
            f"  Colab:            {self.is_colab()}",
            f"  Python:           {self.python_version().split()[0]}",
            f"  Free-threaded:    {self.is_free_threaded()}",
            f"  Device:           {self.device()}",
        ]

        if gpu:
            lines.append(f"  GPU:              {gpu}")
            if gpu_mem is not None:
                lines.append(f"  GPU Memory:       {gpu_mem:.1f} GB total")
            if gpu_free is not None:
                lines.append(f"  GPU Free:         {gpu_free:.1f} GB")
        else:
            lines.append("  GPU:              Not available")

        if self.has_tpu():
            lines.append("  TPU:              Available")

        # Check key packages
        for pkg_name in ("torch", "timm", "phytoveda"):
            try:
                mod = __import__(pkg_name)
                ver = getattr(mod, "__version__", "installed")
                lines.append(f"  {pkg_name:18s}{ver}")
            except ImportError:
                lines.append(f"  {pkg_name:18s}NOT INSTALLED")

        return "\n".join(lines)

    def check_ready(self) -> dict[str, bool]:
        """Check if the environment is ready for training.

        Returns:
            Dict with readiness checks.
        """
        checks: dict[str, bool] = {}

        # Python version
        major, minor, _ = self.python_version_tuple()
        checks["python_3.14+"] = (major, minor) >= (3, 14)

        # GPU
        checks["gpu_available"] = self.has_gpu()

        # Key packages
        for pkg in ("torch", "torchvision", "timm", "phytoveda"):
            try:
                __import__(pkg)
                checks[f"pkg_{pkg}"] = True
            except ImportError:
                checks[f"pkg_{pkg}"] = False

        return checks
