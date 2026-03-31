"""FastAPI inference server for PhytoVeda.

Endpoints:
    POST /identify       — upload plant leaf image, get full diagnosis
    GET  /health         — service health check
    GET  /model/version  — current model version and performance metrics

Usage:
    uvicorn phytoveda.api.server:app --host 0.0.0.0 --port 8000
    python -m phytoveda.api.server --checkpoint checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import io
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from phytoveda.api.inference import InferencePipeline
from phytoveda.rag.report_generator import ReportGenerator
from phytoveda.rag.retriever import AyurvedicRetriever

# ─── App State ─────────────────────────────────────────────────────────────────

_state: dict = {
    "pipeline": None,
    "model_version": "unknown",
    "checkpoint_path": None,
    "metrics": {},
    "start_time": None,
}


def _load_pipeline(
    checkpoint_path: str | Path,
    chromadb_dir: str | Path | None = None,
    gemini_api_key: str | None = None,
    device: str | None = None,
) -> None:
    """Load the inference pipeline from a checkpoint into app state."""
    dev = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    retriever = None
    if chromadb_dir and Path(chromadb_dir).exists():
        retriever = AyurvedicRetriever(persist_dir=chromadb_dir)

    report_gen = ReportGenerator(api_key=gemini_api_key) if gemini_api_key else None

    checkpoint = torch.load(checkpoint_path, map_location=dev, weights_only=False)

    pipeline = InferencePipeline.from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=dev,
        retriever=retriever,
        report_generator=report_gen,
    )

    _state["pipeline"] = pipeline
    _state["checkpoint_path"] = str(checkpoint_path)
    _state["model_version"] = checkpoint.get("config", {}).get("version", "v0.1")
    _state["metrics"] = checkpoint.get("metrics", {})
    _state["start_time"] = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler — loads model on startup if configured via env/cli."""
    # Model loading happens in main() before uvicorn.run().
    # This is a no-op placeholder for adding startup/shutdown hooks.
    yield


# ─── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="PhytoVeda",
    description=(
        "AI-Driven Botanical Identification and Textual Mapping in Ayurveda. "
        "Upload a plant leaf image to receive species identification, disease diagnosis, "
        "Vrikshayurveda Dosha assessment, and traditional treatment recommendations."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    """Service health check.

    Returns:
        JSON with status, model_loaded flag, device, and uptime.
    """
    pipeline = _state.get("pipeline")
    uptime = time.time() - _state["start_time"] if _state["start_time"] else 0

    return {
        "status": "healthy" if pipeline is not None else "no_model",
        "model_loaded": pipeline is not None,
        "device": str(pipeline.device) if pipeline else None,
        "uptime_seconds": round(uptime, 1),
    }


@app.get("/model/version")
async def model_version() -> dict:
    """Current model version and performance metrics.

    Returns:
        JSON with version, checkpoint path, metrics, and device info.
    """
    pipeline = _state.get("pipeline")

    return {
        "version": _state.get("model_version", "unknown"),
        "checkpoint": _state.get("checkpoint_path"),
        "metrics": _state.get("metrics", {}),
        "device": str(pipeline.device) if pipeline else None,
        "num_species": pipeline.taxonomy.num_species if pipeline else None,
    }


@app.post("/identify")
async def identify(
    file: UploadFile = File(  # noqa: B008
        ..., description="Plant leaf image (JPEG/PNG)"
    ),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of top predictions"),
    generate_report: bool = Query(
        default=False, description="Generate full botanical report (requires LLM)"
    ),
) -> JSONResponse:
    """Identify plant species and diagnose health from a leaf image.

    Accepts a plant leaf image and returns:
    - Species identification (top-K predictions with confidence)
    - Pathology/disease diagnosis (top-K with confidence)
    - Vrikshayurveda Tridosha assessment
    - Uncertainty score for active learning
    - Optional full botanical report (with RAG + LLM)

    Args:
        file: Uploaded image file (JPEG or PNG).
        top_k: Number of top predictions to return per head.
        generate_report: Whether to generate a full LLM-powered botanical report.

    Returns:
        JSON response with complete diagnosis.
    """
    pipeline: InferencePipeline | None = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Start server with --checkpoint.",
        )

    # Validate file type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {file.content_type}. Use JPEG or PNG.",
        )

    # Read and decode image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}") from e

    # Run inference
    start = time.time()
    result = pipeline.predict(
        image, top_k=top_k, generate_report=generate_report
    )
    inference_ms = round((time.time() - start) * 1000, 1)

    response = result.to_dict()
    response["inference_time_ms"] = inference_ms

    return JSONResponse(content=response)


# ─── CLI Entry Point ───────────────────────────────────────────────────────────


def main() -> None:
    """Launch the PhytoVeda inference API server."""
    parser = argparse.ArgumentParser(description="PhytoVeda Inference API Server")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--device", type=str, default=None, help="Torch device (cpu/cuda)")
    parser.add_argument(
        "--chromadb-dir", type=str, default=None,
        help="ChromaDB persist directory for RAG retrieval",
    )
    parser.add_argument(
        "--gemini-api-key", type=str, default=None,
        help="Gemini API key for LLM report generation",
    )
    parser.add_argument("--workers", type=int, default=1, help="Number of uvicorn workers")
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    _load_pipeline(
        checkpoint_path=args.checkpoint,
        chromadb_dir=args.chromadb_dir,
        gemini_api_key=args.gemini_api_key,
        device=args.device,
    )
    print(f"Model loaded on {_state['pipeline'].device}")

    import uvicorn
    uvicorn.run(
        "phytoveda.api.server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
