"""
Downloader utilities for Kaggle LLM benchmarking
================================================
Provides a minimal, reproducible way to pull a single HuggingFace model
into the Kaggle environment, clean up afterwards, and report the on‑disk
size.  This is used in Phase 1 pilot validation.
"""

import os
import json
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download

def download_model(repo_id: str, revision: Optional[str] = None, cache_dir: Optional[str] = None) -> Path:
    """Download a model repo to a local directory.

    Args:
        repo_id: The HuggingFace repository identifier, e.g. ``"microsoft/Phi-3.5-mini-instruct"``.
        revision: Optional git revision (branch, tag, or commit SHA) to pin the model.
        cache_dir: Optional directory for the HF cache; defaults to the standard ``~/.cache/huggingface``.

    Returns:
        Path to the root directory containing the downloaded files.
    """
    # ``snapshot_download`` respects the HF cache and will reuse existing files.
    local_path = snapshot_download(repo_id=repo_id, revision=revision, cache_dir=cache_dir, resume_download=True)
    return Path(local_path)

def get_dir_size(path: Path) -> int:
    """Recursively compute the total size (in bytes) of *path*.

    Returns:
        Integer byte count.
    """
    total = 0
    for entry in path.rglob('*'):
        if entry.is_file():
            total += entry.stat().st_size
    return total

def human_readable_bytes(num_bytes: int) -> str:
    """Convert a byte count to a human‑readable string (KB/MB/GB)."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"

def main():
    # Example pilot: Phi‑3.5‑mini‑instruct (≈3.8 B params, ~2 GB safetensors)
    repo = "microsoft/Phi-3.5-mini-instruct"
    # The revision can be pinned to a known commit; for the demo we leave it as None.
    model_path = download_model(repo)
    size_bytes = get_dir_size(model_path)
    result = {
        "repo_id": repo,
        "local_path": str(model_path),
        "size_bytes": size_bytes,
        "size_human": human_readable_bytes(size_bytes),
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
