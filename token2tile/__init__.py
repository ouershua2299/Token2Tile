"""Token2Tile: per-query sparse attention executed over GPU tiles."""
from pathlib import Path

__version__ = "0.1.0"


def operator_dir() -> str:
    """Directory of the attention operator (deploy_step.py, Triton helpers, CUDA kernel sources)."""
    return str(Path(__file__).resolve().parent / "operator")
