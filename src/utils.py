import os
import gc
from typing import Iterable


def ensure_dir(path: str) -> None:
    """Create a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def cleanup(*objs: Iterable[object]) -> None:
    """
    Best-effort cleanup to reduce memory pressure between runs.
    This is mainly helpful for long grid searches on CPU.
    """
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
