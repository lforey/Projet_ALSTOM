import os
import gc


def ensure_dir(path: str) -> None:
    """Create a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def cleanup(*_objs) -> None:
    """
    Trigger Python garbage collection.

    Passing objects here is only cosmetic:
    deleting local references inside this function does not delete
    the caller's references.
    """
    gc.collect()

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass