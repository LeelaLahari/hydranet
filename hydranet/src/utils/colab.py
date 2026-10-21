"""Helpers for Google Colab free tier: Drive mount, GPU check, path wiring."""
from pathlib import Path
import os

def setup_colab(mount_drive: bool = True, project_name: str = "hydranet"):
    """Mount Drive, return dict of paths. Call once at the top of every notebook."""
    if mount_drive:
        try:
            from google.colab import drive
            drive.mount("/content/drive")
        except ImportError:
            pass  # running locally
    import torch
    gpu = torch.cuda.is_available()
    print(f"[colab] GPU available: {gpu}"
          + (f" ({torch.cuda.get_device_name(0)})" if gpu else " -> enable Runtime > GPU"))
    root = Path(f"/content/drive/MyDrive/{project_name}") if mount_drive else Path(".")
    for sub in ["data/raw", "data/processed", "checkpoints", "reports/figures"]:
        (root / sub).mkdir(parents=True, exist_ok=True)
    return root
