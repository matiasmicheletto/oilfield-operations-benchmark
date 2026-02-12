"""
Utility functions for the distance matrix generation script.
"""

from pathlib import Path
import yaml
import numpy as np

def load_config(path: str) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file '{path}' not found.")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def create_rng(seed):
    return np.random.default_rng(seed)