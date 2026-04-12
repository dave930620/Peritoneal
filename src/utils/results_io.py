"""
results_io.py — Save and load experiment results as JSON.

Design
------
- Each experiment script appends its own key to a shared JSON file.
- If the file already exists, existing entries are preserved (safe to re-run).
- Thread-safe for single-process sequential use (no locking needed for research).
"""

import json
import os
from pathlib import Path


def save_result(json_path: str, key: str, metrics: dict) -> None:
    """Append or update a single result entry in a JSON file.

    If the file does not exist, it is created.
    If the key already exists, it is overwritten.

    Parameters
    ----------
    json_path : str   Path to the JSON file (e.g. "results/f1/baselines.json").
    key       : str   Identifier for this result (e.g. "ridge", "A-F1-1").
    metrics   : dict  Flat dict of metric names → scalar values.
    """
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = {}

    existing[key] = metrics

    with open(json_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"[results_io] Saved '{key}' → {json_path}")


def load_results(json_path: str) -> dict:
    """Load all results from a JSON file.

    Returns an empty dict if the file does not exist.
    """
    if not os.path.exists(json_path):
        return {}
    with open(json_path, "r") as f:
        return json.load(f)


def result_exists(json_path: str, key: str) -> bool:
    """Return True if the key already exists in the JSON file (skip re-running)."""
    return key in load_results(json_path)
