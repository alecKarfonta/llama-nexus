"""
Persistent storage for fine-tuning datasets.

Stores dataset metadata in a JSON index and the raw dataset files on disk. This
is intentionally lightweight so we can iterate quickly; we can migrate to a DB
later if needed.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from enhanced_logger import enhanced_logger as logger

from .config import Dataset


class DatasetStore:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_dir / "datasets_index.json"
        if not self.index_path.exists():
            self._write_index([])

    def _read_index(self) -> List[Dict]:
        try:
            with self.index_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except FileNotFoundError:
            return []
        except json.JSONDecodeError:
            logger.warning("Dataset index corrupted, resetting to empty list")
            return []

    def _write_index(self, items: List[Dict]) -> None:
        tmp_path = self.index_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            json.dump(items, fh, default=str, indent=2)
        tmp_path.replace(self.index_path)

    def list(self) -> List[Dataset]:
        return [Dataset(**item) for item in self._read_index()]

    def get(self, dataset_id: str) -> Optional[Dataset]:
        for item in self._read_index():
            if item.get("id") == dataset_id:
                return Dataset(**item)
        return None

    def upsert(self, dataset: Dataset) -> Dataset:
        items = self._read_index()
        replaced = False
        for idx, item in enumerate(items):
            if item.get("id") == dataset.id:
                items[idx] = dataset.dict()
                replaced = True
                break
        if not replaced:
            items.append(dataset.dict())
        self._write_index(items)
        return dataset

    def delete(self, dataset_id: str) -> bool:
        items = self._read_index()
        target: Optional[Dict] = None
        remaining: List[Dict] = []
        for item in items:
            if item.get("id") == dataset_id:
                target = item
            else:
                remaining.append(item)

        removed = target is not None
        if removed:
            self._write_index(remaining)
            file_path = Path(target.get("file_path", ""))
            if file_path.exists():
                try:
                    file_path.unlink()
                except Exception as exc:
                    logger.warning(
                        "Failed to delete dataset file",
                        extra={"dataset_id": dataset_id, "error": str(exc)},
                    )
        return removed

    def dataset_path(self, dataset_id: str, original_name: str) -> Path:
        suffix = Path(original_name).suffix or ".jsonl"
        return self.base_dir / f"{dataset_id}{suffix}"


def default_dataset_dir() -> Path:
    env_path = os.getenv("FINETUNE_DATASET_DIR")
    if env_path:
        return Path(env_path).expanduser().resolve()
    # Default to backend/data/finetune_datasets relative to this file
    return (
        Path(__file__).resolve().parent.parent.parent
        / "data"
        / "finetune_datasets"
    )


def init_dataset_store() -> DatasetStore:
    base_dir = default_dataset_dir()
    base_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Initialized dataset store", extra={"path": str(base_dir)})
    return DatasetStore(base_dir=base_dir)
