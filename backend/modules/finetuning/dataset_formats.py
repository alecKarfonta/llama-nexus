"""
Dataset format helpers for LoRA fine-tuning.

These handlers encapsulate minimal schema validation for the supported dataset
formats described in the design document (Alpaca, ShareGPT, ChatML, Completion).
They will be expanded with richer validation and conversion logic in future
iterations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

from .config import DatasetFormat


@dataclass
class DatasetFormatHandler:
    name: DatasetFormat
    required_keys: List[str]
    description: str
    example_keys: List[str] = field(default_factory=list)

    def validate_record(self, record: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        for key in self.required_keys:
            if key not in record:
                errors.append(f"Missing required field '{key}'")
            elif not isinstance(record[key], str):
                errors.append(f"Field '{key}' must be a string")
            elif not record[key].strip():
                errors.append(f"Field '{key}' cannot be empty")
        return errors


DATASET_FORMAT_HANDLERS: Dict[DatasetFormat, DatasetFormatHandler] = {
    DatasetFormat.ALPACA: DatasetFormatHandler(
        name=DatasetFormat.ALPACA,
        required_keys=["instruction", "output"],
        example_keys=["input"],
        description="Instruction-following format with optional input field.",
    ),
    DatasetFormat.SHAREGPT: DatasetFormatHandler(
        name=DatasetFormat.SHAREGPT,
        required_keys=["conversations"],
        description="Multi-turn conversation list with 'from' and 'value' entries.",
    ),
    DatasetFormat.CHATML: DatasetFormatHandler(
        name=DatasetFormat.CHATML,
        required_keys=["messages"],
        description="OpenAI-style message list with role/content pairs.",
    ),
    DatasetFormat.COMPLETION: DatasetFormatHandler(
        name=DatasetFormat.COMPLETION,
        required_keys=["text"],
        description="Pre-tokenized completion-style text entries.",
    ),
}


def detect_format(sample: Dict[str, Any]) -> DatasetFormat:
    """
    Infer dataset format from a single record. Raises ValueError when
    detection fails to avoid silently misclassifying data.
    """
    if "conversations" in sample:
        return DatasetFormat.SHAREGPT
    if "messages" in sample:
        return DatasetFormat.CHATML
    if "instruction" in sample or "input" in sample:
        return DatasetFormat.ALPACA
    if "text" in sample:
        return DatasetFormat.COMPLETION
    raise ValueError("Unable to detect dataset format from sample keys")


def validate_records(
    records: Iterable[Dict[str, Any]], expected_format: DatasetFormat
) -> List[str]:
    """
    Run minimal structural validation across the provided records. This is a
    lightweight guardrail to catch obvious schema issues before deeper checks.
    """
    handler = DATASET_FORMAT_HANDLERS[expected_format]
    errors: List[str] = []
    for idx, record in enumerate(records):
        record_errors = handler.validate_record(record)
        if record_errors:
            prefixed = [f"Record {idx}: {msg}" for msg in record_errors]
            errors.extend(prefixed)
    return errors
