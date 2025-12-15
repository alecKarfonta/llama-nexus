"""
Dataset parsing and lightweight validation for LoRA fine-tuning datasets.

This module provides an initial implementation focused on structure checks and
basic statistics. It intentionally avoids heavy processing so we can iterate on
the interface while wiring the API surface.
"""

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any

from enhanced_logger import enhanced_logger as logger

from .config import DatasetFormat
from .dataset_formats import detect_format, validate_records


@dataclass
class FieldStats:
    """Statistics for a single field in the dataset."""
    name: str
    present_count: int
    total_count: int
    min_length: int = 0
    max_length: int = 0
    avg_length: float = 0.0
    empty_count: int = 0
    
    @property
    def completeness(self) -> float:
        if self.total_count == 0:
            return 0.0
        return (self.present_count - self.empty_count) / self.total_count


@dataclass
class DatasetStatistics:
    """Comprehensive statistics for a dataset."""
    total_records: int
    detected_format: str
    format_confidence: float
    
    # Token statistics (estimated using simple word count * 1.3 factor)
    total_tokens: int = 0
    avg_tokens_per_record: float = 0.0
    min_tokens: int = 0
    max_tokens: int = 0
    token_distribution: Dict[str, int] = field(default_factory=dict)  # bucket -> count
    
    # Sequence length statistics (character count)
    avg_sequence_length: float = 0.0
    min_sequence_length: int = 0
    max_sequence_length: int = 0
    sequence_distribution: Dict[str, int] = field(default_factory=dict)  # bucket -> count
    
    # Field statistics
    field_stats: List[FieldStats] = field(default_factory=list)
    
    # Format detection details
    format_indicators: Dict[str, bool] = field(default_factory=dict)
    
    # Validation summary
    validation_errors: List[str] = field(default_factory=list)
    is_valid: bool = True


def estimate_tokens(text: str) -> int:
    """Estimate token count using word count * 1.3 factor (rough approximation)."""
    if not text:
        return 0
    # Split on whitespace and punctuation
    words = re.findall(r'\b\w+\b', text)
    # Factor of 1.3 accounts for subword tokenization
    return int(len(words) * 1.3)


def get_text_content(record: Dict, format_type: DatasetFormat) -> str:
    """Extract text content from a record based on format."""
    texts = []
    
    if format_type == DatasetFormat.ALPACA:
        texts.extend([
            record.get("instruction", ""),
            record.get("input", ""),
            record.get("output", ""),
        ])
    elif format_type == DatasetFormat.SHAREGPT:
        conversations = record.get("conversations", [])
        for conv in conversations:
            texts.append(conv.get("value", ""))
    elif format_type == DatasetFormat.CHATML:
        messages = record.get("messages", [])
        for msg in messages:
            texts.append(msg.get("content", ""))
    elif format_type == DatasetFormat.COMPLETION:
        texts.extend([
            record.get("prompt", ""),
            record.get("completion", ""),
        ])
    else:
        # Generic: extract all string values
        for value in record.values():
            if isinstance(value, str):
                texts.append(value)
    
    return " ".join(texts)


def create_histogram_buckets(values: List[int], num_buckets: int = 10) -> Dict[str, int]:
    """Create histogram buckets from a list of values."""
    if not values:
        return {}
    
    min_val = min(values)
    max_val = max(values)
    
    if min_val == max_val:
        return {f"{min_val}": len(values)}
    
    bucket_size = (max_val - min_val) / num_buckets
    buckets: Counter = Counter()
    
    for val in values:
        bucket_idx = min(int((val - min_val) / bucket_size), num_buckets - 1)
        bucket_start = int(min_val + bucket_idx * bucket_size)
        bucket_end = int(min_val + (bucket_idx + 1) * bucket_size)
        bucket_label = f"{bucket_start}-{bucket_end}"
        buckets[bucket_label] += 1
    
    # Sort by bucket start value
    sorted_buckets = dict(sorted(buckets.items(), key=lambda x: int(x[0].split("-")[0])))
    return sorted_buckets


class DatasetProcessor:
    def __init__(self, max_preview: int = 50):
        """
        Args:
            max_preview: Number of records to inspect for validation/statistics.
        """
        self.max_preview = max_preview

    def _read_jsonl(self, file_path: Path) -> Iterable[Dict]:
        with file_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                yield json.loads(line)

    def _read_json_array(self, file_path: Path) -> Iterable[Dict]:
        with file_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError("JSON dataset must be a list of records")
        return data

    def load_records(self, file_path: str) -> Iterable[Dict]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        if path.suffix.lower() == ".jsonl":
            return self._read_jsonl(path)
        if path.suffix.lower() == ".json":
            return self._read_json_array(path)
        raise ValueError(f"Unsupported dataset file extension: {path.suffix}")

    def inspect(
        self, file_path: str, declared_format: Optional[DatasetFormat] = None
    ) -> Tuple[DatasetFormat, int, List[str]]:
        """
        Load a dataset file, infer or validate its format, and run basic checks.

        Returns:
            (format, example_count, validation_errors)
        """
        records_iterable = list(self.load_records(file_path))
        if not records_iterable:
            raise ValueError("Dataset is empty")

        sample = records_iterable[0]
        detected_format = declared_format or detect_format(sample)
        if declared_format and declared_format != detected_format:
            raise ValueError(
                f"Declared format {declared_format} does not match detected "
                f"format {detected_format}"
            )

        preview = records_iterable[: self.max_preview]
        validation_errors = validate_records(preview, detected_format)
        logger.info(
            "Dataset inspection completed",
            extra={
                "format": detected_format.value,
                "examples_scanned": len(preview),
                "total_examples": len(records_iterable),
                "validation_errors": len(validation_errors),
            },
        )
        return detected_format, len(records_iterable), validation_errors

    def preview(self, file_path: str, limit: int = 20) -> List[Dict]:
        """
        Return the first N records from a dataset file for UI preview.
        """
        records_iterable = self.load_records(file_path)
        preview: List[Dict] = []
        for idx, record in enumerate(records_iterable):
            if idx >= limit:
                break
            preview.append(record)
        if not preview:
            raise ValueError("Dataset is empty")
        return preview

    def analyze(self, file_path: str, max_records: int = 5000) -> DatasetStatistics:
        """
        Perform comprehensive analysis of a dataset file.
        
        Args:
            file_path: Path to the dataset file
            max_records: Maximum number of records to analyze (for performance)
            
        Returns:
            DatasetStatistics with token distributions, field completeness, etc.
        """
        records_iterable = list(self.load_records(file_path))
        if not records_iterable:
            raise ValueError("Dataset is empty")
        
        # Limit records for analysis
        records = records_iterable[:max_records]
        total_records = len(records_iterable)
        
        # Detect format
        sample = records[0]
        detected_format = detect_format(sample)
        
        # Format confidence and indicators
        format_indicators = self._detect_format_indicators(sample)
        format_confidence = self._calculate_format_confidence(records[:100], detected_format)
        
        # Validate records
        validation_errors = validate_records(records[:self.max_preview], detected_format)
        
        # Analyze tokens and sequences
        token_counts = []
        sequence_lengths = []
        
        for record in records:
            text = get_text_content(record, detected_format)
            tokens = estimate_tokens(text)
            seq_len = len(text)
            
            token_counts.append(tokens)
            sequence_lengths.append(seq_len)
        
        # Calculate token statistics
        total_tokens = sum(token_counts)
        avg_tokens = total_tokens / len(records) if records else 0
        min_tokens = min(token_counts) if token_counts else 0
        max_tokens = max(token_counts) if token_counts else 0
        
        # Calculate sequence statistics
        avg_seq = sum(sequence_lengths) / len(records) if records else 0
        min_seq = min(sequence_lengths) if sequence_lengths else 0
        max_seq = max(sequence_lengths) if sequence_lengths else 0
        
        # Create distributions
        token_distribution = create_histogram_buckets(token_counts)
        sequence_distribution = create_histogram_buckets(sequence_lengths)
        
        # Analyze field completeness
        field_stats = self._analyze_fields(records, detected_format)
        
        stats = DatasetStatistics(
            total_records=total_records,
            detected_format=detected_format.value,
            format_confidence=format_confidence,
            total_tokens=total_tokens,
            avg_tokens_per_record=avg_tokens,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            token_distribution=token_distribution,
            avg_sequence_length=avg_seq,
            min_sequence_length=min_seq,
            max_sequence_length=max_seq,
            sequence_distribution=sequence_distribution,
            field_stats=field_stats,
            format_indicators=format_indicators,
            validation_errors=validation_errors,
            is_valid=len(validation_errors) == 0,
        )
        
        logger.info(
            "Dataset analysis completed",
            extra={
                "format": detected_format.value,
                "total_records": total_records,
                "analyzed_records": len(records),
                "total_tokens": total_tokens,
                "avg_tokens": avg_tokens,
            },
        )
        
        return stats

    def _detect_format_indicators(self, sample: Dict) -> Dict[str, bool]:
        """Detect which format indicators are present in a sample record."""
        return {
            "has_instruction": "instruction" in sample,
            "has_input": "input" in sample,
            "has_output": "output" in sample,
            "has_conversations": "conversations" in sample,
            "has_messages": "messages" in sample,
            "has_prompt": "prompt" in sample,
            "has_completion": "completion" in sample,
            "has_text": "text" in sample,
            "has_system": "system" in sample or any(
                msg.get("role") == "system"
                for msg in sample.get("messages", [])
            ),
        }

    def _calculate_format_confidence(
        self, records: List[Dict], detected_format: DatasetFormat
    ) -> float:
        """Calculate confidence score for the detected format."""
        if not records:
            return 0.0
        
        matches = 0
        for record in records:
            record_format = detect_format(record)
            if record_format == detected_format:
                matches += 1
        
        return matches / len(records)

    def _analyze_fields(
        self, records: List[Dict], format_type: DatasetFormat
    ) -> List[FieldStats]:
        """Analyze field presence and statistics."""
        # Define expected fields based on format
        if format_type == DatasetFormat.ALPACA:
            expected_fields = ["instruction", "input", "output"]
        elif format_type == DatasetFormat.SHAREGPT:
            expected_fields = ["conversations"]
        elif format_type == DatasetFormat.CHATML:
            expected_fields = ["messages"]
        elif format_type == DatasetFormat.COMPLETION:
            expected_fields = ["prompt", "completion"]
        else:
            # Collect all unique fields from records
            expected_fields = list(set().union(*(r.keys() for r in records[:100])))
        
        field_stats = []
        total = len(records)
        
        for field_name in expected_fields:
            present_count = 0
            empty_count = 0
            lengths = []
            
            for record in records:
                if field_name in record:
                    present_count += 1
                    value = record[field_name]
                    
                    if isinstance(value, str):
                        length = len(value)
                        if length == 0:
                            empty_count += 1
                        lengths.append(length)
                    elif isinstance(value, list):
                        # For conversations/messages arrays
                        length = len(value)
                        if length == 0:
                            empty_count += 1
                        lengths.append(length)
            
            stats = FieldStats(
                name=field_name,
                present_count=present_count,
                total_count=total,
                min_length=min(lengths) if lengths else 0,
                max_length=max(lengths) if lengths else 0,
                avg_length=sum(lengths) / len(lengths) if lengths else 0,
                empty_count=empty_count,
            )
            field_stats.append(stats)
        
        return field_stats
