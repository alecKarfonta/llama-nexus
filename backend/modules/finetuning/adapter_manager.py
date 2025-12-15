"""
Adapter management module for LoRA adapters.

Features:
- Adapter registry with metadata and versioning
- Download/export adapters as ZIP archives
- Compare adapters (metrics, configs, benchmarks)
- Version tracking with tagging support
"""

import hashlib
import json
import os
import shutil
import uuid
import zipfile
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from enhanced_logger import enhanced_logger as logger


class AdapterStatus(str, Enum):
    TRAINING = "training"
    READY = "ready"
    MERGED = "merged"
    EXPORTED = "exported"
    ARCHIVED = "archived"


class AdapterVersion(BaseModel):
    """Represents a specific version of an adapter."""
    version: str  # semver-like: "1.0.0", "1.0.1", etc.
    created_at: datetime = Field(default_factory=datetime.utcnow)
    checkpoint_step: Optional[int] = None
    description: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    file_hash: Optional[str] = None
    file_size: int = 0
    tags: List[str] = Field(default_factory=list)
    is_current: bool = False


class AdapterMetadata(BaseModel):
    """Complete adapter metadata with versioning."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    base_model: str
    training_job_id: str
    adapter_path: str
    
    # Versioning
    current_version: str = "1.0.0"
    versions: List[AdapterVersion] = Field(default_factory=list)
    
    # Configuration used for training
    lora_config: Dict[str, Any] = Field(default_factory=dict)
    training_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Status and metrics
    status: AdapterStatus = AdapterStatus.TRAINING
    metrics: Dict[str, Any] = Field(default_factory=dict)
    benchmark_results: Dict[str, Any] = Field(default_factory=dict)
    
    # Export paths
    merged_path: Optional[str] = None
    gguf_path: Optional[str] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Tags for organization
    tags: List[str] = Field(default_factory=list)


class AdapterComparison(BaseModel):
    """Comparison results between two or more adapters."""
    adapter_ids: List[str]
    comparison_date: datetime = Field(default_factory=datetime.utcnow)
    
    # Configuration differences
    config_diffs: Dict[str, Any] = Field(default_factory=dict)
    
    # Metric comparisons
    metric_comparisons: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    # Benchmark comparisons (if available)
    benchmark_comparisons: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    # Summary
    summary: Dict[str, Any] = Field(default_factory=dict)


class AdapterManager:
    """Manages adapter lifecycle, versioning, and export."""
    
    def __init__(
        self,
        adapters_dir: Optional[Path] = None,
        exports_dir: Optional[Path] = None,
    ):
        self.adapters_dir = adapters_dir or Path(
            os.getenv("ADAPTERS_DIR", "data/adapters")
        )
        self.exports_dir = exports_dir or Path(
            os.getenv("ADAPTER_EXPORTS_DIR", "data/adapter_exports")
        )
        self.index_path = self.adapters_dir / "adapters_index.json"
        
        # Ensure directories exist
        self.adapters_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        
        # Load adapter index
        self._adapters: Dict[str, AdapterMetadata] = {}
        self._load_index()
    
    def _load_index(self) -> None:
        """Load adapter index from disk."""
        if self.index_path.exists():
            try:
                with self.index_path.open("r") as f:
                    data = json.load(f)
                for item in data.get("adapters", []):
                    adapter = AdapterMetadata(**item)
                    self._adapters[adapter.id] = adapter
            except (json.JSONDecodeError, IOError) as e:
                logger.error("Failed to load adapter index", extra={"error": str(e)})
    
    def _save_index(self) -> None:
        """Save adapter index to disk."""
        try:
            data = {
                "adapters": [a.dict() for a in self._adapters.values()],
                "updated_at": datetime.utcnow().isoformat(),
            }
            with self.index_path.open("w") as f:
                json.dump(data, f, indent=2, default=str)
        except IOError as e:
            logger.error("Failed to save adapter index", extra={"error": str(e)})
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _get_adapter_size(self, adapter_path: Path) -> int:
        """Get total size of adapter files."""
        total_size = 0
        if adapter_path.is_dir():
            for file in adapter_path.rglob("*"):
                if file.is_file():
                    total_size += file.stat().st_size
        elif adapter_path.is_file():
            total_size = adapter_path.stat().st_size
        return total_size
    
    def register_adapter(
        self,
        name: str,
        base_model: str,
        training_job_id: str,
        adapter_path: str,
        lora_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> AdapterMetadata:
        """Register a new adapter in the registry."""
        adapter_id = str(uuid.uuid4())
        
        # Create initial version
        adapter_path_obj = Path(adapter_path)
        file_size = self._get_adapter_size(adapter_path_obj)
        
        # Try to compute hash from adapter_model.safetensors or similar
        adapter_files = list(adapter_path_obj.glob("*.safetensors")) + \
                       list(adapter_path_obj.glob("*.bin"))
        file_hash = None
        if adapter_files:
            file_hash = self._compute_file_hash(adapter_files[0])
        
        initial_version = AdapterVersion(
            version="1.0.0",
            description="Initial version",
            metrics=metrics or {},
            file_hash=file_hash,
            file_size=file_size,
            is_current=True,
        )
        
        adapter = AdapterMetadata(
            id=adapter_id,
            name=name,
            description=description,
            base_model=base_model,
            training_job_id=training_job_id,
            adapter_path=adapter_path,
            current_version="1.0.0",
            versions=[initial_version],
            lora_config=lora_config or {},
            training_config=training_config or {},
            status=AdapterStatus.READY,
            metrics=metrics or {},
            tags=tags or [],
        )
        
        self._adapters[adapter_id] = adapter
        self._save_index()
        
        logger.info(
            "Registered adapter",
            extra={"adapter_id": adapter_id, "name": name}
        )
        
        return adapter
    
    def get_adapter(self, adapter_id: str) -> Optional[AdapterMetadata]:
        """Get adapter by ID."""
        return self._adapters.get(adapter_id)
    
    def get_adapter_by_name(self, name: str) -> Optional[AdapterMetadata]:
        """Get adapter by name."""
        for adapter in self._adapters.values():
            if adapter.name == name:
                return adapter
        return None
    
    def get_adapter_by_job(self, job_id: str) -> Optional[AdapterMetadata]:
        """Get adapter by training job ID."""
        for adapter in self._adapters.values():
            if adapter.training_job_id == job_id:
                return adapter
        return None
    
    def list_adapters(
        self,
        base_model: Optional[str] = None,
        status: Optional[AdapterStatus] = None,
        tags: Optional[List[str]] = None,
    ) -> List[AdapterMetadata]:
        """List adapters with optional filtering."""
        adapters = list(self._adapters.values())
        
        if base_model:
            adapters = [a for a in adapters if a.base_model == base_model]
        
        if status:
            adapters = [a for a in adapters if a.status == status]
        
        if tags:
            adapters = [
                a for a in adapters
                if any(t in a.tags for t in tags)
            ]
        
        # Sort by updated_at descending
        adapters.sort(key=lambda a: a.updated_at, reverse=True)
        
        return adapters
    
    def update_adapter(
        self,
        adapter_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        benchmark_results: Optional[Dict[str, Any]] = None,
        status: Optional[AdapterStatus] = None,
    ) -> Optional[AdapterMetadata]:
        """Update adapter metadata."""
        adapter = self._adapters.get(adapter_id)
        if not adapter:
            return None
        
        if name is not None:
            adapter.name = name
        if description is not None:
            adapter.description = description
        if tags is not None:
            adapter.tags = tags
        if metrics is not None:
            adapter.metrics.update(metrics)
        if benchmark_results is not None:
            adapter.benchmark_results.update(benchmark_results)
        if status is not None:
            adapter.status = status
        
        adapter.updated_at = datetime.utcnow()
        self._save_index()
        
        return adapter
    
    def create_version(
        self,
        adapter_id: str,
        version: Optional[str] = None,
        description: Optional[str] = None,
        checkpoint_step: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[AdapterVersion]:
        """Create a new version of an adapter."""
        adapter = self._adapters.get(adapter_id)
        if not adapter:
            return None
        
        # Auto-increment version if not specified
        if not version:
            current_parts = adapter.current_version.split(".")
            major, minor, patch = int(current_parts[0]), int(current_parts[1]), int(current_parts[2])
            version = f"{major}.{minor}.{patch + 1}"
        
        # Mark all existing versions as not current
        for v in adapter.versions:
            v.is_current = False
        
        # Get file info
        adapter_path_obj = Path(adapter.adapter_path)
        file_size = self._get_adapter_size(adapter_path_obj)
        adapter_files = list(adapter_path_obj.glob("*.safetensors")) + \
                       list(adapter_path_obj.glob("*.bin"))
        file_hash = None
        if adapter_files:
            file_hash = self._compute_file_hash(adapter_files[0])
        
        new_version = AdapterVersion(
            version=version,
            description=description,
            checkpoint_step=checkpoint_step,
            metrics=metrics or {},
            file_hash=file_hash,
            file_size=file_size,
            tags=tags or [],
            is_current=True,
        )
        
        adapter.versions.append(new_version)
        adapter.current_version = version
        adapter.updated_at = datetime.utcnow()
        self._save_index()
        
        logger.info(
            "Created adapter version",
            extra={"adapter_id": adapter_id, "version": version}
        )
        
        return new_version
    
    def get_version(
        self,
        adapter_id: str,
        version: str,
    ) -> Optional[AdapterVersion]:
        """Get a specific version of an adapter."""
        adapter = self._adapters.get(adapter_id)
        if not adapter:
            return None
        
        for v in adapter.versions:
            if v.version == version:
                return v
        return None
    
    def list_versions(self, adapter_id: str) -> List[AdapterVersion]:
        """List all versions of an adapter."""
        adapter = self._adapters.get(adapter_id)
        if not adapter:
            return []
        return sorted(adapter.versions, key=lambda v: v.created_at, reverse=True)
    
    def tag_version(
        self,
        adapter_id: str,
        version: str,
        tag: str,
    ) -> bool:
        """Add a tag to a specific version."""
        adapter = self._adapters.get(adapter_id)
        if not adapter:
            return False
        
        for v in adapter.versions:
            if v.version == version:
                if tag not in v.tags:
                    v.tags.append(tag)
                    self._save_index()
                return True
        return False
    
    def export_adapter(
        self,
        adapter_id: str,
        include_config: bool = True,
        include_metrics: bool = True,
    ) -> Optional[Tuple[Path, str]]:
        """
        Export adapter as a ZIP archive.
        
        Returns tuple of (zip_path, filename) or None if failed.
        """
        adapter = self._adapters.get(adapter_id)
        if not adapter:
            return None
        
        adapter_path = Path(adapter.adapter_path)
        if not adapter_path.exists():
            logger.error("Adapter path not found", extra={"path": str(adapter_path)})
            return None
        
        # Create export filename
        safe_name = adapter.name.replace(" ", "_").replace("/", "-")
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"{safe_name}_v{adapter.current_version}_{timestamp}.zip"
        zip_path = self.exports_dir / zip_filename
        
        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                # Add adapter files
                if adapter_path.is_dir():
                    for file in adapter_path.rglob("*"):
                        if file.is_file():
                            arcname = file.relative_to(adapter_path)
                            zf.write(file, arcname)
                else:
                    zf.write(adapter_path, adapter_path.name)
                
                # Add metadata
                if include_config:
                    metadata = {
                        "id": adapter.id,
                        "name": adapter.name,
                        "description": adapter.description,
                        "base_model": adapter.base_model,
                        "version": adapter.current_version,
                        "lora_config": adapter.lora_config,
                        "training_config": adapter.training_config,
                        "created_at": adapter.created_at.isoformat(),
                        "exported_at": datetime.utcnow().isoformat(),
                    }
                    zf.writestr("adapter_metadata.json", json.dumps(metadata, indent=2))
                
                # Add metrics
                if include_metrics and adapter.metrics:
                    zf.writestr("metrics.json", json.dumps(adapter.metrics, indent=2))
                
                # Add benchmark results if available
                if adapter.benchmark_results:
                    zf.writestr("benchmark_results.json", json.dumps(adapter.benchmark_results, indent=2))
            
            logger.info(
                "Exported adapter",
                extra={"adapter_id": adapter_id, "zip_path": str(zip_path)}
            )
            
            return zip_path, zip_filename
            
        except Exception as e:
            logger.error("Failed to export adapter", extra={"error": str(e)})
            if zip_path.exists():
                zip_path.unlink()
            return None
    
    def get_export_path(self, adapter_id: str) -> Optional[Path]:
        """Get the most recent export for an adapter."""
        adapter = self._adapters.get(adapter_id)
        if not adapter:
            return None
        
        safe_name = adapter.name.replace(" ", "_").replace("/", "-")
        exports = list(self.exports_dir.glob(f"{safe_name}_*.zip"))
        
        if not exports:
            return None
        
        # Return most recent
        exports.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return exports[0]
    
    def list_exports(self, adapter_id: str) -> List[Dict[str, Any]]:
        """List all exports for an adapter."""
        adapter = self._adapters.get(adapter_id)
        if not adapter:
            return []
        
        safe_name = adapter.name.replace(" ", "_").replace("/", "-")
        exports = list(self.exports_dir.glob(f"{safe_name}_*.zip"))
        
        result = []
        for export in exports:
            stat = export.stat()
            result.append({
                "filename": export.name,
                "path": str(export),
                "size": stat.st_size,
                "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
        
        result.sort(key=lambda x: x["created_at"], reverse=True)
        return result
    
    def compare_adapters(
        self,
        adapter_ids: List[str],
    ) -> Optional[AdapterComparison]:
        """
        Compare multiple adapters.
        
        Returns detailed comparison of configs, metrics, and benchmarks.
        """
        if len(adapter_ids) < 2:
            return None
        
        adapters = []
        for aid in adapter_ids:
            adapter = self._adapters.get(aid)
            if adapter:
                adapters.append(adapter)
        
        if len(adapters) < 2:
            return None
        
        comparison = AdapterComparison(adapter_ids=adapter_ids)
        
        # Compare configurations
        config_diffs = {
            "lora_config": {},
            "training_config": {},
            "base_model": {},
        }
        
        # LoRA config comparison
        lora_keys = set()
        for a in adapters:
            lora_keys.update(a.lora_config.keys())
        
        for key in lora_keys:
            values = {a.id: a.lora_config.get(key) for a in adapters}
            if len(set(str(v) for v in values.values())) > 1:
                config_diffs["lora_config"][key] = values
        
        # Training config comparison
        train_keys = set()
        for a in adapters:
            train_keys.update(a.training_config.keys())
        
        for key in train_keys:
            values = {a.id: a.training_config.get(key) for a in adapters}
            if len(set(str(v) for v in values.values())) > 1:
                config_diffs["training_config"][key] = values
        
        # Base model comparison
        base_models = {a.id: a.base_model for a in adapters}
        if len(set(base_models.values())) > 1:
            config_diffs["base_model"] = base_models
        
        comparison.config_diffs = config_diffs
        
        # Compare metrics
        metric_keys = set()
        for a in adapters:
            metric_keys.update(a.metrics.keys())
        
        for key in metric_keys:
            comparison.metric_comparisons[key] = {
                a.id: a.metrics.get(key) for a in adapters
            }
        
        # Compare benchmark results
        benchmark_keys = set()
        for a in adapters:
            benchmark_keys.update(a.benchmark_results.keys())
        
        for key in benchmark_keys:
            comparison.benchmark_comparisons[key] = {
                a.id: a.benchmark_results.get(key) for a in adapters
            }
        
        # Generate summary
        summary = {
            "num_adapters": len(adapters),
            "adapters": [
                {
                    "id": a.id,
                    "name": a.name,
                    "base_model": a.base_model,
                    "version": a.current_version,
                    "status": a.status.value,
                }
                for a in adapters
            ],
            "config_differences": len([k for k, v in config_diffs.items() if v]),
            "shared_base_model": len(set(a.base_model for a in adapters)) == 1,
        }
        
        # Find best performer for key metrics
        for metric in ["final_loss", "accuracy", "perplexity"]:
            values = {
                a.id: a.metrics.get(metric)
                for a in adapters
                if a.metrics.get(metric) is not None
            }
            if values:
                if metric == "final_loss":
                    best_id = min(values, key=values.get)
                else:
                    best_id = max(values, key=values.get)
                summary[f"best_{metric}"] = {
                    "adapter_id": best_id,
                    "value": values[best_id],
                }
        
        comparison.summary = summary
        
        return comparison
    
    def delete_adapter(self, adapter_id: str) -> bool:
        """Delete an adapter from the registry."""
        if adapter_id not in self._adapters:
            return False
        
        adapter = self._adapters[adapter_id]
        
        # Delete exports
        safe_name = adapter.name.replace(" ", "_").replace("/", "-")
        for export in self.exports_dir.glob(f"{safe_name}_*.zip"):
            try:
                export.unlink()
            except IOError:
                pass
        
        del self._adapters[adapter_id]
        self._save_index()
        
        logger.info("Deleted adapter", extra={"adapter_id": adapter_id})
        
        return True
    
    def archive_adapter(self, adapter_id: str) -> Optional[AdapterMetadata]:
        """Archive an adapter (soft delete)."""
        return self.update_adapter(adapter_id, status=AdapterStatus.ARCHIVED)


# Module-level instance
_adapter_manager: Optional[AdapterManager] = None


def get_adapter_manager() -> AdapterManager:
    """Get the global adapter manager instance."""
    global _adapter_manager
    if _adapter_manager is None:
        _adapter_manager = AdapterManager()
    return _adapter_manager
