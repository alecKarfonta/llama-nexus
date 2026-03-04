import os
import asyncio
import subprocess
import json
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
import psutil
import httpx
from collections import deque
import re

from huggingface_hub import hf_hub_url, HfApi
from enhanced_logger import enhanced_logger as logger
from modules.managers.base import DOCKER_AVAILABLE, docker_client


from dataclasses import dataclass, asdict

@dataclass
class DownloadRecord:
    model_id: str
    repo_id: str
    filename: str
    status: str  # queued | downloading | completed | failed | cancelled
    progress: float
    total_size: int
    downloaded_size: int
    speed: float
    eta: float
    error: Optional[str] = None
    parts_info: Optional[Dict[str, Any]] = None  # For multi-part downloads



class ModelDownloadManager:
    def __init__(self):
        self._downloads: Dict[str, DownloadRecord] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._cancel_events: Dict[str, asyncio.Event] = {}
        self._lock = asyncio.Lock()
        # Models directory shared with llamacpp
        self.models_dir = Path("/home/llamacpp/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        # Metadata directory for storing model source information
        self.metadata_dir = self.models_dir / ".metadata"
        self.metadata_dir.mkdir(exist_ok=True)

    def _derive_model_id(self, filename: str) -> str:
        # Strip extension
        base = filename[:-5] if filename.endswith('.gguf') else Path(filename).stem
        return base

    def _save_model_metadata(self, model_name: str, variant: str, repo_id: str, filename: str):
        """Save model metadata to track repository source."""
        metadata = {
            "model_name": model_name,
            "variant": variant,
            "repo_id": repo_id,
            "filename": filename,
            "downloaded_at": datetime.now().isoformat(),
            "source": "huggingface"
        }
        
        # Use model_name-variant as the metadata filename
        metadata_file = self.metadata_dir / f"{model_name}-{variant}.json"
        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved metadata for {model_name}-{variant} from {repo_id}")
        except Exception as e:
            logger.warning(f"Failed to save metadata for {model_name}-{variant}: {e}")

    def _load_model_metadata(self, model_name: str, variant: str) -> Optional[Dict[str, Any]]:
        """Load model metadata if it exists."""
        metadata_file = self.metadata_dir / f"{model_name}-{variant}.json"
        try:
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load metadata for {model_name}-{variant}: {e}")
        return None

    def get_model_repository(self, model_name: str, variant: str) -> Optional[str]:
        """Get the repository ID for a model, or None if unknown."""
        metadata = self._load_model_metadata(model_name, variant)
        return metadata.get("repo_id") if metadata else None

    async def list_local_models(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        seen_models = set()  # Track models to avoid duplicates for multi-part files
        
        # Scan both root directory and subdirectories for .gguf files
        for entry in self.models_dir.rglob('*.gguf'):
            try:
                stat = entry.stat()
                name, variant = self._parse_name_variant(entry.name)
                
                # Create a unique identifier for this model (combining name and variant)
                model_key = f"{name}:{variant or 'unknown'}"
                
                # For multi-part files, only add the model once
                if self._is_multipart_file(entry.name):
                    # Only include the first part to represent the complete model
                    if '-00001-of-' not in entry.name:
                        continue
                    
                    # Calculate total size for multi-part files
                    all_parts = self._get_multipart_files(entry.name)
                    total_size = 0
                    for part_file in all_parts:
                        part_path = entry.parent / part_file
                        if part_path.exists():
                            total_size += part_path.stat().st_size
                    
                    # Use total size and latest modification time
                    latest_mtime = stat.st_mtime
                    for part_file in all_parts:
                        part_path = entry.parent / part_file
                        if part_path.exists():
                            part_stat = part_path.stat()
                            if part_stat.st_mtime > latest_mtime:
                                latest_mtime = part_stat.st_mtime
                else:
                    total_size = stat.st_size
                    latest_mtime = stat.st_mtime
                
                # Only add if we haven't seen this model before
                if model_key not in seen_models:
                    seen_models.add(model_key)
                    # Get relative path for linking to local files
                    relative_path = str(entry.relative_to(self.models_dir))
                    items.append({
                        "name": name,
                        "variant": variant or "unknown",
                        "size": total_size,
                        "status": "available",
                        "lastModified": datetime.fromtimestamp(latest_mtime).isoformat(),
                        "localPath": relative_path,
                        "filename": entry.name
                    })
                    
            except Exception as e:
                # Skip any unreadable files but log the error
                logger.debug(f"Skipping unreadable model file {entry}: {e}")
                continue
        
        # Scan for Transformers-style model directories (safetensors/bin with config.json)
        for config_file in self.models_dir.rglob('config.json'):
            try:
                # Skip if in .metadata, checkpoint, or cache directories
                config_path_str = str(config_file)
                if any(skip in config_path_str for skip in ['.metadata', 'checkpoint', '.cache', '__pycache__']):
                    continue
                    
                model_dir = config_file.parent
                
                # Check if directory contains model weights
                weight_files = list(model_dir.glob('*.safetensors')) + list(model_dir.glob('model*.bin'))
                if not weight_files:
                    continue
                    
                # Parse config.json for metadata
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Skip if this looks like an adapter config (LoRA)
                if 'base_model_name_or_path' in config or 'peft_type' in config:
                    continue
                
                # Extract parameters count
                parameters = self._extract_parameter_count(config)
                context_length = config.get('max_position_embeddings', 2048)
                
                # Get architecture type
                architectures = config.get('architectures', [])
                architecture = config.get('model_type', architectures[0] if architectures else 'unknown')
                
                # Calculate total size from weight files
                total_size = sum(wf.stat().st_size for wf in weight_files)
                latest_mtime = max(wf.stat().st_mtime for wf in weight_files)
                
                # Use directory name as model name
                model_name = model_dir.name
                
                # Determine variant from weight file type
                has_safetensors = any(wf.suffix == '.safetensors' for wf in weight_files)
                variant = "safetensors" if has_safetensors else "bin"
                
                # Check for training metadata (indicates fine-tuned model)
                training_metadata = None
                training_metadata_path = model_dir / 'training_metadata.json'
                if training_metadata_path.exists():
                    try:
                        with open(training_metadata_path, 'r') as f:
                            training_metadata = json.load(f)
                    except Exception as e:
                        logger.debug(f"Could not load training metadata: {e}")
                
                model_key = f"{model_name}:transformers"
                if model_key not in seen_models:
                    seen_models.add(model_key)
                    model_info = {
                        "name": model_name,
                        "variant": variant,
                        "size": total_size,
                        "status": "available",
                        "lastModified": datetime.fromtimestamp(latest_mtime).isoformat(),
                        "localPath": str(model_dir.relative_to(self.models_dir)),
                        "filename": model_name,
                        "framework": "transformers",
                        "parameters": parameters,
                        "contextLength": context_length,
                        "architecture": architecture
                    }
                    
                    # Include training metadata if this is a fine-tuned model
                    if training_metadata:
                        model_info["source"] = training_metadata.get("source", "trained")
                        model_info["trainingJobId"] = training_metadata.get("training_job_id")
                        model_info["baseModel"] = training_metadata.get("base_model")
                        model_info["mergedAt"] = training_metadata.get("merged_at")
                        model_info["finalLoss"] = training_metadata.get("final_loss")
                        model_info["totalSteps"] = training_metadata.get("total_steps")
                        model_info["loraConfig"] = training_metadata.get("lora_config")
                    
                    items.append(model_info)
            except Exception as e:
                logger.debug(f"Skipping invalid model directory {config_file.parent}: {e}")
                continue
                
        # Merge in any actively downloading items (only queued/downloading, not completed)
        async with self._lock:
            for rec in self._downloads.values():
                # Only show actively downloading items - completed ones should appear in local scan
                if rec.status not in ("queued", "downloading"):
                    continue
                
                # For Transformers models (safetensors/bin), use repo_id as model name
                is_transformers = any(rec.filename.lower().endswith(ext) for ext in ('.safetensors', '.bin', '.pth', '.pt'))
                
                if is_transformers:
                    # Use repo_id formatted as model name (same as subdirectory)
                    name = rec.repo_id.replace('/', '_')
                    variant = "safetensors" if rec.filename.lower().endswith('.safetensors') else "bin"
                else:
                    name, variant = self._parse_name_variant(rec.filename)
                    variant = variant or "unknown"
                
                model_key = f"{name}:{variant}"
                
                # Only add if not already present from local scan
                if model_key not in seen_models:
                    seen_models.add(model_key)
                    items.append({
                        "name": name,
                        "variant": variant,
                        "size": rec.total_size or 0,
                        "status": "downloading",
                        "downloadProgress": rec.progress,
                        "framework": "transformers" if is_transformers else "gguf"
                    })
                        
        return items

    def _parse_name_variant(self, filename: str) -> (str, Optional[str]):
        # Remove extension
        stem = filename[:-5] if filename.endswith('.gguf') else Path(filename).stem
        
        # Handle multi-part files by removing the part suffix first
        import re
        multipart_pattern = r'(.+)-(\d{5}-of-\d{5})$'
        multipart_match = re.match(multipart_pattern, stem)
        if multipart_match:
            stem = multipart_match.group(1)
        
        # Try dot pattern: ModelName.Q4_K_M
        if '.' in stem:
            parts = stem.split('.')
            if parts[-1].startswith('Q'):
                return ('.'.join(parts[:-1]), parts[-1])
        
        # Try hyphen pattern: ModelName-Q4_K_M
        if '-' in stem:
            parts = stem.split('-')
            if parts[-1].startswith('Q'):
                return ('-'.join(parts[:-1]), parts[-1])
        
        # Try to extract variant from end of filename (e.g., devstralQ4_K_M)
        # Look for common quantization patterns like Q4_K_M, Q5_K_S, etc.
        pattern = r'(.*?)(Q\d+_[KS](_[MS])?)$'
        match = re.match(pattern, stem)
        if match:
            return (match.group(1), match.group(2))
        
        # If no pattern matches, check if the entire name contains a quantization pattern
        # This handles cases where the quantization is embedded in the name
        if re.search(r'Q\d+_[KS](_[MS])?', stem):
            # Extract the quantization pattern
            quant_match = re.search(r'(Q\d+_[KS](_[MS])?)', stem)
            if quant_match:
                quant = quant_match.group(1)
                # Remove the quantization from the name
                name = stem.replace(quant, '')
                # Clean up any trailing underscores or hyphens
                name = name.rstrip('_-')
                if name:
                    return (name, quant)
        
        return (stem, None)

    def _extract_parameter_count(self, config: Dict[str, Any]) -> str:
        """Extract or estimate parameter count from Transformers config.json."""
        # Check if explicitly provided
        if 'num_parameters' in config:
            return self._format_param_count(config['num_parameters'])
        
        # Estimate from architecture parameters
        hidden_size = config.get('hidden_size', 0)
        num_hidden_layers = config.get('num_hidden_layers', 0)
        vocab_size = config.get('vocab_size', 0)
        intermediate_size = config.get('intermediate_size', hidden_size * 4 if hidden_size else 0)
        
        if hidden_size and num_hidden_layers and vocab_size:
            # Rough estimate: embeddings + transformer layers
            embedding_params = vocab_size * hidden_size * 2  # input + output embeddings
            layer_params = num_hidden_layers * (
                4 * hidden_size * hidden_size +  # attention (Q, K, V, O)
                2 * hidden_size * intermediate_size +  # FFN
                4 * hidden_size  # layer norms
            )
            total_params = embedding_params + layer_params
            return self._format_param_count(total_params)
        
        return "?B"

    def _format_param_count(self, num_params: int) -> str:
        """Format parameter count as human-readable string."""
        if num_params >= 1e12:
            return f"{num_params/1e12:.1f}T"
        elif num_params >= 1e9:
            return f"{num_params/1e9:.1f}B"
        elif num_params >= 1e6:
            return f"{num_params/1e6:.0f}M"
        else:
            return f"{num_params/1e3:.0f}K"

    def _is_multipart_file(self, filename: str) -> bool:
        """Check if filename indicates a multi-part model file (GGUF or safetensors)"""
        import re
        # Pattern for multi-part GGUF: filename-00001-of-00002.gguf
        # Pattern for multi-part safetensors: model-00001-of-00002.safetensors
        return bool(re.search(r'-\d{5}-of-\d{5}\.(gguf|safetensors)$', filename))
    
    def _get_multipart_files(self, filename: str) -> List[str]:
        """Get all part filenames for a multi-part download (GGUF or safetensors)"""
        import re
        
        # Extract the base name, part info, and extension
        match = re.match(r'(.+)-(\d{5})-of-(\d{5})\.(gguf|safetensors)$', filename)
        if not match:
            return [filename]  # Not a multi-part file
        
        base_name = match.group(1)
        total_parts = int(match.group(3))
        extension = match.group(4)
        
        # Generate all part filenames
        part_files = []
        for i in range(1, total_parts + 1):
            part_filename = f"{base_name}-{i:05d}-of-{total_parts:05d}.{extension}"
            part_files.append(part_filename)
        
        return part_files
    


    async def get_downloads(self) -> List[Dict[str, Any]]:
        async with self._lock:
            return [asdict(rec) for rec in self._downloads.values()]

    async def start_download(self, repo_id: str, filename: str) -> DownloadRecord:
        model_id = self._derive_model_id(filename)
        dest_path = self.models_dir / filename
        async with self._lock:
            if model_id in self._downloads and self._downloads[model_id].status in ("queued", "downloading"):
                raise HTTPException(status_code=400, detail=f"Download already in progress for {model_id}")
            
            # Check if this is a multi-part file and handle accordingly
            is_multipart = self._is_multipart_file(filename)
            if is_multipart:
                logger.info(f"Detected multi-part file: {filename}")
                # For multi-part files, we need to download all parts
                part_files = self._get_multipart_files(filename)
                logger.info(f"Multi-part files to download: {part_files}")
                
                # Create a download record for the combined model
                rec = DownloadRecord(
                    model_id=model_id,
                    repo_id=repo_id,
                    filename=filename,
                    status="queued",
                    progress=0.0,
                    total_size=0,
                    downloaded_size=0,
                    speed=0.0,
                    eta=0.0,
                )
                self._downloads[model_id] = rec
                cancel_event = asyncio.Event()
                self._cancel_events[model_id] = cancel_event
                
                # Launch background task for multi-part download
                task = asyncio.create_task(self._run_multipart_download(model_id, repo_id, part_files, cancel_event))
                self._tasks[model_id] = task
                return rec
            else:
                # Initialize record for single file
                rec = DownloadRecord(
                    model_id=model_id,
                    repo_id=repo_id,
                    filename=filename,
                    status="queued",
                    progress=0.0,
                    total_size=0,
                    downloaded_size=0,
                    speed=0.0,
                    eta=0.0,
                )
                self._downloads[model_id] = rec
                cancel_event = asyncio.Event()
                self._cancel_events[model_id] = cancel_event
                # Launch background task
                task = asyncio.create_task(self._run_download(model_id, repo_id, filename, dest_path, cancel_event))
                self._tasks[model_id] = task
                return rec

    async def cancel_download(self, model_id: str) -> bool:
        async with self._lock:
            if model_id not in self._downloads:
                raise HTTPException(status_code=404, detail=f"No download found for {model_id}")
            rec = self._downloads[model_id]
            if rec.status not in ("queued", "downloading"):
                raise HTTPException(status_code=400, detail=f"Download not in progress for {model_id}")
            self._cancel_events[model_id].set()
            return True

    async def _download_transformers_metadata(self, repo_id: str, model_dir: Path, cancel_event: asyncio.Event):
        """Download config.json and other essential metadata files for Transformers models."""
        metadata_files = ['config.json', 'tokenizer_config.json', 'tokenizer.json']
        
        headers = {"User-Agent": "llama-nexus1.0"}
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
        
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            for meta_file in metadata_files:
                if cancel_event.is_set():
                    return
                try:
                    url = hf_hub_url(repo_id=repo_id, filename=meta_file)
                    response = await client.get(url, headers=headers)
                    if response.status_code == 200:
                        (model_dir / meta_file).write_bytes(response.content)
                        logger.info(f"Downloaded {meta_file} for {repo_id}")
                except Exception as e:
                    logger.debug(f"Could not download {meta_file} for {repo_id}: {e}")

    async def _run_download(self, model_id: str, repo_id: str, filename: str, dest_path: Path, cancel_event: asyncio.Event):
        # Check if this is a Transformers model (safetensors/bin) - needs special handling
        is_transformers_model = any(filename.lower().endswith(ext) for ext in ('.safetensors', '.bin', '.pth', '.pt')) and not filename.endswith('.gguf')
        
        if is_transformers_model:
            # Create a subdirectory for the model based on repo_id
            model_subdir = repo_id.replace('/', '_')
            model_dir = self.models_dir / model_subdir
            model_dir.mkdir(parents=True, exist_ok=True)
            dest_path = model_dir / filename
            
            # Download config.json and tokenizer files for metadata
            logger.info(f"Downloading Transformers model to {model_dir}")
            await self._download_transformers_metadata(repo_id, model_dir, cancel_event)
        
        url = hf_hub_url(repo_id=repo_id, filename=filename)
        headers = {"User-Agent": "llama-nexus1.0"}
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
        temp_path = dest_path.with_suffix('.part')
        
        # Ensure parent directories exist
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        start_time = time.time()
        last_report_time = start_time
        last_reported = 0

        async with self._lock:
            rec = self._downloads[model_id]
            rec.status = "downloading"

        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=None) as client:
                # HEAD to get size
                head = await client.head(url, headers=headers)
                if head.status_code >= 400:
                    raise HTTPException(status_code=502, detail=f"Failed to fetch file headers: HTTP {head.status_code}")
                total_size = int(head.headers.get('Content-Length', '0'))
                async with self._lock:
                    rec.total_size = total_size
                # Stream GET
                async with client.stream('GET', url, headers=headers) as resp:
                    if resp.status_code >= 400:
                        raise HTTPException(status_code=502, detail=f"Failed to start download: HTTP {resp.status_code}")
                    with open(temp_path, 'wb') as f:
                        async for chunk in resp.aiter_bytes(chunk_size=1024 * 1024):
                            if cancel_event.is_set():
                                raise asyncio.CancelledError()
                            if not chunk:
                                continue
                            f.write(chunk)
                            async with self._lock:
                                rec.downloaded_size += len(chunk)
                                if rec.total_size > 0:
                                    rec.progress = min(100.0, (rec.downloaded_size / rec.total_size) * 100)
                            # update speed periodically
                            now = time.time()
                            if now - last_report_time >= 0.5:
                                bytes_delta = rec.downloaded_size - last_reported
                                time_delta = now - last_report_time
                                async with self._lock:
                                    rec.speed = bytes_delta / time_delta if time_delta > 0 else 0
                                    remaining = max(0, rec.total_size - rec.downloaded_size)
                                    rec.eta = (remaining / rec.speed) if rec.speed > 0 else 0
                                last_report_time = now
                                last_reported = rec.downloaded_size
                # Move temp to final
                temp_path.replace(dest_path)
                async with self._lock:
                    rec.status = "completed"
                    rec.progress = 100.0
                    rec.speed = (rec.total_size / max(1e-6, (time.time() - start_time))) if rec.total_size else 0.0
                    rec.eta = 0.0
                
                # Save metadata for the downloaded model
                try:
                    model_name, variant = self._parse_name_variant(filename)
                    if model_name and variant:
                        self._save_model_metadata(model_name, variant, repo_id, filename)
                except Exception as e:
                    logger.warning(f"Failed to save metadata for {filename}: {e}")
        except asyncio.CancelledError:
            # Cleanup and mark cancelled
            try:
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass
            async with self._lock:
                rec.status = "cancelled"
                rec.error = "Cancelled by user"
                rec.eta = 0.0
        except HTTPException as e:
            try:
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass
            async with self._lock:
                rec.status = "failed"
                rec.error = e.detail if isinstance(e.detail, str) else str(e.detail)
        except Exception as e:
            try:
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass
            async with self._lock:
                rec.status = "failed"
                rec.error = str(e)
        finally:
            # Cleanup task maps
            async with self._lock:
                self._tasks.pop(model_id, None)
                self._cancel_events.pop(model_id, None)

    async def _run_multipart_download(self, model_id: str, repo_id: str, part_files: List[str], cancel_event: asyncio.Event):
        """Download all parts of a multi-part model file (GGUF or safetensors)."""
        start_time = time.time()
        
        # Check if this is a Transformers model (safetensors) - needs subdirectory
        first_file = part_files[0] if part_files else ""
        is_transformers_model = first_file.lower().endswith('.safetensors')
        
        if is_transformers_model:
            # Create a subdirectory for the model based on repo_id
            model_subdir = repo_id.replace('/', '_')
            model_dir = self.models_dir / model_subdir
            model_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Downloading multi-part Transformers model to {model_dir}")
            await self._download_transformers_metadata(repo_id, model_dir, cancel_event)
        else:
            model_dir = self.models_dir
        
        async with self._lock:
            rec = self._downloads[model_id]
            rec.status = "downloading"
            # Add parts info for better progress tracking
            rec.parts_info = {"total": len(part_files), "completed": 0, "current": ""}
        
        try:
            # Get total size of all parts
            total_size = 0
            headers = {"User-Agent": "llama-nexus1.0"}
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            if hf_token:
                headers["Authorization"] = f"Bearer {hf_token}"
            
            async with httpx.AsyncClient(follow_redirects=True, timeout=None) as client:
                # Get sizes of all parts
                logger.info(f"Getting file sizes for {len(part_files)} parts...")
                for part_file in part_files:
                    url = hf_hub_url(repo_id=repo_id, filename=part_file)
                    head = await client.head(url, headers=headers)
                    if head.status_code >= 400:
                        raise HTTPException(status_code=502, detail=f"Failed to fetch headers for {part_file}: HTTP {head.status_code}")
                    part_size = int(head.headers.get('Content-Length', '0'))
                    total_size += part_size
                
                async with self._lock:
                    rec.total_size = total_size
                
                # Download all parts
                total_downloaded = 0
                
                for i, part_file in enumerate(part_files):
                    if cancel_event.is_set():
                        raise asyncio.CancelledError()
                    
                    logger.info(f"Downloading part {i+1}/{len(part_files)}: {part_file}")
                    async with self._lock:
                        rec.parts_info["current"] = f"Part {i+1}/{len(part_files)}: {part_file}"
                    
                    url = hf_hub_url(repo_id=repo_id, filename=part_file)
                    temp_path = model_dir / f"{part_file}.part"
                    final_path = model_dir / part_file
                    
                    # Ensure parent directories exist
                    temp_path.parent.mkdir(parents=True, exist_ok=True)
                    final_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    async with client.stream('GET', url, headers=headers) as resp:
                        if resp.status_code >= 400:
                            raise HTTPException(status_code=502, detail=f"Failed to download {part_file}: HTTP {resp.status_code}")
                        
                        with open(temp_path, 'wb') as f:
                            async for chunk in resp.aiter_bytes(chunk_size=1024 * 1024):
                                if cancel_event.is_set():
                                    raise asyncio.CancelledError()
                                if not chunk:
                                    continue
                                f.write(chunk)
                                total_downloaded += len(chunk)
                                
                                async with self._lock:
                                    rec.downloaded_size = total_downloaded
                                    if rec.total_size > 0:
                                        rec.progress = min(100.0, (total_downloaded / rec.total_size) * 100)
                    
                    # Move temp to final
                    temp_path.replace(final_path)
                    
                    async with self._lock:
                        rec.parts_info["completed"] = i + 1
                    
                    logger.info(f"Completed part {i+1}/{len(part_files)}: {part_file}")
                
                logger.info(f"Successfully downloaded all {len(part_files)} parts. llamacpp will automatically load all parts when using the first part.")
                
                async with self._lock:
                    rec.status = "completed"
                    rec.progress = 100.0
                    rec.speed = (rec.total_size / max(1e-6, (time.time() - start_time))) if rec.total_size else 0.0
                    rec.eta = 0.0
                    rec.parts_info["current"] = f"Completed all {len(part_files)} parts"
                
                # Save metadata for the downloaded model (use first part filename for parsing)
                try:
                    first_part = part_files[0]
                    model_name, variant = self._parse_name_variant(first_part)
                    if model_name and variant:
                        self._save_model_metadata(model_name, variant, repo_id, first_part)
                except Exception as e:
                    logger.warning(f"Failed to save metadata for multipart download: {e}")
                    
        except asyncio.CancelledError:
            # Cleanup any partial downloads
            for part_file in part_files:
                try:
                    temp_path = self.models_dir / f"{part_file}.part"
                    final_path = self.models_dir / part_file
                    if temp_path.exists():
                        temp_path.unlink()
                    if final_path.exists():
                        final_path.unlink()
                except Exception:
                    pass
            async with self._lock:
                rec.status = "cancelled"
                rec.error = "Cancelled by user"
                rec.eta = 0.0
        except Exception as e:
            # Cleanup any partial downloads
            for part_file in part_files:
                try:
                    temp_path = self.models_dir / f"{part_file}.part"
                    final_path = self.models_dir / part_file
                    if temp_path.exists():
                        temp_path.unlink()
                    if final_path.exists():
                        final_path.unlink()
                except Exception:
                    pass
            async with self._lock:
                rec.status = "failed"
                rec.error = str(e)
        finally:
            # Cleanup task maps
            async with self._lock:
                self._tasks.pop(model_id, None)
                self._cancel_events.pop(model_id, None)
    
    async def read_subprocess_logs(self):
        """Read logs from subprocess and broadcast to websocket clients"""
        if not self.process:
            return
        
        try:
            # Create tasks to read both stdout and stderr
            async def read_stdout():
                if self.process.stdout:
                    while self.process and self.process.poll() is None:
                        line = self.process.stdout.readline()
                        if line:
                            await self.add_log_line(f"[STDOUT] {line.strip()}")
                        else:
                            await asyncio.sleep(0.1)
            
            async def read_stderr():
                if self.process.stderr:
                    while self.process and self.process.poll() is None:
                        line = self.process.stderr.readline()
                        if line:
                            await self.add_log_line(f"[STDERR] {line.strip()}")
                        else:
                            await asyncio.sleep(0.1)
            
            # Run both readers concurrently
            await asyncio.gather(read_stdout(), read_stderr(), return_exceptions=True)
            
        except asyncio.CancelledError:
            logger.info("Log reader task was cancelled")
        except Exception as e:
            logger.error(f"Error reading subprocess logs: {e}", exc_info=True)
    
    async def read_docker_logs(self):
        """Read logs from Docker container using docker CLI as fallback"""
        try:
            if docker_client and DOCKER_AVAILABLE:
                # Try using Python Docker client first
                container = self.docker_container or docker_client.containers.get(self.container_name)
                # Stream logs
                for line in container.logs(stream=True, follow=True):
                    if isinstance(line, bytes):
                        line = line.decode('utf-8')
                    await self.add_log_line(line.strip())
            else:
                # Fallback to docker CLI subprocess
                await self.read_docker_logs_cli()
                
        except Exception as e:
            logger.warning(f"Error reading Docker logs with Python client: {e}")
            # Fallback to CLI method
            try:
                await self.read_docker_logs_cli()
            except Exception as cli_error:
                logger.error(f"Error reading Docker logs with CLI: {cli_error}")
    
    async def read_docker_logs_cli(self):
        """Read logs from Docker container using CLI"""
        try:
            # Use docker logs with follow flag to stream logs
            process = await asyncio.create_subprocess_exec(
                'docker', 'logs', '-f', self.container_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )
            
            logger.info(f"Started reading Docker logs for container: {self.container_name}")
            
            async for line in process.stdout:
                if line:
                    line_str = line.decode('utf-8').strip()
                    if line_str:  # Only add non-empty lines
                        await self.add_log_line(line_str)
                        
        except Exception as e:
            logger.error(f"Error reading Docker logs via CLI: {e}")
    
    async def check_existing_container(self):
        """Check if target container is already running and start log reading"""
        try:
            # Wait a bit for the manager to initialize
            await asyncio.sleep(2)
            
            # Check if container is running using CLI
            process = await asyncio.create_subprocess_exec(
                'docker', 'ps', '--filter', f'name={self.container_name}', '--format', '{{.Names}}',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                container_names = stdout.decode().strip().split('\n')
                if self.container_name in container_names:
                    logger.info(f"Found existing container {self.container_name}, starting log reading")
                    self.log_reader_task = asyncio.create_task(self.read_docker_logs_cli())
                else:
                    logger.info(f"Container {self.container_name} not found or not running")
            else:
                logger.warning(f"Failed to check for existing container: {stderr.decode()}")
                
        except Exception as e:
            logger.warning(f"Error checking for existing container: {e}")

    async def add_log_line(self, line: str):
        """Add a log line and broadcast to websocket clients"""
        # Add to buffer
        self.log_buffer.append({
            "message": line,
            "timestamp": datetime.now().isoformat()
        })
        
        # Broadcast to websocket clients
        disconnected_clients = []
        for client in self.websocket_clients:
            try:
                await client.send_json({
                    "type": "log",
                    "data": line,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception:
                disconnected_clients.append(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            self.websocket_clients.remove(client)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of llamacpp"""
        is_running = False
        pid = None
        
        if self.use_docker:
            if DOCKER_AVAILABLE and docker_client:
                # Use Python SDK
                try:
                    container = docker_client.containers.get(self.container_name)
                    is_running = container.status == "running"
                    if is_running:
                        pid = container.attrs['State']['Pid']
                except:
                    pass
            else:
                # Use Docker CLI
                try:
                    import subprocess
                    result = subprocess.run(
                        ['docker', 'ps', '--filter', f'name={self.container_name}', '--format', '{{.Names}}'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        container_names = result.stdout.strip().split('\n')
                        is_running = self.container_name in container_names and result.stdout.strip()
                        
                        if is_running:
                            # Get PID using docker inspect
                            pid_result = subprocess.run(
                                ['docker', 'inspect', '--format', '{{.State.Pid}}', self.container_name],
                                capture_output=True,
                                text=True,
                                timeout=5
                            )
                            if pid_result.returncode == 0:
                                try:
                                    pid = int(pid_result.stdout.strip())
                                except:
                                    pass
                except Exception as e:
                    logger.warning(f"Error checking Docker container status: {e}")
        else:
            is_running = self.process is not None and self.process.poll() is None
            if is_running:
                pid = self.process.pid
        
        status = {
            "running": is_running,
            "pid": pid,
            "uptime": (datetime.now() - self.start_time).total_seconds() if self.start_time and is_running else 0,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "mode": "docker" if self.use_docker else "subprocess",
            "config": self.config,
            "model": {
                "name": self.config["model"]["name"],
                "variant": self.config["model"]["variant"],
                "context_size": self.config["model"]["context_size"],
                "gpu_layers": self.config["model"]["gpu_layers"],
                "n_cpu_moe": self.config["model"]["n_cpu_moe"],
            }
        }
        
        # Add resource usage if running
        if is_running and pid:
            try:
                process = psutil.Process(pid)
                memory_info = process.memory_info()
                status["resources"] = {
                    "cpu_percent": process.cpu_percent(),
                    "memory_mb": memory_info.rss / (1024 * 1024),
                    "memory_percent": process.memory_percent(),
                    "num_threads": process.num_threads(),
                }
            except:
                pass
        
        # Add GPU info
        try:
            # First check if nvidia-smi is available
            which_result = subprocess.run(
                ["which", "nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if which_result.returncode == 0:
                # nvidia-smi is available, try to get GPU info
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu", 
                     "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    values = result.stdout.strip().split(", ")
                    if len(values) >= 4:
                        status["gpu"] = {
                            "vram_used_mb": int(values[0]),
                            "vram_total_mb": int(values[1]),
                            "gpu_usage_percent": float(values[2]),
                            "temperature_c": float(values[3]),
                        }
                else:
                    logger.warning(f"nvidia-smi command failed with return code: {result.returncode}")
            else:
                logger.warning("nvidia-smi not found in PATH")
                status["gpu"] = {
                    "status": "unavailable",
                    "reason": "nvidia-smi not found"
                }
        except Exception as e:
            logger.warning(f"Could not get GPU info: {str(e)}")
            status["gpu"] = {
                "status": "error",
                "reason": str(e)
            }
        
        return status
    
    def get_logs(self, lines: int = 100) -> List[Dict[str, str]]:
        """Get recent log lines"""
        try:
            lines = int(lines)
        except Exception:
            lines = 100
        # Return a shallow copy slice to avoid exposing internal deque refs
        return list(self.log_buffer)[-max(0, lines):]
    
    def update_config(self, new_config: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration"""
        # Deep merge configs, filtering out None values
        for category in ["model", "sampling", "performance", "server"]:
            if category in new_config:
                if category not in self.config:
                    self.config[category] = {}
                # Filter out None values before updating
                filtered_category = {k: v for k, v in new_config[category].items() if v is not None}
                self.config[category].update(filtered_category)
        
        # Save config to file for persistence
        config_file = Path("/tmp/llamacpp_config.json")
        with open(config_file, "w") as f:
            json.dump(self.config, f, indent=2)
        
        return self.config
    
    async def get_llamacpp_health(self) -> Dict[str, Any]:
        """Check health of the actual llamacpp service"""
        try:
            async with httpx.AsyncClient() as client:
                # Try to reach llamacpp health endpoint
                response = await client.get(
                    f"http://{'localhost' if not self.use_docker else self.container_name}:8080/health",
                    timeout=5.0
                )
                return {
                    "healthy": response.status_code == 200,
                    "status_code": response.status_code,
                    "response": response.text
                }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }
