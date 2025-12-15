#!/usr/bin/env python3
"""
Quantization worker process.

This worker monitors a Redis queue for quantization jobs
and executes them using various quantization tools.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Optional

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import redis
    from huggingface_hub import snapshot_download
    from modules.quantization import (
        QuantizationJob,
        QuantizationStatus,
        QuantizationFormat,
        GGUFQuantType,
    )
    from modules.quantization.manager import QuantizationManager
    from enhanced_logger import enhanced_logger as logger
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    print("Make sure all dependencies are installed")
    sys.exit(1)


class QuantizationWorker:
    """Worker that processes quantization jobs."""

    def __init__(self):
        self.manager = QuantizationManager()
        self.redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        self.models_dir = Path(os.getenv("MODELS_DIR", "/home/llamacpp/models"))
        self.output_dir = Path(os.getenv("QUANTIZATION_OUTPUT_DIR", "/app/quantization_output"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.redis_client = redis.from_url(self.redis_url)
            self.redis_client.ping()
            logger.info("Connected to Redis", extra={"url": self.redis_url})
        except Exception as e:
            logger.error("Failed to connect to Redis", extra={"error": str(e)})
            raise

    def run(self):
        """Main worker loop."""
        logger.info("Quantization worker started")
        
        while True:
            try:
                # Check for queued jobs
                jobs = self.manager.list_jobs(status=QuantizationStatus.QUEUED.value, limit=1)
                
                if jobs:
                    job = jobs[0]
                    logger.info("Processing quantization job", extra={"job_id": job.id})
                    self.process_job(job)
                else:
                    # No jobs available, sleep for a bit
                    time.sleep(5)
                    
            except KeyboardInterrupt:
                logger.info("Worker shutting down...")
                break
            except Exception as e:
                logger.error("Worker error", extra={"error": str(e)})
                time.sleep(10)

    def process_job(self, job: QuantizationJob):
        """Process a single quantization job."""
        try:
            # Update status to downloading
            self.manager.update_status(
                job.id,
                QuantizationStatus.DOWNLOADING,
                progress=0.0,
                current_step="Downloading source model",
            )

            # Download model if needed
            model_path = self.download_model(job.source_model)
            if not model_path:
                self.manager.update_status(
                    job.id,
                    QuantizationStatus.FAILED,
                    error="Failed to download model",
                )
                return

            # Update to preparing
            self.manager.update_status(
                job.id,
                QuantizationStatus.PREPARING,
                progress=10.0,
                current_step="Preparing for quantization",
            )

            # Process each format
            if QuantizationFormat.GGUF in job.output_formats:
                self.quantize_to_gguf(job, model_path)

            if QuantizationFormat.GPTQ in job.output_formats:
                self.quantize_to_gptq(job, model_path)

            if QuantizationFormat.AWQ in job.output_formats:
                self.quantize_to_awq(job, model_path)

            if QuantizationFormat.ONNX in job.output_formats:
                self.export_to_onnx(job, model_path)

            # Mark as completed
            self.manager.update_status(
                job.id,
                QuantizationStatus.COMPLETED,
                progress=100.0,
                current_step="Quantization completed",
            )
            logger.info("Quantization job completed", extra={"job_id": job.id})

        except Exception as e:
            logger.error("Failed to process job", extra={"job_id": job.id, "error": str(e)})
            self.manager.update_status(
                job.id,
                QuantizationStatus.FAILED,
                error=str(e),
            )

    def download_model(self, model_id: str) -> Optional[Path]:
        """Download model from HuggingFace."""
        try:
            logger.info("Downloading model", extra={"model_id": model_id})
            local_dir = self.models_dir / model_id.replace("/", "_")
            
            if local_dir.exists():
                logger.info("Model already downloaded", extra={"path": str(local_dir)})
                return local_dir

            snapshot_download(
                repo_id=model_id,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )
            logger.info("Model downloaded", extra={"path": str(local_dir)})
            return local_dir
        except Exception as e:
            logger.error("Failed to download model", extra={"error": str(e)})
            return None

    def quantize_to_gguf(self, job: QuantizationJob, model_path: Path):
        """Quantize model to GGUF format."""
        try:
            self.manager.update_status(
                job.id,
                QuantizationStatus.QUANTIZING,
                progress=20.0,
                current_step="Converting to GGUF FP16",
            )

            # Convert to FP16 GGUF first
            fp16_path = self.output_dir / f"{job.source_model.replace('/', '_')}_fp16.gguf"
            
            # Use llama.cpp convert script
            convert_script = Path("/opt/llama.cpp/convert_hf_to_gguf.py")
            if convert_script.exists():
                subprocess.run(
                    [
                        "python3",
                        str(convert_script),
                        str(model_path),
                        "--outfile",
                        str(fp16_path),
                        "--outtype",
                        "f16",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                logger.info("Created FP16 GGUF", extra={"path": str(fp16_path)})
            else:
                logger.warning("Conversion script not found, skipping GGUF conversion")
                return

            # Now quantize to each requested type
            base_progress = 30.0
            progress_per_quant = 60.0 / len(job.gguf_quant_types)
            
            for i, quant_type in enumerate(job.gguf_quant_types):
                output_id = f"{job.id}-gguf-{quant_type.value}"
                output_path = self.output_dir / f"{job.source_model.replace('/', '_')}_{quant_type.value}.gguf"
                
                self.manager.update_status(
                    job.id,
                    QuantizationStatus.QUANTIZING,
                    progress=base_progress + (i * progress_per_quant),
                    current_step=f"Quantizing to {quant_type.value}",
                )

                # Use llama-quantize
                subprocess.run(
                    [
                        "llama-quantize",
                        str(fp16_path),
                        str(output_path),
                        quant_type.value,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )

                file_size = output_path.stat().st_size
                self.manager.update_output(
                    job.id,
                    output_id,
                    file_path=str(output_path),
                    file_size=file_size,
                    status="completed",
                )
                logger.info("Created quantized GGUF", extra={
                    "type": quant_type.value,
                    "path": str(output_path),
                    "size": file_size,
                })

        except Exception as e:
            logger.error("Failed to quantize to GGUF", extra={"error": str(e)})
            raise

    def quantize_to_gptq(self, job: QuantizationJob, model_path: Path):
        """Quantize model to GPTQ format."""
        logger.info("GPTQ quantization not yet implemented")
        # TODO: Implement GPTQ quantization using AutoGPTQ

    def quantize_to_awq(self, job: QuantizationJob, model_path: Path):
        """Quantize model to AWQ format."""
        logger.info("AWQ quantization not yet implemented")
        # TODO: Implement AWQ quantization using AutoAWQ

    def export_to_onnx(self, job: QuantizationJob, model_path: Path):
        """Export model to ONNX format."""
        logger.info("ONNX export not yet implemented")
        # TODO: Implement ONNX export using Optimum


if __name__ == "__main__":
    worker = QuantizationWorker()
    worker.run()
