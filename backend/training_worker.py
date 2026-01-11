"""
LoRA/QLoRA training worker.

Reads a job config (JSON) and performs PEFT training with optional 4-bit
quantization. Publishes status/log messages to Redis to keep the API informed.

Supports resuming from checkpoints when a previous training run was interrupted.
"""

import json
import os
import re
import sys
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import redis
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)

# Force CPU training via environment variable
FORCE_CPU_TRAINING = os.getenv("FORCE_CPU_TRAINING", "false").lower() in ("true", "1", "yes")

# Minimum CUDA capability required (7.0 = Volta architecture)
MIN_CUDA_CAPABILITY = 7.0


def is_gpu_compatible() -> bool:
    """
    Check if the GPU is compatible with the current PyTorch/bitsandbytes installation.
    Returns False if no compatible GPU is available.
    """
    if FORCE_CPU_TRAINING:
        print("[training] FORCE_CPU_TRAINING=true, using CPU mode", flush=True)
        return False
    
    if not torch.cuda.is_available():
        print("[training] No CUDA available, using CPU mode", flush=True)
        return False
    
    try:
        # Check CUDA capability of all available GPUs
        for i in range(torch.cuda.device_count()):
            capability = torch.cuda.get_device_capability(i)
            capability_version = float(f"{capability[0]}.{capability[1]}")
            device_name = torch.cuda.get_device_name(i)
            
            if capability_version < MIN_CUDA_CAPABILITY:
                print(f"[training] GPU {i} ({device_name}) has CUDA capability {capability_version}, "
                      f"but {MIN_CUDA_CAPABILITY}+ is required. Using CPU mode.", flush=True)
                return False
        
        print(f"[training] GPU compatible, using CUDA", flush=True)
        return True
    except Exception as e:
        print(f"[training] Error checking GPU compatibility: {e}. Using CPU mode.", flush=True)
        return False



def load_job_config() -> Optional[Dict[str, Any]]:
    """Load job config from path or return None if in polling mode."""
    path = os.getenv("JOB_CONFIG_PATH") or (sys.argv[1] if len(sys.argv) > 1 else None)
    if not path:
        return None  # Polling mode
    config_path = Path(path)
    if not config_path.exists():
        raise SystemExit(f"Job config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def find_checkpoints(output_dir: str) -> List[Path]:
    """Find all checkpoint directories in the output directory."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return []
    
    checkpoints = []
    for item in output_path.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            # Verify it's a valid checkpoint (has trainer_state.json)
            if (item / "trainer_state.json").exists():
                checkpoints.append(item)
    
    return checkpoints


def get_latest_checkpoint(output_dir: str) -> Optional[Path]:
    """Get the latest checkpoint from the output directory."""
    checkpoints = find_checkpoints(output_dir)
    if not checkpoints:
        return None
    
    # Sort by checkpoint number (checkpoint-100, checkpoint-200, etc.)
    def get_step(path: Path) -> int:
        match = re.search(r"checkpoint-(\d+)", path.name)
        return int(match.group(1)) if match else 0
    
    checkpoints.sort(key=get_step, reverse=True)
    return checkpoints[0]


def get_checkpoint_step(checkpoint_path: Path) -> int:
    """Extract the step number from a checkpoint path."""
    match = re.search(r"checkpoint-(\d+)", checkpoint_path.name)
    return int(match.group(1)) if match else 0


def load_checkpoint_info(checkpoint_path: Path) -> Dict[str, Any]:
    """Load trainer state from a checkpoint."""
    state_file = checkpoint_path / "trainer_state.json"
    if not state_file.exists():
        return {}
    
    with state_file.open("r") as f:
        return json.load(f)


def connect_redis() -> redis.Redis:
    url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    return redis.from_url(url, decode_responses=True)


def publish(redis_client: redis.Redis, channel: str, payload: Dict[str, Any]) -> None:
    redis_client.publish(channel, json.dumps(payload))


def stream_log(redis_client: redis.Redis, job_id: str, message: str) -> None:
    publish(redis_client, f"training:logs:{job_id}", {"timestamp": time.time(), "message": message})


def stream_status(
    redis_client: redis.Redis,
    job_id: str,
    step: int,
    total_steps: int,
    loss: Optional[float],
    status: str,
    adapter_path: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    publish(
        redis_client,
        f"training:status:{job_id}",
        {
            "step": step,
            "total_steps": total_steps,
            "loss": loss,
            "status": status,
            "adapter_path": adapter_path,
            "metrics": metrics,
        },
    )


def load_dataset_for_job(job: Dict[str, Any]) -> Any:
    path = job["dataset_path"]
    fmt = job["dataset_format"]
    ds = load_dataset("json", data_files=path, split="train")

    def format_record(rec):
        if fmt == "alpaca":
            instr = rec.get("instruction", "")
            inp = rec.get("input", "")
            out = rec.get("output", "")
            return f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
        if fmt == "chatml":
            msgs = rec.get("messages", [])
            parts = [f"{m.get('role')}: {m.get('content')}" for m in msgs]
            return "\n".join(parts)
        if fmt == "sharegpt":
            conv = rec.get("conversations", [])
            parts = [f"{c.get('from')}: {c.get('value')}" for c in conv]
            return "\n".join(parts)
        if fmt == "completion":
            return rec.get("text", "")
        return json.dumps(rec)

    ds = ds.map(lambda r: {"text": format_record(r)})
    return ds


def tokenize_function(tokenizer, max_length: int):
    def _tok(batch):
        return tokenizer(
            batch["text"] if "text" in batch else batch,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    return _tok


def build_model(
    job: Dict[str, Any],
    checkpoint_path: Optional[Path] = None,
) -> Tuple[Any, Any, bool]:
    """
    Build or load the model for training.
    
    Args:
        job: Job configuration dict
        checkpoint_path: Path to checkpoint to resume from (if any)
    
    Returns:
        Tuple of (model, tokenizer, is_resumed)
    """
    base_model = job["base_model"]
    qlora = job.get("qlora_config", {}).get("enabled", False)
    lora_cfg = job["lora_config"]
    
    # Check GPU compatibility - force CPU mode if incompatible
    use_gpu = is_gpu_compatible()
    
    # QLoRA requires bitsandbytes which needs compatible GPU
    if qlora and not use_gpu:
        print("[training] QLoRA disabled - bitsandbytes requires compatible GPU. Using standard LoRA on CPU.", flush=True)
        qlora = False
    
    # Load base model
    if qlora:
        # GPU mode with 4-bit quantization
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_4bit=True,
            device_map="auto",
            trust_remote_code=True,
        )
        # Prepare quantized model for training (enables gradients on LoRA params)
        model = prepare_model_for_kbit_training(model)
    elif use_gpu:
        # GPU mode without quantization
        model = AutoModelForCausalLM.from_pretrained(
            base_model, 
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        # CPU mode - load to CPU explicitly, use float32 for compatibility
        print("[training] Loading model on CPU (this may take a while and use significant RAM)...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map=None,  # Don't auto-map, keep on CPU
            torch_dtype=torch.float32,  # Full precision for CPU
            trust_remote_code=True,
            low_cpu_mem_usage=True,  # Load in chunks to reduce peak RAM
        )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    
    is_resumed = False
    
    # Check if we're resuming from a checkpoint
    if checkpoint_path and checkpoint_path.exists():
        # Check if the checkpoint has adapter weights
        adapter_config_path = checkpoint_path / "adapter_config.json"
        if adapter_config_path.exists():
            # Load the PEFT model from checkpoint
            print(f"[resume] Loading adapter from checkpoint: {checkpoint_path}", flush=True)
            model = PeftModel.from_pretrained(
                model,
                str(checkpoint_path),
                is_trainable=True,
            )
            is_resumed = True
        else:
            # No adapter in checkpoint, create fresh LoRA config
            print(f"[resume] Checkpoint has no adapter, creating new LoRA config", flush=True)
            peft_cfg = LoraConfig(
                r=lora_cfg["rank"],
                lora_alpha=lora_cfg["alpha"],
                lora_dropout=lora_cfg["dropout"],
                target_modules=lora_cfg["target_modules"],
                bias=lora_cfg.get("bias", "none"),
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_cfg)
    else:
        # Fresh training - create new LoRA config
        peft_cfg = LoraConfig(
            r=lora_cfg["rank"],
            lora_alpha=lora_cfg["alpha"],
            lora_dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            bias=lora_cfg.get("bias", "none"),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)
    
    return model, tokenizer, is_resumed


def train(job: Dict[str, Any]) -> None:
    redis_client = connect_redis()
    job_id = job["id"]
    
    # Check for resume mode
    resume_from = job.get("resume_from_checkpoint")
    output_dir = job.get("output_dir", f"/app/checkpoints/{job_id}")
    
    # Determine checkpoint to resume from
    checkpoint_path: Optional[Path] = None
    if resume_from == "latest":
        # Find the latest checkpoint automatically
        checkpoint_path = get_latest_checkpoint(output_dir)
        if checkpoint_path:
            stream_log(redis_client, job_id, f"Resuming from latest checkpoint: {checkpoint_path}")
        else:
            stream_log(redis_client, job_id, "No checkpoint found, starting fresh")
    elif resume_from:
        # Explicit checkpoint path provided
        checkpoint_path = Path(resume_from)
        if checkpoint_path.exists():
            stream_log(redis_client, job_id, f"Resuming from checkpoint: {checkpoint_path}")
        else:
            stream_log(redis_client, job_id, f"Checkpoint not found: {checkpoint_path}, starting fresh")
            checkpoint_path = None
    else:
        stream_log(redis_client, job_id, f"Starting job {job_id}")
    
    # Load dataset
    dataset = load_dataset_for_job(job)
    
    # Build model (with checkpoint if resuming)
    model, tokenizer, is_resumed = build_model(job, checkpoint_path)
    
    if is_resumed:
        stream_log(redis_client, job_id, "Model loaded from checkpoint successfully")
    
    train_cfg = job["training_config"]
    tokenized = dataset.map(tokenize_function(tokenizer, train_cfg["max_seq_length"]), batched=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Calculate remaining steps if resuming
    starting_step = 0
    if checkpoint_path:
        checkpoint_info = load_checkpoint_info(checkpoint_path)
        starting_step = checkpoint_info.get("global_step", 0)
        stream_log(redis_client, job_id, f"Resuming from step {starting_step}")

    # Check GPU compatibility for training args
    use_gpu = is_gpu_compatible()
    
    # On CPU, we must disable bf16/fp16 as they require GPU
    use_fp16 = train_cfg.get("fp16", False) if use_gpu else False
    use_bf16 = train_cfg.get("bf16", True) if use_gpu else False
    
    if not use_gpu:
        stream_log(redis_client, job_id, "Running in CPU-only mode (training will be slow)")

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        num_train_epochs=train_cfg["num_epochs"],
        learning_rate=train_cfg["learning_rate"],
        fp16=use_fp16,
        bf16=use_bf16,
        no_cuda=not use_gpu,  # Force CPU mode when GPU incompatible
        max_steps=train_cfg.get("max_steps") or -1,  # -1 means no limit
        warmup_steps=train_cfg.get("warmup_steps", 0),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        logging_steps=1,
        save_steps=train_cfg.get("save_steps", 50),
        save_total_limit=3,  # Keep more checkpoints for resume capability
        report_to=[],
        # Enable checkpoint saving of optimizer and scheduler state
        save_safetensors=True,
        # Gradient checkpointing reduces memory usage by recomputing activations
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True) if use_gpu else False,
        # Reduce memory usage on CPU
        dataloader_pin_memory=use_gpu,
    )


    stop_flag = {"stop": False, "paused": False}

    def listen_for_commands():
        pubsub = redis_client.pubsub()
        pubsub.subscribe(f"training:commands:{job_id}")
        for message in pubsub.listen():
            if message["type"] != "message":
                continue
            raw_data = message.get("data")
            # Handle simple string commands
            if raw_data == "stop":
                stop_flag["stop"] = True
                stop_flag["paused"] = False
                stream_log(redis_client, job_id, "Stop command received; will halt training")
                break
            elif raw_data == "pause":
                stop_flag["paused"] = True
                stream_log(redis_client, job_id, "Pause command received; saving checkpoint and pausing...")
                stream_status(redis_client, job_id, 0, 0, None, "paused")
            elif raw_data == "resume":
                if stop_flag["paused"]:
                    stop_flag["paused"] = False
                    stream_log(redis_client, job_id, "Resume command received; continuing training...")
                    stream_status(redis_client, job_id, 0, 0, None, "training")
            # Handle JSON commands
            try:
                cmd_data = json.loads(raw_data)
                cmd = cmd_data.get("command")
                if cmd == "stop":
                    stop_flag["stop"] = True
                    stop_flag["paused"] = False
                    stream_log(redis_client, job_id, "Stop command received; will halt training")
                    break
                elif cmd == "pause":
                    stop_flag["paused"] = True
                    stream_log(redis_client, job_id, "Pause command received; saving checkpoint and pausing...")
                    stream_status(redis_client, job_id, 0, 0, None, "paused")
                elif cmd == "resume":
                    if stop_flag["paused"]:
                        stop_flag["paused"] = False
                        stream_log(redis_client, job_id, "Resume command received; continuing training...")
                        stream_status(redis_client, job_id, 0, 0, None, "training")
            except (json.JSONDecodeError, TypeError):
                pass
        pubsub.close()

    threading.Thread(target=listen_for_commands, daemon=True).start()

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    # Calculate total steps - use Trainer's calculation
    # The Trainer already accounts for gradient accumulation internally
    num_batches = len(trainer.get_train_dataloader())
    if args.max_steps is not None and args.max_steps > 0:
        total_steps = args.max_steps
    else:
        # Total optimizer steps = ceil(num_batches / grad_accum) * epochs
        import math
        steps_per_epoch = math.ceil(num_batches / args.gradient_accumulation_steps)
        total_steps = steps_per_epoch * int(args.num_train_epochs)
    
    stream_status(redis_client, job_id, starting_step, total_steps, None, "training")

    # Custom callback for progress reporting with resume and pause support
    class ProgressCallback(TrainerCallback):
        def __init__(self, hook_fn, stop_flag_ref, redis_client_ref, job_id_ref):
            self.hook_fn = hook_fn
            self.stop_flag = stop_flag_ref
            self.redis_client = redis_client_ref
            self.job_id = job_id_ref
            self.last_paused_step = None
        
        def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
            if logs is None:
                return
            loss = logs.get("loss")
            if loss is not None:
                self.hook_fn(state.global_step, float(loss))
        
        def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
            # Handle stop
            if self.stop_flag["stop"]:
                control.should_training_stop = True
                control.should_save = True  # Save checkpoint before stopping
                return control
            
            # Handle pause - save checkpoint and wait
            if self.stop_flag["paused"]:
                # Only save once when entering pause
                if self.last_paused_step != state.global_step:
                    self.last_paused_step = state.global_step
                    control.should_save = True  # Save checkpoint when pausing
                    stream_log(self.redis_client, self.job_id, f"Training paused at step {state.global_step}")
                
                # Wait loop while paused
                import time
                while self.stop_flag["paused"] and not self.stop_flag["stop"]:
                    time.sleep(1)  # Check every second
                
                if self.stop_flag["stop"]:
                    control.should_training_stop = True
                else:
                    stream_log(self.redis_client, self.job_id, f"Training resumed from step {state.global_step}")
            
            return control

    def hook(step, loss):
        # Only report if not paused
        if not stop_flag["paused"]:
            stream_status(redis_client, job_id, step, total_steps, loss, "training")
            stream_log(redis_client, job_id, f"step={step} loss={loss:.4f}")

    trainer.add_callback(ProgressCallback(hook, stop_flag, redis_client, job_id))
    
    # Resume from checkpoint if available
    if checkpoint_path and checkpoint_path.exists():
        stream_log(redis_client, job_id, f"Calling trainer.train with resume_from_checkpoint={checkpoint_path}")
        trainer.train(resume_from_checkpoint=str(checkpoint_path))
    else:
        trainer.train()

    # Save final adapter
    adapter_dir = job.get("adapter_dir", job.get("output_dir", f"/app/checkpoints/{job_id}"))
    Path(adapter_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    
    # Also save the final checkpoint info for future reference
    final_info = {
        "job_id": job_id,
        "base_model": job["base_model"],
        "total_steps": trainer.state.global_step,
        "final_epoch": trainer.state.epoch,
        "resumed_from": str(checkpoint_path) if checkpoint_path else None,
    }
    with (Path(adapter_dir) / "training_info.json").open("w") as f:
        json.dump(final_info, f, indent=2)

    # Final metrics
    last_log = next((l for l in reversed(trainer.state.log_history) if "loss" in l), {})
    final_loss = float(last_log.get("loss")) if last_log else None
    metrics = {
        "final_loss": final_loss,
        "total_steps": trainer.state.global_step,
        "resumed_from_step": starting_step if checkpoint_path else 0,
    }

    stream_status(
        redis_client,
        job_id,
        trainer.state.global_step,
        total_steps,
        final_loss,
        "completed",
        adapter_path=adapter_dir,
        metrics=metrics,
    )
    stream_log(redis_client, job_id, f"Training completed. Adapter saved to {adapter_dir}")




def merge_adapter(adapter_path: str, base_model: str, output_dir: str, job_id: str = None) -> str:
    """
    Merge LoRA adapter weights into the base model.
    Returns the path to the merged model.
    
    Args:
        adapter_path: Path to the LoRA adapter directory
        base_model: HuggingFace model ID or path to base model
        output_dir: Directory to save the merged model
        job_id: Optional training job ID for metadata tracking
    """
    from peft import PeftModel
    from datetime import datetime
    
    print(f"[merge] Loading base model: {base_model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    
    print(f"[merge] Loading adapter from: {adapter_path}", flush=True)
    model = PeftModel.from_pretrained(model, adapter_path)
    
    print(f"[merge] Merging adapter into base model...", flush=True)
    merged_model = model.merge_and_unload()
    
    print(f"[merge] Saving merged model to: {output_dir}", flush=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Load training info from adapter directory if available
    training_info = {}
    training_info_path = Path(adapter_path) / "training_info.json"
    if training_info_path.exists():
        try:
            with open(training_info_path, "r") as f:
                training_info = json.load(f)
        except Exception as e:
            print(f"[merge] Could not load training_info.json: {e}", flush=True)
    
    # Load adapter config for LoRA details
    lora_config = {}
    adapter_config_path = Path(adapter_path) / "adapter_config.json"
    if adapter_config_path.exists():
        try:
            with open(adapter_config_path, "r") as f:
                lora_config = json.load(f)
        except Exception as e:
            print(f"[merge] Could not load adapter_config.json: {e}", flush=True)
    
    # Save training metadata for the Models page to discover
    training_metadata = {
        "source": "trained",
        "training_job_id": job_id or training_info.get("job_id"),
        "base_model": base_model,
        "merged_at": datetime.utcnow().isoformat(),
        "adapter_path": adapter_path,
        "final_loss": training_info.get("final_loss"),
        "total_steps": training_info.get("total_steps"),
        "lora_config": {
            "r": lora_config.get("r"),
            "lora_alpha": lora_config.get("lora_alpha"),
            "lora_dropout": lora_config.get("lora_dropout"),
            "target_modules": lora_config.get("target_modules"),
        } if lora_config else None,
    }
    
    metadata_path = Path(output_dir) / "training_metadata.json"
    try:
        with open(metadata_path, "w") as f:
            json.dump(training_metadata, f, indent=2)
        print(f"[merge] Saved training metadata to: {metadata_path}", flush=True)
    except Exception as e:
        print(f"[merge] Warning: Could not save training metadata: {e}", flush=True)
    
    print(f"[merge] Merge completed: {output_dir}", flush=True)
    return output_dir


def run_merge_worker():
    """
    Background worker that listens for merge commands on Redis.
    """
    redis_client = connect_redis()
    pubsub = redis_client.pubsub()
    pubsub.psubscribe("training:commands:*")
    
    print("[merge-worker] Listening for merge commands...", flush=True)
    
    for message in pubsub.listen():
        if message["type"] != "pmessage":
            continue
        
        try:
            raw_data = message.get("data")
            cmd_data = json.loads(raw_data)
            
            if cmd_data.get("command") != "merge":
                continue
            
            channel = message.get("channel", "")
            job_id = channel.split(":")[-1]
            
            adapter_path = cmd_data.get("adapter_path")
            base_model = cmd_data.get("base_model")
            output_dir = cmd_data.get("output_dir")
            
            if not all([adapter_path, base_model, output_dir]):
                print(f"[merge-worker] Invalid merge command: missing fields", flush=True)
                continue
            
            print(f"[merge-worker] Processing merge for job {job_id}", flush=True)
            
            try:
                merged_path = merge_adapter(adapter_path, base_model, output_dir, job_id)
                # Publish success
                publish(
                    redis_client,
                    f"training:status:{job_id}",
                    {
                        "status": "merge_completed",
                        "merged_path": merged_path,
                    },
                )
                stream_log(redis_client, job_id, f"Merge completed: {merged_path}")
            except Exception as exc:
                print(f"[merge-worker] Merge failed: {exc}", flush=True)
                publish(
                    redis_client,
                    f"training:status:{job_id}",
                    {
                        "status": "merge_failed",
                        "error": str(exc),
                    },
                )
                stream_log(redis_client, job_id, f"Merge failed: {exc}")
                
        except (json.JSONDecodeError, TypeError):
            continue


def poll_for_jobs():
    """Poll for new training jobs in the job directory."""
    job_dir = Path(os.getenv("FINETUNE_JOB_DIR", "/data/finetune_jobs"))
    print(f"[training-worker] Starting in polling mode, watching: {job_dir}", flush=True)
    
    # Start merge worker in background thread
    import threading
    merge_thread = threading.Thread(target=run_merge_worker, daemon=True)
    merge_thread.start()
    print("[training-worker] Started merge worker in background thread", flush=True)
    
    processed_jobs = set()
    redis_client = connect_redis()
    
    # Load previously processed jobs from marker files
    if job_dir.exists():
        for marker in job_dir.glob("*/.completed"):
            processed_jobs.add(marker.parent.name)
        print(f"[training-worker] Found {len(processed_jobs)} already completed jobs", flush=True)
    
    while True:
        try:
            # Look for job config files
            if job_dir.exists():
                for config_file in job_dir.glob("*.json"):
                    job_id = config_file.stem
                    if job_id in processed_jobs or job_id == "jobs_index":
                        continue
                    
                    # Check if already completed via marker file
                    job_output_dir = job_dir / job_id
                    completed_marker = job_output_dir / ".completed"
                    if completed_marker.exists():
                        processed_jobs.add(job_id)
                        continue
                    
                    # Check if job is in queued state
                    try:
                        with config_file.open("r") as f:
                            job_config = json.load(f)
                        
                        # Skip if this is not a training config (needs certain fields)
                        if "base_model" not in job_config or "dataset_path" not in job_config:
                            processed_jobs.add(job_id)
                            continue
                        
                        print(f"[training-worker] Found new job: {job_id}", flush=True)
                        processed_jobs.add(job_id)
                        
                        # Run the training
                        try:
                            train(job_config)
                            # Mark as completed
                            job_output_dir.mkdir(parents=True, exist_ok=True)
                            completed_marker.touch()
                        except Exception as exc:
                            print(f"[training-worker] Job {job_id} failed: {exc}", flush=True)
                            publish(
                                redis_client,
                                f"training:status:{job_id}",
                                {"status": "failed", "error": str(exc)},
                            )
                    except (json.JSONDecodeError, IOError) as e:
                        print(f"[training-worker] Error reading {config_file}: {e}", flush=True)
                        processed_jobs.add(job_id)
            
            time.sleep(5)  # Poll every 5 seconds
            
        except KeyboardInterrupt:
            print("[training-worker] Shutting down...", flush=True)
            break
        except Exception as e:
            print(f"[training-worker] Error in polling loop: {e}", flush=True)
            time.sleep(10)


def main():
    # Check if running in merge mode
    if len(sys.argv) > 1 and sys.argv[1] == "--merge-worker":
        run_merge_worker()
        return
    
    job = load_job_config()
    if job is None:
        # No config provided - run in polling mode
        poll_for_jobs()
    else:
        train(job)


if __name__ == "__main__":
    main()
