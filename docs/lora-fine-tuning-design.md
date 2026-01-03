# LoRA Fine-Tuning Design Document for Llama Nexus

## Executive Summary

This document outlines the design and implementation plan for integrating LoRA (Low-Rank Adaptation) fine-tuning capabilities into the Llama Nexus application. The feature will enable users to fine-tune models directly from the UI, prepare datasets, leverage model distillation from larger models, and deploy fine-tuned adapters.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Best Practices Research Summary](#2-best-practices-research-summary)
3. [Dataset Preparation](#3-dataset-preparation)
4. [Model Distillation](#4-model-distillation)
5. [Architecture Design](#5-architecture-design)
6. [Implementation Plan](#6-implementation-plan)
7. [API Specification](#7-api-specification)
8. [Frontend Design](#8-frontend-design)
9. [Docker Integration](#9-docker-integration)
10. [Model Evaluation and Comparison](#10-model-evaluation-and-comparison)
11. [Testing Strategy](#11-testing-strategy)

---

## 1. Overview

### 1.1 What is LoRA?

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that:
- Injects trainable low-rank matrices into transformer layers
- Freezes original pre-trained weights
- Reduces trainable parameters by 10,000x compared to full fine-tuning
- Enables fine-tuning on consumer GPUs (16-24GB VRAM)
- Produces small adapter files (10-100MB) vs full model copies (GB)

### 1.2 Goals

1. **Enable in-app fine-tuning**: Users can fine-tune models without leaving Llama Nexus
2. **Dataset management**: Tools for preparing, validating, and managing training data
3. **Model distillation**: Generate training data from larger teacher models
4. **Adapter management**: Store, version, and deploy LoRA adapters
5. **Export compatibility**: Export fine-tuned models to GGUF for llama.cpp deployment

### 1.3 Scope

- LoRA and QLoRA (4-bit quantized) fine-tuning
- Support for Llama, Mistral, Qwen, and other popular architectures
- Supervised fine-tuning (SFT) for instruction following
- Integration with existing model management system

---

## 2. Best Practices Research Summary

### 2.1 Dataset Size and Quality

| Dataset Size | Use Case | Quality Requirements |
|--------------|----------|---------------------|
| 1,000-5,000 | Task-specific adaptation | Very high quality, curated |
| 5,000-10,000 | General instruction tuning | High quality |
| 10,000-50,000 | Comprehensive fine-tuning | Good quality |
| 50,000-100,000 | Production models | Mixed quality acceptable |

**Key Findings:**
- Quality > Quantity for LoRA fine-tuning
- 1,000-10,000 high-quality examples often outperform 100,000 noisy examples
- Diverse, task-representative data is critical
- Data deduplication improves results

### 2.2 LoRA Hyperparameters

| Parameter | Recommended Range | Notes |
|-----------|------------------|-------|
| Rank (r) | 8-128 | Start with 32-64, increase for complex tasks |
| Alpha (lora_alpha) | 16-128 | Typically 2x rank, or equal to rank |
| Dropout | 0.05-0.1 | Prevents overfitting |
| Target Modules | q_proj, k_proj, v_proj, o_proj | Add gate_proj, up_proj, down_proj for better results |
| Learning Rate | 1e-5 to 2e-4 | Lower than full fine-tuning |
| Batch Size | 4-32 | Depends on VRAM |
| Epochs | 1-5 | Watch for overfitting |

**Hyperparameter Guidelines:**
- **Rank 8-16**: Simple tasks, style transfer
- **Rank 32-64**: Instruction tuning, domain adaptation
- **Rank 128+**: Complex reasoning, multi-task learning

### 2.3 QLoRA Specifics (4-bit Quantized LoRA)

QLoRA enables fine-tuning 7B-70B models on consumer GPUs:

| Model Size | Full Fine-Tune | QLoRA (4-bit) | Savings |
|------------|---------------|---------------|---------|
| 7B | ~56GB | ~6GB | 9x |
| 13B | ~104GB | ~10GB | 10x |
| 70B | ~560GB | ~48GB | 12x |

**QLoRA Configuration:**
- Quantization: NF4 (Normal Float 4-bit)
- Double quantization: Enabled
- Compute dtype: bfloat16 or float16

---

## 3. Dataset Preparation

### 3.1 Supported Dataset Formats

#### 3.1.1 Alpaca Format (Instruction-Input-Output)

```json
{
  "instruction": "Summarize the following text",
  "input": "Long article text here...",
  "output": "Concise summary of the article..."
}
```

Best for: Instruction-following tasks with optional context

#### 3.1.2 ShareGPT Format (Multi-turn Conversations)

```json
{
  "conversations": [
    {"from": "human", "value": "What is machine learning?"},
    {"from": "gpt", "value": "Machine learning is a subset of AI..."},
    {"from": "human", "value": "Can you give an example?"},
    {"from": "gpt", "value": "Sure! Consider email spam filtering..."}
  ]
}
```

Best for: Conversational AI, multi-turn dialogues

#### 3.1.3 ChatML Format (OpenAI-style)

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help you today?"}
  ]
}
```

Best for: Chat models, system prompt customization

#### 3.1.4 Completion Format (Simple Text)

```json
{
  "text": "<s>[INST] What is 2+2? [/INST] 2+2 equals 4.</s>"
}
```

Best for: Pre-formatted text with model-specific tokens

### 3.2 Dataset Validation Rules

The system will validate datasets for:

1. **Format Compliance**
   - Required fields present
   - Correct data types
   - Valid JSON structure

2. **Quality Checks**
   - Minimum text length per field
   - Maximum text length (context window limits)
   - No empty responses
   - Language detection (optional)

3. **Deduplication**
   - Exact duplicates removal
   - Near-duplicate detection (MinHash)
   - Instruction overlap detection

4. **Statistics Generation**
   - Total examples
   - Token distribution
   - Average turns (for conversations)
   - Estimated training time

### 3.3 Dataset Processing Pipeline

```
[Upload] -> [Parse] -> [Validate] -> [Clean] -> [Tokenize] -> [Split] -> [Store]
    |          |          |            |           |            |
    v          v          v            v           v            v
 JSONL/CSV   Detect    Validate    Remove      Count       Train/Val/Test
             Format    Schema      Duplicates  Tokens       (80/10/10)
```

### 3.4 Data Augmentation Options

1. **Instruction Rephrasing**: Use LLM to generate alternative phrasings
2. **Response Expansion**: Add reasoning steps to outputs
3. **Negative Examples**: Generate incorrect responses for contrast learning
4. **Back-translation**: Translate to another language and back

---

## 4. Model Distillation

### 4.1 What is Model Distillation?

Knowledge distillation transfers capabilities from a larger "teacher" model to a smaller "student" model by using the teacher's outputs as training data.

### 4.2 Distillation Strategies

#### 4.2.1 Response Distillation (Recommended)

```
[User Prompts] -> [Teacher Model (GPT-4, Claude, etc.)]
                           |
                           v
                  [High-Quality Responses]
                           |
                           v
               [Student Model (Llama 8B)]
                   via LoRA Fine-Tuning
```

**Process:**
1. Prepare a set of prompts/instructions
2. Generate responses using a powerful teacher model
3. Create instruction-output pairs
4. Fine-tune the student model on this dataset

#### 4.2.2 Chain-of-Thought Distillation

Include reasoning traces from the teacher:

```json
{
  "instruction": "What is 15% of 80?",
  "output": "Let me solve this step by step:\n1. 15% means 15/100 = 0.15\n2. Multiply: 0.15 x 80 = 12\nThe answer is 12."
}
```

#### 4.2.3 Multi-Teacher Distillation

Combine outputs from multiple teachers for robustness:
- Teacher A: Strong at reasoning
- Teacher B: Strong at creativity
- Ensemble: Best of both

### 4.3 Distillation Pipeline in Llama Nexus

```
                    +-------------------+
                    | Prompt Templates  |
                    +-------------------+
                            |
                            v
+------------+      +-------------------+      +------------------+
| Seed Data  | ---> | Prompt Generator  | ---> | Teacher API      |
+------------+      +-------------------+      | (GPT-4, Claude)  |
                                               +------------------+
                                                        |
                                                        v
                                               +------------------+
                                               | Response Filter  |
                                               | (Quality Check)  |
                                               +------------------+
                                                        |
                                                        v
                                               +------------------+
                                               | Dataset Builder  |
                                               +------------------+
                                                        |
                                                        v
                                               +------------------+
                                               | LoRA Fine-Tuning |
                                               +------------------+
```

### 4.4 Distillation Configuration

```python
class DistillationConfig:
    teacher_model: str           # "gpt-4", "claude-3-opus", "deployed:model-id"
    teacher_api_key: str         # API key for external services
    prompts_source: str          # File path or template
    num_examples: int            # Target dataset size
    temperature: float           # Teacher sampling temperature
    max_tokens: int              # Max response length
    include_reasoning: bool      # Request chain-of-thought
    quality_threshold: float     # Min quality score (0-1)
    batch_size: int              # Parallel requests
    rate_limit: int              # Requests per minute
```

### 4.5 Self-Distillation (Using Deployed Models)

Users can also distill from models deployed in Llama Nexus:

```
[Larger Model in Nexus] -> [Generate Training Data] -> [Fine-tune Smaller Model]
        70B                                                    8B
```

Benefits:
- No external API costs
- Full control over outputs
- Custom system prompts

---

## 5. Architecture Design

### 5.1 Component Overview

```
+------------------------------------------------------------------+
|                        Llama Nexus Frontend                       |
+------------------------------------------------------------------+
|  FineTuningPage  |  DatasetManager  |  AdapterManager  |  Monitor |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                         Backend API                               |
+------------------------------------------------------------------+
| /api/v1/finetune/*  |  /api/v1/datasets/*  |  /api/v1/adapters/* |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                    Fine-Tuning Engine                             |
+------------------------------------------------------------------+
| DatasetProcessor | TrainingManager | AdapterStore | ExportManager |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                    Training Container                             |
+------------------------------------------------------------------+
| PyTorch + Transformers + PEFT + bitsandbytes + unsloth           |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|                         GPU Resources                             |
+------------------------------------------------------------------+
```

### 5.2 New Backend Modules

```
backend/
├── modules/
│   └── finetuning/
│       ├── __init__.py
│       ├── config.py           # Training configuration models
│       ├── dataset_processor.py # Dataset validation and processing
│       ├── dataset_formats.py   # Format parsers (Alpaca, ShareGPT, etc.)
│       ├── distillation.py      # Model distillation pipeline
│       ├── training_manager.py  # Training job orchestration
│       ├── adapter_store.py     # LoRA adapter management
│       ├── export_manager.py    # GGUF export functionality
│       └── metrics.py           # Training metrics collection
├── routes/
│   └── finetuning.py           # API endpoints
```

### 5.3 Data Models

```python
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime

class DatasetFormat(str, Enum):
    ALPACA = "alpaca"
    SHAREGPT = "sharegpt"
    CHATML = "chatml"
    COMPLETION = "completion"

class DatasetStatus(str, Enum):
    UPLOADING = "uploading"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"

class TrainingStatus(str, Enum):
    QUEUED = "queued"
    PREPARING = "preparing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class Dataset(BaseModel):
    id: str
    name: str
    description: Optional[str]
    format: DatasetFormat
    status: DatasetStatus
    num_examples: int
    total_tokens: int
    file_path: str
    created_at: datetime
    updated_at: datetime
    validation_errors: Optional[List[str]]
    statistics: Optional[Dict[str, Any]]

class LoRAConfig(BaseModel):
    rank: int = 32                          # LoRA rank
    alpha: int = 64                         # LoRA alpha
    dropout: float = 0.05                   # LoRA dropout
    target_modules: List[str] = [           # Modules to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    bias: str = "none"                      # Bias training

class TrainingConfig(BaseModel):
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    max_steps: Optional[int] = None
    warmup_steps: int = 100
    weight_decay: float = 0.01
    optimizer: str = "adamw_8bit"
    lr_scheduler: str = "cosine"
    max_seq_length: int = 2048
    fp16: bool = False
    bf16: bool = True

class QLoRAConfig(BaseModel):
    enabled: bool = True
    bits: int = 4                           # Quantization bits
    quant_type: str = "nf4"                 # nf4 or fp4
    double_quant: bool = True               # Double quantization
    compute_dtype: str = "bfloat16"

class FineTuningJob(BaseModel):
    id: str
    name: str
    description: Optional[str]
    base_model: str                         # Model repo ID or local path
    dataset_id: str
    lora_config: LoRAConfig
    training_config: TrainingConfig
    qlora_config: QLoRAConfig
    status: TrainingStatus
    progress: float                         # 0-100
    current_step: int
    total_steps: int
    current_loss: Optional[float]
    best_loss: Optional[float]
    metrics: Dict[str, Any]
    adapter_path: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error: Optional[str]

class LoRAAdapter(BaseModel):
    id: str
    name: str
    description: Optional[str]
    base_model: str
    training_job_id: str
    lora_config: LoRAConfig
    file_path: str
    file_size: int
    metrics: Dict[str, Any]
    created_at: datetime
    is_merged: bool = False                 # If merged with base model
    gguf_path: Optional[str]                # Exported GGUF path
```

### 5.4 Training Container Architecture

The training workload runs in a separate Docker container with GPU access:

```dockerfile
# Dockerfile.training
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3.10 python3-pip git

# Install PyTorch with CUDA support
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu121

# Install fine-tuning dependencies
RUN pip3 install \
    transformers>=4.40.0 \
    peft>=0.10.0 \
    bitsandbytes>=0.43.0 \
    accelerate>=0.28.0 \
    datasets>=2.18.0 \
    trl>=0.8.0 \
    unsloth[cu121]>=2024.4

# Install llama.cpp for GGUF export
RUN git clone https://github.com/ggerganov/llama.cpp /opt/llama.cpp && \
    cd /opt/llama.cpp && \
    make LLAMA_CUDA=1

WORKDIR /app
COPY training_worker.py /app/

CMD ["python3", "training_worker.py"]
```

---

## 6. Implementation Plan

### Phase 1: Dataset Management (Week 1-2)

#### 6.1.1 Backend Tasks
- [ ] Create `backend/modules/finetuning/` module structure
- [ ] Implement dataset format parsers (Alpaca, ShareGPT, ChatML)
- [ ] Build dataset validation and statistics generation
- [ ] Create dataset storage with SQLite metadata
- [ ] Implement train/validation split functionality
- [ ] Add dataset CRUD API endpoints

#### 6.1.2 Frontend Tasks
- [ ] Create `DatasetManagementPage.tsx`
- [ ] Build dataset upload component (drag-drop, file select)
- [ ] Create dataset preview/viewer component
- [ ] Add dataset statistics visualization
- [ ] Implement validation error display

### Phase 2: Model Distillation (Week 2-3)

#### 6.2.1 Backend Tasks
- [ ] Implement distillation pipeline manager
- [ ] Add support for external teacher APIs (OpenAI, Anthropic)
- [ ] Integrate with deployed Llama Nexus models as teachers
- [ ] Build prompt generation and batching system
- [ ] Add response quality filtering
- [ ] Create distillation job tracking

#### 6.2.2 Frontend Tasks
- [ ] Create `DistillationPage.tsx`
- [ ] Build teacher model configuration UI
- [ ] Add prompt template editor
- [ ] Create distillation progress monitor
- [ ] Implement generated data review interface

### Phase 3: Training Infrastructure (Week 3-4)

#### 6.3.1 Docker Setup
- [ ] Create training container Dockerfile
- [ ] Add training service to docker-compose.yml
- [ ] Configure GPU passthrough
- [ ] Set up volume mounts for models and datasets
- [ ] Implement container health checks

#### 6.3.2 Backend Tasks
- [ ] Build training job orchestrator
- [ ] Implement training worker communication (Redis/WebSocket)
- [ ] Create checkpoint management system
- [ ] Add training metrics streaming
- [ ] Implement job cancellation and resumption
- [ ] Build VRAM estimation for training

### Phase 4: Training UI and Monitoring (Week 4-5)

#### 6.4.1 Frontend Tasks
- [ ] Create `FineTuningPage.tsx` (main page)
- [ ] Build training configuration wizard
- [ ] Add hyperparameter preset selector
- [ ] Create real-time training monitor
- [ ] Implement loss curve visualization
- [ ] Add training logs viewer

### Phase 5: Adapter Management and Export (Week 5-6)

#### 6.5.1 Backend Tasks
- [ ] Implement LoRA adapter storage
- [ ] Build adapter merging functionality
- [ ] Create GGUF export pipeline
- [ ] Add adapter deployment integration
- [ ] Implement adapter A/B testing support

#### 6.5.2 Frontend Tasks
- [ ] Create `AdapterManagerPage.tsx`
- [ ] Build adapter browser/selector
- [ ] Add merge and export UI
- [ ] Create deployment integration
- [ ] Implement adapter comparison view

### Phase 6: Model Evaluation (Week 6-7)

#### 6.6.1 Backend Tasks
- [ ] Implement comparison session management
- [ ] Build benchmark runner with standard suites (MMLU, HellaSwag, etc.)
- [ ] Create custom benchmark support
- [ ] Implement LLM-as-judge evaluation
- [ ] Build performance comparison metrics
- [ ] Add evaluation results storage

#### 6.6.2 Frontend Tasks
- [ ] Create `ModelEvaluationPage.tsx`
- [ ] Build `ComparisonPage.tsx` for side-by-side comparison
- [ ] Implement blind comparison mode
- [ ] Create human rating interface
- [ ] Build benchmark results visualization
- [ ] Add evaluation summary dashboard

### Phase 7: Testing and Polish (Week 7-8)

- [ ] End-to-end testing with various model sizes
- [ ] Performance optimization
- [ ] Documentation
- [ ] Error handling improvements
- [ ] UI/UX refinements

---

## 7. API Specification

### 7.1 Dataset Endpoints

```
# Dataset Management
POST   /api/v1/finetune/datasets                    # Upload dataset
GET    /api/v1/finetune/datasets                    # List datasets
GET    /api/v1/finetune/datasets/{id}               # Get dataset details
DELETE /api/v1/finetune/datasets/{id}               # Delete dataset
PUT    /api/v1/finetune/datasets/{id}               # Update dataset metadata
GET    /api/v1/finetune/datasets/{id}/preview       # Preview samples
GET    /api/v1/finetune/datasets/{id}/statistics    # Get statistics
POST   /api/v1/finetune/datasets/{id}/validate      # Re-validate dataset
POST   /api/v1/finetune/datasets/{id}/split         # Create train/val split
POST   /api/v1/finetune/datasets/convert            # Convert between formats
```

### 7.2 Distillation Endpoints

```
# Model Distillation
POST   /api/v1/finetune/distillation/jobs           # Start distillation job
GET    /api/v1/finetune/distillation/jobs           # List distillation jobs
GET    /api/v1/finetune/distillation/jobs/{id}      # Get job status
DELETE /api/v1/finetune/distillation/jobs/{id}      # Cancel job
GET    /api/v1/finetune/distillation/jobs/{id}/preview  # Preview generated data
POST   /api/v1/finetune/distillation/jobs/{id}/approve  # Approve and create dataset
GET    /api/v1/finetune/distillation/templates      # List prompt templates
POST   /api/v1/finetune/distillation/templates      # Create prompt template
```

### 7.3 Training Endpoints

```
# Fine-Tuning Jobs
POST   /api/v1/finetune/jobs                        # Create training job
GET    /api/v1/finetune/jobs                        # List training jobs
GET    /api/v1/finetune/jobs/{id}                   # Get job details
DELETE /api/v1/finetune/jobs/{id}                   # Cancel job
POST   /api/v1/finetune/jobs/{id}/resume            # Resume training
GET    /api/v1/finetune/jobs/{id}/logs              # Stream training logs
GET    /api/v1/finetune/jobs/{id}/metrics           # Get training metrics
GET    /api/v1/finetune/jobs/{id}/checkpoints       # List checkpoints

# Presets
GET    /api/v1/finetune/presets                     # List hyperparameter presets
POST   /api/v1/finetune/presets                     # Create custom preset
GET    /api/v1/finetune/presets/{id}                # Get preset details

# Estimation
POST   /api/v1/finetune/estimate                    # Estimate VRAM/time
```

### 7.4 Adapter Endpoints

```
# Adapter Management
GET    /api/v1/finetune/adapters                    # List adapters
GET    /api/v1/finetune/adapters/{id}               # Get adapter details
DELETE /api/v1/finetune/adapters/{id}               # Delete adapter
POST   /api/v1/finetune/adapters/{id}/merge         # Merge with base model
POST   /api/v1/finetune/adapters/{id}/export        # Export to GGUF
POST   /api/v1/finetune/adapters/{id}/deploy        # Deploy adapter
GET    /api/v1/finetune/adapters/{id}/compare       # Compare with base model
```

### 7.5 Example API Requests

#### Create Training Job

```json
POST /api/v1/finetune/jobs
{
  "name": "customer-support-llama-8b",
  "description": "Fine-tuned for customer support responses",
  "base_model": "meta-llama/Meta-Llama-3-8B",
  "dataset_id": "ds_abc123",
  "lora_config": {
    "rank": 64,
    "alpha": 128,
    "dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  },
  "training_config": {
    "learning_rate": 2e-4,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_epochs": 3,
    "max_seq_length": 2048
  },
  "qlora_config": {
    "enabled": true,
    "bits": 4,
    "quant_type": "nf4"
  }
}
```

#### Start Distillation Job

```json
POST /api/v1/finetune/distillation/jobs
{
  "name": "gpt4-distillation-coding",
  "teacher_config": {
    "model": "gpt-4-turbo",
    "api_key": "sk-...",
    "temperature": 0.7,
    "max_tokens": 2048
  },
  "prompts_config": {
    "template_id": "coding-assistant",
    "num_examples": 5000,
    "include_reasoning": true
  },
  "output_format": "sharegpt"
}
```

---

## 8. Frontend Design

### 8.1 Page Structure

```
/finetune
  /datasets              # Dataset management
    /upload              # Upload new dataset
    /:id                 # Dataset details
  /distillation          # Model distillation
    /new                 # New distillation job
    /:id                 # Distillation job details
  /training              # Training jobs
    /new                 # New training job (wizard)
    /:id                 # Training job details/monitor
  /adapters              # Adapter management
    /:id                 # Adapter details
```

### 8.2 Key Components

```
frontend/src/
├── pages/
│   ├── FineTuningPage.tsx            # Main dashboard
│   ├── DatasetManagementPage.tsx     # Dataset CRUD
│   ├── DatasetUploadPage.tsx         # Upload wizard
│   ├── DistillationPage.tsx          # Distillation jobs
│   ├── TrainingWizardPage.tsx        # Training configuration
│   ├── TrainingMonitorPage.tsx       # Live training monitor
│   └── AdapterManagementPage.tsx     # Adapter browser
├── components/
│   └── finetuning/
│       ├── DatasetPreview.tsx        # Sample viewer
│       ├── DatasetStatistics.tsx     # Charts and stats
│       ├── FormatSelector.tsx        # Dataset format picker
│       ├── LoRAConfigEditor.tsx      # LoRA hyperparameter UI
│       ├── TrainingConfigEditor.tsx  # Training hyperparameter UI
│       ├── PresetSelector.tsx        # Hyperparameter presets
│       ├── LossChart.tsx             # Training loss visualization
│       ├── MetricsPanel.tsx          # Training metrics display
│       ├── VRAMEstimator.tsx         # VRAM requirement display
│       ├── AdapterCard.tsx           # Adapter summary card
│       ├── ExportDialog.tsx          # GGUF export options
│       └── TeacherModelConfig.tsx    # Distillation teacher setup
```

### 8.3 Training Configuration Wizard Flow

```
Step 1: Select Base Model
  - Browse available models
  - Filter by size, architecture
  - Show VRAM requirements

Step 2: Select Dataset
  - Choose from uploaded datasets
  - Preview samples
  - Verify token counts

Step 3: Configure LoRA
  - Preset selector (Quick Start, Balanced, High Quality)
  - Advanced: rank, alpha, target modules
  - VRAM estimation update

Step 4: Configure Training
  - Preset selector
  - Advanced: learning rate, batch size, epochs
  - Time estimation

Step 5: Review and Start
  - Summary of all settings
  - Estimated resources
  - Start training button
```

### 8.4 Training Monitor UI

```
+------------------------------------------------------------------+
| Training Job: customer-support-llama-8b                    [Stop] |
+------------------------------------------------------------------+
| Status: Training | Progress: 67% | Step: 1,340 / 2,000           |
+------------------------------------------------------------------+
|                                                                   |
|  Loss Curve                      |  Current Metrics              |
|  [Interactive Chart]             |  Loss: 0.342                  |
|                                  |  LR: 1.8e-4                   |
|                                  |  Speed: 2.1 steps/s           |
|                                  |  ETA: 5m 23s                  |
|                                  |                               |
+------------------------------------------------------------------+
| Training Logs                                            [Export] |
+------------------------------------------------------------------+
| [2024-03-15 14:23:45] Step 1340/2000 | Loss: 0.342 | LR: 1.8e-4  |
| [2024-03-15 14:23:43] Step 1339/2000 | Loss: 0.348 | LR: 1.8e-4  |
| ...                                                               |
+------------------------------------------------------------------+
| Checkpoints                                                       |
| [x] checkpoint-500  | Loss: 0.89  | [Load] [Export]              |
| [x] checkpoint-1000 | Loss: 0.52  | [Load] [Export]              |
+------------------------------------------------------------------+
```

---

## 9. Docker Integration

### 9.1 Docker Compose Addition

```yaml
# docker-compose.yml additions

services:
  # ... existing services ...

  training:
    build:
      context: ./backend
      dockerfile: Dockerfile.training
    volumes:
      - ./models:/home/llamacpp/models
      - ./datasets:/app/datasets
      - ./adapters:/app/adapters
      - ./checkpoints:/app/checkpoints
      - huggingface_cache:/root/.cache/huggingface
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - HF_HOME=/root/.cache/huggingface
      - REDIS_URL=redis://redis:6379
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    depends_on:
      - redis
    networks:
      - llama-network

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    networks:
      - llama-network

volumes:
  huggingface_cache:
  redis_data:
```

### 9.2 Training Worker Communication

The training worker communicates with the main backend via Redis:

```python
# Training status updates
CHANNEL: "training:status:{job_id}"
MESSAGE: {
    "step": 1340,
    "total_steps": 2000,
    "loss": 0.342,
    "learning_rate": 1.8e-4,
    "tokens_per_second": 1250,
    "eta_seconds": 323
}

# Training logs
CHANNEL: "training:logs:{job_id}"
MESSAGE: {
    "timestamp": "2024-03-15T14:23:45Z",
    "level": "INFO",
    "message": "Step 1340/2000 | Loss: 0.342"
}

# Job commands
CHANNEL: "training:commands:{job_id}"
MESSAGE: {
    "command": "stop" | "pause" | "resume" | "checkpoint"
}
```

---

## 10. Model Evaluation and Comparison

A critical part of the fine-tuning workflow is evaluating whether the fine-tuned model actually improves upon the base model for your specific use case. This section covers comprehensive evaluation tools.

### 10.1 Evaluation Overview

```
+------------------------------------------------------------------+
|                     Evaluation Framework                          |
+------------------------------------------------------------------+
|                                                                   |
|  +------------------+     +------------------+                    |
|  |   Base Model     |     | Fine-Tuned Model |                    |
|  |   (Original)     |     |  (with LoRA)     |                    |
|  +--------+---------+     +--------+---------+                    |
|           |                        |                              |
|           +------------------------+                              |
|                        |                                          |
|           +------------v------------+                             |
|           |   Evaluation Engine     |                             |
|           +------------+------------+                             |
|                        |                                          |
|     +------------------+------------------+                       |
|     |                  |                  |                       |
|     v                  v                  v                       |
| +---------+     +-----------+     +-------------+                 |
| | Prompt  |     | Benchmark |     | Human       |                 |
| | Compare |     | Suite     |     | Evaluation  |                 |
| +---------+     +-----------+     +-------------+                 |
|                                                                   |
+------------------------------------------------------------------+
```

### 10.2 Side-by-Side Response Comparison

#### 10.2.1 Comparison Modes

1. **Single Prompt Comparison**: Enter one prompt, see both model responses
2. **Batch Comparison**: Run a set of test prompts through both models
3. **Conversation Comparison**: Compare multi-turn conversation handling
4. **Blind Comparison**: Randomize model order for unbiased human evaluation

#### 10.2.2 Comparison Data Model

```python
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime

class ComparisonType(str, Enum):
    SINGLE = "single"
    BATCH = "batch"
    CONVERSATION = "conversation"
    BLIND = "blind"

class ModelResponse(BaseModel):
    model_id: str
    model_name: str
    is_finetuned: bool
    adapter_id: Optional[str]
    response: str
    tokens_generated: int
    generation_time_ms: float
    tokens_per_second: float
    finish_reason: str

class PromptComparison(BaseModel):
    id: str
    prompt: str
    system_prompt: Optional[str]
    base_response: ModelResponse
    finetuned_response: ModelResponse
    created_at: datetime
    # Human evaluation fields
    preferred_model: Optional[str]  # "base", "finetuned", "tie"
    preference_reason: Optional[str]
    ratings: Optional[Dict[str, Dict[str, int]]]  # model -> criterion -> score

class ComparisonSession(BaseModel):
    id: str
    name: str
    description: Optional[str]
    comparison_type: ComparisonType
    base_model: str
    adapter_id: str
    test_prompts: List[str]
    comparisons: List[PromptComparison]
    sampling_params: Dict[str, Any]
    created_at: datetime
    completed_at: Optional[datetime]
    summary: Optional[Dict[str, Any]]
```

#### 10.2.3 Evaluation Criteria

The comparison UI allows rating responses on multiple criteria:

| Criterion | Description | Scale |
|-----------|-------------|-------|
| **Relevance** | How well does the response address the prompt? | 1-5 |
| **Accuracy** | Is the information correct and factual? | 1-5 |
| **Coherence** | Is the response logically structured? | 1-5 |
| **Completeness** | Does it fully answer the question? | 1-5 |
| **Style** | Does it match the desired tone/format? | 1-5 |
| **Helpfulness** | How useful is the response overall? | 1-5 |

### 10.3 Automated Benchmarking

#### 10.3.1 Built-in Benchmark Suites

| Benchmark | Description | Metrics |
|-----------|-------------|---------|
| **MMLU** | Multi-task language understanding | Accuracy % |
| **HellaSwag** | Commonsense reasoning | Accuracy % |
| **ARC** | Science question answering | Accuracy % |
| **TruthfulQA** | Truthfulness evaluation | MC1/MC2 scores |
| **Winogrande** | Pronoun resolution | Accuracy % |
| **GSM8K** | Grade school math | Accuracy % |
| **HumanEval** | Code generation | Pass@1, Pass@10 |
| **Custom** | User-defined test sets | Configurable |

#### 10.3.2 Benchmark Configuration

```python
class BenchmarkConfig(BaseModel):
    benchmark_name: str
    num_samples: Optional[int] = None      # None = full benchmark
    num_few_shot: int = 5                   # Few-shot examples
    batch_size: int = 8
    max_tokens: int = 256
    temperature: float = 0.0                # Deterministic for benchmarks
    include_reasoning: bool = False         # Chain-of-thought

class BenchmarkJob(BaseModel):
    id: str
    name: str
    base_model: str
    adapter_id: Optional[str]               # None = evaluate base only
    benchmarks: List[BenchmarkConfig]
    status: str                             # queued, running, completed, failed
    progress: float
    results: Optional[Dict[str, Any]]
    created_at: datetime
    completed_at: Optional[datetime]

class BenchmarkResult(BaseModel):
    benchmark_name: str
    base_model_score: float
    finetuned_score: Optional[float]
    improvement: Optional[float]            # Percentage improvement
    num_samples: int
    detailed_results: Dict[str, Any]        # Per-category breakdowns
```

#### 10.3.3 Custom Benchmark Creation

Users can create domain-specific benchmarks:

```json
{
  "name": "customer-support-eval",
  "description": "Evaluate customer support response quality",
  "test_cases": [
    {
      "id": "cs_001",
      "prompt": "Customer: My order hasn't arrived after 2 weeks. Order #12345",
      "expected_elements": ["apologize", "order lookup", "resolution offer"],
      "evaluation_type": "element_check"
    },
    {
      "id": "cs_002", 
      "prompt": "How do I reset my password?",
      "reference_answer": "To reset your password, go to Settings > Security > Reset Password...",
      "evaluation_type": "similarity"
    },
    {
      "id": "cs_003",
      "prompt": "I want to cancel my subscription",
      "forbidden_elements": ["sorry to see you go", "are you sure"],
      "required_tone": "helpful_neutral",
      "evaluation_type": "constraint_check"
    }
  ],
  "scoring": {
    "element_check": {"weight": 0.4},
    "similarity": {"weight": 0.3, "threshold": 0.8},
    "constraint_check": {"weight": 0.3}
  }
}
```

### 10.4 LLM-as-Judge Evaluation

Use a powerful LLM to automatically judge response quality.

#### 10.4.1 Judge Configuration

```python
class JudgeConfig(BaseModel):
    judge_model: str                        # "gpt-4", "claude-3-opus", or deployed model
    judge_api_key: Optional[str]
    evaluation_criteria: List[str]
    scoring_rubric: Dict[str, str]
    num_evaluations: int = 1                # Multiple judgments for reliability
    temperature: float = 0.0

class JudgeEvaluation(BaseModel):
    prompt: str
    base_response: str
    finetuned_response: str
    judge_verdict: str                      # "base", "finetuned", "tie"
    judge_reasoning: str
    criteria_scores: Dict[str, Dict[str, int]]  # model -> criterion -> score
    confidence: float
```

#### 10.4.2 Judge Prompt Template

```
You are an expert evaluator comparing two AI assistant responses.

## Evaluation Criteria
{criteria_list}

## Prompt
{user_prompt}

## Response A
{response_a}

## Response B
{response_b}

## Instructions
1. Evaluate each response against the criteria above
2. Provide scores (1-5) for each criterion for each response
3. Determine which response is better overall, or if they are tied
4. Explain your reasoning

## Output Format
```json
{
  "response_a_scores": {"relevance": X, "accuracy": X, ...},
  "response_b_scores": {"relevance": X, "accuracy": X, ...},
  "verdict": "A" | "B" | "tie",
  "reasoning": "..."
}
```
```

### 10.5 Performance Comparison

#### 10.5.1 Inference Metrics

Compare inference performance between base and fine-tuned models:

| Metric | Description |
|--------|-------------|
| **Tokens/Second** | Generation speed |
| **Time to First Token** | Latency before streaming starts |
| **Total Generation Time** | End-to-end response time |
| **Memory Usage** | VRAM consumption |
| **Cache Efficiency** | KV cache utilization |

#### 10.5.2 Performance Test Configuration

```python
class PerformanceTest(BaseModel):
    test_prompts: List[str]                 # Various length prompts
    output_lengths: List[int]               # Target generation lengths
    batch_sizes: List[int]                  # Concurrent request counts
    num_iterations: int = 10                # Runs per configuration
    warmup_iterations: int = 2

class PerformanceResults(BaseModel):
    base_model_metrics: Dict[str, float]
    finetuned_metrics: Dict[str, float]
    comparison: Dict[str, float]            # % difference
```

### 10.6 Evaluation API Endpoints

```
# Comparison Sessions
POST   /api/v1/finetune/eval/compare                    # Start comparison session
GET    /api/v1/finetune/eval/compare                    # List comparison sessions
GET    /api/v1/finetune/eval/compare/{id}               # Get session details
DELETE /api/v1/finetune/eval/compare/{id}               # Delete session
POST   /api/v1/finetune/eval/compare/{id}/prompts       # Add prompts to session
POST   /api/v1/finetune/eval/compare/{id}/run           # Run comparison
POST   /api/v1/finetune/eval/compare/{id}/rate          # Submit human rating
GET    /api/v1/finetune/eval/compare/{id}/export        # Export results

# Benchmarking
POST   /api/v1/finetune/eval/benchmark                  # Start benchmark job
GET    /api/v1/finetune/eval/benchmark                  # List benchmark jobs
GET    /api/v1/finetune/eval/benchmark/{id}             # Get job status/results
DELETE /api/v1/finetune/eval/benchmark/{id}             # Cancel benchmark
GET    /api/v1/finetune/eval/benchmark/suites           # List available benchmarks
POST   /api/v1/finetune/eval/benchmark/custom           # Create custom benchmark

# LLM-as-Judge
POST   /api/v1/finetune/eval/judge                      # Run LLM judge evaluation
GET    /api/v1/finetune/eval/judge/{id}                 # Get judge results
GET    /api/v1/finetune/eval/judge/config               # Get judge configuration

# Performance Testing
POST   /api/v1/finetune/eval/performance                # Run performance test
GET    /api/v1/finetune/eval/performance/{id}           # Get performance results
```

### 10.7 Evaluation Frontend

#### 10.7.1 New Pages

```
frontend/src/pages/
├── ModelEvaluationPage.tsx           # Main evaluation dashboard
├── ComparisonPage.tsx                # Side-by-side comparison
├── BenchmarkPage.tsx                 # Benchmark runner (extend existing)
├── EvaluationResultsPage.tsx         # Detailed results view
```

#### 10.7.2 New Components

```
frontend/src/components/evaluation/
├── ResponseCompare.tsx               # Side-by-side response viewer
├── BlindCompare.tsx                  # Randomized blind comparison
├── RatingPanel.tsx                   # Human rating interface
├── BenchmarkSelector.tsx             # Choose benchmarks to run
├── BenchmarkResults.tsx              # Display benchmark scores
├── ImprovementChart.tsx              # Visualize base vs finetuned
├── PerformanceComparison.tsx         # Speed/memory comparison
├── JudgeConfig.tsx                   # Configure LLM judge
├── EvalSummary.tsx                   # Overall evaluation summary
```

#### 10.7.3 Comparison UI Mockup

```
+------------------------------------------------------------------+
| Model Comparison: customer-support-llama-8b                       |
+------------------------------------------------------------------+
| Base Model: meta-llama/Llama-3-8B                                 |
| Fine-tuned: customer-support-llama-8b (adapter)                   |
+------------------------------------------------------------------+
|                                                                   |
| System Prompt: [You are a helpful customer support agent...]      |
|                                                                   |
| User Prompt:                                                      |
| +--------------------------------------------------------------+ |
| | My package hasn't arrived and it's been 10 days. Order #789  | |
| +--------------------------------------------------------------+ |
|                                                        [Compare]  |
+------------------------------------------------------------------+
|                                                                   |
| +---------------------------+  +-------------------------------+  |
| | BASE MODEL               |  | FINE-TUNED MODEL              |  |
| +---------------------------+  +-------------------------------+  |
| | I understand you're      |  | I sincerely apologize for    |  |
| | waiting for your package.|  | the delay with your order    |  |
| | Package delivery times   |  | #789. Let me look into this  |  |
| | can vary...              |  | immediately for you...       |  |
| |                          |  |                               |  |
| | 127 tokens | 2.3s        |  | 156 tokens | 2.8s            |  |
| | 55.2 tok/s               |  | 55.7 tok/s                   |  |
| +---------------------------+  +-------------------------------+  |
|                                                                   |
| Rate the responses:                                               |
| +---------------------------+  +-------------------------------+  |
| | Relevance:   [****-]     |  | Relevance:   [*****]         |  |
| | Accuracy:    [***--]     |  | Accuracy:    [****-]         |  |
| | Helpfulness: [***--]     |  | Helpfulness: [*****]         |  |
| | Style:       [**---]     |  | Style:       [*****]         |  |
| +---------------------------+  +-------------------------------+  |
|                                                                   |
| Which response is better?  [ Base ] [ Tie ] [Fine-tuned]         |
| Reason: [                                                    ]    |
|                                                        [Submit]   |
+------------------------------------------------------------------+
```

#### 10.7.4 Benchmark Results UI Mockup

```
+------------------------------------------------------------------+
| Benchmark Results: customer-support-llama-8b                      |
+------------------------------------------------------------------+
|                                                                   |
| Overall Summary                                                   |
| +--------------------------------------------------------------+ |
| |  Base Model Average: 62.4%    Fine-tuned Average: 71.8%      | |
| |  Improvement: +9.4%                                          | |
| +--------------------------------------------------------------+ |
|                                                                   |
| Detailed Results                                                  |
| +------------------+------------+-------------+------------+      |
| | Benchmark        | Base Model | Fine-tuned  | Change     |      |
| +------------------+------------+-------------+------------+      |
| | MMLU             | 58.2%      | 59.1%       | +0.9%      |      |
| | HellaSwag        | 71.5%      | 72.0%       | +0.5%      |      |
| | Custom: Support  | 45.0%      | 78.5%       | +33.5%  ^^ |      |
| | Custom: Tone     | 52.0%      | 89.0%       | +37.0%  ^^ |      |
| | TruthfulQA       | 61.2%      | 60.8%       | -0.4%      |      |
| +------------------+------------+-------------+------------+      |
|                                                                   |
| [Chart: Bar graph comparing scores]                               |
|                                                                   |
| Insights:                                                         |
| - Significant improvement on domain-specific tasks (+33-37%)      |
| - General capabilities maintained (< 1% change)                   |
| - No regression on truthfulness                                   |
|                                                                   |
+------------------------------------------------------------------+
```

### 10.8 Evaluation Best Practices

#### 10.8.1 Test Set Guidelines

1. **Hold out test data**: Never evaluate on training data
2. **Diverse prompts**: Cover different aspects of your use case
3. **Edge cases**: Include challenging or unusual inputs
4. **Representative**: Match production query distribution
5. **Sufficient size**: At least 100-500 test cases for statistical significance

#### 10.8.2 Avoiding Common Pitfalls

| Pitfall | How to Avoid |
|---------|--------------|
| **Overfitting evaluation** | Use held-out test sets, not validation data |
| **Cherry-picking** | Run full benchmark suites, report all metrics |
| **Single metric focus** | Evaluate multiple dimensions (quality, safety, speed) |
| **Ignoring regressions** | Check general capability benchmarks too |
| **Biased human eval** | Use blind comparison mode |

#### 10.8.3 Evaluation Workflow

```
1. Before Fine-Tuning
   - Establish baseline metrics on all benchmarks
   - Create domain-specific test set (held out from training)
   - Run base model on custom prompts for comparison

2. After Training
   - Run same benchmarks on fine-tuned model
   - Compare side-by-side on representative prompts
   - Check for regressions on general tasks
   - Conduct blind human evaluation if needed

3. Iterate
   - If results unsatisfactory, adjust training (data, hyperparams)
   - Re-evaluate after changes
   - Document what worked and what didn't

4. Before Deployment
   - Final evaluation on full test suite
   - A/B test in production (if possible)
   - Monitor post-deployment metrics
```

### 10.9 Evaluation Data Models (Additional)

```python
class EvaluationSummary(BaseModel):
    """Overall summary of model evaluation."""
    adapter_id: str
    adapter_name: str
    base_model: str
    evaluation_date: datetime
    
    # Benchmark results
    benchmarks_run: List[str]
    base_model_avg_score: float
    finetuned_avg_score: float
    improvement_percentage: float
    
    # Human evaluation
    human_comparisons: int
    human_preference_base: float        # % preferring base
    human_preference_finetuned: float   # % preferring fine-tuned
    human_preference_tie: float         # % tie
    
    # Performance
    speed_comparison: float             # % faster/slower
    memory_comparison: float            # % more/less memory
    
    # Verdict
    recommendation: str                 # "deploy", "needs_work", "regression"
    key_improvements: List[str]
    key_regressions: List[str]
    notes: Optional[str]
```

---

## 11. Testing Strategy

### 11.1 Unit Tests

- Dataset format parsers
- Validation logic
- Configuration models
- Adapter management
- Evaluation scoring logic

### 11.2 Integration Tests

- Dataset upload flow
- Training job creation
- Distillation pipeline
- Export functionality
- Benchmark execution
- Comparison sessions

### 11.3 End-to-End Tests

```python
def test_complete_finetuning_workflow():
    # 1. Upload dataset
    dataset = upload_dataset("test_data.jsonl", format="alpaca")
    assert dataset.status == "ready"
    
    # 2. Create training job
    job = create_training_job(
        base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        dataset_id=dataset.id,
        lora_config={"rank": 8, "alpha": 16},
        training_config={"num_epochs": 1, "max_steps": 100}
    )
    
    # 3. Wait for completion
    job = wait_for_job(job.id, timeout=600)
    assert job.status == "completed"
    
    # 4. Verify adapter created
    adapter = get_adapter(job.adapter_id)
    assert adapter is not None
    
    # 5. Export to GGUF
    gguf_path = export_adapter(adapter.id, quantization="q4_k_m")
    assert Path(gguf_path).exists()
```

### 11.4 Performance Tests

- Training throughput benchmarks
- Memory usage validation
- Export time benchmarks
- Evaluation benchmark speed
- Comparison session throughput

---

## Appendix A: Hyperparameter Presets

### A.1 Quick Start (Fast Training)

```python
PRESET_QUICK = {
    "lora": {
        "rank": 8,
        "alpha": 16,
        "dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"]
    },
    "training": {
        "learning_rate": 3e-4,
        "batch_size": 8,
        "num_epochs": 1,
        "warmup_steps": 50
    }
}
```

### A.2 Balanced (Recommended Default)

```python
PRESET_BALANCED = {
    "lora": {
        "rank": 32,
        "alpha": 64,
        "dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
    },
    "training": {
        "learning_rate": 2e-4,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "num_epochs": 3,
        "warmup_steps": 100
    }
}
```

### A.3 High Quality (Best Results)

```python
PRESET_HIGH_QUALITY = {
    "lora": {
        "rank": 128,
        "alpha": 256,
        "dropout": 0.1,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"]
    },
    "training": {
        "learning_rate": 1e-4,
        "batch_size": 2,
        "gradient_accumulation_steps": 8,
        "num_epochs": 5,
        "warmup_steps": 200,
        "weight_decay": 0.01
    }
}
```

---

## Appendix B: VRAM Requirements

### B.1 QLoRA Training VRAM Estimates

| Model Size | Rank 8 | Rank 32 | Rank 64 | Rank 128 |
|------------|--------|---------|---------|----------|
| 1B | ~4GB | ~4GB | ~5GB | ~6GB |
| 3B | ~5GB | ~6GB | ~7GB | ~8GB |
| 7B | ~8GB | ~9GB | ~10GB | ~12GB |
| 8B | ~9GB | ~10GB | ~11GB | ~13GB |
| 13B | ~12GB | ~14GB | ~16GB | ~18GB |
| 34B | ~24GB | ~26GB | ~28GB | ~32GB |
| 70B | ~48GB | ~52GB | ~56GB | ~64GB |

*Note: Estimates include model weights, optimizer states, and gradient buffers.*

### B.2 VRAM Estimation Formula

```python
def estimate_training_vram(
    model_params_b: float,
    quant_bits: int,
    lora_rank: int,
    batch_size: int,
    seq_length: int
) -> float:
    # Base model memory (quantized)
    model_mem = model_params_b * (quant_bits / 8)
    
    # LoRA adapter memory (full precision)
    # Rough estimate: 2 * rank * hidden_dim * num_layers * 2 (A and B matrices)
    hidden_dim = int(model_params_b ** 0.5 * 1000)  # Approximate
    num_layers = int(model_params_b * 4)  # Approximate
    lora_mem = 2 * lora_rank * hidden_dim * num_layers * 4 / 1e9  # GB
    
    # Gradient and optimizer states (for LoRA only)
    optimizer_mem = lora_mem * 4  # Adam has 2 state tensors
    
    # Activation memory
    activation_mem = batch_size * seq_length * hidden_dim * num_layers * 2 / 1e9
    
    total = model_mem + lora_mem + optimizer_mem + activation_mem
    return total * 1.1  # 10% overhead
```

---

## Appendix C: Supported Model Architectures

| Architecture | Models | Target Modules |
|--------------|--------|----------------|
| LLaMA | Llama 2/3, CodeLlama, Vicuna | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Mistral | Mistral, Mixtral | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Qwen | Qwen, Qwen2 | c_attn, c_proj, w1, w2 |
| Phi | Phi-2, Phi-3 | q_proj, k_proj, v_proj, dense, fc1, fc2 |
| Gemma | Gemma | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |

---

## Appendix D: Dataset Format Templates

### D.1 Alpaca Template

```json
[
  {
    "instruction": "Write a Python function to calculate factorial",
    "input": "",
    "output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
  },
  {
    "instruction": "Translate the following sentence to French",
    "input": "Hello, how are you?",
    "output": "Bonjour, comment allez-vous?"
  }
]
```

### D.2 ShareGPT Template

```json
[
  {
    "conversations": [
      {"from": "human", "value": "What is recursion in programming?"},
      {"from": "gpt", "value": "Recursion is a programming technique where a function calls itself..."},
      {"from": "human", "value": "Can you show me an example?"},
      {"from": "gpt", "value": "Here's a classic example with factorial:\n\n```python\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)\n```"}
    ]
  }
]
```

### D.3 ChatML Template

```json
[
  {
    "messages": [
      {"role": "system", "content": "You are a helpful coding assistant."},
      {"role": "user", "content": "Write a hello world in Python"},
      {"role": "assistant", "content": "print('Hello, World!')"}
    ]
  }
]
```

---

## Conclusion

This design document provides a comprehensive blueprint for implementing LoRA fine-tuning in Llama Nexus. The implementation prioritizes:

1. **User Experience**: Wizard-based configuration, presets, real-time monitoring
2. **Flexibility**: Multiple dataset formats, configurable hyperparameters, distillation options
3. **Efficiency**: QLoRA support for consumer GPUs, containerized training
4. **Integration**: Seamless adapter deployment with existing llama.cpp infrastructure

The phased implementation approach allows for incremental delivery while maintaining a clear path to the full feature set.

