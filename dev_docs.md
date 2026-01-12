# Developer Documentation

Technical overview of the Text2SQL codebase.

## Project Structure

```
text2sql/
├── src/
│   ├── prepare_dataset.py   # Dataset preprocessing pipeline
│   └── s3_client.py         # S3 upload utilities
├── config/                   # Training configurations
├── data/                     # Processed datasets (gitignored)
├── pyproject.toml            # Dependencies
└── uv.lock                   # Locked dependencies
```

## Data Pipeline

### prepare_dataset.py

Preprocesses the SynSQL-2.5M dataset for distributed training.

**Pipeline steps:**
1. Load SynSQL-2.5M from HuggingFace Hub
2. Filter samples without chain-of-thought (CoT) reasoning
3. Split into train/val/test sets
4. Format into chat template with system prompt, user query, and assistant response
5. Tokenize with prompt masking (labels=-100 for prompt tokens)
6. Filter samples exceeding max_length
7. Save as memory-mapped Arrow files

**Chat format:**
- **System**: SQL expert prompt
- **User**: Database schema + optional context + question
- **Assistant**: CoT reasoning + SQL query in code block

**Output:** `./data/processed/` with train/val/test splits in Arrow format

## Utilities

### s3_client.py

S3 upload utilities for checkpoint backup during training.

**S3UploadCallback**: A HuggingFace `TrainerCallback` that automatically uploads checkpoints to S3 after each save.

- Triggered on `on_save` event
- Uploads entire checkpoint directory recursively
- Preserves folder structure in S3

**Requires:** AWS credentials configured (via environment variables or AWS profile)

## Configuration

Production configuration optimized for 7x A100 GPUs on RunPod. These settings were tuned through experimentation to maximize throughput without OOM.

### train.yaml

Main training configuration file.

| Section | Parameters |
|---------|------------|
| `model` | Model name/path |
| `quantization` | Enable/disable quantization (4-bit/8-bit) |
| `lora` | LoRA hyperparameters (r, alpha, dropout, target_modules) |
| `training` | Epochs, batch size, learning rate, scheduler, DeepSpeed config path |
| `data` | Dataset path, max samples |
| `checkpointing` | Output dir, save frequency, resume path |
| `logging` | W&B project/run name, log frequency |
| `s3` | Bucket and prefix for checkpoint uploads |

**Key training parameters:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| `quantization.enabled` | `false` | BF16 native on A100s, no need for quantization |
| `per_device_batch_size` | `7` | Max batch size fitting A100 80GB VRAM |
| `gradient_accumulation_steps` | `4` | Accumulate before optimizer step |
| `num_gpus` | `7` | RunPod 7x A100 pod |
| **Effective batch size** | **196** | 7 × 4 × 7 = 196 samples per optimizer step |
| `learning_rate` | `2e-4` | Standard for LoRA fine-tuning |
| `warmup_ratio` | `0.03` | 3% of training steps for LR warmup |
| `lora.r` | `64` | LoRA rank |
| `lora.alpha` | `128` | LoRA alpha (scaling = alpha/r = 2) |

### accelerate.yml

HuggingFace Accelerate configuration for distributed training.

- `distributed_type`: DEEPSPEED
- `num_processes`: 8 (configurable per setup)
- `deepspeed_config_file`: Path to DeepSpeed config

### ds_config.json

DeepSpeed ZeRO-2 optimization config.

- BF16 mixed precision
- ZeRO Stage 2 (optimizer state partitioning)
- No CPU offloading (GPU-only)
- Auto batch size detection
