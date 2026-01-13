# Developer Documentation

Technical overview of the Text2SQL codebase.

## Project Structure

```
text2sql/
├── src/
│   ├── prepare_dataset.py   # Dataset preprocessing pipeline
│   ├── train.py             # Distributed training script
│   ├── evaluate.py          # Model evaluation script
│   ├── merge_weights.py     # Merge LoRA weights for deployment
│   └── s3_client.py         # S3 upload utilities
├── config/
│   ├── train.yaml           # Training hyperparameters
│   ├── accelerate.yml       # Accelerate distributed config
│   └── ds_config.json       # DeepSpeed ZeRO config
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

## Training

### train.py

Distributed LoRA fine-tuning script using HuggingFace Trainer with DeepSpeed.

**Features:**
- LoRA fine-tuning via PEFT (parameter-efficient fine-tuning)
- Optional 4-bit/8-bit quantization via bitsandbytes
- DeepSpeed ZeRO-2 for distributed training
- Flash Attention 2 for memory-efficient attention
- Gradient checkpointing for reduced memory usage
- W&B experiment tracking (main process only)
- S3 checkpoint uploads (main process only)
- Resume training or initialize from existing checkpoint

**CLI arguments:**
- `--config`: Path to training config YAML (required)
- `--resume`: Resume from checkpoint (keeps optimizer/scheduler state)
- `--init-from`: Initialize LoRA weights from checkpoint but start fresh (step 0, new optimizer)

**Distributed training considerations:**
- Logging silenced on worker processes (only rank 0 logs)
- W&B init/finish only on main process
- S3 uploads only on main process
- Uses `RANK` environment variable (set by accelerate/deepspeed)

### evaluate.py

Evaluates fine-tuned model on the test set.

**Metrics computed:**
- **Loss**: Average cross-entropy loss on non-masked tokens
- **Perplexity**: Exp of average loss
- **Exact Match**: Percentage of generated SQL matching target exactly (after normalization)

**SQL comparison:**
1. Extract SQL from markdown code blocks (```sql ... ```)
2. Normalize: lowercase, collapse whitespace
3. Compare strings for exact match

**CLI arguments:**
- `--config`: Path to training config YAML (required)
- `--adapter-path`: Path to LoRA adapter (default: `checkpoints/final`)
- `--checkpoint`: Checkpoint name, e.g., `checkpoint-500` (overrides --adapter-path)
- `--max-samples`: Limit number of test samples
- `--num-examples`: Number of sample outputs to display (default: 5)

### merge_weights.py

Merges LoRA adapter weights into the base model for deployment/serving.

**Why merge?**
- LoRA adapters require loading base model + adapter at inference time
- Merged model is a single artifact that can be served directly
- No PEFT dependency needed at inference time

**Process:**
1. Load base model from config
2. Load LoRA adapter from checkpoint
3. Merge adapter weights into base model via `merge_and_unload()`
4. Save merged model and tokenizer

**CLI arguments:**
- `--config`: Path to training config YAML (required, used to get base model name)
- `--adapter`: Path to LoRA adapter checkpoint (required)
- `--output`: Output path for merged model (required)
- `--dtype`: Data type for merged model - `bf16`, `fp16`, or `fp32` (default: `bf16`)
- `--force`: Overwrite output path if it exists

**Validation:**
- Checks adapter path exists and contains `adapter_config.json`
- Refuses to overwrite existing output without `--force` flag

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
