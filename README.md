# Text2SQL

Fine-tuning Qwen2.5-Coder-7B-Instruct for text-to-SQL generation using the SynSQL-2.5M dataset with distributed training via DeepSpeed and HuggingFace Accelerate.

## Setup

```bash
uv sync
cp .env.example .env  # Configure environment variables
```

## Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Description |
|----------|-------------|
| `AWS_ACCESS_KEY_ID` | AWS credentials for S3 checkpoint uploads |
| `AWS_SECRET_ACCESS_KEY` | AWS credentials for S3 checkpoint uploads |
| `WANDB_API_KEY` | Weights & Biases API key for experiment tracking |

## Data Preparation

Preprocesses SynSQL-2.5M dataset into memory-mapped Arrow files for efficient distributed loading.

```bash
python src/prepare_dataset.py --output-dir ./data/processed
```

Options:
- `--model-name`: Tokenizer model (default: `Qwen/Qwen2.5-Coder-7B-Instruct`)
- `--max-length`: Maximum sequence length (default: `4096`)
- `--num-proc`: Number of parallel workers (default: `8`)
- `--val-ratio`: Validation split ratio (default: `0.01`)
- `--test-ratio`: Test split ratio (default: `0.01`)

## Training

Launch distributed training with Accelerate:

```bash
accelerate launch --config_file config/accelerate.yml src/train.py --config config/train.yaml
```

Resume from checkpoint:
```bash
accelerate launch --config_file config/accelerate.yml src/train.py --config config/train.yaml --resume ./checkpoints/checkpoint-500
```

Initialize from checkpoint (fresh optimizer state):
```bash
accelerate launch --config_file config/accelerate.yml src/train.py --config config/train.yaml --init-from ./checkpoints/checkpoint-500
```
