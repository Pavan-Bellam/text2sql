# Text2SQL

Fine-tuning Qwen2.5-Coder-7B-Instruct for text-to-SQL generation using the SynSQL-2.5M dataset with distributed training via DeepSpeed and HuggingFace Accelerate.

## Setup

```bash
uv sync
```

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
