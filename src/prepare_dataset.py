"""
Preprocesses SynSQL-2.5M dataset for distributed training.
Run once before training to create memory-mapped Arrow files.

Filters:
- Drops samples without chain-of-thought (CoT)
- Drops samples that exceed max_length after tokenization
"""

from pathlib import Path
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from loguru import logger

SYSTEM_PROMPT = "You are a data science expert. Generate valid SQLite queries based on the provided database schema."

USER_TEMPLATE = """Database Schema:
{schema}

{external_knowledge}Question:
{question}

Generate a SQL query to answer this question."""


def has_cot(sample: dict) -> bool:
    """Filter: keeps only samples with non-empty CoT."""
    cot = sample.get("cot", "")
    return cot is not None and cot.strip() != ""


def format_sample(sample: dict) -> dict:
    """Converts raw sample to chat format with CoT."""
    external = ""
    if sample.get("external_knowledge", "").strip():
        external = f"Context:\n{sample['external_knowledge']}\n\n"
    
    user_content = USER_TEMPLATE.format(
        schema=sample["schema"],
        external_knowledge=external,
        question=sample["question"]
    )
    
    assistant_content = f"{sample['cot']}\n\n```sql\n{sample['sql']}\n```"
    
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    }


def tokenize_sample(sample: dict, tokenizer, max_length: int) -> dict:
    """
    Tokenizes chat messages with prompt masking.
    Returns None-equivalent (empty input_ids) if exceeds max_length.
    """
    messages = sample["messages"]
    
    # Tokenize full conversation (no truncation - we'll filter instead)
    full_ids = tokenizer.apply_chat_template(messages, tokenize=True)
    
    if len(full_ids) > max_length:
        # Mark for filtering
        return {"input_ids": [], "attention_mask": [], "labels": []}
    
    # Get prompt length for masking
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True
    )
    prompt_len = len(prompt_ids)
    
    labels = [-100] * prompt_len + full_ids[prompt_len:]
    attention_mask = [1] * len(full_ids)
    
    return {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def is_valid_tokenization(sample: dict) -> bool:
    """Filter: keeps only samples that didn't exceed max_length."""
    return len(sample["input_ids"]) > 0


def prepare(
    output_dir: str = "./data",
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    max_length: int = 4096,
    num_proc: int = 8,
    seed: int = 42,
    val_ratio: float = 0.01,
    test_ratio: float = 0.01
):
    """
    Downloads, filters, splits, formats, tokenizes, and saves dataset.
    Output is memory-mapped Arrow format for efficient distributed loading.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load
    logger.info("Loading iNeil77/SynSQL-2.5M")
    raw = load_dataset("iNeil77/SynSQL-2.5M", split="train")
    initial_count = len(raw)
    logger.info(f"Loaded {initial_count:,} samples")
    
    # Filter: require CoT
    logger.info("Filtering samples without CoT")
    raw = raw.filter(has_cot, num_proc=num_proc, desc="Filtering CoT")
    after_cot_filter = len(raw)
    dropped_no_cot = initial_count - after_cot_filter
    logger.info(f"Dropped {dropped_no_cot:,} samples without CoT ({dropped_no_cot/initial_count*100:.1f}%)")
    logger.info(f"Remaining: {after_cot_filter:,} samples")
    
    # Split before tokenization (faster)
    logger.info("Splitting into train/val/test")
    test_val_ratio = val_ratio + test_ratio
    split1 = raw.train_test_split(test_size=test_val_ratio, seed=seed)
    split2 = split1["test"].train_test_split(
        test_size=test_ratio / test_val_ratio, seed=seed
    )
    
    dataset = DatasetDict({
        "train": split1["train"],
        "val": split2["train"],
        "test": split2["test"]
    })
    
    for name, ds in dataset.items():
        logger.info(f"  {name}: {len(ds):,} samples")
    
    # Format
    logger.info("Formatting to chat template")
    dataset = dataset.map(
        format_sample,
        num_proc=num_proc,
        remove_columns=raw.column_names,
        desc="Formatting"
    )
    
    # Tokenize
    logger.info(f"Tokenizing (max_length={max_length})")
    dataset = dataset.map(
        lambda x: tokenize_sample(x, tokenizer, max_length),
        num_proc=num_proc,
        remove_columns=["messages"],
        desc="Tokenizing"
    )
    
    # Filter truncated
    logger.info("Filtering samples exceeding max_length")
    counts_before = {name: len(ds) for name, ds in dataset.items()}
    dataset = dataset.filter(is_valid_tokenization, num_proc=num_proc, desc="Filtering length")
    
    for name, ds in dataset.items():
        dropped = counts_before[name] - len(ds)
        logger.info(f"  {name}: dropped {dropped:,} ({dropped/counts_before[name]*100:.1f}%), kept {len(ds):,}")
    
    # Save
    logger.info(f"Saving to {output_path}")
    dataset.save_to_disk(output_path)
    
    # Summary
    total_kept = sum(len(ds) for ds in dataset.values())
    total_dropped = initial_count - total_kept
    logger.info(f"Complete. Kept {total_kept:,}/{initial_count:,} samples ({total_dropped:,} dropped total)")
    
    # Sample stats
    sample = dataset["train"][0]
    prompt_tokens = sample["labels"].count(-100)
    response_tokens = len(sample["labels"]) - prompt_tokens
    logger.info(f"Sample stats: {len(sample['input_ids'])} tokens, {prompt_tokens} prompt (masked), {response_tokens} response")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare SynSQL-2.5M for training")
    parser.add_argument("--output-dir", type=str, default="./data/processed")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.01)
    parser.add_argument("--test-ratio", type=float, default=0.01)
    
    args = parser.parse_args()
    
    prepare(
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_length=args.max_length,
        num_proc=args.num_proc,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )