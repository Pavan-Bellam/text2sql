"""
Merge LoRA adapter weights into base model for deployment.

Usage:
    python merge.py --config config.yaml --adapter ./checkpoint-1000 --output ./merged_model
    python merge.py --config config.yaml --adapter ./checkpoint-1000 --output ./merged_model --dtype fp16
    python merge.py --config config.yaml --adapter ./checkpoint-1000 --output ./merged_model --force
"""

import argparse
import yaml
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from loguru import logger


DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def validate_paths(adapter_path: Path, output_path: Path, force: bool) -> None:
    """Validate adapter exists and output won't overwrite without --force."""
    
    # Check adapter exists
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")
    
    # Check for required adapter files
    adapter_config = adapter_path / "adapter_config.json"
    if not adapter_config.exists():
        raise FileNotFoundError(
            f"No adapter_config.json found in {adapter_path}. "
            "Are you sure this is a LoRA checkpoint?"
        )
    
    # Check output path
    if output_path.exists():
        if not force:
            raise FileExistsError(
                f"Output path already exists: {output_path}. "
                "Use --force to overwrite."
            )
        logger.warning(f"Output path exists, will overwrite: {output_path}")


def merge(
    config_path: str,
    adapter_path: str,
    output_path: str,
    dtype: str,
    force: bool,
) -> None:
    config = load_config(config_path)
    model_name = config["model"]["name"]
    torch_dtype = DTYPE_MAP[dtype]
    
    adapter_path = Path(adapter_path)
    output_path = Path(output_path)
    
    validate_paths(adapter_path, output_path, force)
    
    logger.info(f"Base model: {model_name}")
    logger.info(f"Adapter: {adapter_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Dtype: {dtype}")
    
    # Load base model
    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load adapter
    logger.info("Loading adapter...")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    
    # Merge
    logger.info("Merging weights...")
    merged = model.merge_and_unload()
    
    # Save
    logger.info(f"Saving merged model to {output_path}...")
    output_path.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter into base model for deployment"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML (used to get base model name)",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Path to LoRA adapter checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for merged model",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Data type for merged model (default: bf16)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output path if it exists",
    )
    
    args = parser.parse_args()
    
    merge(
        config_path=args.config,
        adapter_path=args.adapter,
        output_path=args.output,
        dtype=args.dtype,
        force=args.force,
    )