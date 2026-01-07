"""
Evaluates fine-tuned model on test set.
Computes loss, perplexity, and exact match accuracy on SQL.
"""

import yaml
import argparse
import re

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from loguru import logger


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_quantization(config: dict) -> BitsAndBytesConfig | None:
    if not config.get("enabled", True):
        return None
    
    if config.get("load_in_8bit"):
        return BitsAndBytesConfig(load_in_8bit=True)
    
    if config.get("load_in_4bit"):
        compute_dtype = getattr(torch, config["bnb_4bit_compute_dtype"])
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=True
        )
    
    return None


def load_model_and_tokenizer(config: dict, adapter_path: str):
    model_name = config["model"]["name"]
    quant_config = setup_quantization(config["quantization"])
    
    logger.info(f"Loading base model: {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if quant_config is None else None,
    )
    
    logger.info(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def extract_sql(text: str) -> str:
    """Extracts SQL from markdown code block."""
    match = re.search(r"```sql\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def normalize_sql(sql: str) -> str:
    """Normalizes SQL for comparison."""
    sql = sql.lower()
    sql = re.sub(r'\s+', ' ', sql)
    sql = sql.strip()
    return sql


@torch.no_grad()
def evaluate(config_path: str, adapter_path: str | None = None, max_samples: int | None = None, num_examples: int = 5):
    config = load_config(config_path)
    
    if adapter_path is None:
        adapter_path = f"{config['checkpointing']['output_dir']}/final"
    
    model, tokenizer = load_model_and_tokenizer(config, adapter_path)
    
    # Load test data
    data_path = config["data"]["path"]
    logger.info(f"Loading test data from {data_path}")
    dataset = load_from_disk(data_path)
    test_data = dataset["test"]
    
    if max_samples and max_samples < len(test_data):
        test_data = test_data.select(range(max_samples))
        logger.info(f"Using {len(test_data)} test samples")
    
    total_loss = 0.0
    total_tokens = 0
    exact_matches = 0
    total_samples = 0
    sample_outputs = []  # Store examples for inspection
    
    for sample in tqdm(test_data, desc="Evaluating"):
        input_ids = torch.tensor([sample["input_ids"]], device=model.device)
        attention_mask = torch.tensor([sample["attention_mask"]], device=model.device)
        labels = torch.tensor([sample["labels"]], device=model.device)
        
        # Compute loss
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Count non-masked tokens for proper averaging
        non_masked = (labels != -100).sum().item()
        total_loss += outputs.loss.item() * non_masked
        total_tokens += non_masked
        
        # Generate for exact match
        prompt_len = (labels[0] == -100).sum().item()
        prompt_ids = input_ids[:, :prompt_len]
        prompt_mask = attention_mask[:, :prompt_len]
        
        generated = model.generate(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            max_new_tokens=4096,
            do_sample=False,
            num_beams=1,
            temperature=1.0,  
            top_p=1.0,
            top_k=50,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
                
        generated_text = tokenizer.decode(generated[0][prompt_len:], skip_special_tokens=True)
        target_text = tokenizer.decode(
            [t for t in labels[0].tolist() if t != -100], 
            skip_special_tokens=True
        )
        
        generated_sql = normalize_sql(extract_sql(generated_text))
        target_sql = normalize_sql(extract_sql(target_text))
        
        is_match = generated_sql == target_sql
        if is_match:
            exact_matches += 1
        
        # Collect sample outputs
        if len(sample_outputs) < num_examples:
            prompt_text = tokenizer.decode(prompt_ids[0], skip_special_tokens=True)
            sample_outputs.append({
                "prompt": prompt_text,
                "expected": target_sql,
                "generated": generated_sql,
                "match": is_match
            })
        
        total_samples += 1
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    exact_match_acc = exact_matches / total_samples * 100
    
    logger.info("=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Samples:      {total_samples}")
    logger.info(f"Loss:         {avg_loss:.4f}")
    logger.info(f"Perplexity:   {perplexity:.4f}")
    logger.info(f"Exact Match:  {exact_match_acc:.2f}%")
    logger.info("=" * 50)
    
    # Print sample outputs
    logger.info("")
    logger.info("=" * 50)
    logger.info(f"SAMPLE OUTPUTS ({len(sample_outputs)} examples)")
    logger.info("=" * 50)
    
    for i, ex in enumerate(sample_outputs):
        status = "✓ MATCH" if ex["match"] else "✗ MISMATCH"
        logger.info(f"\n--- Example {i+1} [{status}] ---")
        logger.info(f"PROMPT:\n{ex['prompt'][:500]}...")  # Truncate long prompts
        logger.info(f"\nEXPECTED:\n{ex['expected']}")
        logger.info(f"\nGENERATED:\n{ex['generated']}")
        logger.info("-" * 40)
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "exact_match": exact_match_acc,
        "samples": total_samples,
        "sample_outputs": sample_outputs
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--adapter-path", type=str, default=None, help="Path to adapter (default: checkpoints/final)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint name (e.g., 'checkpoint-500'). Overrides --adapter-path")
    parser.add_argument("--max-samples", type=int, default=None, help="Max test samples to evaluate")
    parser.add_argument("--num-examples", type=int, default=5, help="Number of sample outputs to display")
    args = parser.parse_args()
    
    # Resolve adapter path
    adapter_path = args.adapter_path
    if args.checkpoint:
        config = load_config(args.config)
        adapter_path = f"{config['checkpointing']['output_dir']}/{args.checkpoint}"
        logger.info(f"Using checkpoint: {adapter_path}")
    
    evaluate(args.config, adapter_path, args.max_samples, args.num_examples)
