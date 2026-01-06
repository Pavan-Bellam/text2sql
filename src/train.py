import yaml
import argparse
from pathlib import Path
from datetime import datetime

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import wandb
from loguru import logger
import os


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


def setup_lora(config: dict) -> LoraConfig:
    return LoraConfig(
        r=config['r'],
        lora_alpha=config['alpha'],
        lora_dropout=config['dropout'],
        target_modules=config['target_modules'],
        task_type="CAUSAL_LM",
        bias="none"
    )


def is_deepspeed_zero3(config: dict) -> bool:
    """Check if using DeepSpeed ZeRO-3"""
    ds_config_path = config.get("training", {}).get("deepspeed")
    if not ds_config_path:
        return False
    
    with open(ds_config_path) as f:
        ds_config = yaml.safe_load(f) if ds_config_path.endswith('.yml') else __import__('json').load(f)
    
    return ds_config.get("zero_optimization", {}).get("stage") == 3


def load_model_and_tokenizer(config: dict):
    model_name = config["model"]["name"]
    quant_config = setup_quantization(config["quantization"])
    use_zero3 = is_deepspeed_zero3(config)
    use_gradient_checkpointing = config["training"].get("gradient_checkpointing", False)
    
    logger.info(f"Loading model: {model_name}")
    logger.info(f"DeepSpeed ZeRO-3: {use_zero3}")
    logger.info(f"Gradient checkpointing: {use_gradient_checkpointing}")
    
    if quant_config:
        logger.info(f"Quantization: {'8-bit' if quant_config.load_in_8bit else '4-bit'}")
    else:
        logger.info("Quantization: disabled (full precision)")


    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map=None,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare for kbit training if quantized
    if quant_config:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=use_gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False} if use_gradient_checkpointing else None
        )
    elif use_gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    lora_config = setup_lora(config['lora'])
    model = get_peft_model(model, lora_config)
    

    model.enable_input_require_grads()
    
    model.print_trainable_parameters()

    return model, tokenizer


def load_data(config: dict):
    data_path = config["data"]["path"]
    logger.info(f"Loading data from {data_path}")
    dataset = load_from_disk(data_path)
    
    train_data = dataset["train"]
    val_data = dataset["val"]
    
    max_train = config["data"].get("max_train_samples")
    max_val = config["data"].get("max_val_samples")
    
    if max_train and max_train < len(train_data):
        train_data = train_data.select(range(max_train))
        logger.info(f"Truncated train to {len(train_data):,} samples")
    
    if max_val and max_val < len(val_data):
        val_data = val_data.select(range(max_val))
        logger.info(f"Truncated val to {len(val_data):,} samples")
    
    return train_data, val_data


def train(config_path: str):
    config = load_config(config_path)

    torch.manual_seed(config['seed'])

    global_rank = int(os.environ.get("RANK", 0))
    run_name = config['logging'].get("wandb_run_name")
    if run_name is None:
        run_name = f"qlora-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if global_rank == 0:
        

        wandb.init(
            project=config["logging"]["wandb_project"],
            name=run_name,
            config=config,  # This logs your YAML structure
        )

    model, tokenizer = load_model_and_tokenizer(config)

    train_data, val_data = load_data(config)
    logger.info(f"Train: {len(train_data):,} samples, Val: {len(val_data):,} samples")

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
    )

    s3_config = config.get('s3')
    s3_prefix = f"{s3_config.get("prefix")}/{run_name}"
    s3_callback = S3UploadCallback(
    s3_bucket=s3_config.get('bucket_name'),  # CHANGE THIS
    s3_prefix=s3_prefix  # CHANGE THIS
    )

    train_config = config["training"]
    ckpt_config = config["checkpointing"]
    log_config = config["logging"]

    training_args = TrainingArguments(
        output_dir=ckpt_config["output_dir"],
        num_train_epochs=train_config["num_epochs"],
        per_device_train_batch_size=train_config["per_device_batch_size"],
        per_device_eval_batch_size=train_config["per_device_batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        learning_rate=train_config["learning_rate"],
        lr_scheduler_type=train_config["lr_scheduler"],
        warmup_ratio=train_config["warmup_ratio"],
        weight_decay=train_config["weight_decay"],
        max_grad_norm=train_config["max_grad_norm"],
        bf16=train_config["bf16"],
        gradient_checkpointing=train_config.get("gradient_checkpointing", False),
        gradient_checkpointing_kwargs={"use_reentrant": False} if train_config.get("gradient_checkpointing") else None,
        deepspeed=train_config.get("deepspeed"),
        logging_steps=log_config["log_steps"],
        save_steps=ckpt_config["save_steps"],
        save_total_limit=ckpt_config["save_total_limit"],
        eval_strategy="steps",
        eval_steps=ckpt_config["save_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb",
        run_name=run_name,
        seed=config["seed"],
        dataloader_num_workers=train_config.get("dataloader_num_workers"),
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=collator
    )

    resume_from = ckpt_config.get("resume_from")
    if resume_from:
        logger.info(f"Resuming from {resume_from}")

    logger.info("Starting training")
    trainer.train(resume_from_checkpoint=resume_from)

    final_path = Path(ckpt_config["output_dir"]) / "final"
    trainer.save_model(str(final_path))
    logger.info(f"Saved final model to {final_path}")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    train(args.config)