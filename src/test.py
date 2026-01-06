# src/check_lengths.py
"""
Check token length distribution to decide max_length and filtering threshold.
"""

from datasets import load_dataset
from transformers import AutoTokenizer
from collections import Counter

def check_length_distribution(num_samples: int = 10000):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
    dataset = load_dataset("iNeil77/SynSQL-2.5M", split="train", streaming=True)
    
    lengths = []
    
    for i, sample in enumerate(dataset.take(num_samples)):
        # Build the full training sequence (input + output)
        text = f"""<|im_start|>system
You are a data science expert.<|im_end|>
<|im_start|>user
Database Schema:
{sample['schema']}

Question:
{sample['question']}<|im_end|>
<|im_start|>assistant
````sql
{sample['sql']}
```<|im_end|>"""
        
        tokens = tokenizer.encode(text)
        lengths.append(len(tokens))
        
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1} samples...")
    
    # Stats
    lengths.sort()
    print(f"\n=== Token Length Distribution ({num_samples} samples) ===")
    print(f"Min: {min(lengths)}")
    print(f"Max: {max(lengths)}")
    print(f"Mean: {sum(lengths) / len(lengths):.0f}")
    print(f"Median: {lengths[len(lengths)//2]}")
    print(f"\nPercentiles:")
    for p in [50, 75, 90, 95, 99, 99.5]:
        idx = int(len(lengths) * p / 100)
        print(f"  {p}%: {lengths[idx]}")
    
    # Bucket distribution
    print(f"\nBucket distribution:")
    buckets = [512, 1024, 2048, 4096, 8192, 16384, float('inf')]
    bucket_names = ["<512", "512-1K", "1K-2K", "2K-4K", "4K-8K", "8K-16K", ">16K"]
    counts = [0] * len(buckets)
    
    for length in lengths:
        for i, threshold in enumerate(buckets):
            if length <= threshold:
                counts[i] += 1
                break
    
    for name, count in zip(bucket_names, counts):
        pct = count / len(lengths) * 100
        print(f"  {name}: {count} ({pct:.1f}%)")

if __name__ == "__main__":
    check_length_distribution(10000)