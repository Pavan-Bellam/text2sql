# Developer Documentation

Technical overview of the Text2SQL codebase.

## Project Structure

```
text2sql/
├── src/
│   └── prepare_dataset.py   # Dataset preprocessing pipeline
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
