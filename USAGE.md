# Influence - Usage Guide

## Quick Start

### 1. Search for Models

Find models on HuggingFace:

```bash
# Search for Llama models
cargo run -- search "llama"

# Search with filters
cargo run -- search "text-generation" --limit 10 --author meta-llama

# Search for specific models
cargo run -- search "granite" --author ibm
```

### 2. Download a Model

Download a model to your local machine:

```bash
# Download IBM Granite model
cargo run -- download -m ibm/granite-4-h-small

# Download to custom directory
cargo run -- download -m meta-llama/Llama-2-7b-hf -o ./my-models

# Use custom mirror
cargo run -- download -m mistralai/Mistral-7B-v0.1 -r https://hf-mirror.com
```

Downloaded models are stored in `./models/` by default with the format `models/<org>_<model-name>/`.

### 3. Generate Text with Local Model

Use the downloaded model for text generation:

```bash
# Basic generation
cargo run -- generate "What is Rust?" --model-path ./models/ibm_granite-4-h-small

# With custom parameters
cargo run -- generate "Explain quantum computing" \
  --model-path ./models/ibm_granite-4-h-small \
  --max-tokens 1024 \
  --temperature 0.8
```

## Complete Workflow Example

```bash
# Step 1: Search for a model
cargo run -- search "granite" --author ibm

# Step 2: Download the model
cargo run -- download -m ibm/granite-4-h-small

# Step 3: Generate text
cargo run -- generate "Write a hello world program in Rust" \
  --model-path ./models/ibm_granite-4-h-small
```

## Model Directory Structure

After downloading, your model directory will contain:

```
models/
└── ibm_granite-4-h-small/
    ├── config.json              # Model configuration
    ├── tokenizer.json           # Tokenizer vocabulary
    ├── tokenizer_config.json    # Tokenizer settings
    ├── special_tokens_map.json  # Special tokens
    └── model.safetensors        # Model weights
```

## Advanced Usage

### Custom Output Directory

```bash
cargo run -- download -m ibm/granite-4-h-small -o /path/to/custom/dir
```

### Search with Filters

```bash
# Find models by specific author
cargo run -- search "llama" --author meta-llama --limit 5

# Search for specific task types
cargo run -- search "text-generation" --limit 20
```

### Generation Parameters

- `--max-tokens`: Controls the maximum length of generated text (default: 512)
- `--temperature`: Controls randomness (0.0 = deterministic, 1.0 = very random, default: 0.7)

```bash
# More creative output
cargo run -- generate "Write a story" \
  --model-path ./models/ibm_granite-4-h-small \
  --temperature 0.9 \
  --max-tokens 2048

# More focused output
cargo run -- generate "What is 2+2?" \
  --model-path ./models/ibm_granite-4-h-small \
  --temperature 0.1 \
  --max-tokens 50
```

## Troubleshooting

### Model Not Found

If you get a "Model not found" error:
1. Verify the model name is correct (format: `org/model-name`)
2. Check if the model exists on HuggingFace
3. Some models may be gated - visit the model page to request access

### Download Fails

If downloads fail:
1. Check your internet connection
2. Try using a different mirror with `-r` flag
3. Some files may require authentication for gated models

### Generation Requires Model Path

The generate command requires a local model:
```bash
# ❌ Wrong - no model path
cargo run -- generate "Hello"

# ✅ Correct - with model path
cargo run -- generate "Hello" --model-path ./models/ibm_granite-4-h-small
```

## Tips

1. **Start with smaller models** - They download faster and are easier to test
2. **Use search before download** - Find the right model for your task
3. **Check model size** - Large models may take significant time to download
4. **Keep models organized** - Use the default `./models/` directory for consistency

## Supported Model Formats

- **SafeTensors** (`.safetensors`) - Recommended format
- **GGUF** (`.gguf`) - Quantized models for efficient inference
- **PyTorch** (`.bin`) - Standard PyTorch format

## Next Steps

- Implement full local inference with candle-transformers
- Add support for quantized models (GGUF)
- Enable GPU acceleration
- Add batch processing capabilities
