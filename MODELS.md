# Supported Models Guide

## Overview

The Influence CLI supports local inference for **standard Llama-architecture transformer models**. This guide helps you understand which models are compatible and how to use them.

## ✅ Fully Supported Models

### Llama Family
- **meta-llama/Llama-2-7b-hf** - Standard Llama 2 7B
- **meta-llama/Llama-2-13b-hf** - Llama 2 13B
- **TinyLlama/TinyLlama-1.1B-Chat-v1.0** - Tiny 1.1B model (great for testing)
- **openlm-research/open_llama_3b** - Open Llama 3B
- **openlm-research/open_llama_7b** - Open Llama 7B

### Mistral Family
- **mistralai/Mistral-7B-v0.1** - Mistral 7B base
- **mistralai/Mistral-7B-Instruct-v0.1** - Mistral 7B instruct

### Requirements
- Must use **standard transformer architecture** (self-attention layers)
- Must have **.safetensors** weight files
- Must include `tokenizer.json` and `config.json`

## ❌ Unsupported Models

### Mamba/Hybrid Models
- **ibm/granite-4.0-h-1b** - Uses Mamba+Attention hybrid layers
- **ibm/granite-4.0-h-3b** - Mamba hybrid architecture
- Any model with `layer_types` in config.json

**Why?** These models use State Space Models (Mamba) instead of standard attention, requiring a different implementation.

### MoE Models
- **mixtral-8x7b** - Mixture of Experts architecture
- Models with `num_local_experts > 0`

**Why?** MoE models require routing logic not yet implemented.

### GGUF-Only Models
- Models only available in GGUF format
- Quantized models without .safetensors

**Why?** Current implementation requires .safetensors format.

## How to Check Model Compatibility

### Method 1: Check config.json

```bash
# Download and check config
cargo run -- download -m MODEL_NAME
cat models/MODEL_NAME/config.json | grep -E "model_type|layer_types|num_local_experts"
```

**Compatible if:**
- `"model_type": "llama"` or `"model_type": "mistral"`
- No `layer_types` field
- `num_local_experts` is 0 or absent

**Incompatible if:**
- `layer_types` array present (Mamba hybrid)
- `num_local_experts > 0` (MoE)
- `model_type` is unusual (e.g., "granitemoehybrid")

### Method 2: Try Loading

```bash
cargo run -- generate "test" --model-path ./models/MODEL_NAME
```

You'll get a clear error message if the model is unsupported.

## Recommended Models for Testing

### For Quick Testing (< 2GB)
```bash
# TinyLlama - 1.1B parameters, ~2GB
cargo run -- download -m TinyLlama/TinyLlama-1.1B-Chat-v1.0
cargo run -- generate "Hello!" --model-path ./models/TinyLlama_TinyLlama-1.1B-Chat-v1.0
```

### For Better Quality (3-7GB)
```bash
# Open Llama 3B - ~6GB
cargo run -- download -m openlm-research/open_llama_3b

# Mistral 7B - ~14GB
cargo run -- download -m mistralai/Mistral-7B-v0.1
```

### For Production Use (14GB+)
```bash
# Llama 2 7B - ~14GB
cargo run -- download -m meta-llama/Llama-2-7b-hf

# Llama 2 13B - ~26GB
cargo run -- download -m meta-llama/Llama-2-13b-hf
```

## Model Size vs Performance

| Model Size | RAM Required | Speed (CPU) | Quality |
|------------|--------------|-------------|---------|
| 1B         | ~4GB         | 5-10 tok/s  | Basic   |
| 3B         | ~8GB         | 2-5 tok/s   | Good    |
| 7B         | ~16GB        | 1-2 tok/s   | Great   |
| 13B        | ~32GB        | 0.5-1 tok/s | Excellent |

*Speed estimates for CPU inference*

## Alternative Solutions for Unsupported Models

### For Granite MoE Hybrid Models
```bash
# Use transformers library with Python
pip install transformers torch
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('ibm/granite-4.0-h-1b')
tokenizer = AutoTokenizer.from_pretrained('ibm/granite-4.0-h-1b')
"
```

### For High-Performance Serving
```bash
# Use vLLM
pip install vllm
vllm serve ibm/granite-4.0-h-1b
```

### For Quantized Models
```bash
# Use llama.cpp (if GGUF available)
./llama.cpp/main -m model.gguf -p "Hello"
```

## Finding Compatible Models

### On HuggingFace
1. Search for "llama" or "mistral"
2. Check model card for architecture
3. Look for .safetensors files in "Files and versions"
4. Avoid models mentioning "Mamba", "MoE", or "Hybrid"

### Using the CLI
```bash
# Search for Llama models
cargo run -- search "llama" --limit 10

# Search for Mistral models
cargo run -- search "mistral" --limit 10
```

## Troubleshooting

### "Unsupported model architecture: This model uses mixed Mamba+Attention layers"
**Solution:** Use a standard Llama or Mistral model instead, or use Python transformers library.

### "shape mismatch for model.embed_tokens.weight"
**Solution:** This was fixed - update to latest version. The CLI now parses actual config.json.

### "No .safetensors files found"
**Solution:** Model may only have .bin or .gguf files. Look for a version with .safetensors, or convert using:
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("model-name")
model.save_pretrained("output-dir", safe_serialization=True)
```

### "Model directory not found"
**Solution:** Download the model first:
```bash
cargo run -- download -m MODEL_NAME
```

## Future Support

### Planned
- [ ] GPU acceleration (CUDA/Metal)
- [ ] GGUF format support
- [ ] Quantization (4-bit, 8-bit)
- [ ] More architectures (Phi, Falcon)

### Under Consideration
- [ ] MoE support (Mixtral)
- [ ] Mamba architecture support
- [ ] Custom model implementations

## Summary

**Use these models for best results:**
- ✅ TinyLlama (testing)
- ✅ Llama 2 (production)
- ✅ Mistral (production)
- ✅ Open Llama (open source)

**Avoid these for now:**
- ❌ Granite MoE Hybrid
- ❌ Mixtral MoE
- ❌ Mamba-based models

For unsupported models, use Python transformers library or vLLM instead.
