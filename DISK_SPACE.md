# Disk Space Management

## Current Issue

Your disk is **100% full** (435GB used of 460GB). This caused the BERT model download to fail partway through.

## Immediate Actions

### 1. Free Up Space

```bash
# Remove the incomplete BERT download (not compatible anyway)
rm -rf ./models/google-bert_bert-base-uncased

# This will free ~2.3GB
```

### 2. Check Available Space

```bash
df -h .
```

You need at least **5-10GB free** to download and use models safely.

## Model Sizes

| Model | Size | Recommended Free Space |
|-------|------|------------------------|
| TinyLlama 1.1B | ~2GB | 5GB |
| Llama 2 7B | ~14GB | 20GB |
| Llama 2 13B | ~26GB | 35GB |
| Mistral 7B | ~14GB | 20GB |

## Why BERT Failed

1. **Disk full** - Download stopped at tokenizer files
2. **Wrong model type** - BERT is encoder-only, cannot generate text
3. **Not compatible** - This CLI only supports decoder models (Llama, Mistral)

## What is BERT?

**BERT (Bidirectional Encoder Representations from Transformers)** is an **encoder-only** model:

### ❌ Cannot Do
- Generate text
- Complete sentences
- Chat/conversation
- Creative writing

### ✅ Can Do
- Text classification
- Sentiment analysis
- Named entity recognition
- Question answering (extractive)
- Embeddings

## Compatible Models for Text Generation

### Small (Good for Testing)
```bash
# TinyLlama - 1.1B params, ~2GB
cargo run -- download -m TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

### Medium (Good Quality)
```bash
# Mistral 7B - ~14GB
cargo run -- download -m mistralai/Mistral-7B-v0.1
```

### Large (Best Quality)
```bash
# Llama 2 7B - ~14GB
cargo run -- download -m meta-llama/Llama-2-7b-hf
```

## Cleaning Up Space

### Remove Unused Models
```bash
# List downloaded models
ls -lh ./models/

# Remove specific model
rm -rf ./models/MODEL_NAME

# Remove all models
rm -rf ./models/*
```

### Check What's Using Space
```bash
# Find large files
du -sh ./models/*

# Total space used by models
du -sh ./models/
```

## Best Practices

1. **Check disk space before downloading**
   ```bash
   df -h .
   ```

2. **Download smaller models first** (TinyLlama for testing)

3. **Remove old models** before downloading new ones

4. **Use only one model at a time** if space is limited

5. **Monitor downloads** - Cancel if disk fills up

## Recommended Next Steps

1. **Free up 10GB+ of disk space**
   - Delete old files, applications, or downloads
   - Empty trash
   - Remove unused Docker images/containers

2. **Download a compatible model**
   ```bash
   cargo run -- download -m TinyLlama/TinyLlama-1.1B-Chat-v1.0
   ```

3. **Test generation**
   ```bash
   cargo run -- generate "What is Rust?" \
     --model-path ./models/TinyLlama_TinyLlama-1.1B-Chat-v1.0
   ```

## Error Messages Explained

### "No space left on device"
- Disk is 100% full
- Free up space before downloading

### "Tokenizer file not found"
- Download was incomplete
- Re-download after freeing space

### "Unsupported model type 'bert'"
- BERT cannot generate text
- Use Llama/Mistral instead

## Summary

- ❌ **BERT** - Encoder-only, cannot generate text
- ✅ **Llama** - Decoder, can generate text
- ✅ **Mistral** - Decoder, can generate text
- 🔴 **Disk full** - Free up space first
- 💾 **Need 5-10GB** - For safe model downloads
