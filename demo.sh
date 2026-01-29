#!/bin/bash

echo "=========================================="
echo "Influence CLI - Demonstration"
echo "=========================================="
echo ""

echo "1. Searching for models..."
echo "   Command: cargo run -- search \"granite\" --limit 3"
echo ""
cargo run --quiet -- search "granite" --limit 3
echo ""

echo "=========================================="
echo ""
echo "2. Downloading a model..."
echo "   Command: cargo run -- download -m ibm/granite-4-h-small"
echo ""
echo "   Note: This will download the model to ./models/ibm_granite-4-h-small/"
echo "   Press Ctrl+C to skip download, or wait for it to complete..."
echo ""
read -p "Press Enter to start download (or Ctrl+C to skip)..."
cargo run -- download -m ibm/granite-4-h-small
echo ""

echo "=========================================="
echo ""
echo "3. Generating text with local model..."
echo "   Command: cargo run -- generate \"What is Rust?\" --model-path ./models/ibm_granite-4-h-small"
echo ""
if [ -d "./models/ibm_granite-4-h-small" ]; then
    cargo run -- generate "What is Rust programming language?" --model-path ./models/ibm_granite-4-h-small
else
    echo "   Model not found. Please download it first with:"
    echo "   cargo run -- download -m ibm/granite-4-h-small"
fi
echo ""

echo "=========================================="
echo "Demo complete!"
echo ""
echo "Try these commands yourself:"
echo "  - Search: cargo run -- search \"llama\" --limit 5"
echo "  - Download: cargo run -- download -m <model-name>"
echo "  - Generate: cargo run -- generate \"<prompt>\" --model-path ./models/<model-dir>"
echo ""
