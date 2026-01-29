#!/bin/bash

echo "Influence CLI Test Script"
echo "========================="
echo ""

echo "Building the project..."
cargo build --release

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo ""
echo "Build successful!"
echo ""

echo "Available commands:"
echo ""
echo "1. Show help:"
echo "   ./target/release/influence --help"
echo ""

echo "2. Download a model (requires internet):"
echo "   ./target/release/influence download -m ibm/granite-4-h-small"
echo ""

echo "3. Generate text (requires WATSONX_API_KEY and WATSONX_PROJECT_ID):"
echo "   export WATSONX_API_KEY=\"your-key\""
echo "   export WATSONX_PROJECT_ID=\"your-project-id\""
echo "   ./target/release/influence generate \"What is Rust?\""
echo ""

echo "4. Serve the influencer:"
echo "   ./target/release/influence serve --port 8080"
echo ""

echo "Running help command..."
./target/release/influence --help
