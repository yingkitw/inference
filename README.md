# Influence

A Rust CLI application for downloading models from HuggingFace mirrors and serving them as an LLM influencer using WatsonX.

## Features

- **Download models** from HuggingFace mirrors with progress tracking
- **Serve LLM** as an influencer using WatsonX API
- **Generate text** with streaming support using Granite models
- **CLI interface** with simple commands
- **Test-friendly architecture** with trait-based design

## Installation

```bash
cargo build --release
```

## Configuration

Set the following environment variables:

```bash
export WATSONX_API_KEY="your-api-key"
export WATSONX_PROJECT_ID="your-project-id"
export WATSONX_MODEL_ID="ibm/granite-4-h-small"  # Optional, defaults to granite-4-h-small
```

## Usage

### Download a model

Download a model from HuggingFace mirror:

```bash
cargo run -- download -m ibm/granite-4-h-small
```

With custom mirror and output directory:

```bash
cargo run -- download -m ibm/granite-4-h-small -r https://hf-mirror.com -o ./models
```

### Generate text

Generate text using a prompt:

```bash
cargo run -- generate "What is Rust programming language?"
```

With custom parameters:

```bash
cargo run -- generate "Explain quantum computing" --max-tokens 1024 --temperature 0.8
```

### Serve the influencer

Start the influencer service:

```bash
cargo run -- serve --port 8080
```

## Commands

### `download`

Download a model from HuggingFace mirror.

**Options:**
- `-m, --model <MODEL>` - Model name (e.g., 'ibm/granite-4-h-small')
- `-r, --mirror <MIRROR>` - Mirror URL (default: hf-mirror.com)
- `-o, --output <OUTPUT>` - Output directory

### `serve`

Serve LLM as influencer.

**Options:**
- `-m, --model-path <MODEL_PATH>` - Path to model directory
- `-p, --port <PORT>` - Port to serve on (default: 8080)

### `generate`

Generate text using the LLM.

**Arguments:**
- `<PROMPT>` - Prompt text

**Options:**
- `-m, --model-path <MODEL_PATH>` - Path to model directory
- `--max-tokens <MAX_TOKENS>` - Maximum tokens to generate (default: 512)
- `--temperature <TEMPERATURE>` - Temperature for generation (default: 0.7)

## Architecture

The application follows a modular architecture:

- **`cli`** - Command-line interface using Clap
- **`download`** - Model downloading from HuggingFace mirrors
- **`influencer`** - LLM service using WatsonX with trait-based design
- **`error`** - Error handling with thiserror

## Testing

Run tests:

```bash
cargo test
```

Run with logging:

```bash
RUST_LOG=influence=debug cargo test
```

## Examples

See `examples/basic_usage.rs` for usage examples.

## Dependencies

- **clap** - CLI parsing
- **tokio** - Async runtime
- **reqwest** - HTTP client
- **watsonx-rs** - WatsonX SDK
- **anyrepair** - JSON repair
- **tracing** - Logging

## License

MIT
