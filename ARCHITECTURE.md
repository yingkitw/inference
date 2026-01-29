# Architecture

## Overview

Influence follows a modular, trait-based architecture for maintainability and testability.

## Module Structure

```
influence/
├── src/
│   ├── main.rs           # Entry point, CLI routing
│   ├── cli.rs            # Command definitions
│   ├── download.rs       # Model download logic
│   ├── influencer.rs     # LLM service
│   └── error.rs          # Error types
├── examples/
│   └── basic_usage.rs    # Usage examples
└── tests/                # Integration tests
```

## Component Diagram

```
┌─────────────────────────────────────────┐
│              CLI (main.rs)              │
│         Command Routing Layer           │
└─────────────────┬───────────────────────┘
                  │
        ┌─────────┼─────────┐
        │         │         │
        ▼         ▼         ▼
   ┌────────┐ ┌──────┐ ┌──────────┐
   │Download│ │Serve │ │Generate  │
   │Module  │ │Module│ │Module    │
   └────┬───┘ └───┬──┘ └────┬─────┘
        │         │         │
        │    ┌────▼─────────▼────┐
        │    │  LlmService Trait │
        │    │  (influencer.rs)  │
        │    └────────┬───────────┘
        │             │
        │    ┌────────▼───────────┐
        │    │ WatsonXService     │
        │    │ Implementation     │
        │    └────────────────────┘
        │
   ┌────▼──────────┐
   │ HTTP Client   │
   │ (reqwest)     │
   └───────────────┘
```

## Key Design Patterns

### 1. Trait-Based Service Layer

The `LlmService` trait abstracts LLM operations:

```rust
pub trait LlmService {
    async fn generate_text(&self, prompt: &str, max_tokens: usize, temperature: f32) -> Result<String>;
    async fn generate_stream(&self, prompt: &str, max_tokens: usize, temperature: f32) -> Result<()>;
}
```

Benefits:
- Easy to mock for testing
- Can swap implementations
- Clear contract for LLM operations

### 2. Error Handling

Centralized error types using `thiserror`:

```rust
pub enum InfluenceError {
    DownloadError(String),
    ModelNotFound(String),
    InvalidConfig(String),
    LlmError(String),
    // ...
}
```

### 3. Async/Await

All I/O operations are async:
- HTTP downloads
- File operations
- LLM API calls

### 4. Streaming

Real-time output for LLM generation:
- Uses WatsonX streaming API
- Displays tokens as they're generated
- Better UX for long generations

## Data Flow

### Download Command

```
CLI Input → Parse Args → Determine Output Dir → Create HTTP Client
    → For each file:
        → Download with progress
        → Save to disk
    → Complete
```

### Generate Command

```
CLI Input → Parse Args → Load Config → Create WatsonX Client
    → Format Prompt (Granite-specific)
    → Stream Generation
    → Display Output
    → Complete
```

## Configuration Management

Configuration is loaded from environment variables:
- `WATSONX_API_KEY`
- `WATSONX_PROJECT_ID`
- `WATSONX_MODEL_ID`

No configuration files to keep it simple (KISS principle).

## Logging

Uses `tracing` for structured logging:
- Info level for user-facing messages
- Debug level for internal operations
- Error level for failures

## Testing Strategy

### Unit Tests
- Individual function testing
- Mock external dependencies
- Test error cases

### Integration Tests
- End-to-end command testing
- Mock HTTP servers for downloads
- Mock LLM service for generation

### Test Organization
```
tests/
├── download_tests.rs
├── influencer_tests.rs
└── integration_tests.rs
```

## Dependencies

### Core
- `tokio` - Async runtime
- `clap` - CLI parsing
- `reqwest` - HTTP client
- `watsonx-rs` - WatsonX SDK

### Utilities
- `tracing` - Logging
- `indicatif` - Progress bars
- `directories` - Path management
- `serde` - Serialization

### Error Handling
- `anyhow` - Error context
- `thiserror` - Error types

## Performance Considerations

1. **Streaming Downloads**: Uses chunked streaming to avoid loading entire files in memory
2. **Async I/O**: Non-blocking operations for better concurrency
3. **Progress Tracking**: Minimal overhead with indicatif
4. **Lazy Loading**: Only loads what's needed

## Security Considerations

1. **API Keys**: Must be in environment variables, never hardcoded
2. **HTTPS**: All downloads use HTTPS
3. **File Permissions**: Respects system file permissions
4. **Input Validation**: CLI arguments are validated

## Extensibility

Easy to extend:
- Add new commands: Extend `Commands` enum
- Add new LLM providers: Implement `LlmService` trait
- Add new model formats: Extend `get_model_files()`
- Add new mirrors: Pass as CLI argument

## Maintenance

- **Single Responsibility**: Each module has one clear purpose
- **DRY**: No code duplication
- **KISS**: Simple, straightforward implementations
- **Documentation**: Inline docs for public APIs
- **Tests**: Good coverage for maintainability
