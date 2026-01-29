# Integration Tests

This directory contains integration tests for the Influence API server.

## Prerequisites

Before running the integration tests, you need to:

1. **Start the Influence server** with a model loaded:

```bash
# Download a model first (if you haven't already)
influence download TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Start the server
influence serve --model-path ~/.cache/influence/models/TinyLlama--TinyLlama-1.1B-Chat-v1.0
```

Or use environment variables:

```bash
export INFLUENCE_MODEL_PATH=~/.cache/influence/models/TinyLlama--TinyLlama-1.1B-Chat-v1.0
export INFLUENCE_PORT=3000
influence serve
```

2. **Wait for the server to be ready** - You should see:
```
Server running on http://0.0.0.0:3000
```

## Running the Tests

Once the server is running, execute the integration tests in a separate terminal:

```bash
# Run all integration tests
cargo test --test integration_test

# Run a specific test
cargo test --test integration_test test_v1_generate_endpoint

# Run with output
cargo test --test integration_test -- --nocapture
```

## Test Coverage

The integration tests cover:

### Standard REST Endpoints
- ✅ `POST /v1/generate` - Non-streaming text generation
- ✅ `GET /v1/generate_stream` - Streaming text generation with SSE
- ✅ Parameter validation and error handling

### Ollama-Compatible Endpoints
- ✅ `POST /api/generate` (stream=false) - Non-streaming generation
- ✅ `POST /api/generate` (stream=true) - Streaming with NDJSON
- ✅ `GET /api/tags` - List available models

## Test Behavior

If the server is **not running**, the tests will:
- Print a helpful message indicating the server needs to be started
- Skip the test gracefully (not fail)

If the server **is running**, the tests will:
- Verify HTTP status codes
- Validate response JSON structure
- Check streaming behavior (SSE and NDJSON)
- Ensure proper content types
- Validate timeout handling

## Example Test Output

```
running 8 tests
test test_ollama_generate_non_stream ... ok
test test_ollama_generate_stream ... ok
test test_ollama_tags_endpoint ... ok
test test_v1_generate_endpoint ... ok
test test_v1_generate_invalid_request ... ok
test test_v1_generate_stream_endpoint ... ok
test test_v1_generate_with_parameters ... ok

test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

## Troubleshooting

### Connection Refused
```
Server not running. Start with: influence serve --model-path <path>
Skipping integration test.
```
**Solution:** Start the Influence server before running tests.

### Timeout Errors
If tests timeout, the model might be too slow or the server is overloaded.
**Solution:** 
- Use a smaller/faster model
- Reduce `max_tokens` in test requests
- Increase timeout duration in tests

### Port Already in Use
If port 3000 is already in use:
```bash
export INFLUENCE_PORT=3001
influence serve
```

Then update the test URLs to use port 3001.

## Adding New Tests

To add new integration tests:

1. Add a new test function in `tests/integration_test.rs`
2. Use the `#[tokio::test]` attribute
3. Handle connection errors gracefully (check for `e.is_connect()`)
4. Add assertions for expected behavior

Example:
```rust
#[tokio::test]
async fn test_my_new_endpoint() {
    let client = reqwest::Client::new();
    
    let response = client
        .get("http://localhost:3000/my/endpoint")
        .send()
        .await;

    match response {
        Ok(resp) => {
            assert_eq!(resp.status(), 200);
            // Add more assertions
        }
        Err(e) => {
            if e.is_connect() {
                eprintln!("Server not running. Skipping integration test.");
            } else {
                panic!("Request failed: {}", e);
            }
        }
    }
}
```

## CI/CD Integration

For CI/CD pipelines, you can:

1. Start the server in the background:
```bash
influence serve --model-path $MODEL_PATH &
SERVER_PID=$!
sleep 5  # Wait for server to start
```

2. Run the tests:
```bash
cargo test --test integration_test
```

3. Stop the server:
```bash
kill $SERVER_PID
```

## Notes

- Tests use a 30-second timeout for streaming endpoints
- Tests validate JSON structure but not specific content (model outputs vary)
- Tests are designed to be non-destructive and read-only
- Multiple test runs can be executed without restarting the server
