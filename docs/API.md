# Influence API Documentation

## Overview

Influence provides REST and Server-Sent Events (SSE) APIs for text generation and embeddings. The server supports both standard REST endpoints and Ollama-compatible endpoints.

## Base URL

```
http://localhost:3000
```

Configure the port with the `INFLUENCE_PORT` environment variable.

## Authentication

Currently, no authentication is required. All endpoints are publicly accessible.

## Endpoints

### 1. Generate Text (Non-Streaming)

Generate text completion without streaming.

**Endpoint:** `POST /v1/generate`

**Request Body:**
```json
{
  "prompt": "What is Rust?",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50
}
```

**Parameters:**
- `prompt` (string, required): Input text prompt
- `max_tokens` (integer, optional): Maximum tokens to generate (default: 512)
- `temperature` (float, optional): Sampling temperature 0.0-2.0 (default: 0.7)
- `top_p` (float, optional): Nucleus sampling threshold (default: 0.9)
- `top_k` (integer, optional): Top-k sampling limit (default: none)

**Response:**
```json
{
  "text": "Rust is a systems programming language...",
  "tokens_generated": 45
}
```

**Example with curl:**
```bash
curl -X POST http://localhost:3000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is Rust?",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

---

### 2. Generate Text (Streaming)

Generate text with Server-Sent Events (SSE) streaming.

**Endpoint:** `GET /v1/generate_stream`

**Query Parameters:**
- `prompt` (string, required): Input text prompt
- `max_tokens` (integer, optional): Maximum tokens to generate (default: 512)
- `temperature` (float, optional): Sampling temperature (default: 0.7)
- `top_p` (float, optional): Nucleus sampling threshold (default: 0.9)
- `top_k` (integer, optional): Top-k sampling limit (default: none)

**Response:** Server-Sent Events stream

Each event contains a JSON object:
```json
{"text": "Rust"}
{"text": " is"}
{"text": " a"}
...
```

**Example with curl:**
```bash
curl -N "http://localhost:3000/v1/generate_stream?prompt=What%20is%20Rust?&max_tokens=100&temperature=0.7"
```

**Example with JavaScript:**
```javascript
const eventSource = new EventSource(
  'http://localhost:3000/v1/generate_stream?prompt=What%20is%20Rust?&max_tokens=100'
);

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data.text);
};

eventSource.onerror = () => {
  eventSource.close();
};
```

---

### 3. Ollama-Compatible Generate (Non-Streaming)

Ollama-compatible text generation endpoint.

**Endpoint:** `POST /api/generate`

**Request Body:**
```json
{
  "model": "tinyllama",
  "prompt": "What is Rust?",
  "stream": false,
  "options": {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "num_predict": 100
  }
}
```

**Parameters:**
- `model` (string, optional): Model name (ignored, uses loaded model)
- `prompt` (string, required): Input text prompt
- `stream` (boolean, optional): Enable streaming (default: false)
- `options` (object, optional): Generation options
  - `temperature` (float): Sampling temperature
  - `top_p` (float): Nucleus sampling threshold
  - `top_k` (integer): Top-k sampling limit
  - `num_predict` (integer): Maximum tokens to generate
  - `repeat_penalty` (float): Repetition penalty

**Response:**
```json
{
  "model": "tinyllama",
  "created_at": "2024-01-30T00:00:00Z",
  "response": "Rust is a systems programming language...",
  "done": true
}
```

**Example with curl:**
```bash
curl -X POST http://localhost:3000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is Rust?",
    "stream": false,
    "options": {
      "temperature": 0.7,
      "num_predict": 100
    }
  }'
```

---

### 4. Ollama-Compatible Generate (Streaming)

Ollama-compatible streaming text generation.

**Endpoint:** `POST /api/generate`

**Request Body:**
```json
{
  "model": "tinyllama",
  "prompt": "What is Rust?",
  "stream": true,
  "options": {
    "temperature": 0.7,
    "num_predict": 100
  }
}
```

**Response:** NDJSON (Newline-Delimited JSON) stream

Each line contains a JSON object:
```json
{"model":"tinyllama","created_at":"2024-01-30T00:00:00Z","response":"Rust","done":false}
{"model":"tinyllama","created_at":"2024-01-30T00:00:00Z","response":" is","done":false}
...
{"model":"tinyllama","created_at":"2024-01-30T00:00:00Z","response":"","done":true}
```

**Example with curl:**
```bash
curl -X POST http://localhost:3000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is Rust?",
    "stream": true,
    "options": {
      "temperature": 0.7,
      "num_predict": 100
    }
  }'
```

---

### 5. Ollama-Compatible Tags

List available models (Ollama-compatible).

**Endpoint:** `GET /api/tags`

**Response:**
```json
{
  "models": [
    {
      "name": "loaded-model",
      "modified_at": "2024-01-30T00:00:00Z",
      "size": 0
    }
  ]
}
```

**Example with curl:**
```bash
curl http://localhost:3000/api/tags
```

---

### 6. Ollama-Compatible Embeddings

Generate embeddings for text (BERT models only).

**Endpoint:** `POST /api/embeddings`

**Request Body:**
```json
{
  "model": "bert",
  "prompt": "What is Rust?"
}
```

**Response:**
```json
{
  "embedding": [0.123, -0.456, 0.789, ...]
}
```

**Example with curl:**
```bash
curl -X POST http://localhost:3000/api/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is Rust?"
  }'
```

---

## Error Responses

All endpoints return errors in the following format:

```json
{
  "error": "Error message description"
}
```

**Common HTTP Status Codes:**
- `200 OK`: Success
- `400 Bad Request`: Invalid request parameters
- `500 Internal Server Error`: Server error during processing

---

## Rate Limiting

Currently, no rate limiting is implemented. All requests are processed sequentially.

---

## Examples

### Python Example (Non-Streaming)

```python
import requests

response = requests.post(
    'http://localhost:3000/v1/generate',
    json={
        'prompt': 'What is Rust?',
        'max_tokens': 100,
        'temperature': 0.7
    }
)

result = response.json()
print(result['text'])
```

### Python Example (Streaming)

```python
import requests
import json

response = requests.get(
    'http://localhost:3000/v1/generate_stream',
    params={
        'prompt': 'What is Rust?',
        'max_tokens': 100,
        'temperature': 0.7
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        data = json.loads(line.decode('utf-8').replace('data: ', ''))
        print(data['text'], end='', flush=True)
```

### Node.js Example (Streaming)

```javascript
const EventSource = require('eventsource');

const url = new URL('http://localhost:3000/v1/generate_stream');
url.searchParams.append('prompt', 'What is Rust?');
url.searchParams.append('max_tokens', '100');

const eventSource = new EventSource(url.toString());

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  process.stdout.write(data.text);
};

eventSource.onerror = () => {
  eventSource.close();
  console.log('\nDone');
};
```

---

## Starting the Server

Start the Influence server with:

```bash
influence serve --model-path /path/to/model
```

Or with environment variables:

```bash
export INFLUENCE_MODEL_PATH=/path/to/model
export INFLUENCE_PORT=3000
influence serve
```

---

## Notes

- The server loads a single model at startup specified by `--model-path` or `INFLUENCE_MODEL_PATH`
- All generation requests use the loaded model
- Streaming endpoints provide real-time token generation
- The Ollama-compatible endpoints allow using Influence as a drop-in replacement for Ollama
