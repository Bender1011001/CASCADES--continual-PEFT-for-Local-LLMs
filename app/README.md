# CASCADES Chat App

Local ChatGPT-like interface with personal memory via CASCADES continual learning.

## Quick Start

```bash
# Install dependencies
pip install fastapi uvicorn sse-starlette

# Run (loads model + starts web UI)
python -m app.server --model_id ./abliterated --adapter_weights cascades_v10_weights.pt

# Open http://localhost:8000 in your browser
```

## Architecture

- **Backend**: FastAPI server with OpenAI-compatible `/v1/chat/completions` endpoint
- **Frontend**: Single-page dark-theme chat UI
- **Storage**: SQLite for conversations, `.pt` files for adapter weights
- **Memory**: Export conversations → train on Colab T4 → hot-swap adapters
