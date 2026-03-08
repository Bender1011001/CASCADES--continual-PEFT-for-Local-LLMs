# CASCADES Chat App

## Status
- **Working**: FastAPI server, chat UI, conversation CRUD, SQLite storage, memory flagging, JSONL export
- **Pending**: Model inference (needs GPU), adapter hot-swap (needs trained weights), Self-Synthesizer

## Tech Stack
- FastAPI + uvicorn (backend)
- Vanilla HTML/CSS/JS (frontend, single index.html)
- SQLite (conversation storage)
- PyTorch + bitsandbytes (NF4 inference)

## Key Files
- `server.py` — FastAPI server with OpenAI-compatible endpoints, `--no-model` dev mode
- `model_loader.py` — NF4 model loading, CASCADES adapter injection, streaming generation, hot-swap
- `conversation_store.py` — SQLite CRUD, auto-titling, memory flagging, JSONL export
- `prepare_memory.py` — Conversation → CASCADES training format (ChatML + think tags)
- `static/index.html` — Dark-theme chat UI with sidebar, streaming, memory flagging

## Architecture Quirks
- Server supports `--no-model` for UI development without GPU
- Hot-swap is thread-safe via `_lock` in CASCADESModel
- Chat completions endpoint auto-saves messages and auto-titles conversations
- JSONL export prioritizes flagged memories (sorted first in training data)

## Anti-Patterns (DO NOT)
- Do NOT train on raw chat logs (causes Conversational Collapse) — use Self-Synthesizer to extract Q&A pairs
- Do NOT load the base model twice for hot-swap — only swap adapter weights (~5MB)
- Do NOT modify the base model weights — they stay frozen in NF4

## Build / Verify
```
# Dev mode (no GPU needed)
python -m app.server --no-model --port 8000

# Full mode (needs GPU + model)
python -m app.server --model_id ./abliterated --adapter_weights cascades_v10_weights.pt
```
