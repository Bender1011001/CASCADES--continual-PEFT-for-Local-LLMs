"""
CASCADES Chat — FastAPI server with OpenAI-compatible endpoints.

Endpoints:
    POST /v1/chat/completions  — Stream or batch chat completion
    POST /v1/reload            — Hot-swap adapter weights
    GET  /v1/models            — List available model info
    GET  /v1/conversations     — List saved conversations
    POST /v1/conversations     — Create new conversation
    GET  /v1/conversations/{id}— Get conversation with messages
    DELETE /v1/conversations/{id} — Delete a conversation
    PATCH /v1/conversations/{id}  — Update title/pin
    POST /v1/messages/{id}/flag   — Flag message for memory
    GET  /v1/memory/stats      — Memory/adapter stats
    POST /v1/memory/export     — Export conversations for training
    GET  /                     — Serve the chat UI

Usage:
    python -m app.server --model_id ./abliterated
"""

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.conversation_store import ConversationStore

app = FastAPI(title="CASCADES Chat", version="1.0.0")

# ── Globals (initialized in main) ─────────────────────────────────
store = ConversationStore()
model = None  # Lazy-loaded CASCADESModel

STATIC_DIR = Path(__file__).parent / "static"


# ── Request/Response Models ────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = "user"
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "cascades"
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    stream: bool = False
    conversation_id: Optional[str] = None


class ConversationCreate(BaseModel):
    title: str = "New Chat"


class ConversationUpdate(BaseModel):
    title: Optional[str] = None
    pinned: Optional[bool] = None


class ReloadRequest(BaseModel):
    adapter_path: str


class ExportRequest(BaseModel):
    since: Optional[str] = None
    flagged_only: bool = False
    output_path: str = "app/data/training_export.jsonl"


# ── Chat Completion ────────────────────────────────────────────────

def _make_sse_chunk(content: str, finish_reason: Optional[str] = None) -> str:
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "cascades",
        "choices": [{
            "index": 0,
            "delta": {"content": content} if content else {},
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(chunk)}\n\n"


async def _stream_response(req: ChatCompletionRequest):
    """Generate SSE stream compatible with OpenAI API."""
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    # Initial chunk with role
    yield _make_sse_chunk("")

    full_response = []
    loop = asyncio.get_event_loop()

    # Run synchronous generator in thread pool
    def _generate():
        return list(model.generate_stream(
            messages,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        ))

    chunks = await loop.run_in_executor(None, _generate)

    for chunk_text in chunks:
        full_response.append(chunk_text)
        yield _make_sse_chunk(chunk_text)
        await asyncio.sleep(0)  # Yield control

    yield _make_sse_chunk("", finish_reason="stop")
    yield "data: [DONE]\n\n"

    # Save assistant response to conversation
    if req.conversation_id:
        response_text = "".join(full_response)
        store.add_message(req.conversation_id, "assistant", response_text)


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if model is None or not model._loaded:
        raise HTTPException(503, "Model not loaded yet")

    # Auto-save user message
    if req.conversation_id:
        # Save the last user message
        user_msgs = [m for m in req.messages if m.role == "user"]
        if user_msgs:
            store.add_message(req.conversation_id, "user", user_msgs[-1].content)
            # Auto-title on first message
            conv = store.get_conversation(req.conversation_id)
            if conv and conv["title"] == "New Chat":
                store.auto_title(req.conversation_id)

    if req.stream:
        return StreamingResponse(
            _stream_response(req),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    response_text = model.generate(messages, max_new_tokens=req.max_tokens)

    if req.conversation_id:
        store.add_message(req.conversation_id, "assistant", response_text)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "cascades",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


# ── Model Info ─────────────────────────────────────────────────────

@app.get("/v1/models")
async def list_models():
    status = model.status if model else {"loaded": False}
    return {
        "object": "list",
        "data": [{
            "id": "cascades",
            "object": "model",
            "owned_by": "local",
            **status,
        }],
    }


# ── Adapter Hot-Swap ──────────────────────────────────────────────

@app.post("/v1/reload")
async def reload_adapters(req: ReloadRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    result = model.swap_adapters(req.adapter_path)
    return result


# ── Conversations ─────────────────────────────────────────────────

@app.get("/v1/conversations")
async def get_conversations(limit: int = 50, offset: int = 0):
    return store.list_conversations(limit=limit, offset=offset)


@app.post("/v1/conversations")
async def create_conversation(req: ConversationCreate):
    return store.create_conversation(title=req.title)


@app.get("/v1/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    conv = store.get_conversation(conv_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    messages = store.get_messages(conv_id)
    return {**conv, "messages": messages}


@app.patch("/v1/conversations/{conv_id}")
async def update_conversation(conv_id: str, req: ConversationUpdate):
    conv = store.get_conversation(conv_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    store.update_conversation(conv_id, title=req.title, pinned=req.pinned)
    return {"status": "ok"}


@app.delete("/v1/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    store.delete_conversation(conv_id)
    return {"status": "ok"}


# ── Memory ────────────────────────────────────────────────────────

@app.post("/v1/messages/{msg_id}/flag")
async def flag_message(msg_id: str, flagged: bool = True):
    store.flag_message(msg_id, flagged=flagged)
    return {"status": "ok", "flagged": flagged}


@app.get("/v1/memory/stats")
async def memory_stats():
    stats = store.stats()
    if model:
        stats["model"] = model.status
    return stats


@app.post("/v1/memory/export")
async def export_for_training(req: ExportRequest):
    count = store.export_training_data(
        Path(req.output_path),
        since=req.since,
        flagged_only=req.flagged_only,
    )
    return {"status": "ok", "examples_exported": count, "path": req.output_path}


# ── Static Files (Chat UI) ────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        return HTMLResponse("<h1>CASCADES Chat</h1><p>UI not found. Place index.html in app/static/</p>")
    return HTMLResponse(index.read_text(encoding="utf-8"))


# Mount static assets
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Entrypoint ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CASCADES Chat Server")
    parser.add_argument("--model_id", type=str, default="./abliterated")
    parser.add_argument("--adapter_weights", type=str, default=None)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-model", action="store_true", help="Start without loading model (UI dev mode)")
    args = parser.parse_args()

    global model

    if not args.no_model:
        from app.model_loader import CASCADESModel
        model = CASCADESModel(
            model_id=args.model_id,
            adapter_weights=args.adapter_weights,
            rank=args.rank,
        )
        model.load()
    else:
        print("Starting in UI dev mode (no model loaded)")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
