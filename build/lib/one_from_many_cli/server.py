from __future__ import annotations
import json
import socket
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from .utils import canonical_model_id, get_experts_root, choose_random_expert
from .infer import load_model_and_tokenizer, build_prompt_from_messages, generate, LoadPrefs
from .utils import OFFLOAD_DIR

def find_free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port

def create_app(served_model_id: str) -> FastAPI:
    app = FastAPI()
    served_model_id = canonical_model_id(served_model_id)

    # Cache a single expert per process to avoid reloading each request
    _cache = {"expert": None, "tok": None, "mdl": None}

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{"id": served_model_id, "object": "model"}],
        }

    @app.get("/v1/health")
    async def health():
        return {"status": "ok", "model": served_model_id}

    def _load_any_expert():
        if _cache["mdl"] is not None:
            return _cache["expert"], _cache["tok"], _cache["mdl"]

        expert = choose_random_expert(served_model_id)
        if not expert:
            raise RuntimeError("No experts installed for model. Run `ai dynamoe run deca-ai/3-alpha-ultra` first.")
        expert_dir = Path(get_experts_root(served_model_id)) / expert
        prefs = LoadPrefs.auto(offload_folder=str(OFFLOAD_DIR))
        tok, mdl = load_model_and_tokenizer(str(expert_dir), prefs)
        _cache["expert"], _cache["tok"], _cache["mdl"] = expert, tok, mdl
        return expert, tok, mdl

    @app.post("/v1/completions")
    async def completions(req: Request):
        body = await req.json()
        prompt = body.get("prompt", "")
        temperature = float(body.get("temperature", 0.7))
        max_tokens = int(body.get("max_tokens", 256))
        stream = bool(body.get("stream", False))

        expert, tok, mdl = _load_any_expert()

        if not stream:
            text = "".join(generate(tok, mdl, prompt, max_new_tokens=max_tokens, temperature=temperature, stream=False))
            rid = f"cmpl-{uuid.uuid4().hex[:12]}"
            return JSONResponse({
                "id": rid,
                "object": "text_completion",
                "created": int(time.time()),
                "model": served_model_id,
                "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
            })

        def sse():
            rid = f"cmpl-{uuid.uuid4().hex[:12]}"
            for ch in generate(tok, mdl, prompt, max_new_tokens=max_tokens, temperature=temperature, stream=True):
                payload = {
                    "id": rid,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": served_model_id,
                    "choices": [{"text": ch, "index": 0, "finish_reason": None}],
                }
                yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        messages = body.get("messages", [])
        temperature = float(body.get("temperature", 0.7))
        max_tokens = int(body.get("max_tokens", 256))
        stream = bool(body.get("stream", False))

        prompt = build_prompt_from_messages(messages)
        expert, tok, mdl = _load_any_expert()

        rid = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        if not stream:
            text = "".join(generate(tok, mdl, prompt, max_new_tokens=max_tokens, temperature=temperature, stream=False))
            return JSONResponse({
                "id": rid,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": served_model_id,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop"
                }],
            })

        def sse():
            first_delta = {
                "id": rid,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": served_model_id,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None
                }],
            }
            yield f"data: {json.dumps(first_delta)}\n\n"

            for ch in generate(tok, mdl, prompt, max_new_tokens=max_tokens, temperature=temperature, stream=True):
                payload = {
                    "id": rid,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": served_model_id,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": ch},
                        "finish_reason": None
                    }],
                }
                yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    return app

def serve(model_id: str, host: str = "127.0.0.1", port: Optional[int] = None):
    port = port or find_free_port()
    app = create_app(model_id)
    print(f"Serving OpenAI-compatible API for {canonical_model_id(model_id)} on http://{host}:{port}")
    print("Endpoints: /v1/models, /v1/completions, /v1/chat/completions")
    uvicorn.run(app, host=host, port=port, log_level="info")