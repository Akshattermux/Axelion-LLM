"""
OpenAI-Compatible REST API Server for Axelion LLM.
Supports /v1/chat/completions (streaming and non-streaming), /v1/completions, and /v1/models.
Compatible with OpenAI SDK, LangChain, AutoGen, CrewAI, and web interfaces.
"""

import json
import os
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

# Ensure repo root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from axelion.config import ModelConfig
from axelion.model import Axelion
from axelion.tokenizer import AxelionTokenizer
from axelion.generation import stream_generate, generate_text
from axelion.agent import AxelionAgent, ToolRegistry

app = FastAPI(
    title="Axelion LLM API",
    description="High-performance, OpenAI-compatible API for the Axelion decoder-only Transformer.",
    version="1.0.0",
)

MAX_PROMPT_CHARS = int(os.getenv("AXELION_MAX_PROMPT_CHARS", "24000"))
MAX_NEW_TOKENS = int(os.getenv("AXELION_MAX_NEW_TOKENS", "2048"))
API_KEY = os.getenv("AXELION_API_KEY")
ALLOWED_ORIGINS = [origin.strip() for origin in os.getenv("AXELION_ALLOWED_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000").split(",") if origin.strip()]
_dashboard_path = Path(__file__).with_name("dashboard.html")
_stats = Counter()
_started_at = time.time()

# Enable CORS for external agents and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=bool(API_KEY),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
_model: Optional[Axelion] = None
_tokenizer: Optional[AxelionTokenizer] = None
_device: str = "cpu"
_model_name: str = "axelion"


def require_api_key(authorization: Optional[str] = Header(default=None)):
    """Protect inference routes when AXELION_API_KEY is configured."""
    if not API_KEY:
        return
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Missing or invalid API key")


def _validate_generation(prompt: str, max_tokens: int, temperature: float, top_p: float):
    if len(prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(status_code=413, detail=f"Prompt exceeds {MAX_PROMPT_CHARS} characters")
    if max_tokens < 1 or max_tokens > MAX_NEW_TOKENS:
        raise HTTPException(status_code=422, detail=f"max_tokens must be between 1 and {MAX_NEW_TOKENS}")
    if temperature < 0 or temperature > 2:
        raise HTTPException(status_code=422, detail="temperature must be between 0 and 2")
    if top_p <= 0 or top_p > 1:
        raise HTTPException(status_code=422, detail="top_p must be greater than 0 and at most 1")


def get_model():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    return _model, _tokenizer, _device


def init_model(
    checkpoint_path: Optional[str] = None,
    size: str = "micro",
    device: Optional[str] = None,
):
    global _model, _tokenizer, _device, _model_name
    import torch

    _device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = AxelionTokenizer()

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=_device, weights_only=False)
        cfg = ckpt.get("config", ModelConfig.create_micro(vocab_size=_tokenizer.vocab_size))
        cfg.vocab_size = max(cfg.vocab_size, _tokenizer.vocab_size)
        _model = Axelion(cfg).to(_device)
        _model.load_checkpoint_weights(ckpt["model"])
        _model_name = os.path.basename(checkpoint_path).replace(".pt", "")
    else:
        preset_map = {
            "micro": ModelConfig.create_micro,
            "mini": ModelConfig.create_mini_gpt,
            "medium": ModelConfig.create_medium,
            "large": ModelConfig.create_large,
        }
        cfg_fn = preset_map.get(size.lower(), ModelConfig.create_micro)
        cfg = cfg_fn(vocab_size=_tokenizer.vocab_size)
        print(f"Initializing uncheckpointed {size} model for serving on {_device}...")
        _model = Axelion(cfg).to(_device)
        _model_name = f"axelion-{size}"

    _model.eval()
    print(f"Model ready. Parameters: {_model.get_num_params() / 1e6:.2f}M | Device: {_device}")


# --- Pydantic Request / Response Schemas ---
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "axelion"
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=0.7, ge=0, le=2)
    top_p: Optional[float] = Field(default=0.9, gt=0, le=1)
    max_tokens: Optional[int] = Field(default=256, ge=1, le=MAX_NEW_TOKENS)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None


class CompletionRequest(BaseModel):
    model: Optional[str] = "axelion"
    prompt: str
    temperature: Optional[float] = Field(default=0.7, ge=0, le=2)
    top_p: Optional[float] = Field(default=0.9, gt=0, le=1)
    max_tokens: Optional[int] = Field(default=256, ge=1, le=MAX_NEW_TOKENS)
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None


class AgentRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_PROMPT_CHARS)
    max_steps: int = Field(default=3, ge=1, le=10)


@app.get("/health")
def health_check():
    if _model is None:
        return {"status": "starting", "model": _model_name, "uptime_seconds": int(time.time() - _started_at)}
    model, _, device = get_model()
    return {
        "status": "healthy",
        "model": _model_name,
        "parameters": model.get_num_params(),
        "device": device,
        "max_seq_len": model.config.max_seq_len,
        "uptime_seconds": int(time.time() - _started_at),
    }


@app.get("/ready")
def readiness_check():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    return {"ready": True, "model": _model_name}


@app.get("/metrics")
def metrics(_: None = Depends(require_api_key)):
    return {"uptime_seconds": int(time.time() - _started_at), "requests": dict(_stats)}


@app.get("/dashboard", include_in_schema=False)
def dashboard():
    if not _dashboard_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard asset not found")
    return FileResponse(_dashboard_path)


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": _model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "axelion",
                "root": _model_name,
                "parent": None,
                "permission": [],
            },
            {
                "id": "axelion",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "axelion",
            },
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest, _: None = Depends(require_api_key)):
    model, tokenizer, device = get_model()

    # Convert pydantic messages to dict
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    _validate_generation(
        formatted_prompt,
        req.max_tokens if req.max_tokens is not None else 256,
        req.temperature if req.temperature is not None else 0.7,
        req.top_p if req.top_p is not None else 0.9,
    )
    _stats["chat_completions"] += 1
    stop_ids = [tokenizer.end_id, tokenizer.pad_id]

    req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())

    if req.stream:
        def stream_events():
            for chunk in stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=formatted_prompt,
                max_new_tokens=req.max_tokens or 256,
                temperature=req.temperature if req.temperature is not None else 0.7,
                top_p=req.top_p if req.top_p is not None else 0.9,
                stop_token_ids=stop_ids,
                device=device,
            ):
                payload = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": req.model or _model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(payload)}\n\n"

            # Final chunk indicating end
            done_payload = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": req.model or _model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done_payload)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_events(), media_type="text/event-stream")

    # Non-streaming response
    generated_content = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=formatted_prompt,
        max_new_tokens=req.max_tokens or 256,
        temperature=req.temperature if req.temperature is not None else 0.7,
        top_p=req.top_p if req.top_p is not None else 0.9,
        stop_token_ids=stop_ids,
        device=device,
    )

    prompt_tokens = len(tokenizer.encode(formatted_prompt))
    completion_tokens = len(tokenizer.encode(generated_content))

    return {
        "id": req_id,
        "object": "chat.completion",
        "created": created_ts,
        "model": req.model or _model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": generated_content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@app.post("/v1/completions")
def completions(req: CompletionRequest, _: None = Depends(require_api_key)):
    model, tokenizer, device = get_model()
    _validate_generation(
        req.prompt,
        req.max_tokens if req.max_tokens is not None else 256,
        req.temperature if req.temperature is not None else 0.7,
        req.top_p if req.top_p is not None else 0.9,
    )
    _stats["completions"] += 1

    req_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())
    stop_ids = [tokenizer.pad_id]

    if req.stream:
        def stream_events():
            for chunk in stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=req.prompt,
                max_new_tokens=req.max_tokens or 256,
                temperature=req.temperature if req.temperature is not None else 0.7,
                top_p=req.top_p if req.top_p is not None else 0.9,
                stop_token_ids=stop_ids,
                device=device,
            ):
                payload = {
                    "id": req_id,
                    "object": "text_completion",
                    "created": created_ts,
                    "model": req.model or _model_name,
                    "choices": [
                        {
                            "text": chunk,
                            "index": 0,
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_events(), media_type="text/event-stream")

    text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=req.prompt,
        max_new_tokens=req.max_tokens or 256,
        temperature=req.temperature if req.temperature is not None else 0.7,
        top_p=req.top_p if req.top_p is not None else 0.9,
        stop_token_ids=stop_ids,
        device=device,
    )

    prompt_tokens = len(tokenizer.encode(req.prompt))
    completion_tokens = len(tokenizer.encode(text))

    return {
        "id": req_id,
        "object": "text_completion",
        "created": created_ts,
        "model": req.model or _model_name,
        "choices": [
            {
                "text": text,
                "index": 0,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@app.post("/v1/agent/run")
def agent_run(req: AgentRequest, _: None = Depends(require_api_key)):
    """Run a model-backed agent loop; custom deployments can supply their own ToolRegistry."""
    model, tokenizer, device = get_model()
    _stats["agent_runs"] += 1
    agent = AxelionAgent(
        model=model,
        tokenizer=tokenizer,
        registry=ToolRegistry(),
        max_steps=req.max_steps,
        device=device,
    )
    result = agent.run(req.query, verbose=False)
    return {
        "object": "agent.run",
        "query": req.query,
        "answer": result["final_answer"],
        "completed": result["completed"],
        "steps": [step.__dict__ for step in result["steps"]],
    }
