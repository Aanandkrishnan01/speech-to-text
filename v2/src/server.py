"""
Multi-user web server for realtime speech-to-text.

Architecture:
- Only the default ASR backend is loaded at app startup. No VAD model.
- ASR backends are kept in a per-process pool, lazy-loaded on demand. Each
  WebSocket session can pick its own (backend, model) combination — different
  users can use different ASR backends concurrently. The pool is capped (LRU
  eviction) so memory doesn't grow without bound.
- Inference is serialized through a single asyncio.Lock so concurrent users'
  audio chunks are queued one after another rather than racing on the GPU.
- Browser sends raw float32 PCM @ config.sample_rate (typically 16 kHz)
  as binary WebSocket frames.

Run:
    python main.py serve
"""

from __future__ import annotations

import asyncio
import json
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .asr_backends import ASRBackend, build_asr_backend, make_asr_backend
from .config import get_config
from .logger import get_logger
from .session import Session

log = get_logger(__name__)

# How many ASR backends to keep loaded at once. LRU-evicted beyond this.
ASR_POOL_MAX = 3

_state: dict = {
    "asr_pool": None,         # OrderedDict[(backend, model_name) -> ASRBackend], LRU
    "asr_pool_lock": None,    # protects asr_pool dict mutations
    "model_lock": None,       # serializes model inference across sessions
    "default_asr_key": None,  # ("backend", "model_name") of the startup-default ASR
    "ready": False,           # set once default ASR is loaded
}

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load only the default ASR backend on startup. Nothing else."""
    log.info("Server starting — loading default ASR backend...")
    cfg = get_config()
    default_asr = build_asr_backend(cfg, use_realtime_model=True)
    _state["asr_pool"] = OrderedDict()
    _state["asr_pool_lock"] = asyncio.Lock()
    _state["model_lock"] = asyncio.Lock()
    key = (default_asr.name, default_asr.model_name)
    _state["asr_pool"][key] = default_asr
    _state["default_asr_key"] = key
    _state["ready"] = True
    log.info("Server ready — ASR only: %s/%s",
             default_asr.name, default_asr.model_name)
    yield
    log.info("Server shutting down")


app = FastAPI(title="Speech-To-Text Realtime", lifespan=lifespan)

# Static files (HTML/JS/CSS for the UI)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


_SECURE_HOSTS = {"localhost", "127.0.0.1", "::1"}


@app.get("/")
async def index(request: Request):
    """Serve the single-page UI.

    Browsers gate `getUserMedia` to "secure contexts" — `http://localhost` and
    `http://127.0.0.1` qualify, `http://0.0.0.0` and LAN IPs do not. To save
    users from copy-pasting the uvicorn bind line (`http://0.0.0.0:8000`),
    redirect the entry-point URL to localhost when the request was made to
    a non-secure host. Once the browser lands on localhost, every subsequent
    request (WebSocket, static, health) stays there.
    """
    host_header = request.headers.get("host", "")
    host_only = host_header.split(":")[0].lower().strip("[]")
    if host_only and host_only not in _SECURE_HOSTS:
        port_suffix = ""
        if ":" in host_header:
            port_suffix = ":" + host_header.split(":", 1)[1]
        target = f"http://localhost{port_suffix}/"
        log.info("Redirecting %s → %s (insecure host for getUserMedia)",
                 host_header, target)
        return RedirectResponse(url=target, status_code=307)

    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return {"error": f"static/index.html not found at {STATIC_DIR}"}
    return FileResponse(str(index_path))


@app.get("/health")
async def health():
    cfg = get_config()
    default_key = _state.get("default_asr_key")
    return {
        "status": "ready" if _state.get("ready") else "loading",
        "asr_backend": default_key[0] if default_key else None,
        "asr_model": default_key[1] if default_key else None,
        "sample_rate": cfg.sample_rate,
        "chunk_duration": cfg.chunk_duration,
    }


@app.get("/models")
async def list_models():
    """Return available + currently-loaded ASR backends/models for the UI."""
    cfg = get_config()
    available = cfg.available_models
    loaded = []
    pool = _state.get("asr_pool")
    if pool is not None:
        loaded = [{"backend": b, "model": m} for (b, m) in pool.keys()]
    default_key = _state.get("default_asr_key")
    return {
        "available": available,
        "loaded": loaded,
        "default": (
            {"backend": default_key[0], "model": default_key[1]}
            if default_key else None
        ),
        "pool_max": ASR_POOL_MAX,
    }


async def _get_or_load_asr(backend: str, model: str) -> ASRBackend:
    """
    Look up an ASR backend in the pool; load it if missing. LRU-evicts the
    oldest entry when the pool grows beyond ASR_POOL_MAX (excluding the
    default backend, which is never evicted).
    """
    backend = (backend or "").lower()
    key = (backend, model)
    pool = _state["asr_pool"]
    pool_lock = _state["asr_pool_lock"]

    async with pool_lock:
        if key in pool:
            pool.move_to_end(key)  # mark as most-recently used
            return pool[key]

    # Slow path — load the backend without holding the pool lock so other
    # readers aren't blocked. Hold an inflight lock so two requests for the
    # same key don't load it twice.
    log.info("Loading ASR backend on demand: %s / %s", backend, model)
    cfg = get_config()
    loop = asyncio.get_running_loop()
    asr = await loop.run_in_executor(None, make_asr_backend, backend, model, cfg)

    async with pool_lock:
        # Re-check after load in case a concurrent loader beat us.
        if key in pool:
            pool.move_to_end(key)
            return pool[key]
        pool[key] = asr
        pool.move_to_end(key)
        # LRU evict if pool exceeds capacity. Default ASR is never evicted.
        default_key = _state.get("default_asr_key")
        while len(pool) > ASR_POOL_MAX:
            evict_key, _ = next(iter(pool.items()))
            if evict_key == default_key:
                # default is at the front and we don't evict it; bump it
                # to the end so we look at the next candidate.
                pool.move_to_end(evict_key)
                evict_key, _ = next(iter(pool.items()))
                if evict_key == default_key:
                    break  # only the default is in the pool — stop
            evicted = pool.pop(evict_key)
            log.info("Evicting ASR from pool: %s / %s", *evict_key)
            del evicted
        return asr


@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket):
    """
    WebSocket protocol (streaming).
      Client → Server:
        - binary frames: raw float32 PCM, mono, sample_rate Hz
        - text JSON:
          {"type": "select_model", "backend": "...", "model": "..."}
          {"type": "stop"}
          {"type": "ping"}
      Server → Client (text JSON):
        {"type": "ready", ...}
        {"type": "loading_model", ...}  / {"type": "model_changed", ...}
        {"type": "interim",  "id": "...", "text": "...", "start": ..., "duration": ...}
        {"type": "segment",  "id": "...", "text": "...", "speaker": "...",
         "start": ..., "end": ..., "is_final": true}
        {"type": "error", "severity": "...", "message": "..."}
        {"type": "closed", "total_segments": N, "transcript_path": "..."}
    """
    await websocket.accept()

    model_lock: asyncio.Lock = _state.get("model_lock")
    if not _state.get("ready") or model_lock is None:
        await websocket.send_json({"type": "error",
                                   "message": "Server not ready yet."})
        await websocket.close()
        return

    cfg = get_config()
    default_key = _state["default_asr_key"]
    default_asr = _state["asr_pool"][default_key]

    async def send_emission(result):
        """Forward a StreamingResult as an interim or final WS message."""
        if result.is_final:
            await websocket.send_json({
                "type": "segment",
                "id": result.id,
                "text": result.text,
                "start": round(result.start, 2),
                "end": round(result.start + result.duration, 2),
                "is_final": True,
            })
        else:
            await websocket.send_json({
                "type": "interim",
                "id": result.id,
                "text": result.text,
                "start": round(result.start, 2),
                "duration": round(result.duration, 2),
            })

    session = Session(
        asr_backend=default_asr,
        model_lock=model_lock,
        on_emission=send_emission,
        config=cfg,
    )

    await websocket.send_json({
        "type": "ready",
        "session_id": session.session_id,
        "sample_rate": cfg.sample_rate,
        "chunk_duration": cfg.chunk_duration,
        "asr_backend": session.asr_backend.name,
        "asr_model": session.asr_backend.model_name,
    })

    # Kick off the streaming background loop on the session
    session.start()

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            if "bytes" in msg and msg["bytes"] is not None:
                buf = msg["bytes"]
                pcm = np.frombuffer(buf, dtype=np.float32)
                session.add_audio(pcm)

            elif "text" in msg and msg["text"] is not None:
                try:
                    payload = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue

                t = payload.get("type")
                if t == "stop":
                    break
                elif t == "ping":
                    await websocket.send_json({"type": "pong"})
                elif t == "select_model":
                    await _handle_select_model(websocket, session, payload)
                else:
                    log.warning("Session %s unknown message type %r",
                                session.session_id, t)

    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("Session %s WS error", session.session_id)
    finally:
        try:
            await session.stop()  # graceful — flushes pending utterance
        except Exception:
            log.exception("Session %s stop failed", session.session_id)

        try:
            await websocket.send_json({
                "type": "closed",
                "total_segments": len(session.transcript_log),
                "transcript_path": str(session.transcript_path) if session.transcript_path else None,
            })
        except Exception:
            pass

        try:
            await websocket.close()
        except Exception:
            pass


_MODEL_HINTS = {
    # Approximate first-time download sizes surfaced in the UI overlay so
    # users know what to expect.
    "whisper": {
        "tiny": "~39 MB download on first use",
        "base": "~74 MB download on first use",
        "small": "~244 MB download on first use",
        "medium": "~769 MB download on first use",
        "large-v2": "~1.5 GB download on first use",
        "large-v3": "~1.5 GB download on first use",
        "distil-large-v3": "~1.5 GB download on first use",
    },
    "qwen": {
        "Qwen/Qwen3-ASR-0.6B": "~1.2 GB download on first use",
        "Qwen/Qwen3-ASR-1.7B": "~3.4 GB download on first use",
        "Qwen/Qwen2-Audio-7B-Instruct": "~14 GB download on first use",
    },
    "nemo": {
        "stt_en_conformer_ctc_small": "~50 MB download on first use",
        "stt_en_conformer_ctc_medium": "~300 MB download on first use",
        "stt_en_conformer_ctc_large": "~500 MB download on first use",
    },
}


def _model_hint(backend: str, model: str) -> str:
    return _MODEL_HINTS.get(backend, {}).get(model, "First-time loads can take a minute.")


async def _handle_select_model(websocket: WebSocket, session: Session,
                               payload: dict) -> None:
    """Swap the session's ASR backend, loading it on demand."""
    backend = payload.get("backend")
    model = payload.get("model")
    if not backend or not model:
        await websocket.send_json({
            "type": "error", "severity": "error",
            "message": "select_model requires both 'backend' and 'model' fields.",
        })
        return

    # No-op if already on this backend
    if (session.asr_backend.name == backend.lower() and
            session.asr_backend.model_name == model):
        await websocket.send_json({
            "type": "model_changed",
            "backend": session.asr_backend.name,
            "model": session.asr_backend.model_name,
            "note": "already active",
        })
        return

    pool = _state.get("asr_pool") or {}
    cached = (backend.lower(), model) in pool

    await websocket.send_json({
        "type": "loading_model",
        "backend": backend,
        "model": model,
        "cached": cached,
        "detail": ("Already loaded in memory — switching now."
                   if cached else _model_hint(backend, model)),
    })

    try:
        asr = await _get_or_load_asr(backend, model)
    except Exception as e:
        log.exception("Session %s failed to load %s/%s",
                      session.session_id, backend, model)
        await websocket.send_json({
            "type": "error", "severity": "error",
            "message": f"Failed to load {backend}/{model}: {e}",
        })
        return

    session.set_asr_backend(asr)
    await websocket.send_json({
        "type": "model_changed",
        "backend": asr.name,
        "model": asr.model_name,
        "cached": cached,
    })


def run(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Programmatic entrypoint used by `python main.py serve`."""
    import uvicorn
    uvicorn.run(
        "src.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )
