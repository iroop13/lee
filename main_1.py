"""
LEE backend — FastAPI + Claude Agent SDK
The agent has REAL tools:
  - Read / Write / Edit / Glob / Grep : your filesystem (scoped to WORKSPACE)
  - Bash                              : run shell commands
  - WebFetch / WebSearch              : crawl the web

It also exposes the Mission Hub state (jobs, logs) via simple endpoints
so the frontend stops faking the data.
"""

import asyncio
import json
import os
import time
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uuid
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI(@app.get("/")
async def serve_ui():
    return FileResponse("index.html"))
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from claude_agent_sdk import query, ClaudeAgentOptions

load_dotenv()

# ── config ──────────────────────────────────────────────────────────────────
# WORKSPACE is the only folder the agent can read/write/run scripts in.
# Change this to wherever you want LEE to operate.
WORKSPACE = Path(os.getenv("LEE_WORKSPACE", Path.home() / "lee-workspace")).resolve()
WORKSPACE.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = f"""You are LEE, an AI ops agent running locally on the operator's PC.

You speak in short, precise, terminal-style sentences. Use newlines and • bullets.
You have a playful 'Kid Mode' that translates technical status into plain English.

You have REAL tools — when the operator asks for something, USE THEM:
  • Read, Write, Edit, Glob, Grep — files inside {WORKSPACE}
  • Bash — run shell commands in {WORKSPACE}
  • WebFetch, WebSearch — pull live info from the web

Your workspace is: {WORKSPACE}
Stay inside it unless explicitly told otherwise. Never run destructive commands
(rm -rf, mkfs, dd) without confirming with the operator first.
"""

# ── in-memory state (replace with sqlite later if you want) ─────────────────
jobs: list[dict] = []
logs: list[str] = []

def log(msg: str) -> None:
    ts = time.strftime("[%H:%M:%S]")
    line = f"{ts} {msg}"
    logs.append(line)
    # keep last 200
    if len(logs) > 200:
        del logs[:-200]
    print(line)

# ── app ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="LEE backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "workspace": str(WORKSPACE), "jobs": len(jobs), "logs": len(logs)}

@app.get("/jobs")
def list_jobs():
    return {"jobs": jobs}

@app.get("/logs")
def get_logs(tail: int = 50):
    return {"logs": logs[-tail:]}

# ── chat (streaming) ────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    prompt: str
    # which built-in tools the agent is allowed to use this turn
    allowed_tools: list[str] = ["Read", "Write", "Edit", "Glob", "Grep", "Bash", "WebFetch", "WebSearch"]
    # 'acceptEdits' lets the agent edit files without per-call confirmation
    permission_mode: str = "acceptEdits"

async def run_agent(req: ChatRequest) -> AsyncIterator[str]:
    """
    Stream Server-Sent Events back to the frontend.
    Each event is a JSON line: {"type": "...", "data": ...}
    """
    job_id = str(uuid.uuid4())[:8]
    job = {"id": job_id, "status": "running", "name": req.prompt[:60], "started": time.time()}
    jobs.append(job)
    log(f"▶ Job #{job_id} started — {req.prompt[:60]}")

    try:
        options = ClaudeAgentOptions(
            system_prompt=SYSTEM_PROMPT,
            allowed_tools=req.allowed_tools,
            permission_mode=req.permission_mode,
            cwd=str(WORKSPACE),
        )

        async for message in query(prompt=req.prompt, options=options):
            # The SDK yields several message types: SystemMessage, AssistantMessage,
            # UserMessage (tool results), ResultMessage. We forward useful bits.
            payload = serialize_message(message)
            if payload:
                yield f"data: {json.dumps(payload)}\n\n"

        job["status"] = "finished"
        job["duration"] = round(time.time() - job["started"], 2)
        log(f"✓ Job #{job_id} complete in {job['duration']}s")
        yield f"data: {json.dumps({'type': 'done', 'job_id': job_id})}\n\n"

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        log(f"✗ Job #{job_id} failed — {e}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

def serialize_message(message) -> dict | None:
    """Convert SDK message objects to plain dicts for the frontend."""
    # Final assistant text
    if hasattr(message, "result") and message.result:
        return {"type": "result", "text": message.result}

    # Assistant message with content blocks (text + tool uses)
    if hasattr(message, "content"):
        blocks = []
        for block in message.content if isinstance(message.content, list) else []:
            btype = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
            if btype == "text":
                text = getattr(block, "text", None) or block.get("text", "")
                blocks.append({"type": "text", "text": text})
            elif btype == "tool_use":
                name = getattr(block, "name", None) or block.get("name", "?")
                inp = getattr(block, "input", None) or block.get("input", {})
                blocks.append({"type": "tool_use", "name": name, "input": inp})
                log(f"⚙ tool: {name}")
            elif btype == "tool_result":
                content = getattr(block, "content", None) or block.get("content", "")
                blocks.append({"type": "tool_result", "content": str(content)[:500]})
        if blocks:
            return {"type": "assistant", "blocks": blocks}

    return None

@app.post("/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(run_agent(req), media_type="text/event-stream")

# ── run ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    log(f"LEE backend booting — workspace: {WORKSPACE}")
    uvicorn.run(app, host="127.0.0.1", port=8000)
