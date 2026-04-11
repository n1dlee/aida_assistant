"""
server.py
FastAPI WebSocket server — bridges the React frontend with AIDA's orchestrator.
Run: python server.py
Then open http://localhost:5173 (React dev) or http://localhost:8000 (static build).
"""
import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from core.orchestrator import Orchestrator

log = logging.getLogger("aida.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

orchestrator = Orchestrator()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("AIDA server starting...")
    yield
    log.info("AIDA server shutting down.")


app = FastAPI(title="AIDA API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    log.info("Client connected.")
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            user_text = msg.get("text", "").strip()

            if not user_text:
                continue

            # Stream status back immediately
            await ws.send_text(json.dumps({"type": "thinking"}))

            try:
                response = await orchestrator.process(user_text)
            except Exception as e:
                log.error("Orchestrator error: %s", e)
                response = f"Sorry, something went wrong: {e}"

            await ws.send_text(json.dumps({
                "type": "response",
                "text": response,
            }))

    except WebSocketDisconnect:
        log.info("Client disconnected.")


@app.get("/api/status")
async def status():
    return {
        "status": "online",
        "local_llm": orchestrator.selector.local.is_available(),
        "cloud_llm": orchestrator.selector.cloud.is_available(),
        "memory_entries": orchestrator.vector_store.count(),
        "tools": orchestrator.tools.list_tools(),
    }


@app.get("/api/history")
async def history():
    return {"history": orchestrator.buffer.get_history()}


@app.delete("/api/history")
async def clear_history():
    orchestrator.buffer.clear()
    return {"status": "cleared"}


# Serve built React app from frontend/dist if it exists
_dist = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.isdir(_dist):
    app.mount("/", StaticFiles(directory=_dist, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
