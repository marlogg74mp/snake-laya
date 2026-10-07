"""
Snake Duel — human vs Laya (VERSUS_RULES.md). Standalone web service:
- serves the game (versus/) at /
- /api/predict proxies the AI's moves to the Laya model server (web_server.py, MODEL_URL)
- /api/leaderboard keeps results in SQLite (duel.db)

Run: python versus_server.py   ->  http://localhost:9700
"""

import os
import sqlite3
import time

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_URL = os.environ.get("MODEL_URL", "http://127.0.0.1:9500/predict")
DB_PATH = os.path.join(ROOT, "duel.db")
PORT = int(os.environ.get("VERSUS_PORT", "9700"))

app = FastAPI(title="Snake Duel")
http = requests.Session()

def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

with db() as con:
    con.execute("""CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, human INTEGER NOT NULL, ai INTEGER NOT NULL,
        kills INTEGER NOT NULL, deaths INTEGER NOT NULL, seed INTEGER, created REAL NOT NULL)""")
    if "speed" not in [r[1] for r in con.execute("PRAGMA table_info(results)")]:
        con.execute("ALTER TABLE results ADD COLUMN speed TEXT NOT NULL DEFAULT 'normal'")

MODELS = ("laya", "laya_convai", "laya_v11w", "laya_v12")  # our ModernBERT-base v10 / the real Laya 421M fine-tuned on Snake

class PredictRequest(BaseModel):
    state: dict
    model: str = "laya"

SPEEDS = ("slow", "normal", "fast")

class Result(BaseModel):
    name: str = Field(min_length=1, max_length=16)
    speed: str = "normal"
    human: int = Field(ge=0, le=999)
    ai: int = Field(ge=0, le=999)
    kills: int = Field(ge=0, le=99)
    deaths: int = Field(ge=0, le=99)
    seed: int | None = None

@app.post("/api/predict")
def predict(req: PredictRequest):
    try:
        model = req.model if req.model in MODELS else "laya"
        r = http.post(MODEL_URL, json={"model": model, "state": req.state}, timeout=2)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"model server unavailable: {e}")

@app.get("/api/leaderboard")
def leaderboard(limit: int = 10, speed: str = "normal"):
    with db() as con:
        rows = con.execute("SELECT id, name, human, ai FROM results WHERE speed = ? "
                           "ORDER BY human - ai DESC, human DESC, created ASC LIMIT ?",
                           (speed if speed in SPEEDS else "normal", max(1, min(limit, 50)))).fetchall()
    return [dict(r) for r in rows]

@app.post("/api/leaderboard")
def submit(res: Result):
    name = " ".join(res.name.split())[:16] or "Игрок"
    speed = res.speed if res.speed in SPEEDS else "normal"
    with db() as con:
        cur = con.execute("INSERT INTO results (name, human, ai, kills, deaths, seed, created, speed) "
                          "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                          (name, res.human, res.ai, res.kills, res.deaths, res.seed, time.time(), speed))
    return {"id": cur.lastrowid}

@app.get("/api/health")
def health():
    try:
        h = http.get(MODEL_URL.rsplit("/", 1)[0] + "/health", timeout=2).json()
    except requests.RequestException:
        h = {}
    return {"status": "ok", "model_ready": h.get("laya_loaded", False), "laya_convai_ready": h.get("laya_convai_loaded", False),
            "window_models": h.get("window_models", [])}

@app.middleware("http")
async def no_cache_static(request, call_next):
    response = await call_next(request)
    if not request.url.path.startswith("/api"):
        response.headers["Cache-Control"] = "no-cache"
    return response

# in-browser Laya (ONNX int4 + tokenizer, built by export_onnx_web.py) is shared with the 3D page
app.mount("/model", StaticFiles(directory=os.path.join(ROOT, "web3d", "model")), name="model")
app.mount("/model_laya421", StaticFiles(directory=os.path.join(ROOT, "web3d", "model_laya421")), name="model_laya421")
app.mount("/", StaticFiles(directory=os.path.join(ROOT, "versus"), html=True), name="versus")

if __name__ == "__main__":
    print(f"Snake Duel on http://localhost:{PORT} (model: {MODEL_URL})")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
