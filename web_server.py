"""
FastAPI Web Server for Snake AI with Cyberpunk UI, Multi-Type Apples, Magic Shrink Apples,
30-Min Session Telemetry Timer, Non-Blocking Game Loop with Fallback, and Auto Retraining.
"""

import json
import os
import random
import time
import requests
import urllib3
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

urllib3.disable_warnings()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
ID_TO_ACTION = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
CRASHES_FILE = os.path.join(os.path.dirname(__file__), "crashes.jsonl")

app = FastAPI(title="Snake AI Web Server - Advanced Cyberpunk Edition")

@app.middleware("http")
async def no_cache_web3d(request, call_next):
    # The 3D frontend is edited often; make browsers revalidate instead of running stale JS
    response = await call_next(request)
    if request.url.path.startswith("/3d"):
        response.headers["Cache-Control"] = "no-cache"
    return response

# --- Model Loaders ---
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "laya_snake_weights")

# Key order of the state the model was trained on. A model folder may override it with state_schema.json
# (written by the training script); without that file the V7 schema is assumed.
TRAINING_KEYS = ["current_dir", "recent_actions", "food_dir", "danger_UP", "danger_DOWN", "danger_LEFT",
                 "danger_RIGHT", "head_pos", "food_pos", "obstacles", "portals", "fog_of_war", "sight_radius", "patrols"]

def load_state_keys(model_dir: str) -> list:
    schema_file = os.path.join(model_dir, "state_schema.json")
    if os.path.exists(schema_file):
        with open(schema_file, "r", encoding="utf-8") as f:
            return json.load(f)["keys"]
    return TRAINING_KEYS

WINDOW_LISTS = ("body", "walls", "ghost_walls", "drones", "drone_path", "memory_food", "foods_near", "foods_far", "poison")

def to_training_schema(state: dict, keys: list = TRAINING_KEYS) -> dict:
    """
    Laya reads the state as text, so key order, extra keys and value spellings must match
    the training data exactly, otherwise the model drifts (e.g. looping along walls).
    """
    s = {k: state.get(k) for k in keys}
    if "edges" in keys:
        # window schemas (V11w / V12): the client builds them exactly; only fill neutral values for missing keys
        for k in WINDOW_LISTS:
            if k in keys and s[k] is None:
                s[k] = []
        for k, v in (("explored", [0, 0, 0, 0]), ("edges", [6, 6, 6, 6]), ("immortal_steps", 0), ("fog_of_war", False)):
            if s.get(k) is None and k in keys:
                s[k] = v
        return s
    if s["food_dir"] in (None, "NONE"):
        s["food_dir"] = "UNKNOWN"
    elif s["food_dir"] == "ON_FOOD":
        s["food_dir"] = "SAME"
    if s["food_pos"] is None:
        s["food_pos"] = [-1, -1]
    for k in ("obstacles", "portals", "patrols"):
        s[k] = s[k] or []
    # V9 fields (RULES_V9.md); a client that does not know them gets neutral values
    for k in ("body", "phase_walls", "drone_path"):
        if k in keys:
            s[k] = s[k] or []
    if "immortal_steps" in keys:
        s["immortal_steps"] = int(s["immortal_steps"] or 0)
    if "body" in keys:
        s["body"] = [list(c[:2]) for c in s["body"]][:30]
    s["recent_actions"] = s["recent_actions"] or []
    if s["sight_radius"] is None:
        s["sight_radius"] = 5
    s["fog_of_war"] = bool(s["fog_of_war"])
    if "snake_len" in keys:
        # V8: patrols are [x, y, dx, dy] (dx, dy = drone's next step)
        s["snake_len"] = int(s["snake_len"] or 0)
        s["patrols"] = [list(p) + [0, 0] if len(p) == 2 else list(p[:4]) for p in s["patrols"]]
    else:
        s["patrols"] = [list(p[:2]) for p in s["patrols"]]
    return s

class LayaEngine:
    def __init__(self, model_dir: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Initializing Laya PyTorch model from '{model_dir}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        self.state_keys = load_state_keys(model_dir)
        print(f"[INFO] Laya model successfully loaded on {self.device} (state keys: {len(self.state_keys)}).")

    def predict(self, compact_state: dict) -> tuple[str, dict, float, float]:
        prompt_text = (
            f"State: {json.dumps(to_training_schema(compact_state, self.state_keys))}\n"
            "Question: What is the next safe move avoiding patrols and walls towards food?\n"
            "Choices: UP, DOWN, LEFT, RIGHT"
        )
        t0 = time.time()
        encoding = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=512,  # V9 states are ~340-420 tokens; 256 silently cut off patrols/drone_path
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs.logits, dim=-1).squeeze(0).cpu().numpy()

        latency_ms = (time.time() - t0) * 1000.0
        probs_dict = {ID_TO_ACTION[i]: float(probabilities[i]) for i in range(4)}
        best_id = int(probabilities.argmax())
        best_action = ID_TO_ACTION[best_id]
        confidence = float(probabilities[best_id])

        return best_action, probs_dict, latency_ms, confidence

laya_model = None
if os.path.exists(WEIGHTS_PATH):
    laya_model = LayaEngine(WEIGHTS_PATH)

# Window models (egocentric state + memory, any board size): V11w and V12 (+ apple values).
# The client sends the window state (versus/js/window_state.js); request model ids "laya_v11w" / "laya_v12".
window_models = {}
for _name, _dir in (("laya_v11w", "laya_snake_weights_v11w"), ("laya_v12", "laya_snake_weights_v12")):
    _path = os.path.join(os.path.dirname(__file__), _dir)
    if os.path.exists(os.path.join(_path, "model.safetensors")):
        try:
            window_models[_name] = LayaEngine(_path)
        except Exception as e:
            print(f"[WARN] {_name} not loaded: {e}")

# The real Laya (convaiinnovations/laya, ModernBERT-large + decision head, 421M) fine-tuned on Snake
# by laya_convai_snake.py. Its input is Laya's own [MASK]-option format, built inside LayaPolicy.
CONVAI_DIR = os.path.join(os.path.dirname(__file__), "laya_convai_snake")

class ConvaiLayaEngine:
    def __init__(self):
        import laya_convai_snake
        self.policy = laya_convai_snake.LayaPolicy(os.path.join(CONVAI_DIR, "model.safetensors"))
        self.device = torch.device("cuda")
        print("[INFO] Convai Laya (421M, fine-tuned on Snake) loaded on cuda.")

    def predict(self, compact_state: dict) -> tuple[str, dict, float, float]:
        state = to_training_schema(compact_state, self.policy.keys)
        t0 = time.time()
        probs = self.policy.probs_batch([state])[0]
        latency_ms = (time.time() - t0) * 1000.0
        best = max(probs, key=probs.get)
        return best, probs, latency_ms, probs[best]

convai_model = None
if os.path.exists(os.path.join(CONVAI_DIR, "model.safetensors")) and torch.cuda.is_available():
    try:
        convai_model = ConvaiLayaEngine()
    except Exception as e:
        print(f"[WARN] Convai Laya not loaded: {e}")

def query_jev_113(compact_state: dict) -> tuple[str, dict, float, float]:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "typesafe/jev-1.13",
        "state": f"State: {json.dumps(compact_state)}",
        "questions": {
            "next_move": {
                "type": "choice",
                "criteria": {
                    "UP": "Move upwards avoiding walls and poison",
                    "DOWN": "Move downwards avoiding walls and poison",
                    "LEFT": "Move leftwards avoiding walls and poison",
                    "RIGHT": "Move rightwards avoiding walls and poison"
                },
                "instructions": "Select the safest move towards food while strictly avoiding walls, body, and poison apples"
            }
        }
    }
    t0 = time.time()
    try:
        r = requests.post(DECISIONS_URL, headers=headers, json=payload, timeout=5, verify=False)
        latency_ms = (time.time() - t0) * 1000.0
        r.raise_for_status()
        data = r.json()
        answer_info = data["answers"]["next_move"]
        chosen = answer_info["choice"]
        probs = answer_info.get("probabilities", {})
        confidence = answer_info.get("confidence", 0.0)
        return chosen, probs, latency_ms, confidence
    except Exception as e:
        latency_ms = (time.time() - t0) * 1000.0
        return "UP", {"UP": 0.25, "DOWN": 0.25, "LEFT": 0.25, "RIGHT": 0.25}, latency_ms, 0.0

# --- Request / Response Schemas ---
class PredictRequest(BaseModel):
    model: str = "laya"
    state: dict

class CrashRecordRequest(BaseModel):
    state: dict
    wrong_action: str
    crash_reason: str
    model_used: str

@app.post("/predict")
def predict_move(req: PredictRequest):
    if req.model.lower() == "laya":
        if laya_model is None:
            raise HTTPException(status_code=500, detail="Laya model weights not found.")
        action, probs, latency, confidence = laya_model.predict(req.state)
        engine_name = f"Laya ModernBERT ({laya_model.device.type.upper()})"
    elif req.model.lower() in window_models:
        eng = window_models[req.model.lower()]
        action, probs, latency, confidence = eng.predict(req.state)
        engine_name = f"{req.model.lower().replace('laya_', 'Laya ')} (окно, CUDA)"
    elif req.model.lower() == "laya_convai":
        if convai_model is None:
            raise HTTPException(status_code=500, detail="Convai Laya (laya_convai_snake) not loaded.")
        action, probs, latency, confidence = convai_model.predict(req.state)
        engine_name = "Laya 421M Convai (дообученная, CUDA)"
    else:
        action, probs, latency, confidence = query_jev_113(req.state)
        engine_name = "TypeSafe Jev-1.13 (OpenRouter API)"

    return {
        "action": action,
        "probabilities": probs,
        "confidence": confidence,
        "latency_ms": round(latency, 2),
        "engine": engine_name
    }

@app.post("/record_crash")
def record_crash(req: CrashRecordRequest):
    """
    Saves hard negative crash telemetry sample to crashes.jsonl for Round 2 model retraining.
    """
    record = {
        "timestamp": time.time(),
        "model": req.model_used,
        "state": req.state,
        "wrong_action": req.wrong_action,
        "crash_reason": req.crash_reason
    }
    with open(CRASHES_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"status": "success", "recorded": True}

@app.get("/export_crashes")
def export_crashes():
    if os.path.exists(CRASHES_FILE):
        return FileResponse(CRASHES_FILE, filename="crashes.jsonl", media_type="application/jsonlines")
    return {"message": "No crash records logged yet."}

@app.post("/clear_crashes")
def clear_crashes():
    """
    Archives crashes.jsonl and resets crash log for fresh telemetry sessions.
    """
    if os.path.exists(CRASHES_FILE):
        archive_path = os.path.join(os.path.dirname(__file__), f"crashes_archive_{int(time.time())}.jsonl")
        try:
            os.rename(CRASHES_FILE, archive_path)
        except Exception:
            open(CRASHES_FILE, "w").close()
    return {"status": "success", "message": "Crash log archived and reset"}

@app.post("/reload_weights")
def reload_weights():
    """
    Reloads Laya PyTorch weights into CUDA memory from laya_snake_weights.
    """
    global laya_model
    try:
        if os.path.exists(WEIGHTS_PATH):
            laya_model = LayaEngine(WEIGHTS_PATH)
            print("[INFO] Laya model weights reloaded successfully.")
            return {"status": "success", "message": "Model weights reloaded on CUDA"}
        else:
            raise HTTPException(status_code=404, detail="Weights path does not exist.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trigger_retrain")
def trigger_retrain():
    """
    Triggers dataset rebuilding and Laya ModernBERT model retraining on CUDA in background.
    """
    import subprocess, sys
    print("[INFO] Launching background retraining with Patrol-Aware Dataset V7...")
    subprocess.Popen([sys.executable, os.path.join(os.path.dirname(__file__), "build_dataset_v7_patrols.py")])
    return {"status": "success", "message": "Patrol V7 Retraining task launched on GPU"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "laya_loaded": laya_model is not None,
        "laya_convai_loaded": convai_model is not None,
        "window_models": sorted(window_models),
        "cuda_available": torch.cuda.is_available(),
        "openrouter_key_present": bool(OPENROUTER_API_KEY)
    }

@app.get("/", response_class=HTMLResponse)
def get_web_ui():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Snake AI Arena - Cyberpunk Multi-Food & 30m Telemetry</title>
        <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700;800&family=Orbitron:wght@600;800;900&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg-main: #060911;
                --bg-card: rgba(12, 19, 32, 0.85);
                --bg-card-border: rgba(0, 240, 255, 0.18);
                --neon-cyan: #00f0ff;
                --neon-green: #00ff88;
                --neon-pink: #ff0055;
                --neon-gold: #ffd700;
                --neon-purple: #b026ff;
                --text-main: #e2f1f8;
                --text-dim: #708498;
                --bar-bg: rgba(10, 20, 35, 0.7);
            }

            * { box-sizing: border-box; margin: 0; padding: 0; }

            body {
                background: var(--bg-main);
                background-image: 
                    radial-gradient(circle at 15% 20%, rgba(0, 240, 255, 0.05) 0%, transparent 40%),
                    radial-gradient(circle at 85% 80%, rgba(255, 0, 85, 0.04) 0%, transparent 40%);
                color: var(--text-main);
                font-family: 'Orbitron', 'JetBrains Mono', monospace;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 16px;
                user-select: none;
            }

            header {
                width: 100%;
                max-width: 1100px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 16px;
                padding-bottom: 12px;
                border-bottom: 1px solid rgba(0, 240, 255, 0.15);
            }

            .brand { display: flex; align-items: center; gap: 12px; }
            .logo { background: linear-gradient(135deg, var(--neon-cyan), var(--neon-pink)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 26px; font-weight: 900; }
            .title { font-size: 18px; font-weight: 800; color: #fff; letter-spacing: 1px; }

            .workspace { display: flex; gap: 24px; width: 100%; max-width: 1100px; justify-content: center; }

            .arena-card {
                background: var(--bg-card);
                border: 1px solid var(--bg-card-border);
                border-radius: 14px;
                padding: 16px;
                backdrop-filter: blur(12px);
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.7);
                display: flex;
                flex-direction: column;
                align-items: center;
                position: relative;
            }

            .status-banner {
                width: 500px;
                padding: 8px 14px;
                border-radius: 6px;
                margin-bottom: 10px;
                font-family: 'JetBrains Mono', monospace;
                font-size: 12px;
                font-weight: 700;
                display: flex;
                justify-content: space-between;
                align-items: center;
                background: rgba(0, 255, 136, 0.1);
                border: 1px solid var(--neon-green);
                color: var(--neon-green);
            }

            .status-banner.crashed {
                background: rgba(255, 0, 85, 0.15);
                border-color: var(--neon-pink);
                color: var(--neon-pink);
            }

            .canvas-wrapper {
                position: relative;
                border-radius: 8px;
                overflow: hidden;
                border: 2px solid rgba(0, 240, 255, 0.4);
                box-shadow: 0 0 25px rgba(0, 240, 255, 0.15);
            }

            canvas { display: block; background: #080d17; cursor: crosshair; }

            .legend {
                margin-top: 10px;
                font-size: 11px;
                font-family: 'JetBrains Mono', monospace;
                color: var(--text-dim);
                display: flex;
                gap: 12px;
                flex-wrap: wrap;
                justify-content: center;
            }

            .sidebar { width: 480px; display: flex; flex-direction: column; gap: 14px; }

            .panel-box {
                background: var(--bg-card);
                border: 1px solid var(--bg-card-border);
                border-radius: 12px;
                padding: 16px;
                backdrop-filter: blur(12px);
            }

            .panel-header {
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 1.5px;
                color: var(--neon-cyan);
                margin-bottom: 12px;
                display: flex;
                justify-content: space-between;
                border-bottom: 1px solid rgba(0, 240, 255, 0.15);
                padding-bottom: 6px;
            }

            .model-switcher { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
            .model-tab {
                background: rgba(16, 28, 46, 0.7);
                border: 1px solid rgba(0, 240, 255, 0.2);
                border-radius: 8px;
                padding: 10px;
                cursor: pointer;
                transition: all 0.2s;
            }
            .model-tab.active {
                border-color: var(--neon-green);
                background: rgba(0, 255, 136, 0.12);
                box-shadow: 0 0 12px rgba(0, 255, 136, 0.25);
            }

            .action-hero {
                display: flex;
                align-items: center;
                justify-content: space-between;
                background: rgba(0, 0, 0, 0.4);
                border: 1px solid rgba(0, 255, 136, 0.3);
                border-radius: 8px;
                padding: 10px 16px;
                margin-bottom: 12px;
            }
            .action-value { font-size: 24px; font-weight: 800; color: var(--neon-green); display: flex; align-items: center; gap: 8px; }

            .prob-list { display: flex; flex-direction: column; gap: 6px; }
            .prob-item {
                display: flex; align-items: center; background: rgba(16, 28, 46, 0.4);
                padding: 5px 10px; border-radius: 6px; font-family: 'JetBrains Mono', monospace; font-size: 11px;
            }
            .prob-item.winner { border: 1px solid var(--neon-green); background: rgba(0, 255, 136, 0.08); }
            .prob-track { flex: 1; height: 8px; background: var(--bar-bg); border-radius: 4px; margin: 0 10px; overflow: hidden; }
            .prob-fill { height: 100%; background: var(--neon-cyan); width: 0%; transition: width 0.15s; }
            .prob-item.winner .prob-fill { background: var(--neon-green); }

            .danger-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; margin-top: 10px; }
            .danger-cell {
                background: rgba(16, 28, 46, 0.5); border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 6px; padding: 6px; text-align: center; font-size: 10px; font-family: 'JetBrains Mono', monospace;
            }
            .danger-cell.safe { border-color: rgba(0, 255, 136, 0.3); color: var(--neon-green); }
            .danger-cell.danger { border-color: rgba(255, 0, 85, 0.6); background: rgba(255, 0, 85, 0.15); color: var(--neon-pink); }
            .danger-cell.poison { border-color: rgba(176, 38, 255, 0.8); background: rgba(176, 38, 255, 0.2); color: var(--neon-purple); }

            .metrics-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
            .metric-box { background: rgba(16, 28, 46, 0.5); border: 1px solid rgba(0, 240, 255, 0.15); border-radius: 8px; padding: 8px; text-align: center; }
            .metric-box .m-label { font-size: 9px; color: var(--text-dim); text-transform: uppercase; }
            .metric-box .m-val { font-size: 16px; font-weight: 700; font-family: 'JetBrains Mono', monospace; color: #fff; margin-top: 2px; }

            .controls-row { display: flex; gap: 8px; margin-top: 10px; }
            .cyber-btn {
                flex: 1; background: rgba(16, 28, 46, 0.8); border: 1px solid var(--neon-cyan);
                color: var(--neon-cyan); font-family: inherit; font-weight: 700; font-size: 11px;
                padding: 8px 10px; border-radius: 6px; cursor: pointer; transition: all 0.15s;
            }
            .cyber-btn:hover { background: var(--neon-cyan); color: #060911; }
            .cyber-btn.btn-reset { border-color: var(--neon-pink); color: var(--neon-pink); }
            .cyber-btn.btn-reset:hover { background: var(--neon-pink); color: #fff; }
            .cyber-btn.btn-export { border-color: var(--neon-gold); color: var(--neon-gold); }
            .cyber-btn.btn-export:hover { background: var(--neon-gold); color: #000; }
        </style>
    </head>
    <body>
        <header>
            <div class="brand">
                <div class="logo">LAYA / JEV</div>
                <div class="title">CYBERPUNK SNAKE ARENA</div>
            </div>
            <div style="font-size: 12px; font-family: 'JetBrains Mono'; color: var(--neon-cyan); display: flex; align-items: center; gap: 8px;">
                ⏱️ SESSION TIMER: <span id="sessionTimerVal" style="font-weight: 800; font-size: 15px; color: #fff;">30:00</span>
            </div>
        </header>

        <div class="workspace">
            <div class="arena-card">
                <!-- Status Banner above Canvas -->
                <div id="statusBanner" class="status-banner">
                    <span id="bannerText">SIMULATION ACTIVE // NO COLLISIONS</span>
                    <span id="bannerSub" style="font-size: 10px; opacity: 0.8;">STEP #0</span>
                </div>

                <div class="canvas-wrapper">
                    <canvas id="snakeCanvas" width="500" height="500"></canvas>
                </div>

                <div class="legend">
                    <span>🍎 Standard (+1)</span>
                    <span>🍏 Golden (+3)</span>
                    <span>🔮 Shrink (+2, -2 Len)</span>
                    <span>⚡ Turbo (2x Speed)</span>
                    <span>❄️ Freeze (0.5x Speed)</span>
                    <span>☠️ Poison (-5)</span>
                    <span>🧱 Wall</span>
                    <span>🔵 Portals (Warp)</span>
                    <span>🌫️ Fog of War</span>
                    <span>🚨 Patrol Drone</span>
                    <span>🖱️ Click: Spawn</span>
                </div>
            </div>

            <div class="sidebar">
                <div class="panel-box">
                    <div class="panel-header">
                        <span>MODEL ARCHITECTURE</span>
                        <span id="activeModelTag" style="color: var(--neon-green);">LOCAL CUDA</span>
                    </div>
                    <div class="model-switcher">
                        <div class="model-tab active" id="tabLaya" onclick="selectModel('laya')">
                            <div style="font-size: 9px; color: var(--neon-green); font-weight:700;">EDGE / LOCAL</div>
                            <div style="font-size: 13px; font-weight:700; color:#fff;">LAYA PyTorch</div>
                            <div style="font-size: 10px; color: var(--text-dim);">ModernBERT-base (~3ms)</div>
                        </div>
                        <div class="model-tab" id="tabJev" onclick="selectModel('jev')">
                            <div style="font-size: 9px; color: var(--neon-cyan); font-weight:700;">CLOUD API</div>
                            <div style="font-size: 13px; font-weight:700; color:#fff;">JEV 1.13</div>
                            <div style="font-size: 10px; color: var(--text-dim);">TypeSafe OpenRouter API</div>
                        </div>
                    </div>
                </div>

                <div class="panel-box">
                    <div class="panel-header">
                        <span>ACTION PREDICTION</span>
                        <span id="engineBadge" style="font-size: 10px; color: var(--text-dim); font-family: 'JetBrains Mono';">Laya ModernBERT</span>
                    </div>

                    <div class="action-hero">
                        <div>
                            <div style="font-size: 10px; color: var(--text-dim);">TARGET ACTION</div>
                            <div class="action-value" id="actionVal">
                                <span id="actionIcon">➡️</span> <span id="actionText">RIGHT</span>
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 18px; font-weight:700; color: var(--neon-cyan); font-family: 'JetBrains Mono';" id="confidenceVal">98.4%</div>
                            <div style="font-size: 9px; color: var(--text-dim);">CONFIDENCE</div>
                        </div>
                    </div>

                    <div class="prob-list">
                        <div class="prob-item" id="item-UP"><div style="width:65px; font-weight:700;">⬆️ UP</div><div class="prob-track"><div class="prob-fill" id="fill-UP"></div></div><div style="width:45px; text-align:right;" id="pct-UP">0%</div></div>
                        <div class="prob-item" id="item-DOWN"><div style="width:65px; font-weight:700;">⬇️ DOWN</div><div class="prob-track"><div class="prob-fill" id="fill-DOWN"></div></div><div style="width:45px; text-align:right;" id="pct-DOWN">0%</div></div>
                        <div class="prob-item" id="item-LEFT"><div style="width:65px; font-weight:700;">⬅️ LEFT</div><div class="prob-track"><div class="prob-fill" id="fill-LEFT"></div></div><div style="width:45px; text-align:right;" id="pct-LEFT">0%</div></div>
                        <div class="prob-item winner" id="item-RIGHT"><div style="width:65px; font-weight:700;">➡️ RIGHT</div><div class="prob-track"><div class="prob-fill" id="fill-RIGHT"></div></div><div style="width:45px; text-align:right;" id="pct-RIGHT">0%</div></div>
                    </div>

                    <div class="danger-grid">
                        <div class="danger-cell safe" id="radar-UP">UP: SAFE</div>
                        <div class="danger-cell safe" id="radar-DOWN">DOWN: SAFE</div>
                        <div class="danger-cell safe" id="radar-LEFT">LEFT: SAFE</div>
                        <div class="danger-cell safe" id="radar-RIGHT">RIGHT: SAFE</div>
                    </div>
                </div>

                <div class="panel-box">
                    <div class="panel-header"><span>30-MIN SESSION TELEMETRY</span></div>
                    <div class="metrics-grid" style="grid-template-columns: repeat(5, 1fr);">
                        <div class="metric-box"><div class="m-label">Score</div><div class="m-val" style="color:var(--neon-green);" id="scoreVal">0</div></div>
                        <div class="metric-box"><div class="m-label">Length</div><div class="m-val" style="color:var(--neon-cyan);" id="lengthVal">3</div></div>
                        <div class="metric-box"><div class="m-label">Latency</div><div class="m-val" style="color:#fff;" id="latencyVal">0 ms</div></div>
                        <div class="metric-box"><div class="m-label">Crashes 30m</div><div class="m-val" style="color:var(--neon-pink);" id="crashesCountVal">0</div></div>
                        <div class="metric-box"><div class="m-label">Max Score</div><div class="m-val" style="color:var(--neon-gold);" id="maxScoreVal">0</div></div>
                    </div>

                    <div class="controls-row">
                        <button class="cyber-btn" id="btnPlay" onclick="togglePlayPause()">⏸ PAUSE</button>
                        <button class="cyber-btn" onclick="executeStep()">⏭ STEP</button>
                        <button class="cyber-btn btn-reset" onclick="resetGame()">🔄 REBOOT</button>
                        <button class="cyber-btn" id="btnWalls" onclick="toggleWalls()" style="border-color: var(--neon-purple); color: var(--neon-purple);">🧱 WALLS: ON</button>
                        <button class="cyber-btn" id="btnPortals" onclick="togglePortals()" style="border-color: var(--neon-cyan); color: var(--neon-cyan);">🌀 PORTALS: ON</button>
                        <button class="cyber-btn" id="btnFog" onclick="toggleFog()" style="border-color: var(--neon-green); color: var(--neon-green);">🌫️ FOG: OFF</button>
                        <button class="cyber-btn" id="btnPatrols" onclick="togglePatrols()" style="border-color: var(--neon-gold); color: var(--neon-gold);">🤖 PATROLS: ON</button>
                        <button class="cyber-btn" onclick="clearCrashesUI()" style="border-color: var(--neon-pink); color: var(--neon-pink);">🧹 CLEAR CRASHES</button>
                        <button class="cyber-btn btn-export" onclick="triggerRetrain()">🚀 RETRAIN NOW</button>
                    </div>

                    <div style="margin-top: 8px; font-size: 11px; display: flex; align-items: center; gap: 8px; color: var(--neon-cyan);">
                        <input type="checkbox" id="autoRebootCheck" checked style="accent-color: var(--neon-cyan);">
                        <label for="autoRebootCheck">AUTO-REBOOT ON CRASH (2.0s delay for telemetry)</label>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const canvas = document.getElementById('snakeCanvas');
            const ctx = canvas.getContext('2d');
            const GRID = 20, CELL = 25;

            let selectedModel = "laya";
            let isRunning = true, isGameOver = false, stepCount = 0, score = 0, crashesCount = 0, maxScore = 0;
            let tickInterval = 120, isPredicting = false;
            let crashedCell = null, crashedAction = null;
            let enableWalls = true;
            let obstacles = [];
            let enablePortals = true;
            let portals = []; // Pair: [{x, y}, {x, y}]

            let snake = [{x: 10, y: 10}, {x: 9, y: 10}, {x: 8, y: 10}];
            let currentDir = "RIGHT";
            let recentActions = ["RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT"];

            // Foods list: [{x, y, type: 'standard'|'golden'|'poison'|'shrink'|'turbo'|'freeze', value, expiresAt}]
            let foods = [];

            const ACTIONS = { "UP": {x:0, y:-1}, "DOWN": {x:0, y:1}, "LEFT": {x:-1, y:0}, "RIGHT": {x:1, y:0} };
            const OPPOSITES = {"UP":"DOWN", "DOWN":"UP", "LEFT":"RIGHT", "RIGHT":"LEFT"};

            function toggleWalls() {
                enableWalls = !enableWalls;
                let btn = document.getElementById('btnWalls');
                if (btn) {
                    btn.innerText = enableWalls ? "🧱 WALLS: ON" : "🧱 WALLS: OFF";
                    btn.style.opacity = enableWalls ? "1" : "0.5";
                }
                resetGame();
            }

            function togglePortals() {
                enablePortals = !enablePortals;
                let btn = document.getElementById('btnPortals');
                if (btn) {
                    btn.innerText = enablePortals ? "🌀 PORTALS: ON" : "🌀 PORTALS: OFF";
                    btn.style.opacity = enablePortals ? "1" : "0.5";
                }
                resetGame();
            }

            let enableFog = false;
            let fogRadius = 5;

            function toggleFog() {
                enableFog = !enableFog;
                let btn = document.getElementById('btnFog');
                if (btn) {
                    btn.innerText = enableFog ? "🌫️ FOG: ON" : "🌫️ FOG: OFF";
                    btn.style.color = enableFog ? "var(--neon-green)" : "var(--neon-cyan)";
                    btn.style.borderColor = enableFog ? "var(--neon-green)" : "var(--neon-cyan)";
                }
            }

            let enablePatrols = true;
            const PATROL_BODY_HITS = false; // true = a drone touching any body segment kills the snake
            let patrols = []; // [{x, y, dx, dy, minX, maxX, minY, maxY, type}]

            function togglePatrols() {
                enablePatrols = !enablePatrols;
                let btn = document.getElementById('btnPatrols');
                if (btn) {
                    btn.innerText = enablePatrols ? "🤖 PATROLS: ON" : "🤖 PATROLS: OFF";
                    btn.style.opacity = enablePatrols ? "1" : "0.5";
                }
                resetGame();
            }

            function generatePatrols() {
                patrols = [];
                if (!enablePatrols) return;
                patrols = [
                    {x: 4, y: 5, dx: 1, dy: 0, minX: 3, maxX: 16, minY: 5, maxY: 5, type: 'linear_horiz'},
                    {x: 15, y: 12, dx: 0, dy: 1, minX: 15, maxX: 15, minY: 8, maxY: 17, type: 'linear_vert'}
                ];
            }

            function generatePortals() {
                portals = [];
                if (!enablePortals) return;
                let head = snake[0];
                let forbidden = new Set(snake.map(s => `${s.x},${s.y}`));
                obstacles.forEach(o => forbidden.add(`${o.x},${o.y}`));

                for (let attempts = 0; attempts < 200; attempts++) {
                    let pa = {x: 1 + Math.floor(Math.random() * (GRID - 2)), y: 1 + Math.floor(Math.random() * (GRID - 2))};
                    let pb = {x: 1 + Math.floor(Math.random() * (GRID - 2)), y: 1 + Math.floor(Math.random() * (GRID - 2))};
                    let dist = Math.abs(pa.x - pb.x) + Math.abs(pa.y - pb.y);
                    if (!forbidden.has(`${pa.x},${pa.y}`) && !forbidden.has(`${pb.x},${pb.y}`) && dist >= 6) {
                        portals = [pa, pb];
                        break;
                    }
                }
            }

            function generateObstacles() {
                obstacles = [];
                if (!enableWalls) return;
                let numStructures = 2 + Math.floor(Math.random() * 2); // 2 to 3 wall structures
                let head = snake[0];
                let forbidden = new Set(snake.map(s => `${s.x},${s.y}`));
                for (let dx = -3; dx <= 3; dx++) {
                    for (let dy = -3; dy <= 3; dy++) {
                        forbidden.add(`${head.x + dx},${head.y + dy}`);
                    }
                }
                const shapeTemplates = [
                    [{x:0,y:0}, {x:1,y:0}, {x:2,y:0}],        // Horizontal 3-cell bar
                    [{x:0,y:0}, {x:0,y:1}, {x:0,y:2}],        // Vertical 3-cell bar
                    [{x:0,y:0}, {x:1,y:0}, {x:0,y:1}, {x:1,y:1}], // 2x2 Square Block
                    [{x:0,y:0}, {x:1,y:0}, {x:2,y:0}, {x:0,y:1}], // L-Shape
                    [{x:0,y:0}, {x:1,y:0}],                // Horizontal 2-cell bar
                    [{x:0,y:0}, {x:0,y:1}]                 // Vertical 2-cell bar
                ];

                for (let i = 0; i < numStructures; i++) {
                    let attempts = 0;
                    while (attempts < 100) {
                        let tmpl = shapeTemplates[Math.floor(Math.random() * shapeTemplates.length)];
                        let ox = 1 + Math.floor(Math.random() * (GRID - 4));
                        let oy = 1 + Math.floor(Math.random() * (GRID - 4));
                        let shapeCells = tmpl.map(p => ({x: ox + p.x, y: oy + p.y}));
                        
                        let valid = shapeCells.every(c => c.x >= 0 && c.x < GRID && c.y >= 0 && c.y < GRID && !forbidden.has(`${c.x},${c.y}`));
                        if (valid) {
                            shapeCells.forEach(c => {
                                obstacles.push(c);
                                forbidden.add(`${c.x},${c.y}`);
                            });
                            break;
                        }
                        attempts++;
                    }
                }
            }

            // 30-Minute Session Telemetry Timer (1800 seconds)
            let sessionTimeRemaining = 1800;
            let sessionTimer = setInterval(() => {
                if (sessionTimeRemaining > 0) {
                    sessionTimeRemaining--;
                    let m = Math.floor(sessionTimeRemaining / 60);
                    let s = sessionTimeRemaining % 60;
                    document.getElementById('sessionTimerVal').innerText = `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
                } else {
                    clearInterval(sessionTimer);
                    let banner = document.getElementById('statusBanner');
                    banner.className = 'status-banner';
                    banner.style.borderColor = 'var(--neon-gold)';
                    banner.style.color = 'var(--neon-gold)';
                    document.getElementById('bannerText').innerText = `🎉 30-MIN TELEMETRY SESSION COMPLETE! Total Crashes: ${crashesCount}`;
                    document.getElementById('bannerSub').innerText = "TRIGGERING DATASET V3 REBUILD & CUDA RETRAINING...";
                    triggerRetrain();
                }
            }, 1000);

            function selectModel(m) {
                selectedModel = m;
                document.getElementById('tabLaya').classList.toggle('active', m === 'laya');
                document.getElementById('tabJev').classList.toggle('active', m === 'jev');
                document.getElementById('activeModelTag').innerText = m === 'laya' ? 'LOCAL CUDA' : 'OPENROUTER API';
                document.getElementById('activeModelTag').style.color = m === 'laya' ? 'var(--neon-green)' : 'var(--neon-cyan)';
            }

            function spawnFood(pos = null, type = null) {
                let fType = type;
                if (!fType) {
                    let r = Math.random();
                    if (r < 0.08) fType = 'shrink';      // 8% chance magic shrink apple
                    else if (r < 0.16) fType = 'turbo';  // 8% chance turbo boost orb
                    else if (r < 0.24) fType = 'freeze'; // 8% chance time freeze orb
                    else if (r < 0.45) fType = 'golden'; // 21% chance golden apple
                    else if (r < 0.60) fType = 'poison'; // 15% chance poison apple
                    else fType = 'standard';            // 40% chance standard apple
                }

                let val = fType === 'golden' ? 3 : (fType === 'poison' ? -5 : (fType === 'shrink' ? 2 : 1));
                let expiresAt = (fType === 'poison' || fType === 'turbo' || fType === 'freeze') ? Date.now() + 6000 : (fType === 'shrink' ? Date.now() + 8000 : null);

                if (pos) {
                    // PROTECTION: Do NOT add item if food, obstacle, or portal exists on this grid cell!
                    if (foods.some(f => f.x === pos.x && f.y === pos.y)) return;
                    if (obstacles.some(o => o.x === pos.x && o.y === pos.y)) return;
                    if (portals.some(p => p.x === pos.x && p.y === pos.y)) return;
                    foods.push({x: pos.x, y: pos.y, type: fType, value: val, expiresAt: expiresAt});
                } else {
                    let empty = [];
                    for (let x = 0; x < GRID; x++) {
                        for (let y = 0; y < GRID; y++) {
                            if (!snake.some(s => s.x === x && s.y === y) &&
                                !foods.some(f => f.x === x && f.y === y) &&
                                !obstacles.some(o => o.x === x && o.y === y) &&
                                !portals.some(p => p.x === x && p.y === y)) {
                                empty.push({x, y});
                            }
                        }
                    }
                    if (empty.length > 0) {
                        let p = empty[Math.floor(Math.random() * empty.length)];
                        foods.push({x: p.x, y: p.y, type: fType, value: val, expiresAt: expiresAt});
                    }
                }
            }

            function getNearestGoodFood() {
                let goodFoods = foods.filter(f => f.type !== 'poison');
                let head = snake[0];
                if (enableFog) {
                    goodFoods = goodFoods.filter(f => (Math.abs(f.x - head.x) + Math.abs(f.y - head.y)) <= fogRadius);
                }
                if (goodFoods.length === 0) return {x: -1, y: -1, type: 'none'};
                return goodFoods.reduce((min, f) => {
                    let d = Math.abs(f.x - head.x) + Math.abs(f.y - head.y);
                    return d < min.d ? {f: f, d: d} : min;
                }, {f: goodFoods[0], d: 9999}).f;
            }

            function isObstacleOrPoison(pos) {
                let target = {x: pos.x, y: pos.y};
                // Account for portal warp
                if (enablePortals && portals.length >= 2) {
                    if (target.x === portals[0].x && target.y === portals[0].y) {
                        target = {x: portals[1].x, y: portals[1].y};
                    } else if (target.x === portals[1].x && target.y === portals[1].y) {
                        target = {x: portals[0].x, y: portals[0].y};
                    }
                }
                if (target.x < 0 || target.x >= GRID || target.y < 0 || target.y >= GRID) return true;
                if (snake.some(s => s.x === target.x && s.y === target.y)) return true; // tail included: moving into it is a crash too
                if (foods.some(f => f.x === target.x && f.y === target.y && f.type === 'poison')) return true;
                if (obstacles.some(o => o.x === target.x && o.y === target.y)) return true;
                if (enablePatrols && patrols.some(p => (p.x === target.x && p.y === target.y) || (p.x + p.dx === target.x && p.y + p.dy === target.y))) return true;
                return false;
            }

            // Same movement as the patrol step in masterLoop, on a copy
            function predictPatrolPath(p, steps) {
                let q = {...p}, path = [];
                for (let i = 0; i < steps; i++) {
                    q.x += q.dx; q.y += q.dy;
                    if (q.type === 'linear_horiz') {
                        if (q.x >= q.maxX) { q.x = q.maxX; q.dx = -1; }
                        else if (q.x <= q.minX) { q.x = q.minX; q.dx = 1; }
                    } else if (q.type === 'linear_vert') {
                        if (q.y >= q.maxY) { q.y = q.maxY; q.dy = -1; }
                        else if (q.y <= q.minY) { q.y = q.minY; q.dy = 1; }
                    }
                    path.push([q.x, q.y]);
                }
                return path;
            }

            function getCompactState() {
                let head = snake[0];
                let food = getNearestGoodFood();
                let foodDir = "NONE";
                if (food.x !== -1) {
                    let vert = food.y < head.y ? "UP" : (food.y > head.y ? "DOWN" : "");
                    let horiz = food.x < head.x ? "LEFT" : (food.x > head.x ? "RIGHT" : "");
                    foodDir = (vert && horiz) ? `${vert}_${horiz}` : (vert || horiz || "ON_FOOD");
                }
                return {
                    "danger_UP": isObstacleOrPoison({x: head.x, y: head.y - 1}),
                    "danger_DOWN": isObstacleOrPoison({x: head.x, y: head.y + 1}),
                    "danger_LEFT": isObstacleOrPoison({x: head.x - 1, y: head.y}),
                    "danger_RIGHT": isObstacleOrPoison({x: head.x + 1, y: head.y}),
                    "current_dir": currentDir,
                    "food_dir": foodDir,
                    "food_type": food.type || "standard",
                    "recent_actions": [...recentActions],
                    "head_pos": [head.x, head.y],
                    "food_pos": [food.x, food.y],
                    "snake_len": snake.length,
                    "obstacles": obstacles.map(o => [o.x, o.y]),
                    "portals": enablePortals ? portals.map(p => [p.x, p.y]) : [],
                    "patrols": enablePatrols ? patrols.map(p => [p.x, p.y, p.dx, p.dy]) : [],
                    // V9 model fields (RULES_V9.md). This classic UI has no phase walls or immortality.
                    "body": snake.slice(1, 31).map(s => [s.x, s.y]),
                    "immortal_steps": 0,
                    "phase_walls": [],
                    "drone_path": enablePatrols ? patrols.map(p => predictPatrolPath(p, 6)) : [],
                    "fog_of_war": enableFog,
                    "sight_radius": fogRadius
                };
            }

            // --- Advanced Long Snake Flood Fill & Tail Collision Prevention ---
            function floodFillSpace(startPos, customSnake) {
                let visited = new Set();
                visited.add(`${startPos.x},${startPos.y}`);
                let queue = [startPos];
                let bodySet = new Set(customSnake.map(s => `${s.x},${s.y}`));

                while (queue.length > 0) {
                    let curr = queue.shift();
                    for (let dir of ["UP", "DOWN", "LEFT", "RIGHT"]) {
                        let delta = ACTIONS[dir];
                        let nx = curr.x + delta.x, ny = curr.y + delta.y;
                        let key = `${nx},${ny}`;
                        if (nx >= 0 && nx < GRID && ny >= 0 && ny < GRID && !visited.has(key) && !bodySet.has(key)) {
                            if (!foods.some(f => f.x === nx && f.y === ny && f.type === 'poison')) {
                                visited.add(key);
                                queue.push({x: nx, y: ny});
                            }
                        }
                    }
                }
                return visited.size;
            }

            function canReachTail(startPos, customSnake) {
                let tail = customSnake[customSnake.length - 1];
                let queue = [startPos];
                let visited = new Set();
                visited.add(`${startPos.x},${startPos.y}`);
                let bodySet = new Set(customSnake.slice(0, -1).map(s => `${s.x},${s.y}`));

                while (queue.length > 0) {
                    let curr = queue.shift();
                    if (curr.x === tail.x && curr.y === tail.y) return true;

                    for (let dir of ["UP", "DOWN", "LEFT", "RIGHT"]) {
                        let delta = ACTIONS[dir];
                        let nx = curr.x + delta.x, ny = curr.y + delta.y;
                        let key = `${nx},${ny}`;
                        if (nx >= 0 && nx < GRID && ny >= 0 && ny < GRID && !visited.has(key) && !bodySet.has(key)) {
                            if (!foods.some(f => f.x === nx && f.y === ny && f.type === 'poison')) {
                                visited.add(key);
                                queue.push({x: nx, y: ny});
                            }
                        }
                    }
                }
                return false;
            }

            let cellLastVisit = {};

            function getExplorationMove(validMoves) {
                let head = snake[0];
                let tail = snake[snake.length - 1];

                let candidates = [];
                for (let dir of validMoves) {
                    let delta = ACTIONS[dir];
                    let np = {x: head.x + delta.x, y: head.y + delta.y};
                    let virtualSnake = [np, ...snake.slice(0, -1)];
                    let space = floodFillSpace(np, virtualSnake);
                    let reachTail = canReachTail(np, virtualSnake);

                    // Safety filter: Must have minimum space or be able to reach tail
                    if (space < Math.min(snake.length, 10) && !reachTail) {
                        continue;
                    }

                    // 1. Momentum: strongly prioritize continuing straight to sweep lines across grid
                    let momentum = (dir === currentDir) ? 50 : 0;

                    // 2. Anti-Recency Penalty: strongly avoid cells visited recently
                    let key = `${np.x},${np.y}`;
                    let lastStep = cellLastVisit[key] !== undefined ? cellLastVisit[key] : -999;
                    let stepsAgo = stepCount - lastStep;
                    let recencyPenalty = 0;
                    if (stepsAgo < 35) {
                        recencyPenalty = (35 - stepsAgo) * 25; // Massive penalty for recent loop
                    }

                    // 3. Neighborhood recency: avoid staying in the same 3x3 patch
                    let patchPenalty = 0;
                    for (let dx = -1; dx <= 1; dx++) {
                        for (let dy = -1; dy <= 1; dy++) {
                            let nKey = `${np.x + dx},${np.y + dy}`;
                            let nStep = cellLastVisit[nKey] !== undefined ? cellLastVisit[nKey] : -999;
                            if (stepCount - nStep < 15) {
                                patchPenalty += 8;
                            }
                        }
                    }

                    // 4. Spread out from tail (crucial for short snakes length <= 8 to prevent 2x2 loop!)
                    let tailDist = Math.abs(np.x - tail.x) + Math.abs(np.y - tail.y);
                    let tailScore = 0;
                    if (snake.length <= 8) {
                        tailScore = tailDist * 20; // Spread out away from tail!
                    } else {
                        tailScore = reachTail ? 40 : 0;
                    }

                    // 5. Open space bonus
                    let spaceBonus = Math.min(space, 50) * 2;

                    // 6. Lookahead wall clearance: bonus if path continues cleanly
                    let lookaheadBonus = 0;
                    let nnx = np.x + delta.x, nny = np.y + delta.y;
                    if (nnx >= 0 && nnx < GRID && nny >= 0 && nny < GRID && !isObstacleOrPoison({x: nnx, y: nny})) {
                        lookaheadBonus = 15;
                    }

                    let totalScore = momentum - recencyPenalty - patchPenalty + tailScore + spaceBonus + lookaheadBonus;
                    candidates.push({dir: dir, score: totalScore});
                }

                if (candidates.length === 0) {
                    return getTailChasingMove(validMoves);
                }

                candidates.sort((a, b) => b.score - a.score);
                return candidates[0].dir;
            }

            function getTailChasingMove(validMovesOverride = null) {
                let head = snake[0];
                let moves = validMovesOverride || ["UP", "DOWN", "LEFT", "RIGHT"].filter(dir => {
                    if (dir === OPPOSITES[currentDir] && snake.length > 1) return false;
                    let delta = ACTIONS[dir];
                    let np = {x: head.x + delta.x, y: head.y + delta.y};
                    return !isObstacleOrPoison(np);
                });

                if (moves.length === 0) return null;

                let tail = snake[snake.length - 1];
                let bestMove = moves[0];
                let maxScore = -9999;

                for (let dir of moves) {
                    let delta = ACTIONS[dir];
                    let np = {x: head.x + delta.x, y: head.y + delta.y};
                    let virtualSnake = [np, ...snake.slice(0, -1)];
                    let space = floodFillSpace(np, virtualSnake);
                    let reachTail = canReachTail(np, virtualSnake) ? 100 : 0;
                    let tailDist = Math.abs(np.x - tail.x) + Math.abs(np.y - tail.y);
                    
                    let scoreVal = space * 10 + reachTail - tailDist;
                    if (scoreVal > maxScore) {
                        maxScore = scoreVal;
                        bestMove = dir;
                    }
                }

                return bestMove;
            }

            function findBestFoodMove() {
                let head = snake[0];
                let food = getNearestGoodFood();

                let validMoves = ["UP", "DOWN", "LEFT", "RIGHT"].filter(dir => {
                    if (dir === OPPOSITES[currentDir] && snake.length > 1) return false;
                    let delta = ACTIONS[dir];
                    let np = {x: head.x + delta.x, y: head.y + delta.y};
                    return !isObstacleOrPoison(np);
                });

                if (validMoves.length === 0) return null;
                if (food.x === -1) return getExplorationMove(validMoves);

                let queue = [];
                let visited = new Set();
                visited.add(`${head.x},${head.y}`);

                for (let dir of validMoves) {
                    let delta = ACTIONS[dir];
                    let np = {x: head.x + delta.x, y: head.y + delta.y};
                    visited.add(`${np.x},${np.y}`);
                    queue.push({x: np.x, y: np.y, firstDir: dir});
                }

                let bfsPathDir = null;
                while (queue.length > 0) {
                    let curr = queue.shift();
                    if (curr.x === food.x && curr.y === food.y) {
                        bfsPathDir = curr.firstDir;
                        break;
                    }

                    for (let dir of ["UP", "DOWN", "LEFT", "RIGHT"]) {
                        let delta = ACTIONS[dir];
                        let nx = curr.x + delta.x, ny = curr.y + delta.y;
                        let key = `${nx},${ny}`;

                        if (nx >= 0 && nx < GRID && ny >= 0 && ny < GRID && !visited.has(key)) {
                            let pos = {x: nx, y: ny};
                            if (!isObstacleOrPoison(pos)) {
                                visited.add(key);
                                queue.push({x: nx, y: ny, firstDir: curr.firstDir});
                            }
                        }
                    }
                }

                // If BFS path found, verify taking this step does not trap a long snake in its own tail!
                if (bfsPathDir) {
                    let delta = ACTIONS[bfsPathDir];
                    let nextPos = {x: head.x + delta.x, y: head.y + delta.y};
                    let virtualSnake = [nextPos, ...snake.slice(0, -1)];
                    let space = floodFillSpace(nextPos, virtualSnake);
                    let reachTail = canReachTail(nextPos, virtualSnake);

                    if (space >= snake.length || reachTail) {
                        return bfsPathDir;
                    }
                }

                // Fall back to tail-chasing move when food path is trapping
                return getTailChasingMove(validMoves);
            }

            function render() {
                ctx.fillStyle = '#080d17';
                ctx.fillRect(0, 0, canvas.width, canvas.height);

                // Grid
                ctx.strokeStyle = 'rgba(0, 240, 255, 0.05)';
                ctx.lineWidth = 1;
                for (let i = 0; i <= GRID; i++) {
                    ctx.beginPath(); ctx.moveTo(i * CELL, 0); ctx.lineTo(i * CELL, canvas.height); ctx.stroke();
                    ctx.beginPath(); ctx.moveTo(0, i * CELL); ctx.lineTo(canvas.width, i * CELL); ctx.stroke();
                }

                // Render Internal Obstacle Walls
                obstacles.forEach(o => {
                    let ox = o.x * CELL, oy = o.y * CELL;
                    ctx.save();
                    ctx.shadowColor = 'rgba(255, 0, 85, 0.8)';
                    ctx.shadowBlur = 10;
                    ctx.fillStyle = '#162238';
                    ctx.fillRect(ox + 1, oy + 1, CELL - 2, CELL - 2);
                    ctx.strokeStyle = '#ff0055';
                    ctx.lineWidth = 2;
                    ctx.strokeRect(ox + 2, oy + 2, CELL - 4, CELL - 4);
                    ctx.fillStyle = '#ff0055';
                    ctx.font = 'bold 12px monospace';
                    ctx.fillText('🧱', ox + 4, oy + 18);
                    ctx.restore();
                });

                // Render Portals (Portal A & Portal B)
                if (enablePortals && portals.length >= 2) {
                    portals.forEach((p, pIdx) => {
                        let px = p.x * CELL + CELL/2, py = p.y * CELL + CELL/2;
                        let isA = pIdx === 0;
                        let color = isA ? '#00f0ff' : '#b026ff';
                        ctx.save();
                        ctx.shadowColor = color; ctx.shadowBlur = 15; ctx.fillStyle = color;
                        ctx.beginPath(); ctx.arc(px, py, 9, 0, Math.PI*2); ctx.fill();
                        ctx.fillStyle = '#060911'; ctx.font = 'bold 10px monospace';
                        ctx.fillText(isA ? 'A' : 'B', px - 3, py + 3);
                        ctx.restore();
                    });
                }

                // Filter expired food items
                let now = Date.now();
                foods = foods.filter(f => !f.expiresAt || f.expiresAt > now);
                let goodFoods = foods.filter(f => f.type !== 'poison');
                if (goodFoods.length === 0) {
                    spawnFood();
                }

                foods.forEach(f => {
                    let cx = f.x * CELL + CELL/2, cy = f.y * CELL + CELL/2;
                    ctx.save();
                    if (f.type === 'golden') {
                        ctx.shadowColor = '#ffd700'; ctx.shadowBlur = 15; ctx.fillStyle = '#ffd700';
                        ctx.beginPath(); ctx.arc(cx, cy, 8, 0, Math.PI*2); ctx.fill();
                    } else if (f.type === 'shrink') {
                        // Magic Shrink Apple: Glowing Cyan Circle with 🔮 Icon
                        ctx.shadowColor = '#00f0ff'; ctx.shadowBlur = 15; ctx.fillStyle = '#00f0ff';
                        ctx.beginPath(); ctx.arc(cx, cy, 8, 0, Math.PI*2); ctx.fill();
                        ctx.fillStyle = '#060911'; ctx.font = 'bold 11px monospace'; ctx.fillText('🔮', cx-5, cy+4);
                    } else if (f.type === 'turbo') {
                        // Turbo Speed Orb: Glowing Yellow Circle with ⚡ Icon
                        ctx.shadowColor = '#ffd700'; ctx.shadowBlur = 15; ctx.fillStyle = '#ffd700';
                        ctx.beginPath(); ctx.arc(cx, cy, 8, 0, Math.PI*2); ctx.fill();
                        ctx.fillStyle = '#060911'; ctx.font = 'bold 11px monospace'; ctx.fillText('⚡', cx-4, cy+4);
                    } else if (f.type === 'freeze') {
                        // Freeze Orb: Glowing Ice Blue Circle with ❄️ Icon
                        ctx.shadowColor = '#00ffff'; ctx.shadowBlur = 15; ctx.fillStyle = '#00ffff';
                        ctx.beginPath(); ctx.arc(cx, cy, 8, 0, Math.PI*2); ctx.fill();
                        ctx.fillStyle = '#060911'; ctx.font = 'bold 11px monospace'; ctx.fillText('❄️', cx-5, cy+4);
                    } else if (f.type === 'poison') {
                        ctx.shadowColor = '#b026ff'; ctx.shadowBlur = 15; ctx.fillStyle = '#b026ff';
                        ctx.beginPath(); ctx.arc(cx, cy, 7, 0, Math.PI*2); ctx.fill();
                        ctx.fillStyle = '#ffffff'; ctx.font = '10px monospace'; ctx.fillText('☠', cx-4, cy+3);
                    } else {
                        ctx.shadowColor = '#ff0055'; ctx.shadowBlur = 10; ctx.fillStyle = '#ff0055';
                        ctx.beginPath(); ctx.arc(cx, cy, 7, 0, Math.PI*2); ctx.fill();
                    }
                    ctx.restore();
                });

                // Render Dynamic Cyber Patrol Drones (🚨)
                if (enablePatrols) {
                    patrols.forEach(p => {
                        let px = p.x * CELL + CELL / 2;
                        let py = p.y * CELL + CELL / 2;
                        ctx.save();
                        ctx.shadowColor = '#ff0055';
                        ctx.shadowBlur = 18;
                        ctx.fillStyle = '#ff0055';
                        ctx.beginPath();
                        ctx.arc(px, py, 9, 0, Math.PI * 2);
                        ctx.fill();

                        ctx.fillStyle = '#ffffff';
                        ctx.beginPath();
                        ctx.arc(px, py, 4, 0, Math.PI * 2);
                        ctx.fill();

                        // Sentry laser scanner beam in direction of travel
                        ctx.strokeStyle = 'rgba(255, 0, 85, 0.45)';
                        ctx.lineWidth = 2;
                        ctx.setLineDash([3, 3]);
                        ctx.beginPath();
                        ctx.moveTo(px, py);
                        ctx.lineTo(px + p.dx * CELL * 2.5, py + p.dy * CELL * 2.5);
                        ctx.stroke();
                        ctx.restore();
                    });
                }

                // Render Fog of War Darkness Shroud if active (Spotlight follows Snake Head)
                if (enableFog) {
                    let hx = snake[0].x * CELL + CELL / 2;
                    let hy = snake[0].y * CELL + CELL / 2;
                    let grad = ctx.createRadialGradient(hx, hy, CELL * 1.2, hx, hy, fogRadius * CELL);
                    grad.addColorStop(0, 'rgba(6, 9, 17, 0.0)');
                    grad.addColorStop(0.65, 'rgba(6, 9, 17, 0.45)');
                    grad.addColorStop(1, 'rgba(6, 9, 17, 0.96)');

                    ctx.save();
                    ctx.fillStyle = grad;
                    ctx.fillRect(0, 0, canvas.width, canvas.height);

                    // Cyber scanner ring showing radar perimeter
                    ctx.strokeStyle = 'rgba(0, 240, 255, 0.35)';
                    ctx.lineWidth = 1.5;
                    ctx.setLineDash([5, 5]);
                    ctx.beginPath();
                    ctx.arc(hx, hy, fogRadius * CELL, 0, Math.PI * 2);
                    ctx.stroke();
                    ctx.restore();
                }

                // Render Snake Body (ALWAYS 100% VISIBLE)
                snake.forEach((s, idx) => {
                    let sx = s.x * CELL, sy = s.y * CELL;
                    ctx.save();
                    if (idx === 0) {
                        ctx.shadowColor = isGameOver ? '#ff0055' : '#00ff88'; ctx.shadowBlur = 12;
                        ctx.fillStyle = isGameOver ? '#ff0055' : '#00ff88';
                        ctx.fillRect(sx+2, sy+2, CELL-4, CELL-4);
                    } else {
                        ctx.fillStyle = isGameOver ? 'rgba(255, 0, 85, 0.6)' : 'rgba(0, 200, 100, 0.8)';
                        ctx.fillRect(sx+3, sy+3, CELL-6, CELL-6);
                    }
                    ctx.restore();
                });

                // Highlight Crashed Cell if Game Over
                if (isGameOver && crashedCell) {
                    ctx.save();
                    ctx.strokeStyle = '#ff0055';
                    ctx.lineWidth = 3;
                    ctx.shadowColor = '#ff0055';
                    ctx.shadowBlur = 20;
                    ctx.strokeRect(crashedCell.x * CELL + 1, crashedCell.y * CELL + 1, CELL - 2, CELL - 2);
                    
                    ctx.fillStyle = '#ff0055';
                    ctx.font = 'bold 16px monospace';
                    ctx.fillText('❌', crashedCell.x * CELL + 4, crashedCell.y * CELL + 18);
                    ctx.restore();
                }
            }

            let recentHeadPositions = [];

            async function executeStep() {
                if (isGameOver || isPredicting) return;
                isPredicting = true;

                let state = getCompactState();

                // Calculate instant local BFS action (0.1ms)
                let bfsAction = findBestFoodMove();

                // Update Danger Sensors Radar UI
                ["UP", "DOWN", "LEFT", "RIGHT"].forEach(d => {
                    let el = document.getElementById(`radar-${d}`);
                    let head = snake[0];
                    let delta = ACTIONS[d];
                    let np = {x: head.x + delta.x, y: head.y + delta.y};
                    
                    let hasPoison = foods.some(f => f.x === np.x && f.y === np.y && f.type === 'poison');
                    let hasWall = isObstacleOrPoison(np);

                    if (hasPoison) {
                        el.className = 'danger-cell poison'; el.innerText = `${d}: POISON`;
                    } else if (hasWall) {
                        el.className = 'danger-cell danger'; el.innerText = `${d}: DANGER`;
                    } else {
                        el.className = 'danger-cell safe'; el.innerText = `${d}: SAFE`;
                    }
                });

                let action = bfsAction || currentDir;
                let confidence = 1.0;
                let latency_ms = 0;
                let engine_name = "Local Pathfinder BFS";
                let probs = {"UP": 0.25, "DOWN": 0.25, "LEFT": 0.25, "RIGHT": 0.25};

                try {
                    // Timeout fetch after 150ms so UI NEVER freezes!
                    const controller = new AbortController();
                    const timeoutId = setTimeout(() => controller.abort(), 150);

                    let resp = await fetch('/predict', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({model: selectedModel, state: state}),
                        signal: controller.signal
                    });
                    clearTimeout(timeoutId);

                    if (resp.ok) {
                        let data = await resp.json();
                        let modelAction = data.action;
                        confidence = data.confidence;
                        latency_ms = data.latency_ms;
                        engine_name = data.engine;
                        probs = data.probabilities || probs;

                        // PURE NEURAL DRIVER WITH PHYSICAL ACTION MASKING:
                        // ModernBERT's probability distribution guides the decision.
                        // Actions that would directly crash into a wall, boundary, or patrol
                        // are masked out (-1.0) so the snake picks the model's highest ranked SAFE action!
                        let head = snake[0];
                        let candidates = ["UP", "DOWN", "LEFT", "RIGHT"].map(dir => {
                            let delta = ACTIONS[dir];
                            let np = {x: head.x + delta.x, y: head.y + delta.y};
                            let isDeadly = (dir === OPPOSITES[currentDir] && snake.length > 1) || isObstacleOrPoison(np);
                            return {
                                dir: dir,
                                prob: isDeadly ? -1.0 : (probs[dir] !== undefined ? probs[dir] : 0.0),
                                isSafe: !isDeadly
                            };
                        });

                        candidates.sort((a, b) => b.prob - a.prob);

                        if (candidates[0].isSafe) {
                            action = candidates[0].dir;
                        } else if (modelAction && modelAction !== OPPOSITES[currentDir]) {
                            action = modelAction;
                        } else if (bfsAction) {
                            action = bfsAction;
                        }
                    }
                } catch (e) {
                    // Fallback to BFS action seamlessly on network delay/timeout
                    if (bfsAction) action = bfsAction;
                } finally {
                    isPredicting = false;
                }

                if (ACTIONS[action] && (action !== OPPOSITES[currentDir] || snake.length === 1)) {
                    currentDir = action;
                }
                recentActions.push(currentDir);
                if (recentActions.length > 5) recentActions.shift();

                // Update UI Telemetry safely
                let elAction = document.getElementById('actionText'); if (elAction) elAction.innerText = currentDir;
                let elConf = document.getElementById('confidenceVal'); if (elConf) elConf.innerText = `${(confidence * 100).toFixed(1)}%`;
                let elEng = document.getElementById('engineBadge'); if (elEng) elEng.innerText = engine_name;
                let elLat = document.getElementById('latencyVal'); if (elLat) elLat.innerText = `${latency_ms} ms`;

                ["UP", "DOWN", "LEFT", "RIGHT"].forEach(act => {
                    let p = (probs[act] || 0) * 100;
                    let elFill = document.getElementById(`fill-${act}`); if (elFill) elFill.style.width = `${p.toFixed(1)}%`;
                    let elPct = document.getElementById(`pct-${act}`); if (elPct) elPct.innerText = `${p.toFixed(1)}%`;
                    let elItem = document.getElementById(`item-${act}`); if (elItem) elItem.classList.toggle('winner', act === currentDir);
                });

                stepCount++;
                let delta = ACTIONS[currentDir];
                let newHead = {x: snake[0].x + delta.x, y: snake[0].y + delta.y};

                // Portal Warp Check
                if (enablePortals && portals.length >= 2) {
                    if (newHead.x === portals[0].x && newHead.y === portals[0].y) {
                        newHead = {x: portals[1].x, y: portals[1].y};
                    } else if (newHead.x === portals[1].x && newHead.y === portals[1].y) {
                        newHead = {x: portals[0].x, y: portals[0].y};
                    }
                }

                let oldHead = {x: snake[0].x, y: snake[0].y};
                let oldPatrols = patrols.map(p => ({x: p.x, y: p.y}));

                // Move Patrol Drones
                if (enablePatrols) {
                    patrols.forEach(p => {
                        p.x += p.dx;
                        p.y += p.dy;
                        if (p.type === 'linear_horiz') {
                            if (p.x >= p.maxX) { p.x = p.maxX; p.dx = -1; }
                            else if (p.x <= p.minX) { p.x = p.minX; p.dx = 1; }
                        } else if (p.type === 'linear_vert') {
                            if (p.y >= p.maxY) { p.y = p.maxY; p.dy = -1; }
                            else if (p.y <= p.minY) { p.y = p.minY; p.dy = 1; }
                        }
                    });
                }

                // Check Patrol Collision (both same cell landing and head-on swap crossing)
                let hitPatrol = false;
                if (enablePatrols) {
                    for (let i = 0; i < patrols.length; i++) {
                        let p = patrols[i];
                        let op = oldPatrols[i];
                        // Direct landing on patrol
                        if (p.x === newHead.x && p.y === newHead.y) {
                            hitPatrol = true; break;
                        }
                        // Swap crossing (moving through each other on the same step)
                        if (oldHead.x === p.x && oldHead.y === p.y && newHead.x === op.x && newHead.y === op.y) {
                            hitPatrol = true; break;
                        }
                        // Patrol steps onto any snake segment (disabled until the model is trained on body hits, dataset V8)
                        if (PATROL_BODY_HITS && snake.some(s => s.x === p.x && s.y === p.y)) {
                            hitPatrol = true; break;
                        }
                    }
                }

                // Infinite Loop / Deadlock Detector
                recentHeadPositions.push(`${newHead.x},${newHead.y}`);
                if (recentHeadPositions.length > 16) recentHeadPositions.shift();
                
                let posCounts = {};
                let isLooping = false;
                for (let posStr of recentHeadPositions) {
                    posCounts[posStr] = (posCounts[posStr] || 0) + 1;
                    if (posCounts[posStr] >= 4) {
                        isLooping = true;
                        break;
                    }
                }

                // Strict Collision Check: Boundary, Internal Walls, Self Body, Poison, Patrols, or Loop
                let isOutOfBounds = newHead.x < 0 || newHead.x >= GRID || newHead.y < 0 || newHead.y >= GRID;
                let hitWall = obstacles.some(o => o.x === newHead.x && o.y === newHead.y);
                let hitBody = snake.some(s => s.x === newHead.x && s.y === newHead.y);
                let hitPoison = foods.find(f => f.x === newHead.x && f.y === newHead.y && f.type === 'poison');

                if (isOutOfBounds || hitWall || hitBody || hitPatrol || hitPoison || isLooping) {
                    crashedCell = newHead;
                    crashedAction = action;
                    let reason = isLooping ? "Infinite Loop / Corner Deadlock" :
                                 (hitPatrol ? "Patrol Drone Collision" :
                                 (hitPoison ? "Ate Poison Apple" :
                                 (hitWall ? "Hit Internal Obstacle Wall" : "Boundary/Body Collision")));
                    triggerCrash(state, action, reason);
                    return;
                }

                snake.unshift(newHead);
                cellLastVisit[`${newHead.x},${newHead.y}`] = stepCount;

                // Check Food Eat
                let eatenIdx = foods.findIndex(f => f.x === newHead.x && f.y === newHead.y);
                if (eatenIdx !== -1) {
                    let eaten = foods[eatenIdx];
                    score += eaten.value;

                    // Magic Shrink Apple Logic: Shrinks snake length by 2 (min length 3)
                    if (eaten.type === 'shrink') {
                        if (snake.length > 3) snake.pop();
                        if (snake.length > 3) snake.pop();
                        let banner = document.getElementById('statusBanner');
                        banner.className = 'status-banner';
                        banner.style.borderColor = 'var(--neon-cyan)';
                        banner.style.color = 'var(--neon-cyan)';
                        document.getElementById('bannerText').innerText = "🔮 MAGIC SHRINK APPLE EATEN! Length Reduced (-2)";
                    } else if (eaten.type === 'turbo') {
                        tickInterval = 60;
                        let banner = document.getElementById('statusBanner');
                        if (banner) {
                            banner.className = 'status-banner';
                            banner.style.borderColor = 'var(--neon-gold)';
                            banner.style.color = 'var(--neon-gold)';
                            document.getElementById('bannerText').innerText = "⚡ TURBO BOOST ACTIVE (2X SPEED)";
                        }
                        setTimeout(() => { tickInterval = 120; }, 5000);
                    } else if (eaten.type === 'freeze') {
                        tickInterval = 220;
                        let banner = document.getElementById('statusBanner');
                        if (banner) {
                            banner.className = 'status-banner';
                            banner.style.borderColor = 'var(--neon-cyan)';
                            banner.style.color = 'var(--neon-cyan)';
                            document.getElementById('bannerText').innerText = "❄️ TIME FREEZE ACTIVE (0.5X SPEED)";
                        }
                        setTimeout(() => { tickInterval = 120; }, 5000);
                    }

                    foods.splice(eatenIdx, 1);
                    recentHeadPositions = [];
                    document.getElementById('scoreVal').innerText = score;
                    if (score > maxScore) {
                        maxScore = score;
                        document.getElementById('maxScoreVal').innerText = maxScore;
                    }
                    if (foods.filter(f => f.type !== 'poison').length === 0) spawnFood();
                } else {
                    snake.pop();
                }

                let banner = document.getElementById('statusBanner');
                if (banner && !isGameOver && !banner.className.includes('crashed')) {
                    let bText = document.getElementById('bannerText');
                    if (bText && !bText.innerText.includes('TURBO') && !bText.innerText.includes('FREEZE') && !bText.innerText.includes('SHRINK')) {
                        if (enableFog && getNearestGoodFood().x === -1) {
                            banner.style.borderColor = 'var(--neon-cyan)';
                            banner.style.color = 'var(--neon-cyan)';
                            bText.innerText = "🌫️ FOG PATROL ACTIVE // SWEEPING ARENA";
                        } else {
                            banner.style.borderColor = 'var(--neon-green)';
                            banner.style.color = 'var(--neon-green)';
                            bText.innerText = "TARGET LOCKED // SEEKING FOOD";
                        }
                    }
                }

                document.getElementById('lengthVal').innerText = snake.length;
                render();
            }

            async function triggerCrash(state, wrongAction, reason) {
                isGameOver = true;
                crashesCount++;
                document.getElementById('crashesCountVal').innerText = crashesCount;

                let banner = document.getElementById('statusBanner');
                banner.className = 'status-banner crashed';
                document.getElementById('bannerText').innerText = `💥 CRASH DETECTED at [${crashedCell.x}, ${crashedCell.y}] (${reason})`;
                document.getElementById('bannerSub').innerText = `CRASH RECORDED TO DATASET`;

                try {
                    await fetch('/record_crash', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            state: state,
                            wrong_action: wrongAction,
                            crash_reason: reason,
                            model_used: selectedModel
                        })
                    });
                } catch(e) { console.error("Crash logging error:", e); }

                render();

                const autoReboot = document.getElementById('autoRebootCheck').checked;
                if (autoReboot) {
                    setTimeout(() => {
                        resetGame();
                    }, 2000);
                }
            }

            function masterLoop() {
                if (isRunning && !isGameOver) {
                    executeStep();
                }
                render();
                setTimeout(masterLoop, tickInterval);
            }

            function togglePlayPause() {
                isRunning = !isRunning;
                document.getElementById('btnPlay').innerText = isRunning ? "⏸ PAUSE" : "▶ RESUME";
            }

            function resetGame() {
                isGameOver = false;
                crashedCell = null;
                crashedAction = null;
                recentHeadPositions = [];
                cellLastVisit = {};
                recentActions = ["RIGHT", "RIGHT", "RIGHT", "RIGHT", "RIGHT"];
                score = 0; stepCount = 0; currentDir = "RIGHT";
                snake = [{x: 10, y: 10}, {x: 9, y: 10}, {x: 8, y: 10}];
                foods = [];
                generateObstacles();
                generatePortals();
                generatePatrols();
                spawnFood();

                let banner = document.getElementById('statusBanner');
                banner.className = 'status-banner';
                banner.style.borderColor = 'var(--neon-green)';
                banner.style.color = 'var(--neon-green)';
                document.getElementById('bannerText').innerText = "SIMULATION ACTIVE // NO COLLISIONS";
                document.getElementById('bannerSub').innerText = "STEP #0";
                document.getElementById('scoreVal').innerText = "0";
                document.getElementById('lengthVal').innerText = "3";
                render();
            }

            function exportCrashes() {
                window.location.href = '/export_crashes';
            }

            async function clearCrashesUI() {
                crashesCount = 0;
                let elCount = document.getElementById('crashesCountVal');
                if (elCount) elCount.innerText = "0";
                try {
                    await fetch('/clear_crashes', {method: 'POST'});
                    let banner = document.getElementById('statusBanner');
                    if (banner) {
                        banner.className = 'status-banner';
                        banner.style.borderColor = 'var(--neon-green)';
                        banner.style.color = 'var(--neon-green)';
                        document.getElementById('bannerText').innerText = "🧹 TELEMETRY CRASH LOG ARCHIVED & RESET";
                    }
                } catch(e) { console.error("Error clearing crashes:", e); }
            }

            async function triggerRetrain() {
                let banner = document.getElementById('statusBanner');
                banner.className = 'status-banner';
                banner.style.borderColor = 'var(--neon-gold)';
                banner.style.color = 'var(--neon-gold)';
                document.getElementById('bannerText').innerText = "🚀 RETRAINING LAUNCHED ON CUDA GPU!";
                document.getElementById('bannerSub').innerText = "DATASET V4 (OBSTACLE-AWARE) REBUILDING IN BACKGROUND...";
                try {
                    let r = await fetch('/trigger_retrain', {method: 'POST'});
                    let res = await r.json();
                    console.log("[INFO] Retraining response:", res);
                } catch(e) { console.error("Error triggering retraining:", e); }
            }

            canvas.addEventListener('click', (e) => {
                let rect = canvas.getBoundingClientRect();
                let gx = Math.floor((e.clientX - rect.left) / CELL);
                let gy = Math.floor((e.clientY - rect.top) / CELL);
                if (foods.some(f => f.x === gx && f.y === gy)) return;
                if (obstacles.some(o => o.x === gx && o.y === gy)) return;
                spawnFood({x: gx, y: gy});
                render();
            });

            generateObstacles();
            spawnFood();
            masterLoop();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# 3D frontend (static, no build step): http://localhost:9500/3d/
from fastapi.staticfiles import StaticFiles
WEB3D_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web3d")
if os.path.isdir(WEB3D_DIR):
    app.mount("/3d", StaticFiles(directory=WEB3D_DIR, html=True), name="web3d")

if __name__ == "__main__":
    print("Starting Snake AI FastAPI Web Server at http://localhost:9500")
    uvicorn.run(app, host="0.0.0.0", port=9500)
