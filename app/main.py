from fastapi import FastAPI, APIRouter
from typing import List
import numpy as np
import json
from stable_baselines3 import PPO
from app.rl.agent import train_agent, run_inference, compute_performance_metrics
from app.utils.performance import compute_kpis
import os
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Adaptive Trading Bot API"}

@app.post("/train")
def train_agent_route():
    result = train_agent()
    return {"status": "Training completed", "details": result}

@app.post("/simulate")
def simulate_trading():
    result = run_inference()
    return result

@app.get("/performance")
def get_performance():
    log_path = "logs/simulation_log.json"
    if not os.path.exists(log_path):
        return {"error": "Simulation log not found. Run /simulate first."}

    with open(log_path, "r") as f:
        log = json.load(f)

    kpis = compute_kpis(log["rewards"])
    return {"performance": kpis}

@app.post("/reset")
def reset_agent():
    return {"status": "Agent/environment reset (not implemented yet)"}

@app.post("/predict")
def predict_action(observation: List[float]):
    model = PPO.load("models/ppo_trading_best.zip")  # ✅ Correct model path
    obs = np.array(observation).reshape(1, -1)
    action, _states = model.predict(obs, deterministic=True)
    return {"action": int(action)}


@app.get("/log")
def get_action_log():
    result = run_inference()
    return {
        "actions": result["actions"],
        "rewards": result["rewards"],
        "num_steps": result["num_steps"]
    }
