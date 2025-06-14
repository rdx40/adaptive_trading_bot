from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from typing import List, Optional
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from app.rl.trading_env import TradingEnv
from stable_baselines3.common.monitor import Monitor
import numpy as np
import json
from stable_baselines3 import PPO
from app.rl.agent import run_inference
from app.utils.performance import compute_kpis
import os
from fastapi.responses import JSONResponse
from io import BytesIO

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Adaptive Trading Bot API"}

@app.post("/train")
async def train(file: UploadFile = File(...)):
    try:
        # Validate file type
        if file.content_type not in ['text/csv', 'application/vnd.ms-excel']:
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Only CSV files are allowed"
            )
        
        # Read CSV
        contents = await file.read()
        try:
            df = pd.read_csv(BytesIO(contents))
        except Exception as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Error parsing CSV: {str(e)}"
            )
        
        print("DEBUG: CSV columns ->", df.columns.tolist())

        # Create a copy of original columns for reference
        original_columns = df.columns.tolist()
        lower_columns = [col.lower() for col in original_columns]
        
        # Find potential price columns
        price_candidates = []
        for candidate in ['close', 'price', 'last', 'value']:
            if candidate in lower_columns:
                idx = lower_columns.index(candidate)
                price_candidates.append(original_columns[idx])
        
        # Ensure we have at least one price column
        if not price_candidates:
            raise HTTPException(
                status_code=400,
                detail="CSV missing price column. Need 'Close' or equivalent"
            )
        
        # Use the first candidate as our Close column
        close_col = price_candidates[0]
        
        # Rename if needed
        if close_col != 'Close':
            df.rename(columns={close_col: 'Close'}, inplace=True)
            print(f"Renamed column '{close_col}' to 'Close'")
        
        # Ensure Close is numeric
        try:
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            nan_count = df['Close'].isnull().sum()
            if nan_count > 0:
                print(f"Warning: {nan_count} non-numeric values found in Close column")
                df = df.dropna(subset=['Close'])  # Remove rows with invalid prices
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid Close column: {str(e)}"
            )
        
        # Check if we have enough data
        if len(df) < 100:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data after cleaning. Only {len(df)} rows remain."
            )
        
        print(f"Using {len(df)} rows with Close prices ranging from {df['Close'].min()} to {df['Close'].max()}")

        # Create environment with normalization
        env = DummyVecEnv([lambda df=df: Monitor(TradingEnv(df))])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        print("✅ Environment created successfully")

        # Train model
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            tensorboard_log="./tensorboard"
        )
        print("✅ Model created successfully. Starting training...")
        model.learn(total_timesteps=500_000)
        print("✅ Training completed successfully")
        
        # Save model and normalization stats
        os.makedirs("models", exist_ok=True)
        model.save("models/ppo_trading_best")
        env.save("models/ppo_trading_env_stats.pkl")
        print("💾 Model and environment stats saved")
        
        return {"message": "Training completed"}
    
    except Exception as e:
        import traceback
        traceback.print_exc()  # Print detailed error to console
        raise HTTPException(
            status_code=500, 
            detail=f"Training failed: {str(e)}"
        )
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
    try:
        model = PPO.load("models/ppo_trading_best.zip")
        obs = np.array(observation).reshape(1, -1)
        action, _states = model.predict(obs, deterministic=True)
        return {"action": int(action[0])}  # Return the action value
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/log")
def get_action_log():
    try:
        result = run_inference()
        return {
            "actions": result.get("actions", []),
            "rewards": result.get("rewards", []),
            "num_steps": result.get("num_steps", 0)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get action log: {str(e)}"
        )