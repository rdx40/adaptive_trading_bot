import os
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from app.rl.trading_env import TradingEnv
import matplotlib.pyplot as plt
import json


def train_agent():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root_dir, "data", "AAPL.csv")
    print(f"Reading data from: {data_path}")

    df = pd.read_csv(data_path)

    # Setup training environment
    env = DummyVecEnv([lambda df=df: Monitor(TradingEnv(df, legacy_mode=legacy_mode))])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # Train model
    # In train_agent() and main.py /train endpoint
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
    model.learn(total_timesteps=500_000)

    # Save model and normalization stats
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_trading_best")
    env.save("models/ppo_trading_env_stats.pkl")
    version_info = {
        "legacy_mode": legacy_mode,
        "window_size": 10,
        "observation_shape": env.observation_space.shape
    }
    with open("models/version_info.json", "w") as f:
        json.dump(version_info, f)
    return {
        "message": "Training complete",
        "model_path": "models/ppo_trading_best.zip",
        "env_stats_path": "models/ppo_trading_env_stats.pkl",
        "tensorboard_log": "ppo_trading_tensorboard"
    }


def run_inference(model_path="models/ppo_trading_best.zip", 
                 env_stats_path="models/ppo_trading_env_stats.pkl", 
                 test_data_path=None):
    # Determine data path
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = test_data_path or os.path.join(root_dir, "data", "AAPL.csv")
    df = pd.read_csv(data_path)

    # Set up test environment
    env = DummyVecEnv([lambda df=df: Monitor(TradingEnv(df))])
    
    # Try to load normalization stats, but handle shape mismatches
    try:
        if os.path.exists(env_stats_path):
            # Attempt to load normalization
            env = VecNormalize.load(env_stats_path, env)
            env.training = False
            env.norm_reward = False
            print("✅ Loaded normalization stats successfully")
        else:
            print("⚠️ No normalization stats found. Using unnormalized environment")
    except (AssertionError, ValueError) as e:
        print(f"❌ Normalization load error: {str(e)}")
        print("🔄 Creating new normalization environment")
        env = VecNormalize(env, training=False, norm_reward=False)

    # Load the trained model
    try:
        model = PPO.load(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}")

    # Run simulation
    obs = env.reset()
    done = False
    total_reward = 0.0
    rewards = []
    actions = []

    while not done:
        try:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            total_reward += reward[0]
            rewards.append(float(reward[0]))
            actions.append(int(action[0]))
        except Exception as e:
            print(f"❌ Simulation error at step {len(rewards)}: {str(e)}")
            done = True

    # Save simulation logs
    log_path = os.path.join("logs", "simulation_log.json")
    os.makedirs("logs", exist_ok=True)
    with open(log_path, "w") as f:
        json.dump({
            "rewards": rewards,
            "actions": actions
        }, f, indent=4)

    print(f"✅ Total reward from simulation: {total_reward:.2f}")
    return {
        "total_reward": float(total_reward),
        "average_step_reward": float(np.mean(rewards)),
        "num_steps": len(rewards),
        "rewards": rewards
    }





def compute_performance_metrics(rewards: list[float]) -> dict:
    rewards = np.array(rewards)
    total_reward = float(np.sum(rewards))
    avg_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    
    # Simple Sharpe ratio approximation (risk-free rate = 0)
    sharpe_ratio = float(avg_reward / std_reward) if std_reward != 0 else 0.0

    return {
        "total_reward": total_reward,
        "average_reward": avg_reward,
        "num_steps": len(rewards),
        "sharpe_ratio": sharpe_ratio
    }
