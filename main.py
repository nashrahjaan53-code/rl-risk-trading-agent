import os
from stable_baselines3 import PPO
from src.data_pipeline import fetch_and_prepare_data
from src.env import RiskConstrainedTradingEnv

def main():
    ticker = "AAPL"
    start_train = "2022-01-01"
    end_train = "2025-01-01"  
    start_test = "2025-01-02"
    end_test = "2026-06-01"    
    
    print(" Phase 1: Data Preparation ")
    train_df = fetch_and_prepare_data(ticker, start_train, end_train)
    test_df = fetch_and_prepare_data(ticker, start_test, end_test)
  
    train_env = RiskConstrainedTradingEnv(train_df, initial_balance=10000, max_drawdown_limit=0.10)
    test_env = RiskConstrainedTradingEnv(test_df, initial_balance=10000, max_drawdown_limit=0.10)
    
    
    print("\n Phase 2: Initializing PPO Agent")
    model = PPO(
        "MlpPolicy", 
        train_env, 
        verbose=1, 
        learning_rate=0.0003,
        tensorboard_log="./tensorboard_logs/"
    )
    
    print("\n Phase 3: Training Agent ")
    total_timesteps = 50000 
    model.learn(total_timesteps=total_timesteps)
    print("Training finished! Saving model...")
    model.save("ppo_risk_trading_agent")
    print("\n Phase 4: Backtesting on Unseen Data")
    obs, info = test_env.reset()
    done = False
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        
        done = terminated or truncated
        
    print("\n Backtest Results Summary ")
    print(f"Final Portfolio Net Worth: ${info.get('net_worth', 10000):.2f}")
    print(f"Final Episode Max Drawdown Encountered: {info.get('drawdown', 0):.2%}")

if __name__ == "__main__":
    main()