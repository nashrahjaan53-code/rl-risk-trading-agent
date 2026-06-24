import yfinance as yf
import pandas as pd
import numpy as np
# Switched from pandas_ta to ta
import ta
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

def fetch_and_prepare_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical market data and calculates technical indicators 
    using the 'ta' library to serve as features for the RL agent.
    """
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    
    # 1. Download data from Yahoo Finance
    df = yf.download(ticker, start=start_date, end=end_date)
    
    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}. Check the symbol and dates.")
        
    # Flatten multi-index columns if necessary
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    # 2. Feature Engineering using 'ta'
    # Relative Strength Index (RSI)
    rsi_ind = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi_ind.rsi()
    
    # Moving Average Convergence Divergence (MACD)
    macd_ind = MACD(close=df['Close'], window_fast=12, window_slow=26, window_sign=9)
    df['MACD'] = macd_ind.macd()
    df['MACD_signal'] = macd_ind.macd_signal()
    
    # Bollinger Bands
    bb_ind = BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_upper'] = bb_ind.bollinger_hband()
    df['BB_lower'] = bb_ind.bollinger_lband()
    
    # Daily Returns
    df['Daily_Return'] = df['Close'].pct_change()
    
    # 3. Clean up data
    # Drop rows with NaN values caused by indicator lookback windows
    df.dropna(inplace=True)
    
    print(f"Data preparation complete. Shape: {df.shape}")
    return df

if __name__ == "__main__":

    try:
        data = fetch_and_prepare_data("AAPL", "2023-01-01", "2026-01-01")
        print("\nFirst few rows of processed features:")
        print(data[['Close', 'RSI', 'MACD', 'BB_upper']].head())
    except Exception as e:
        print(f"An error occurred: {e}")