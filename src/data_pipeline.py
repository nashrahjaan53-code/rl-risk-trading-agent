import yfinance as yf
import pandas as pd
import pandas_ta as ta

def fetch_and_prepare_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:

    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    

    df = yf.download(ticker, start=start_date, end=end_date)
    
    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}. Check the symbol and dates.")
        
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    
    # Bollinger Bands for volatility tracking
    bbands = ta.bbands(df['Close'], length=20, std=2)
    df['BB_upper'] = bbands['BBU_20_2.0']
    df['BB_lower'] = bbands['BBL_20_2.0']
    
    # Daily Returns (useful for calculating reward metrics later)
    df['Daily_Return'] = df['Close'].pct_change()
    
    # 3. Clean up data
    # Drop rows with NaN values caused by indicator lookback windows
    df.dropna(inplace=True)
    
    print(f"Data preparation complete. Shape: {df.shape}")
    return df

if __name__ == "__main__":
    data = fetch_and_prepare_data("AAPL", "2023-01-01", "2026-01-01")
    print(data[['Close', 'RSI', 'MACD', 'BB_upper']].head())