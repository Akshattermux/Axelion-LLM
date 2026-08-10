import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_stock_data(ticker_symbol: str, days: int = 30) -> str:
    """
    Fetches historical stock data and formats it into a prompt string for the LLM.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Download data using yfinance
    data = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False)
    
    if data.empty:
        return f"[TICKER: {ticker_symbol}] [ERROR: NO DATA FOUND]"

    formatted_text = f"Financial Report for {ticker_symbol}:\n\n"
    
    # Convert numerical data to a textual sequence
    for date, row in data.iterrows():
        date_str = date.strftime('%Y-%m-%d')
        close_price = row['Close']
        if isinstance(close_price, pd.Series):
             close_price = close_price.iloc[0]
        close_price = round(float(close_price), 2)
        
        open_price = row['Open']
        if isinstance(open_price, pd.Series):
             open_price = open_price.iloc[0]
        open_price = round(float(open_price), 2)
        
        movement = "UP" if close_price > open_price else "DOWN"
        
        # Example format the LLM can learn from
        formatted_text += f"[DATE: {date_str}] [OPEN: {open_price}] [CLOSE: {close_price}] -> [MOVEMENT: {movement}]\n"
        
    return formatted_text

def create_training_dataset(tickers: list[str], output_file: str):
    """
    Fetches data for multiple tickers and saves it to a text file for LLM pre-tokenization.
    """
    print(f"Generating financial training dataset to {output_file}...")
    with open(output_file, 'w') as f:
        for ticker in tickers:
            try:
                text = fetch_stock_data(ticker, days=365) # 1 year of data
                f.write(text + "\n<|endoftext|>\n")
                print(f"Processed {ticker}")
            except Exception as e:
                print(f"Failed to process {ticker}: {e}")

if __name__ == '__main__':
    # Simple test
    test_tickers = ['AAPL', 'MSFT', 'TSLA']
    create_training_dataset(test_tickers, 'financial_train_data.txt')
    print("Dataset generated. You can now use dataset.py to convert this to a .bin file for training.")
