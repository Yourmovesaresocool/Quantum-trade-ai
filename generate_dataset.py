import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# List of 100+ company symbols
COMPANIES = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'BRK.B', 'JPM', 'JNJ',
    'V', 'WMT', 'PG', 'MA', 'UNH', 'HD', 'DIS', 'PYPL', 'NFLX', 'ADBE',
    'CMCSA', 'VZ', 'INTC', 'T', 'PFE', 'MRK', 'KO', 'PEP', 'ABT', 'NKE',
    'CSCO', 'TMO', 'ABBV', 'ACN', 'AVGO', 'TXN', 'COST', 'MDT', 'NEE', 'BMY',
    'UNP', 'LIN', 'HON', 'QCOM', 'UPS', 'AMT', 'RTX', 'LOW', 'BA', 'SBUX',
    'CHTR', 'IBM', 'INTU', 'CVS', 'AMAT', 'CAT', 'DE', 'GS', 'BLK', 'AXP',
    'ISRG', 'GILD', 'LMT', 'BKNG', 'MMM', 'TJX', 'MDLZ', 'CI', 'MO', 'SYK',
    'CB', 'TMUS', 'ZTS', 'CL', 'SO', 'USB', 'PLD', 'DUK', 'ADI', 'EOG',
    'CSX', 'NSC', 'NOC', 'TGT', 'SPGI', 'CCI', 'ITW', 'GD', 'EL', 'AON',
    'BTC', 'ETH', 'BNB', 'XRP', 'ADA', 'DOGE', 'SOL', 'DOT', 'MATIC', 'LINK',
    # Add more symbols
    'F', 'GM', 'UBER', 'LYFT', 'SNAP', 'TWTR', 'SQ', 'SHOP', 'SPOT', 'ZM',
    'DOCU', 'CRWD', 'PLTR', 'COIN', 'RBLX', 'ROKU', 'PINS', 'DKNG', 'ABNB', 'DASH'
]

def generate_stock_prices(symbol, start_date, end_date, initial_price=100):
    """
    Generate realistic stock price data using random walk with drift
    """
    # Calculate number of days
    days = (end_date - start_date).days
    
    # Parameters for price generation
    drift = random.uniform(-0.0002, 0.0005)  # Daily drift (trend)
    volatility = random.uniform(0.01, 0.03)  # Daily volatility
    
    # Initialize
    prices = [initial_price]
    dates = [start_date]
    
    # Generate prices using geometric Brownian motion
    for i in range(1, days):
        # Random daily return
        daily_return = drift + volatility * np.random.randn()
        
        # Calculate new price
        new_price = prices[-1] * (1 + daily_return)
        
        # Ensure price doesn't go below $1
        new_price = max(new_price, 1.0)
        
        prices.append(new_price)
        dates.append(start_date + timedelta(days=i))
    
    # Generate OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Generate realistic OHLC based on close price
        high = close * random.uniform(1.0, 1.02)
        low = close * random.uniform(0.98, 1.0)
        open_price = random.uniform(low, high)
        
        # Volume (higher volume = more volatile)
        base_volume = random.randint(1000000, 10000000)
        volume = int(base_volume * (1 + abs(daily_return) if i > 0 else 1))
        
        data.append({
            'symbol': symbol,
            'timestamp': date.strftime('%Y-%m-%d %H:%M:%S'),
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume
        })
    
    return data

def generate_all_stock_data(num_companies=120, years=5):
    """
    Generate stock data for multiple companies
    """
    print(f"Generating data for {num_companies} companies over {years} years...")
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    all_data = []
    
    # Select random companies
    selected_companies = random.sample(COMPANIES, min(num_companies, len(COMPANIES)))
    
    for i, symbol in enumerate(selected_companies, 1):
        print(f"Generating {symbol}... ({i}/{num_companies})")
        
        # Random initial price based on company type
        if symbol in ['BTC', 'ETH']:
            initial_price = random.uniform(1000, 50000)
        elif symbol in ['BRK.B', 'GOOGL', 'AMZN']:
            initial_price = random.uniform(1000, 3000)
        else:
            initial_price = random.uniform(10, 500)
        
        # Generate data
        stock_data = generate_stock_prices(symbol, start_date, end_date, initial_price)
        all_data.extend(stock_data)
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    print(f"\nGenerated {len(df)} total records")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Companies: {df['symbol'].nunique()}")
    
    return df

def save_dataset(df, filename='stock_data.csv'):
    """Save dataset to CSV"""
    df.to_csv(filename, index=False)
    print(f"\n✅ Dataset saved to {filename}")
    print(f"File size: {round(len(df) * 100 / 1024 / 1024, 2)} MB (approx)")

def display_sample(df):
    """Display sample data"""
    print("\n" + "="*50)
    print("SAMPLE DATA")
    print("="*50)
    print(df.head(10))
    print("\n" + "="*50)
    print("STATISTICS")
    print("="*50)
    print(df.groupby('symbol').agg({
        'close': ['count', 'min', 'max', 'mean']
    }).head(10))

if __name__ == "__main__":
    # Generate dataset
    df = generate_all_stock_data(num_companies=120, years=5)
    
    # Display sample
    display_sample(df)
    
    # Save
    save_dataset(df, 'stock_data.csv')
    
    print("\n🎉 Dataset generation complete!")
    print("Next step: Upload this data to your database using upload_to_db.py")