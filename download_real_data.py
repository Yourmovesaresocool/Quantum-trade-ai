"""
REAL STOCK DATA DOWNLOADER
Downloads actual market data from Yahoo Finance (FREE)
Run this ONCE to replace all fake data
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

# ============================================
# STOCKS TO DOWNLOAD (Your existing stocks)
# ============================================
STOCKS = [
    # Tech Giants
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AMD', 'INTC',
    'ORCL', 'CRM', 'ADBE', 'IBM', 'CSCO',
    
    # Auto & EV
    'TSLA', 'F', 'GM',
    
    # Finance
    'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'PYPL',
    
    # Retail
    'WMT', 'TGT', 'COST', 'HD', 'LOW', 'NKE', 'SBUX', 'MCD',
    
    # Media
    'DIS', 'NFLX', 'CMCSA',
    
    # Aerospace
    'BA', 'LMT', 'RTX',
    
    # Healthcare
    'JNJ', 'PFE', 'UNH', 'ABBV', 'TMO',
    
    # Energy
    'XOM', 'CVX',
    
    # Telecom
    'T', 'VZ'
]

CRYPTO = ['BTC-USD', 'ETH-USD']

def download_real_data():
    """Download REAL data from Yahoo Finance"""
    print("=" * 70)
    print("📥 DOWNLOADING REAL MARKET DATA FROM YAHOO FINANCE")
    print("=" * 70)
    
    # Date range: 5 years of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    print(f"\n📅 Date Range: {start_date.date()} to {end_date.date()}")
    print(f"📊 Downloading {len(STOCKS + CRYPTO)} symbols")
    print(f"⏱️  This will take 2-3 minutes...\n")
    
    all_data = []
    success_count = 0
    fail_count = 0
    
    total = len(STOCKS + CRYPTO)
    
    for i, symbol in enumerate(STOCKS + CRYPTO, 1):
        print(f"[{i}/{total}] Downloading {symbol:8s} ... ", end="", flush=True)
        
        try:
            # Download from Yahoo Finance
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, auto_adjust=False)
            
            if df.empty:
                print("❌ No data available")
                fail_count += 1
                continue
            
            # Reset index to get date as column
            df.reset_index(inplace=True)
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Rename columns to match your database schema
            df.rename(columns={
                'Date': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)
            
            # Select only needed columns
            df = df[['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Convert timestamp to string format
            df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            all_data.append(df)
            print(f"✅ {len(df):,} records")
            success_count += 1
            
        except Exception as e:
            print(f"❌ Error: {str(e)[:50]}")
            fail_count += 1
    
    if not all_data:
        print("\n❌ No data downloaded! Check your internet connection.")
        return None
    
    # Combine all data
    print(f"\n📦 Combining all data...")
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Sort by symbol and date
    final_df = final_df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    
    # Save to CSV
    output_file = 'real_stock_data.csv'
    final_df.to_csv(output_file, index=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print("✅ DOWNLOAD COMPLETE!")
    print("=" * 70)
    print(f"📊 Total Records:    {len(final_df):,}")
    print(f"📈 Symbols:          {final_df['symbol'].nunique()}")
    print(f"✅ Successful:       {success_count}/{total}")
    print(f"❌ Failed:           {fail_count}/{total}")
    print(f"💾 Saved to:         {output_file}")
    print(f"📏 File Size:        ~{len(final_df) * 100 / 1024 / 1024:.1f} MB")
    print("=" * 70)
    
    # Show sample data
    print("\n📋 SAMPLE DATA (First 5 records):")
    print(final_df.head())
    
    print("\n📊 DATA SUMMARY BY SYMBOL:")
    summary = final_df.groupby('symbol').agg({
        'close': ['count', 'min', 'max', 'mean'],
        'timestamp': ['min', 'max']
    }).round(2)
    print(summary.head(10))
    
    return final_df

if __name__ == "__main__":
    print("\n🚀 Starting Real Data Download...")
    print("⏱️  Estimated time: 2-3 minutes\n")
    
    df = download_real_data()
    
    if df is not None:
        print("\n" + "=" * 70)
        print("🎉 SUCCESS!")
        print("=" * 70)
        print("\n📝 NEXT STEPS:")
        print("   1. ✅ You now have real_stock_data.csv")
        print("   2. 📤 Run: python upload_to_db.py")
        print("   3. 🗑️  Delete old fake files:")
        print("      - stock_data.csv")
        print("      - stock_data_complete.csv")
        print("   4. 🚀 Restart your backend server")
        print("\n💡 TIP: Your app will now use REAL market data!")
        print("=" * 70)
    else:
        print("\n❌ Download failed. Please check your internet connection.")