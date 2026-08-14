"""
REAL STOCK DATA DOWNLOADER (v2)
Downloads actual market data from Yahoo Finance (FREE)

Changes from v1:
  - Removed BTC-USD / ETH-USD. Crypto trades 24/7 while equities trade
    ~252 days/year on a fixed calendar — mixing the two in one LSTM batch
    misaligns the "days since last close" assumption the model relies on,
    and was dragging down training throughput for no accuracy benefit.
  - Added logging (to console + logs/download.log) instead of print().
  - Added per-symbol error handling so one bad ticker doesn't kill the run.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import logging

# ============================================
# LOGGING SETUP
# ============================================
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================
# STOCKS TO DOWNLOAD (crypto removed)
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


def download_real_data():
    """Download REAL data from Yahoo Finance, with per-symbol error isolation."""
    logger.info("=" * 70)
    logger.info("DOWNLOADING REAL MARKET DATA FROM YAHOO FINANCE")
    logger.info("=" * 70)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=5 * 365)

    logger.info(f"Date Range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Downloading {len(STOCKS)} symbols")

    all_data = []
    success_count = 0
    fail_count = 0
    failed_symbols = []

    total = len(STOCKS)

    for i, symbol in enumerate(STOCKS, 1):
        logger.info(f"[{i}/{total}] Downloading {symbol}...")

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, auto_adjust=False)

            if df.empty:
                logger.warning(f"  {symbol}: no data returned, skipping")
                fail_count += 1
                failed_symbols.append(symbol)
                continue

            df.reset_index(inplace=True)
            df['symbol'] = symbol

            df.rename(columns={
                'Date': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)

            df = df[['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]

            # Basic sanity checks — catch bad rows before they poison training
            if df[['open', 'high', 'low', 'close']].isnull().any().any():
                bad_rows = df[['open', 'high', 'low', 'close']].isnull().any(axis=1).sum()
                logger.warning(f"  {symbol}: dropping {bad_rows} rows with null OHLC values")
                df = df.dropna(subset=['open', 'high', 'low', 'close'])

            if (df['close'] <= 0).any():
                bad_rows = (df['close'] <= 0).sum()
                logger.warning(f"  {symbol}: dropping {bad_rows} rows with non-positive close price")
                df = df[df['close'] > 0]

            df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

            all_data.append(df)
            logger.info(f"  {symbol}: {len(df):,} records OK")
            success_count += 1

        except Exception as e:
            logger.error(f"  {symbol}: failed - {e}")
            fail_count += 1
            failed_symbols.append(symbol)

    if not all_data:
        logger.error("No data downloaded! Check your internet connection.")
        return None

    logger.info("Combining all data...")
    final_df = pd.concat(all_data, ignore_index=True)
    final_df = final_df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)

    output_file = 'real_stock_data.csv'
    final_df.to_csv(output_file, index=False)

    logger.info("=" * 70)
    logger.info("DOWNLOAD COMPLETE")
    logger.info(f"Total Records:  {len(final_df):,}")
    logger.info(f"Symbols:        {final_df['symbol'].nunique()}")
    logger.info(f"Successful:     {success_count}/{total}")
    logger.info(f"Failed:         {fail_count}/{total}")
    if failed_symbols:
        logger.info(f"Failed symbols: {failed_symbols}")
    logger.info(f"Saved to:       {output_file}")
    logger.info("=" * 70)

    return final_df


if __name__ == "__main__":
    df = download_real_data()

    if df is not None:
        logger.info("SUCCESS — next steps:")
        logger.info("  1. python prepare_data.py")
        logger.info("  2. python upload_to_db.py")
    else:
        logger.error("Download failed. Please check your internet connection.")
        exit(1)