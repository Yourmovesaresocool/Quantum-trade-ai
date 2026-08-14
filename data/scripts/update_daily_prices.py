"""
UPDATE_DAILY_PRICES.PY
Additive-only daily price sync. Fetches whatever is new since each
symbol's latest stored date and inserts it — never drops, never
truncates, never touches trades/users/predictions.

Run this on a schedule (Task Scheduler on Windows, cron on Linux/a
server) instead of re-running download_real_data.py + upload_to_db.py,
which starts with DROP TABLE ... CASCADE and would wipe real user data.

What it does, in order:
  1. For each symbol, find the latest timestamp already in
     historical_prices (Postgres).
  2. Pull only the missing days from Yahoo Finance (start = last date + 1).
  3. INSERT ... ON CONFLICT (symbol, timestamp) DO NOTHING — so re-running
     this script twice in a row, or after a weekend, is always safe.
  4. Mirror the same new rows into real_stock_data.csv (append-only,
     de-duplicated) so a future prepare_data.py run has the full history
     without re-downloading 5 years from scratch.

Usage:
    python update_daily_prices.py
"""

import os
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Resolve paths relative to THIS FILE's location, not the caller's
# current directory — so `python update_daily_prices.py` works the same
# whether you run it from the project root, from data/scripts/, or
# anywhere else. Walks upward from this script looking for backend/.env.
SCRIPT_DIR = Path(__file__).resolve().parent


def find_upward(filename: str, start: Path) -> Path | None:
    for folder in [start] + list(start.parents):
        candidate = folder / filename
        if candidate.exists():
            return candidate
    return None


env_path = find_upward('backend/.env', SCRIPT_DIR)
if env_path:
    load_dotenv(env_path)
    PROJECT_ROOT = env_path.parent.parent  # the folder that contains backend/
else:
    PROJECT_ROOT = SCRIPT_DIR

os.makedirs(SCRIPT_DIR / 'logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(SCRIPT_DIR / 'logs' / 'update_daily_prices.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

if env_path:
    logger.info(f"Loaded environment from: {env_path}")
else:
    logger.warning(f"Could not find backend/.env searching upward from {SCRIPT_DIR} — "
                    f"falling back to any DATABASE_URL already in the environment.")

DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    logger.error(
        f"DATABASE_URL not found. Searched for backend/.env starting at {SCRIPT_DIR} "
        f"and walking up through parent folders, but didn't find it. "
        f"Either move this script so backend/ is somewhere above it, "
        f"or set DATABASE_URL directly as an environment variable."
    )
    raise SystemExit(1)

# Same 47 symbols as download_real_data.py / STOCK_DATABASE in App.js.
# Keep this list in sync with both if you ever add/remove a stock.
STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AMD', 'INTC',
    'ORCL', 'CRM', 'ADBE', 'IBM', 'CSCO',
    'TSLA', 'F', 'GM',
    'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'PYPL',
    'WMT', 'TGT', 'COST', 'HD', 'LOW', 'NKE', 'SBUX', 'MCD',
    'DIS', 'NFLX', 'CMCSA',
    'BA', 'LMT', 'RTX',
    'JNJ', 'PFE', 'UNH', 'ABBV', 'TMO',
    'XOM', 'CVX',
    'T', 'VZ',
]

# Prefer an existing real_stock_data.csv wherever it already lives
# (project root or a data/ subfolder); otherwise create a fresh one at
# the project root.
_existing_csv = find_upward('real_stock_data.csv', SCRIPT_DIR) or find_upward('data/real_stock_data.csv', SCRIPT_DIR)
CSV_PATH = str(_existing_csv) if _existing_csv else str(PROJECT_ROOT / 'real_stock_data.csv')


def sync_symbol_to_db(engine, symbol: str) -> pd.DataFrame:
    """Insert only the rows newer than what's already stored for this
    symbol. Returns the new rows (empty DataFrame if nothing new) so the
    caller can also mirror them into the CSV."""
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT MAX(timestamp) FROM historical_prices WHERE symbol = :sym"),
            {"sym": symbol}
        ).fetchone()
        last_date = result[0] if result and result[0] else (datetime.now() - timedelta(days=5 * 365))

        df = yf.Ticker(symbol).history(start=last_date + timedelta(days=1), auto_adjust=False)
        if df.empty:
            logger.info(f"  {symbol}: already up to date (last stored: {last_date.date()})")
            return pd.DataFrame()

        df.reset_index(inplace=True)
        df['symbol'] = symbol
        df.rename(columns={'Date': 'timestamp', 'Open': 'open', 'High': 'high',
                            'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
        df = df[['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]

        # Same sanity checks download_real_data.py applies — don't let a
        # bad row from the API corrupt historical_prices.
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        df = df[df['close'] > 0]

        if df.empty:
            logger.info(f"  {symbol}: no valid new rows after cleaning")
            return pd.DataFrame()

        inserted = 0
        for _, row in df.iterrows():
            res = conn.execute(text("""
                INSERT INTO historical_prices (symbol, timestamp, open, high, low, close, volume)
                VALUES (:symbol, :timestamp, :open, :high, :low, :close, :volume)
                ON CONFLICT (symbol, timestamp) DO NOTHING
            """), row.to_dict())
            inserted += res.rowcount
        conn.commit()

        logger.info(f"  {symbol}: fetched {len(df)} rows, inserted {inserted} new (rest already present)")
        return df


def mirror_into_csv(new_rows_by_symbol: dict):
    """Append newly-fetched rows into real_stock_data.csv, de-duplicated
    by (symbol, timestamp), so a future prepare_data.py run sees the full
    history without a fresh 5-year download. Never rewrites or drops any
    existing row in the CSV."""
    new_frames = [df for df in new_rows_by_symbol.values() if not df.empty]
    if not new_frames:
        logger.info("No new rows to mirror into CSV.")
        return

    new_data = pd.concat(new_frames, ignore_index=True)
    new_data['timestamp'] = pd.to_datetime(new_data['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

    if os.path.exists(CSV_PATH):
        existing = pd.read_csv(CSV_PATH)
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=['symbol', 'timestamp'], keep='first')
    else:
        combined = new_data

    combined = combined.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    combined.to_csv(CSV_PATH, index=False)
    logger.info(f"CSV updated: {CSV_PATH} now has {len(combined):,} total rows "
                f"({len(new_data)} new rows merged in this run)")


def main():
    logger.info("=" * 70)
    logger.info("DAILY PRICE UPDATE — additive only, never touches trades/users")
    logger.info("=" * 70)

    engine = create_engine(DATABASE_URL)
    new_rows_by_symbol = {}

    for i, symbol in enumerate(STOCKS, 1):
        logger.info(f"[{i}/{len(STOCKS)}] Checking {symbol}...")
        try:
            new_rows_by_symbol[symbol] = sync_symbol_to_db(engine, symbol)
        except Exception as e:
            logger.error(f"  {symbol}: failed — {e}")
            new_rows_by_symbol[symbol] = pd.DataFrame()

    mirror_into_csv(new_rows_by_symbol)

    total_new = sum(len(df) for df in new_rows_by_symbol.values())
    logger.info("=" * 70)
    logger.info(f"DONE — {total_new} total new rows added across {len(STOCKS)} symbols")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()