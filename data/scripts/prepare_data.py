"""
STEP 1: PREPARE DATA FOR GOOGLE COLAB / LSTM TRAINING (v5 — 7 features)

CHANGE FROM v4: v4 added volume/RSI/MACD (4 features total). v5 adds
three more, cheap to compute since they're already in your database or
easy derivatives of price:
  - high        (day's high price, normalized)
  - low         (day's low price, normalized)
  - volume_ratio (today's volume / 20-day average volume — flags unusual
                  trading activity better than raw volume alone, since
                  "high volume" means different things for different stocks)

7 features total: close, volume, rsi, macd_hist, high, low, volume_ratio.
Architecture and RETURN target are unchanged — only input richness grows.
Do not add more features on top of this without more training data —
diminishing/negative returns are more likely past this point relative to
~55K training sequences.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import logging
import os

SEQUENCE_LENGTH = 90
FEATURE_WARMUP = 40  # buffer for RSI(14)/MACD(26)/volume_ratio(20) warmup
MIN_ROWS_REQUIRED = SEQUENCE_LENGTH + FEATURE_WARMUP
FEATURE_NAMES = ['close', 'volume', 'rsi', 'macd_hist', 'high', 'low', 'volume_ratio']

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/prepare_data.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Same definition as main.py's calculate_rsi(), computed rolling."""
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(window=period, min_periods=period).mean()
    avg_loss = losses.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50.0)
    rsi[avg_loss == 0] = 100.0
    return rsi


def compute_macd_histogram(close: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """Same simplified signal-line definition as main.py's calculate_macd()."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line * 0.2
    histogram = macd_line - signal_line
    histogram.iloc[:slow] = 0.0
    return histogram


def compute_volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """Today's volume relative to its own recent average — flags unusual
    activity in a way that's comparable across stocks of different sizes."""
    avg_volume = volume.rolling(window=period, min_periods=period).mean()
    ratio = volume / avg_volume.replace(0, np.nan)
    return ratio.fillna(1.0)  # 1.0 = "normal" volume where insufficient history


def create_sequences_multivariate(features_normalized, prices_raw, seq_length):
    sequences = []
    returns = []
    for i in range(len(features_normalized) - seq_length):
        seq = features_normalized[i:i + seq_length]
        current_price = prices_raw[i + seq_length - 1]
        next_price = prices_raw[i + seq_length]
        ret = (next_price - current_price) / current_price
        sequences.append(seq)
        returns.append(ret)
    return np.array(sequences), np.array(returns)


def main():
    logger.info("=" * 70)
    logger.info(f"PREPARING MULTIVARIATE DATA FOR LSTM TRAINING (features={FEATURE_NAMES})")
    logger.info("=" * 70)

    try:
        df = pd.read_csv('real_stock_data.csv')
        logger.info(f"Loaded {len(df):,} records, {df['symbol'].nunique()} stocks")
    except FileNotFoundError:
        logger.error("real_stock_data.csv not found! Run download_real_data.py first.")
        raise SystemExit(1)

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)

    all_sequences = []
    all_targets = []
    all_symbols = []
    scalers = {}
    skipped_symbols = []

    for symbol in df['symbol'].unique():
        try:
            stock_data = df[df['symbol'] == symbol].copy()

            if len(stock_data) < MIN_ROWS_REQUIRED:
                logger.warning(f"  {symbol}: only {len(stock_data)} rows (< {MIN_ROWS_REQUIRED}), skipping")
                skipped_symbols.append(symbol)
                continue

            close = stock_data['close']
            volume = stock_data['volume']
            high = stock_data['high']
            low = stock_data['low']

            rsi = compute_rsi(close)
            macd_hist = compute_macd_histogram(close)
            volume_ratio = compute_volume_ratio(volume)

            feature_df = pd.DataFrame({
                'close': close.values,
                'volume': volume.values,
                'rsi': rsi.values,
                'macd_hist': macd_hist.values,
                'high': high.values,
                'low': low.values,
                'volume_ratio': volume_ratio.values,
            })

            prices_raw = close.values  # for return target — never scaled

            scaler = MinMaxScaler()
            features_normalized = scaler.fit_transform(feature_df.values)  # (n_days, 7)
            scalers[symbol] = scaler

            sequences, returns = create_sequences_multivariate(features_normalized, prices_raw, SEQUENCE_LENGTH)

            if len(sequences) == 0:
                logger.warning(f"  {symbol}: no sequences produced, skipping")
                skipped_symbols.append(symbol)
                continue

            all_sequences.append(sequences)
            all_targets.append(returns)
            all_symbols.extend([symbol] * len(sequences))

            logger.info(f"  {symbol}: {len(sequences)} sequences created")

        except Exception as e:
            logger.error(f"  {symbol}: failed to process - {e}")
            skipped_symbols.append(symbol)
            continue

    if not all_sequences:
        logger.error("No sequences produced for any symbol. Aborting.")
        raise SystemExit(1)

    X = np.vstack(all_sequences)  # (samples, 90, 7)
    y = np.hstack(all_targets)

    logger.info(f"Data prepared: {len(X):,} total sequences, shape {X.shape}")
    logger.info(f"Target (return) stats: mean={y.mean():.5f}, std={y.std():.5f}")
    if skipped_symbols:
        logger.info(f"Skipped symbols: {skipped_symbols}")

    np.save('lstm_X_train.npy', X)
    np.save('lstm_y_train.npy', y)
    np.save('lstm_symbols.npy', np.array(all_symbols))

    with open('price_scalers.pkl', 'wb') as f:
        pickle.dump(scalers, f)

    logger.info(f"Saved lstm_X_train.npy - {X.nbytes / 1024 / 1024:.1f} MB")
    logger.info(f"Saved lstm_y_train.npy - {y.nbytes / 1024 / 1024:.1f} MB")
    logger.info(f"Saved price_scalers.pkl - {len(scalers)} per-symbol scalers (7-feature scalers)")

    info = {
        'num_samples': len(X),
        'num_stocks': len(scalers),
        'sequence_length': SEQUENCE_LENGTH,
        'num_features': X.shape[2],
        'feature_names': FEATURE_NAMES,
        'target_type': 'next_day_return',
        'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
        'stocks': list(scalers.keys()),
        'skipped_symbols': skipped_symbols,
    }

    with open('training_info.txt', 'w') as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")

    logger.info("=" * 70)
    logger.info("DATA PREPARATION COMPLETE — 7 features per day")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()