"""
QUANTUM TRADE ML SERVICE - LSTM ONLY, MULTIVARIATE (v5)

CHANGES FROM v4 (single-feature 'recent_prices'):
  - /predict now expects full OHLCV bars, not just a list of close prices.
    This is required because the model now needs 7 features per day
    (close, volume, rsi, macd_hist, high, low, volume_ratio) to match
    what it was trained on in prepare_data.py v5.
  - RSI/MACD/volume-ratio are computed HERE using the exact same rolling
    formulas as prepare_data.py, so training and live inference stay
    consistent (this consistency is the single most important thing to
    get right — a mismatch here would silently corrupt every prediction).
  - price_scalers.pkl scalers are now fit on 7 columns, not 1 — loading
    and usage code is unchanged, only the data shape going through it.
  - Requires MORE than 90 bars from the caller (90 + warmup buffer for
    RSI/MACD/volume-ratio, same as training) — see MIN_BARS_REQUIRED.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import uvicorn
import os
import pickle
import json
import logging
from datetime import datetime

try:
    import tensorflow as tf
    from tensorflow import keras
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/ml_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

if not LSTM_AVAILABLE:
    logger.warning("TensorFlow not installed — predictions unavailable until it is. Install with: pip install tensorflow")

app = FastAPI(
    title="Quantum Trade ML Service - LSTM Only, Multivariate",
    description="LSTM-only stock prediction using 7 daily features (no fallback math models)",
    version="5.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# REQUEST/RESPONSE MODELS
# ============================================

class Bar(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float

class PredictRequest(BaseModel):
    symbol: str
    bars: List[Bar]  # oldest-first, same order as the DB query

class TradeRequest(BaseModel):
    balance: float
    shares: float
    current_price: float
    price_change: float
    volatility: float
    trend: float
    symbol: str

# ============================================
# GLOBAL STATE
# ============================================

lstm_model = None
price_scalers = None   # dict: {symbol: scaler}, now fit on 7 columns
lstm_metadata = None

MODEL_PATH = 'lstm_model_final.h5'
SCALERS_PATH = 'price_scalers.pkl'
METADATA_PATH = 'lstm_metadata.json'

FEATURE_NAMES = ['close', 'volume', 'rsi', 'macd_hist', 'high', 'low', 'volume_ratio']
SEQUENCE_LENGTH_DEFAULT = 90
WARMUP_DEFAULT = 40
MIN_BARS_REQUIRED = SEQUENCE_LENGTH_DEFAULT + WARMUP_DEFAULT  # 130 — matches prepare_data.py's MIN_ROWS_REQUIRED

# ============================================
# FEATURE ENGINEERING — MUST MATCH prepare_data.py EXACTLY
# ============================================

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
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
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line * 0.2
    histogram = macd_line - signal_line
    histogram.iloc[:slow] = 0.0
    return histogram


def compute_volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    avg_volume = volume.rolling(window=period, min_periods=period).mean()
    ratio = volume / avg_volume.replace(0, np.nan)
    return ratio.fillna(1.0)


def build_feature_matrix(bars: List[Bar]) -> np.ndarray:
    """Builds the (n_bars, 7) RAW (unscaled) feature matrix from OHLCV
    bars — same column order and formulas as prepare_data.py."""
    df = pd.DataFrame([b.dict() for b in bars])

    rsi = compute_rsi(df['close'])
    macd_hist = compute_macd_histogram(df['close'])
    volume_ratio = compute_volume_ratio(df['volume'])

    feature_df = pd.DataFrame({
        'close': df['close'].values,
        'volume': df['volume'].values,
        'rsi': rsi.values,
        'macd_hist': macd_hist.values,
        'high': df['high'].values,
        'low': df['low'].values,
        'volume_ratio': volume_ratio.values,
    })
    return feature_df.values

# ============================================
# LOAD LSTM MODEL
# ============================================

def load_lstm_model() -> bool:
    global lstm_model, price_scalers, lstm_metadata

    if not LSTM_AVAILABLE:
        logger.error("TensorFlow not installed — cannot load LSTM.")
        return False

    if not os.path.exists(MODEL_PATH):
        logger.error(f"LSTM model not found at '{MODEL_PATH}'. Expected files: {MODEL_PATH}, {SCALERS_PATH}, {METADATA_PATH}")
        return False

    if not os.path.exists(SCALERS_PATH):
        logger.error(f"Scaler file not found at '{SCALERS_PATH}'.")
        return False

    try:
        logger.info(f"Loading LSTM model from {MODEL_PATH}...")
        lstm_model = keras.models.load_model(MODEL_PATH, compile=False)
        logger.info(f"LSTM model loaded ({os.path.getsize(MODEL_PATH) / (1024*1024):.1f} MB)")

        with open(SCALERS_PATH, 'rb') as f:
            price_scalers = pickle.load(f)
        logger.info(f"Loaded {len(price_scalers)} per-symbol scalers")

        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'r') as f:
                lstm_metadata = json.load(f)
            logger.info(f"Metadata: seq_len={lstm_metadata.get('sequence_length')}, "
                        f"features={lstm_metadata.get('num_features')}, r2={lstm_metadata.get('r2_score')}")
        else:
            logger.warning(f"No metadata file at '{METADATA_PATH}' — using defaults")
            lstm_metadata = {"sequence_length": SEQUENCE_LENGTH_DEFAULT, "num_features": len(FEATURE_NAMES)}

        return True

    except Exception as e:
        logger.error(f"Failed to load LSTM model: {e}", exc_info=True)
        lstm_model = None
        price_scalers = None
        lstm_metadata = None
        return False


def predict_with_lstm(bars: List[Bar], symbol: Optional[str]) -> float:
    """
    Predict next-day RETURN using the LSTM model ONLY, then convert it to
    a dollar price using the current (most recent) close.
    Raises ValueError with a specific reason on any failure.
    """
    if lstm_model is None or price_scalers is None:
        raise ValueError("LSTM model is not loaded on this service.")

    seq_length = lstm_metadata.get('sequence_length', SEQUENCE_LENGTH_DEFAULT)
    min_bars = seq_length + WARMUP_DEFAULT

    if len(bars) < min_bars:
        raise ValueError(f"Need at least {min_bars} bars (90-day window + indicator warmup), got {len(bars)}.")

    if symbol and symbol in price_scalers:
        scaler = price_scalers[symbol]
    else:
        raise ValueError(
            f"No trained scaler for symbol '{symbol}'. "
            f"Available symbols: {list(price_scalers.keys())[:5]}..."
        )

    raw_features = build_feature_matrix(bars)          # (n_bars, 7), warmup included
    normalized = scaler.transform(raw_features)          # same 7-column scaler as training
    recent_window = normalized[-seq_length:]              # take the last 90 rows AFTER warmup

    X = recent_window.reshape(1, seq_length, recent_window.shape[1])

    predicted_return = float(lstm_model.predict(X, verbose=0)[0][0])

    current_price = float(bars[-1].close)
    predicted_price = current_price * (1 + predicted_return)

    return predicted_price

# ============================================
# TECHNICAL INDICATORS FOR /trade_decision (unchanged — request-time only)
# ============================================

def calculate_rsi(prices: np.ndarray, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def calculate_macd(prices: np.ndarray) -> Dict[str, float]:
    if len(prices) < 26:
        return {"macd": 0.0, "signal": 0.0, "histogram": 0.0}

    def ema(data, period):
        multiplier = 2 / (period + 1)
        val = data[0]
        for price in data[1:]:
            val = (price * multiplier) + (val * (1 - multiplier))
        return val

    ema_12 = ema(prices[-26:], 12)
    ema_26 = ema(prices[-26:], 26)
    macd_line = ema_12 - ema_26
    signal_line = macd_line * 0.2
    return {"macd": float(macd_line), "signal": float(signal_line), "histogram": float(macd_line - signal_line)}


def calculate_bollinger_bands(prices: np.ndarray, period: int = 20) -> Dict[str, float]:
    if len(prices) < period:
        return {"upper": 0.0, "middle": 0.0, "lower": 0.0}
    sma = np.mean(prices[-period:])
    std = np.std(prices[-period:])
    return {"upper": float(sma + 2 * std), "middle": float(sma), "lower": float(sma - 2 * std)}

# ============================================
# ENDPOINTS
# ============================================

@app.get("/")
async def root():
    return {
        "service": "Quantum Trade ML Service",
        "version": "5.0.0",
        "status": "active",
        "model": "LSTM (multivariate, no fallback)",
        "lstm_loaded": lstm_model is not None,
        "tensorflow_installed": LSTM_AVAILABLE,
        "feature_names": FEATURE_NAMES,
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy" if lstm_model is not None else "degraded",
        "lstm_loaded": lstm_model is not None,
        "tensorflow_available": LSTM_AVAILABLE,
        "metadata": lstm_metadata or {},
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/predict")
async def predict(request: PredictRequest):
    if len(request.bars) < 10:
        raise HTTPException(status_code=400, detail="Need at least 10 bars for prediction")

    try:
        predicted_price = predict_with_lstm(request.bars, request.symbol)
    except ValueError as e:
        logger.warning(f"Prediction unavailable for {request.symbol}: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed")

    closes = np.array([b.close for b in request.bars])
    rsi = calculate_rsi(closes)
    macd = calculate_macd(closes)
    bollinger = calculate_bollinger_bands(closes)

    return {
        "predicted_price": predicted_price,
        "confidence": lstm_metadata.get('r2_score', None) if lstm_metadata else None,
        "directional_accuracy": lstm_metadata.get('directional_accuracy', None) if lstm_metadata else None,
        "model": "LSTM",
        "current_price": float(closes[-1]),
        "price_change_pct": float(((predicted_price - closes[-1]) / closes[-1]) * 100),
        "indicators": {"rsi": rsi, "macd": macd, "bollinger_bands": bollinger},
        "metadata": {"timestamp": datetime.now().isoformat(), "bars_received": len(request.bars)},
    }


@app.post("/trade_decision")
async def trade_decision(request: TradeRequest):
    try:
        price_signal = 1 if request.price_change > 0.02 else (-1 if request.price_change < -0.02 else 0)
        trend_signal = 1 if request.trend > 0.05 else (-1 if request.trend < -0.05 else 0)
        vol_signal = -1 if request.volatility > 10 else 0

        estimated_rsi = max(0, min(100, 50 + (request.price_change * 5) + (request.trend * 2)))
        rsi_signal = -1 if estimated_rsi > 70 else (1 if estimated_rsi < 30 else 0)

        macd = {"macd": request.trend * 0.5, "signal": request.trend * 0.3, "histogram": request.trend * 0.2}
        macd_signal = 1 if macd["histogram"] > 0.5 else (-1 if macd["histogram"] < -0.5 else 0)

        total_signal = price_signal * 1.5 + trend_signal * 2.0 + vol_signal * 0.5 + rsi_signal * 1.0 + macd_signal * 1.0

        if total_signal >= 2.5:
            action, confidence = "BUY", min(0.92, 0.65 + (total_signal * 0.08))
        elif total_signal <= -2.5:
            action, confidence = "SELL", min(0.92, 0.65 + (abs(total_signal) * 0.08))
        else:
            action, confidence = "HOLD", 0.50 + (abs(total_signal) * 0.05)

        shares_to_trade = 0
        if action == "BUY" and request.balance > 0:
            shares_to_trade = int((request.balance * 0.10) / request.current_price)
        elif action == "SELL" and request.shares > 0:
            shares_to_trade = int(request.shares * 0.50)

        return {
            "action": action,
            "confidence": float(confidence),
            "shares": max(1, shares_to_trade),
            "indicators": {"rsi": float(estimated_rsi), "trend_strength": request.trend, "volatility_level": request.volatility, "signal_score": total_signal, "macd": macd},
            "metadata": {"timestamp": datetime.now().isoformat(), "symbol": request.symbol},
        }
    except Exception as e:
        logger.error(f"Trade decision error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Trade decision error")

# ============================================
# STARTUP
# ============================================

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 70)
    logger.info("QUANTUM TRADE ML SERVICE - LSTM ONLY, MULTIVARIATE - STARTING")
    logger.info("=" * 70)
    loaded = load_lstm_model()
    if not loaded:
        logger.error(
            f"LSTM model FAILED to load. /predict will return 503 until "
            f"{MODEL_PATH} and {SCALERS_PATH} are present in this directory."
        )
    logger.info("=" * 70)

handler = app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")