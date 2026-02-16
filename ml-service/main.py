"""
QUANTUM TRADE ML SERVICE - LSTM EDITION (FIXED)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from typing import List, Dict, Optional
import uvicorn
import os
import pickle
import json
from datetime import datetime

# Try to import TensorFlow (for LSTM)
try:
    import tensorflow as tf
    from tensorflow import keras
    LSTM_AVAILABLE = True
    print("✅ TensorFlow available - LSTM model can be loaded")
except ImportError:
    LSTM_AVAILABLE = False
    print("⚠️  TensorFlow not available - using fallback prediction")
    print("   Install with: pip install tensorflow")

app = FastAPI(
    title="Quantum Trade ML Service - LSTM Edition",
    description="AI-powered stock prediction with LSTM deep learning",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# REQUEST/RESPONSE MODELS
# ============================================

class PredictRequest(BaseModel):
    recent_prices: List[float]

class TradeRequest(BaseModel):
    balance: float
    shares: float
    current_price: float
    price_change: float
    volatility: float
    trend: float
    symbol: str

# ============================================
# GLOBAL VARIABLES
# ============================================

lstm_model = None
price_scaler = None
lstm_metadata = None

# ============================================
# LOAD LSTM MODEL (FIXED!)
# ============================================

def load_lstm_model():
    """
    Load trained LSTM model from disk
    Returns True if successful, False otherwise
    """
    global lstm_model, price_scaler, lstm_metadata
    
    model_path = 'lstm_stock_model.h5'
    scaler_path = 'lstm_scaler.pkl'
    metadata_path = 'lstm_metadata.json'
    
    if not LSTM_AVAILABLE:
        print("\n⚠️  TensorFlow not installed!")
        print("   Install with: pip install tensorflow")
        print("   Using ensemble fallback methods\n")
        return False
    
    try:
        # Load LSTM model
        if os.path.exists(model_path):
            print(f"\n📂 Loading LSTM model from {model_path}...")
            
            # 🔥 FIX: Load with compile=False to avoid metric deserialization error
            lstm_model = keras.models.load_model(model_path, compile=False)
            print("✅ LSTM model loaded successfully (compile=False)!")
            print(f"   Model size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
            
            # Load scaler
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    price_scaler = pickle.load(f)
                print("✅ Price scaler loaded!")
            else:
                print(f"⚠️  Scaler not found at {scaler_path}")
                lstm_model = None
                return False
            
            # Load metadata
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    lstm_metadata = json.load(f)
                print("✅ Model metadata loaded!")
                print(f"   Model type: {lstm_metadata.get('model_type', 'LSTM')}")
                print(f"   Sequence length: {lstm_metadata.get('sequence_length', 60)}")
                print(f"   Training samples: {lstm_metadata.get('total_samples', 'N/A'):,}")
                print(f"   R² Score: {lstm_metadata.get('r2_score', 0)*100:.2f}%")
                print(f"   MAE: ${lstm_metadata.get('real_mae', 'N/A'):.2f}")
            
            return True
            
        else:
            print(f"\n⚠️  LSTM model not found at {model_path}")
            print("   Expected files:")
            print(f"   - {model_path}")
            print(f"   - {scaler_path}")
            print(f"   - {metadata_path}")
            print("\n   Using ensemble fallback methods")
            print("   To use LSTM:")
            print("   1. Train model in Google Colab")
            print("   2. Download the 3 files above")
            print("   3. Copy them to ml-service/ folder")
            print("   4. Restart this service\n")
            return False
            
    except Exception as e:
        print(f"\n❌ Error loading LSTM model: {e}")
        print("   Using ensemble fallback methods\n")
        lstm_model = None
        price_scaler = None
        lstm_metadata = None
        return False

# ============================================
# LSTM PREDICTION
# ============================================

def predict_with_lstm(prices: np.ndarray) -> Optional[float]:
    """
    Predict next price using LSTM deep learning model
    
    Args:
        prices: Array of historical prices
        
    Returns:
        Predicted price or None if model unavailable
    """
    if lstm_model is None or price_scaler is None:
        return None
    
    try:
        # Get sequence length from metadata
        seq_length = lstm_metadata.get('sequence_length', 60) if lstm_metadata else 60
        
        # Need at least seq_length prices
        if len(prices) < seq_length:
            print(f"⚠️  LSTM needs {seq_length} prices, got {len(prices)}")
            return None
        
        # Take last seq_length prices
        recent_prices = prices[-seq_length:]
        
        # Normalize using trained scaler
        normalized = price_scaler.transform(recent_prices.reshape(-1, 1)).flatten()
        
        # Reshape for LSTM: (batch_size=1, timesteps=seq_length, features=1)
        X = normalized.reshape(1, seq_length, 1)
        
        # Predict (silent mode)
        pred_normalized = lstm_model.predict(X, verbose=0)[0][0]
        
        # Denormalize back to real price
        pred_price = price_scaler.inverse_transform([[pred_normalized]])[0][0]
        
        return float(pred_price)
        
    except Exception as e:
        print(f"❌ LSTM prediction error: {e}")
        return None

# ============================================
# ENSEMBLE FALLBACK PREDICTION
# ============================================

def predict_with_ensemble(prices: np.ndarray) -> float:
    """
    Fallback prediction using ensemble of traditional methods:
    - Linear Regression
    - Simple Moving Average (SMA)
    - Exponential Moving Average (EMA)
    
    Args:
        prices: Array of historical prices
        
    Returns:
        Predicted price
    """
    
    # 1. Linear Regression Prediction
    x = np.arange(len(prices[-10:]))
    coeffs = np.polyfit(x, prices[-10:], 1)
    linear_pred = prices[-1] + coeffs[0]
    
    # 2. SMA Prediction
    sma_10 = np.mean(prices[-10:])
    sma_5 = np.mean(prices[-5:])
    sma_pred = sma_5 + (sma_5 - sma_10)
    
    # 3. EMA Prediction
    def ema(data, period=10):
        multiplier = 2 / (period + 1)
        ema_val = data[0]
        for price in data[1:]:
            ema_val = (price * multiplier) + (ema_val * (1 - multiplier))
        return ema_val
    
    ema_10 = ema(prices[-10:])
    ema_5 = ema(prices[-5:])
    ema_pred = ema_5 + (ema_5 - ema_10)
    
    # Weighted average based on volatility
    volatility = np.std(prices[-10:])
    if volatility > prices[-1] * 0.05:  # High volatility
        weights = [0.3, 0.35, 0.35]
    else:  # Low volatility
        weights = [0.5, 0.25, 0.25]
    
    # Combine predictions
    prediction = (
        linear_pred * weights[0] +
        sma_pred * weights[1] +
        ema_pred * weights[2]
    )
    
    return float(prediction)

# ============================================
# TECHNICAL INDICATORS
# ============================================

def calculate_rsi(prices: np.ndarray, period: int = 14) -> float:
    """Calculate Relative Strength Index"""
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
    rsi = 100 - (100 / (1 + rs))
    return float(rsi)

def calculate_macd(prices: np.ndarray) -> Dict[str, float]:
    """Calculate MACD indicator"""
    if len(prices) < 26:
        return {"macd": 0.0, "signal": 0.0, "histogram": 0.0}
    
    # EMA calculation
    def ema(data, period):
        multiplier = 2 / (period + 1)
        ema_val = data[0]
        for price in data[1:]:
            ema_val = (price * multiplier) + (ema_val * (1 - multiplier))
        return ema_val
    
    ema_12 = ema(prices[-26:], 12)
    ema_26 = ema(prices[-26:], 26)
    macd_line = ema_12 - ema_26
    
    # Signal line (9-day EMA of MACD)
    signal_line = macd_line * 0.2  # Simplified
    histogram = macd_line - signal_line
    
    return {
        "macd": float(macd_line),
        "signal": float(signal_line),
        "histogram": float(histogram)
    }

def calculate_bollinger_bands(prices: np.ndarray, period: int = 20) -> Dict[str, float]:
    """Calculate Bollinger Bands"""
    if len(prices) < period:
        return {"upper": 0.0, "middle": 0.0, "lower": 0.0}
    
    sma = np.mean(prices[-period:])
    std = np.std(prices[-period:])
    
    return {
        "upper": float(sma + 2 * std),
        "middle": float(sma),
        "lower": float(sma - 2 * std)
    }

# ============================================
# AI REASONING GENERATION
# ============================================

def generate_reasoning(
    action: str, 
    symbol: str, 
    rsi: float, 
    trend: float, 
    volatility: float, 
    price_change: float,
    macd: Dict[str, float]
) -> str:
    """
    Generate human-readable explanation for trading decision
    """
    parts = []
    
    # Main action reason
    if action == "BUY":
        parts.append(f"📈 {symbol} shows strong buying opportunity")
        if rsi < 30:
            parts.append("Stock appears oversold (RSI < 30) - potential rebound")
        elif trend > 5:
            parts.append(f"Bullish trend confirmed (+{trend:.1f}% momentum)")
        if macd["histogram"] > 0:
            parts.append("MACD histogram positive - upward momentum")
            
    elif action == "SELL":
        parts.append(f"📉 {symbol} showing sell signals")
        if rsi > 70:
            parts.append("Stock overbought (RSI > 70) - correction likely")
        elif trend < -5:
            parts.append(f"Bearish trend confirmed ({trend:.1f}% momentum)")
        if macd["histogram"] < 0:
            parts.append("MACD histogram negative - downward pressure")
            
    else:  # HOLD
        parts.append(f"⏸️ {symbol} in neutral zone - hold current position")
        parts.append(f"RSI at {rsi:.1f} (neutral range)")
    
    # Volatility warning
    if volatility > 10:
        parts.append(f"⚠️ High volatility ({volatility:.1f}σ) - increased risk")
    
    # Recent momentum
    if abs(price_change) > 3:
        direction = "upward" if price_change > 0 else "downward"
        parts.append(f"Recent {direction} momentum ({price_change:+.2f}%)")
    
    return ". ".join(parts) + "."

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """Root endpoint with service info"""
    model_status = "LSTM Deep Learning" if lstm_model is not None else "Ensemble (Fallback)"
    
    return {
        "service": "Quantum Trade ML Service",
        "version": "2.0.0",
        "status": "active",
        "model": model_status,
        "lstm_available": lstm_model is not None,
        "tensorflow_installed": LSTM_AVAILABLE,
        "features": [
            "LSTM Deep Learning Predictions" if lstm_model else "Ensemble Predictions",
            "Technical Indicators (RSI, MACD, Bollinger)",
            "AI Reasoning",
            "Trading Signals"
        ],
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "trade": "/trade_decision (POST)",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "ml-lstm",
        "timestamp": datetime.now().isoformat(),
        "model": {
            "type": "LSTM" if lstm_model is not None else "Ensemble",
            "loaded": lstm_model is not None,
            "tensorflow_available": LSTM_AVAILABLE,
            "metadata": lstm_metadata if lstm_metadata else {}
        }
    }

@app.post("/predict")
async def predict(request: PredictRequest):
    """
    Predict next day's stock price
    
    Uses LSTM model if available, otherwise falls back to ensemble methods
    """
    try:
        prices = np.array(request.recent_prices)
        
        if len(prices) < 10:
            raise HTTPException(
                status_code=400, 
                detail="Need at least 10 price points for prediction"
            )
        
        # Try LSTM first
        lstm_prediction = None
        if lstm_model is not None:
            lstm_prediction = predict_with_lstm(prices)
        
        # Fallback ensemble prediction
        ensemble_prediction = predict_with_ensemble(prices)
        
        # Determine final prediction and confidence
        if lstm_prediction is not None:
            final_prediction = lstm_prediction
            model_used = "LSTM Deep Learning"
            # Use R² score from metadata if available
            confidence = lstm_metadata.get('r2_score', 0.75) if lstm_metadata else 0.75
        else:
            final_prediction = ensemble_prediction
            model_used = "Hybrid Ensemble (Linear + SMA + EMA)"
            
            # Calculate confidence based on volatility
            volatility = np.std(prices[-10:]) / np.mean(prices[-10:])
            confidence = max(0.5, min(0.85, 0.7 - volatility))
        
        # Add small random variance to prevent identical predictions
        variance = np.random.uniform(-0.005, 0.005)
        final_prediction = final_prediction * (1 + variance)
        
        # Calculate technical indicators
        rsi = calculate_rsi(prices)
        macd = calculate_macd(prices)
        bollinger = calculate_bollinger_bands(prices)
        
        return {
            "predicted_price": float(final_prediction),
            "confidence": float(confidence),
            "model": model_used,
            "components": {
                "lstm_prediction": float(lstm_prediction) if lstm_prediction else None,
                "ensemble_prediction": float(ensemble_prediction),
                "current_price": float(prices[-1]),
                "price_change": float(((final_prediction - prices[-1]) / prices[-1]) * 100)
            },
            "indicators": {
                "rsi": float(rsi),
                "macd": macd,
                "bollinger_bands": bollinger
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "data_points": len(prices)
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/trade_decision")
async def trade_decision(request: TradeRequest):
    """
    Generate BUY/SELL/HOLD trading decision with AI reasoning
    """
    try:
        # Calculate individual signals
        price_signal = 1 if request.price_change > 0.02 else (-1 if request.price_change < -0.02 else 0)
        trend_signal = 1 if request.trend > 0.05 else (-1 if request.trend < -0.05 else 0)
        vol_signal = -1 if request.volatility > 10 else 0
        
        # Estimate RSI from trend and price change
        estimated_rsi = 50 + (request.price_change * 5) + (request.trend * 2)
        estimated_rsi = max(0, min(100, estimated_rsi))
        rsi_signal = -1 if estimated_rsi > 70 else (1 if estimated_rsi < 30 else 0)
        
        # Estimate MACD
        macd = {
            "macd": request.trend * 0.5,
            "signal": request.trend * 0.3,
            "histogram": request.trend * 0.2
        }
        macd_signal = 1 if macd["histogram"] > 0.5 else (-1 if macd["histogram"] < -0.5 else 0)
        
        # Weighted total signal
        total_signal = (
            price_signal * 1.5 +
            trend_signal * 2.0 +
            vol_signal * 0.5 +
            rsi_signal * 1.0 +
            macd_signal * 1.0
        )
        
        # Make decision
        if total_signal >= 2.5:
            action = "BUY"
            confidence = min(0.92, 0.65 + (total_signal * 0.08))
        elif total_signal <= -2.5:
            action = "SELL"
            confidence = min(0.92, 0.65 + (abs(total_signal) * 0.08))
        else:
            action = "HOLD"
            confidence = 0.50 + (abs(total_signal) * 0.05)
        
        # Generate AI reasoning
        reason = generate_reasoning(
            action=action,
            symbol=request.symbol,
            rsi=estimated_rsi,
            trend=request.trend,
            volatility=request.volatility,
            price_change=request.price_change,
            macd=macd
        )
        
        # Calculate position sizing (if buying)
        shares_to_trade = 0
        if action == "BUY" and request.balance > 0:
            # Risk 10% of balance
            max_investment = request.balance * 0.10
            shares_to_trade = int(max_investment / request.current_price)
        elif action == "SELL" and request.shares > 0:
            # Sell 50% of position
            shares_to_trade = int(request.shares * 0.50)
        
        return {
            "action": action,
            "confidence": float(confidence),
            "reason": reason,
            "shares": max(1, shares_to_trade),
            "indicators": {
                "rsi": float(estimated_rsi),
                "trend_strength": float(request.trend),
                "volatility_level": float(request.volatility),
                "signal_score": float(total_signal),
                "macd": macd
            },
            "signals": {
                "price": price_signal,
                "trend": trend_signal,
                "volatility": vol_signal,
                "rsi": rsi_signal,
                "macd": macd_signal,
                "total": float(total_signal)
            },
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": "LSTM" if lstm_model else "Ensemble",
                "symbol": request.symbol
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trade decision error: {str(e)}")

# ============================================
# STARTUP EVENT
# ============================================

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    print("\n" + "=" * 70)
    print("🚀 QUANTUM TRADE ML SERVICE - LSTM EDITION")
    print("=" * 70)
    
    # Try to load LSTM model
    lstm_loaded = load_lstm_model()
    
    print("=" * 70)
    print(f"✅ Service ready!")
    print(f"   Model: {'LSTM Deep Learning' if lstm_loaded else 'Ensemble Fallback'}")
    print(f"   TensorFlow: {'Available' if LSTM_AVAILABLE else 'Not installed'}")
    print("=" * 70 + "\n")

# ============================================
# EXPORT FOR VERCEL
# ============================================

handler = app

# ============================================
# LOCAL DEVELOPMENT
# ============================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 STARTING ML SERVICE")
    print("=" * 70)
    print("🌐 Server: http://localhost:8000")
    print("📚 Docs: http://localhost:8000/docs")
    print("=" * 70 + "\n")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )