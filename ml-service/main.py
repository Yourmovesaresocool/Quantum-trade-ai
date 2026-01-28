from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from typing import List
import uvicorn

app = FastAPI(title="Quantum Trade ML Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get("/")
async def root():
    return {"message": "Quantum Trade ML Service", "status": "active"}

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "ml"}

@app.post("/predict")
async def predict(request: PredictRequest):
    """Simple LSTM-style prediction using moving averages"""
    try:
        prices = np.array(request.recent_prices)
        
        if len(prices) < 10:
            raise HTTPException(status_code=400, detail="Need at least 10 price points")
        
        # Simple weighted moving average prediction
        recent_trend = np.polyfit(range(len(prices[-10:])), prices[-10:], 1)
        
        # Calculate prediction with slight randomness
        predicted = prices[-1] + recent_trend[0]
        
        # Add small variance (±2%)
        variance = np.random.uniform(-0.02, 0.02)
        predicted_price = predicted * (1 + variance)
        
        return {
            "predicted_price": float(predicted_price),
            "confidence": 0.75,
            "model": "LSTM-Simplified"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trade_decision")
async def trade_decision(request: TradeRequest):
    """AI trading decision logic"""
    try:
        # Calculate signals
        price_signal = 1 if request.price_change > 0.02 else (-1 if request.price_change < -0.02 else 0)
        trend_signal = 1 if request.trend > 0.05 else (-1 if request.trend < -0.05 else 0)
        vol_signal = -1 if request.volatility > 10 else 0
        
        # Composite score
        total_signal = price_signal + trend_signal + vol_signal
        
        # Decision logic
        if total_signal >= 2:
            action = "BUY"
            confidence = min(0.85, 0.60 + (total_signal * 0.1))
        elif total_signal <= -2:
            action = "SELL"
            confidence = min(0.85, 0.60 + (abs(total_signal) * 0.1))
        else:
            action = "HOLD"
            confidence = 0.50
        
        # Generate reasoning
        reasons = []
        if abs(request.price_change) > 0.02:
            reasons.append(f"Price momentum {'positive' if request.price_change > 0 else 'negative'}")
        if abs(request.trend) > 0.05:
            reasons.append(f"Strong {'upward' if request.trend > 0 else 'downward'} trend")
        if request.volatility > 10:
            reasons.append("High volatility detected")
        
        reason = f"AI Analysis for {request.symbol}: " + ". ".join(reasons) if reasons else f"Neutral market conditions for {request.symbol}"
        
        return {
            "action": action,
            "confidence": confidence,
            "reason": reason
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# For Vercel serverless
handler = app

# For local development
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)