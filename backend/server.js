const express = require('express');
const cors = require('cors');
const axios = require('axios');
const { Pool } = require('pg');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3001;

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false
});

app.use(cors());
app.use(express.json());

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

console.log('🚀 Server starting...');
console.log('📊 Database:', process.env.DATABASE_URL ? 'Connected' : 'Not configured');
console.log('🔗 ML Service:', ML_SERVICE_URL);

// Health Check
app.get('/api/health', async (req, res) => {
  try {
    await pool.query('SELECT 1');
    res.json({ status: 'healthy', database: 'connected' });
  } catch (error) {
    res.status(500).json({ status: 'unhealthy', error: error.message });
  }
});

// Get historical prices
app.get('/api/prices/:symbol', async (req, res) => {
  try {
    const { symbol } = req.params;
    const { limit = 100 } = req.query;
    
    console.log(`📊 Fetching ${limit} prices for ${symbol}`);
    
    const result = await pool.query(
      'SELECT * FROM historical_prices WHERE symbol = $1 ORDER BY timestamp DESC LIMIT $2',
      [symbol, limit]
    );
    
    console.log(`✅ Found ${result.rows.length} records`);
    res.json({ success: true, data: result.rows });
  } catch (error) {
    console.error('❌ Error fetching prices:', error.message);
    res.status(500).json({ success: false, error: error.message });
  }
});

// Price prediction
app.post('/api/predict', async (req, res) => {
  try {
    const { symbol } = req.body;
    console.log(`🔮 Prediction request for ${symbol}`);
    
    const priceResult = await pool.query(
      'SELECT close FROM historical_prices WHERE symbol = $1 ORDER BY timestamp DESC LIMIT 60',
      [symbol]
    );
    
    if (priceResult.rows.length < 60) {
      console.log(`⚠️ Only ${priceResult.rows.length} records, need 60`);
      return res.status(400).json({ 
        success: false, 
        error: `Need 60 data points, only have ${priceResult.rows.length}` 
      });
    }
    
    const recentPrices = priceResult.rows.reverse().map(row => parseFloat(row.close));
    console.log(`📈 Calling ML service with ${recentPrices.length} prices`);
    
    const mlResponse = await axios.post(`${ML_SERVICE_URL}/predict`, {
      recent_prices: recentPrices
    }, { timeout: 30000 }); // Increased timeout for Vercel
    
    console.log(`✅ Prediction: $${mlResponse.data.predicted_price}`);
    res.json({ success: true, prediction: mlResponse.data });
  } catch (error) {
    console.error('❌ Prediction error:', error.message);
    
    if (error.code === 'ECONNREFUSED') {
      return res.status(503).json({ 
        success: false, 
        error: 'ML Service unavailable' 
      });
    }
    
    res.status(500).json({ 
      success: false, 
      error: error.response?.data?.detail || error.message 
    });
  }
});

// Trading decision
app.post('/api/trade', async (req, res) => {
  try {
    const { symbol, balance = 10000, shares = 0 } = req.body;
    console.log(`💼 Trade decision for ${symbol} (Balance: $${balance}, Shares: ${shares})`);
    
    const priceResult = await pool.query(
      'SELECT close, volume, high, low FROM historical_prices WHERE symbol = $1 ORDER BY timestamp DESC LIMIT 10',
      [symbol]
    );
    
    if (priceResult.rows.length < 5) {
      return res.status(400).json({ 
        success: false, 
        error: 'Insufficient price data' 
      });
    }
    
    const prices = priceResult.rows.map(r => parseFloat(r.close));
    const currentPrice = prices[0];
    const previousPrice = prices[1];
    const priceChange = (currentPrice - previousPrice) / previousPrice;
    
    // Market indicators
    const avgPrice = prices.reduce((a, b) => a + b, 0) / prices.length;
    const volatility = Math.sqrt(
      prices.reduce((sum, p) => sum + Math.pow(p - avgPrice, 2), 0) / prices.length
    );
    const trend = (prices[0] - prices[prices.length - 1]) / prices[prices.length - 1];
    
    console.log(`💰 Price: $${currentPrice}, Change: ${(priceChange*100).toFixed(2)}%, Trend: ${(trend*100).toFixed(2)}%`);
    
    const mlResponse = await axios.post(`${ML_SERVICE_URL}/trade_decision`, {
      balance: parseFloat(balance),
      shares: parseFloat(shares),
      current_price: currentPrice,
      price_change: priceChange,
      volatility: volatility,
      trend: trend,
      symbol: symbol
    }, { timeout: 30000 }); // Increased timeout
    
    const decision = mlResponse.data;
    console.log(`✅ Decision: ${decision.action} (${(decision.confidence*100).toFixed(1)}%)`);
    
    res.json({ 
      success: true, 
      decision: {
        action: decision.action,
        confidence: decision.confidence,
        reason: decision.reason
      },
      current_price: currentPrice,
      market_context: {
        trend: trend > 0 ? 'BULLISH' : 'BEARISH',
        volatility: volatility > 5 ? 'HIGH' : 'NORMAL'
      }
    });
  } catch (error) {
    console.error('❌ Trade error:', error.message);
    
    if (error.code === 'ECONNREFUSED') {
      return res.status(503).json({ 
        success: false, 
        error: 'ML Service unavailable' 
      });
    }
    
    res.status(500).json({ 
      success: false, 
      error: error.response?.data?.detail || error.message 
    });
  }
});

// Get trade history
app.get('/api/trades/:symbol', async (req, res) => {
  try {
    const { symbol } = req.params;
    const result = await pool.query(
      'SELECT * FROM trades WHERE symbol = $1 ORDER BY timestamp DESC LIMIT 50',
      [symbol]
    );
    res.json({ success: true, data: result.rows });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// Portfolio stats
app.get('/api/portfolio', async (req, res) => {
  try {
    const tradesResult = await pool.query(
      'SELECT symbol, action, price, quantity FROM trades ORDER BY timestamp'
    );
    
    const portfolio = {};
    let totalInvested = 0;
    let totalValue = 0;
    
    for (const trade of tradesResult.rows) {
      if (!portfolio[trade.symbol]) {
        portfolio[trade.symbol] = { shares: 0, invested: 0 };
      }
      
      if (trade.action === 'BUY') {
        portfolio[trade.symbol].shares += parseFloat(trade.quantity);
        portfolio[trade.symbol].invested += parseFloat(trade.price) * parseFloat(trade.quantity);
      } else if (trade.action === 'SELL') {
        portfolio[trade.symbol].shares -= parseFloat(trade.quantity);
      }
    }
    
    for (const symbol in portfolio) {
      if (portfolio[symbol].shares > 0) {
        const priceResult = await pool.query(
          'SELECT close FROM historical_prices WHERE symbol = $1 ORDER BY timestamp DESC LIMIT 1',
          [symbol]
        );
        
        if (priceResult.rows.length > 0) {
          const currentPrice = parseFloat(priceResult.rows[0].close);
          portfolio[symbol].currentValue = portfolio[symbol].shares * currentPrice;
          portfolio[symbol].profitLoss = portfolio[symbol].currentValue - portfolio[symbol].invested;
          totalValue += portfolio[symbol].currentValue;
          totalInvested += portfolio[symbol].invested;
        }
      }
    }
    
    res.json({ 
      success: true, 
      portfolio,
      totalInvested,
      totalValue,
      totalProfitLoss: totalValue - totalInvested
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// For Vercel serverless deployment
if (process.env.NODE_ENV !== 'production') {
  app.listen(PORT, () => {
    console.log('');
    console.log('✅ Server running on port', PORT);
    console.log('📊 Ready to accept requests');
    console.log('');
  });
}

// Export for Vercel
module.exports = app;