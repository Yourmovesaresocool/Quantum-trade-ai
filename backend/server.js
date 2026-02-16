/*
 * COMPLETE FIXED BACKEND SERVER.JS
 * 
 * FIXES INCLUDED:
 * 1. ✅ calculateVolatility() function added
 * 2. ✅ calculateTrend() function added
 * 3. ✅ BTC-USD → BTC crypto mapping
 * 4. ✅ Better error handling
 * 5. ✅ All helper functions BEFORE endpoints
 * 
 * REPLACE YOUR backend/server.js WITH THIS ENTIRE FILE
 */

const express = require('express');
const cors = require('cors');
const axios = require('axios');
const { Pool } = require('pg');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3001;

// ============================================
// DATABASE CONNECTION
// ============================================

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: process.env.NODE_ENV === 'production' ? { 
    rejectUnauthorized: false,
    sslmode: 'require'
  } : false
});

// Test database connection
pool.query('SELECT NOW()', (err, res) => {
  if (err) {
    console.error('❌ Database connection error:', err.message);
  } else {
    console.log('✅ Database connected at:', res.rows[0].now);
  }
});

// ============================================
// MIDDLEWARE
// ============================================

app.use(cors());
app.use(express.json());

// ============================================
// CONFIGURATION
// ============================================

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';

console.log('\n' + '='.repeat(60));
console.log('🚀 QUANTUM TRADE BACKEND SERVER');
console.log('='.repeat(60));
console.log('📊 Database:', process.env.DATABASE_URL ? 'Connected' : 'Not configured');
console.log('🔗 ML Service:', ML_SERVICE_URL);
console.log('🌐 Port:', PORT);
console.log('='.repeat(60) + '\n');

// ============================================
// HELPER FUNCTIONS (MUST BE BEFORE ENDPOINTS!)
// ============================================

/**
 * Calculate price volatility (standard deviation of returns)
 * @param {Array<number>} prices - Array of stock prices
 * @returns {number} - Volatility as a percentage
 */
function calculateVolatility(prices) {
  if (!prices || prices.length < 2) {
    return 0;
  }
  
  // Calculate daily returns
  const returns = [];
  for (let i = 1; i < prices.length; i++) {
    const dailyReturn = (prices[i] - prices[i-1]) / prices[i-1];
    returns.push(dailyReturn);
  }
  
  if (returns.length === 0) {
    return 0;
  }
  
  // Calculate mean
  const mean = returns.reduce((sum, val) => sum + val, 0) / returns.length;
  
  // Calculate variance
  const variance = returns.reduce((sum, val) => {
    return sum + Math.pow(val - mean, 2);
  }, 0) / returns.length;
  
  // Standard deviation (volatility)
  const stdDev = Math.sqrt(variance);
  
  // Return as percentage
  return stdDev * 100;
}

/**
 * Calculate price trend (momentum)
 * @param {Array<number>} prices - Array of stock prices
 * @returns {number} - Trend as a percentage
 */
function calculateTrend(prices) {
  if (!prices || prices.length < 20) {
    return 0;
  }
  
  // Compare recent average (last 10 days) to older average (10-20 days ago)
  const recentPrices = prices.slice(0, 10);
  const olderPrices = prices.slice(10, 20);
  
  const recentAvg = recentPrices.reduce((sum, val) => sum + val, 0) / recentPrices.length;
  const olderAvg = olderPrices.reduce((sum, val) => sum + val, 0) / olderPrices.length;
  
  if (olderAvg === 0) {
    return 0;
  }
  
  // Calculate percentage change
  const trend = ((recentAvg - olderAvg) / olderAvg) * 100;
  
  return trend;
}

/**
 * Map crypto symbols (BTC-USD → BTC)
 * @param {string} symbol - Stock symbol
 * @returns {string} - Database symbol
 */
function mapSymbolForDatabase(symbol) {
  // Remove -USD suffix for crypto
  return symbol.replace('-USD', '');
}

// ============================================
// API ENDPOINTS
// ============================================

/**
 * Health check endpoint
 */
app.get('/api/health', async (req, res) => {
  try {
    await pool.query('SELECT 1');
    res.json({ 
      status: 'healthy',
      database: 'connected',
      mlService: ML_SERVICE_URL,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({ 
      status: 'unhealthy',
      error: error.message
    });
  }
});

/**
 * Get historical stock prices
 */
app.get('/api/prices/:symbol', async (req, res) => {
  try {
    let { symbol } = req.params;
    const { limit = 100 } = req.query;
    
    // Map crypto symbols
    const dbSymbol = mapSymbolForDatabase(symbol);
    
    console.log(`📊 Fetching prices for ${symbol} (DB: ${dbSymbol})`);
    
    const result = await pool.query(
      `SELECT * FROM historical_prices 
       WHERE symbol = $1 
       ORDER BY timestamp DESC 
       LIMIT $2`,
      [dbSymbol, parseInt(limit)]
    );
    
    if (result.rows.length === 0) {
      return res.status(404).json({ 
        success: false,
        error: `No data found for ${symbol}`,
        hint: `Database symbol: ${dbSymbol}. Stock may not be in database.`
      });
    }
    
    console.log(`✅ Found ${result.rows.length} records for ${symbol}`);
    
    res.json({ 
      success: true,
      data: result.rows,
      symbol: symbol,
      count: result.rows.length
    });
    
  } catch (error) {
    console.error('❌ Database error:', error.message);
    res.status(500).json({ 
      success: false,
      error: error.message
    });
  }
});

/**
 * Get price prediction from ML service
 */
app.post('/api/predict', async (req, res) => {
  try {
    const { symbol } = req.body;
    
    if (!symbol) {
      return res.status(400).json({
        success: false,
        error: 'Symbol is required'
      });
    }
    
    const dbSymbol = mapSymbolForDatabase(symbol);
    
    console.log(`🔮 Predicting price for ${symbol} (DB: ${dbSymbol})`);
    
    // Get historical prices
    const result = await pool.query(
      `SELECT close FROM historical_prices 
       WHERE symbol = $1 
       ORDER BY timestamp DESC 
       LIMIT 30`,
      [dbSymbol]
    );
    
    if (result.rows.length < 10) {
      return res.status(400).json({
        success: false,
        error: `Insufficient data for ${symbol}. Need at least 10 days.`
      });
    }
    
    const recent_prices = result.rows
      .map(r => parseFloat(r.close))
      .reverse();
    
    // Call ML service
    try {
      const mlResponse = await axios.post(
        `${ML_SERVICE_URL}/predict`,
        { recent_prices },
        { timeout: 10000 }
      );
      
      console.log(`✅ Prediction received for ${symbol}`);
      
      res.json({
        success: true,
        prediction: mlResponse.data,
        symbol: symbol
      });
      
    } catch (mlError) {
      console.error('❌ ML service error:', mlError.message);
      
      if (mlError.code === 'ECONNREFUSED') {
        return res.status(503).json({
          success: false,
          error: 'ML service not running',
          hint: 'Start ML service with: cd ml-service && python main.py'
        });
      }
      
      throw mlError;
    }
    
  } catch (error) {
    console.error('❌ Prediction error:', error.message);
    res.status(500).json({ 
      success: false,
      error: error.message
    });
  }
});

/**
 * Get trading decision from ML service
 */
app.post('/api/trade', async (req, res) => {
  try {
    const { symbol, balance, shares } = req.body;
    
    if (!symbol) {
      return res.status(400).json({
        success: false,
        error: 'Symbol is required'
      });
    }
    
    const dbSymbol = mapSymbolForDatabase(symbol);
    
    console.log(`📈 Analyzing trade for ${symbol} (DB: ${dbSymbol})`);
    
    // Get price history
    const result = await pool.query(
      `SELECT close, timestamp FROM historical_prices 
       WHERE symbol = $1 
       ORDER BY timestamp DESC 
       LIMIT 30`,
      [dbSymbol]
    );
    
    if (result.rows.length === 0) {
      return res.status(404).json({
        success: false,
        error: `No data found for ${symbol}`
      });
    }
    
    const current_price = parseFloat(result.rows[0].close);
    const prices = result.rows.map(r => parseFloat(r.close));
    
    // Calculate metrics using helper functions
    const price_change = prices.length > 1 
      ? ((current_price - prices[1]) / prices[1]) * 100 
      : 0;
    
    const volatility = calculateVolatility(prices);
    const trend = calculateTrend(prices);
    
    console.log(`  💰 Price: $${current_price.toFixed(2)}`);
    console.log(`  📊 Change: ${price_change.toFixed(2)}%`);
    console.log(`  📉 Volatility: ${volatility.toFixed(2)}%`);
    console.log(`  📈 Trend: ${trend.toFixed(2)}%`);
    
    // Call ML service
    try {
      const mlResponse = await axios.post(
        `${ML_SERVICE_URL}/trade_decision`,
        {
          balance: balance || 10000,
          shares: shares || 0,
          current_price,
          price_change,
          volatility,
          trend,
          symbol: dbSymbol
        },
        { timeout: 10000 }
      );
      
      console.log(`✅ Decision: ${mlResponse.data.action} for ${symbol}`);
      
      res.json({
        success: true,
        decision: mlResponse.data,
        current_price,
        market_context: {
          price_change: price_change.toFixed(2),
          volatility: volatility.toFixed(2),
          trend: trend > 0 ? 'BULLISH' : 'BEARISH'
        }
      });
      
    } catch (mlError) {
      console.error('❌ ML service error:', mlError.message);
      
      if (mlError.code === 'ECONNREFUSED') {
        return res.status(503).json({
          success: false,
          error: 'ML service not running',
          hint: 'Start ML service with: cd ml-service && python main.py'
        });
      }
      
      throw mlError;
    }
    
  } catch (error) {
    console.error('❌ Trade analysis error:', error.message);
    res.status(500).json({ 
      success: false,
      error: error.message
    });
  }
});

/**
 * Get trade history
 */
app.get('/api/trades/:symbol', async (req, res) => {
  try {
    const { symbol } = req.params;
    const dbSymbol = mapSymbolForDatabase(symbol);
    
    const result = await pool.query(
      'SELECT * FROM trades WHERE symbol = $1 ORDER BY timestamp DESC LIMIT 50',
      [dbSymbol]
    );
    
    res.json({ 
      success: true,
      data: result.rows,
      count: result.rows.length
    });
    
  } catch (error) {
    console.error('❌ Trades error:', error.message);
    res.status(500).json({ 
      success: false,
      error: error.message
    });
  }
});

/**
 * Get portfolio statistics
 */
app.get('/api/portfolio', async (req, res) => {
  try {
    const tradesResult = await pool.query(
      'SELECT symbol, action, price, quantity FROM trades ORDER BY timestamp'
    );
    
    const portfolio = {};
    let totalInvested = 0;
    let totalValue = 0;
    
    // Calculate holdings
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
    
    // Calculate current values
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
    console.error('❌ Portfolio error:', error.message);
    res.status(500).json({ 
      success: false,
      error: error.message
    });
  }
});

/**
 * Get list of available stocks
 */
app.get('/api/stocks', async (req, res) => {
  try {
    const result = await pool.query(
      'SELECT DISTINCT symbol FROM historical_prices ORDER BY symbol'
    );
    
    const symbols = result.rows.map(row => row.symbol);
    
    res.json({ 
      success: true,
      symbols,
      count: symbols.length
    });
    
  } catch (error) {
    console.error('❌ Stocks error:', error.message);
    res.status(500).json({ 
      success: false,
      error: error.message
    });
  }
});

// ============================================
// ERROR HANDLING
// ============================================

/**
 * Global error handler
 */
app.use((err, req, res, next) => {
  console.error('❌ Unhandled error:', err);
  res.status(500).json({ 
    success: false,
    error: 'Internal server error',
    message: err.message
  });
});

/**
 * 404 handler
 */
app.use((req, res) => {
  res.status(404).json({
    success: false,
    error: 'Endpoint not found',
    availableEndpoints: [
      'GET /api/health',
      'GET /api/prices/:symbol',
      'POST /api/predict',
      'POST /api/trade',
      'GET /api/trades/:symbol',
      'GET /api/portfolio',
      'GET /api/stocks'
    ]
  });
});

// ============================================a
// SERVER STARTUP
// ============================================

if (process.env.NODE_ENV !== 'production') {
  app.listen(PORT, () => {
    console.log('\n' + '='.repeat(60));
    console.log('✅ BACKEND SERVER RUNNING');
    console.log('='.repeat(60));
    console.log(`🌐 URL: http://localhost:${PORT}`);
    console.log('📊 Database: Connected');
    console.log(`🔗 ML Service: ${ML_SERVICE_URL}`);
    console.log('\n📋 Available endpoints:');
    console.log('  GET  /api/health');
    console.log('  GET  /api/prices/:symbol');
    console.log('  POST /api/predict');
    console.log('  POST /api/trade');
    console.log('  GET  /api/trades/:symbol');
    console.log('  GET  /api/portfolio');
    console.log('  GET  /api/stocks');
    console.log('='.repeat(60) + '\n');
  });
}

// Export for Vercel
module.exports = app;