/*
 * QUANTUM TRADE BACKEND SERVER.JS — v6
 *
 * CHANGE FROM v5: predictions are now actually logged and scoreable.
 *
 *   - predictions table gains a UNIQUE(symbol, timestamp) index, and is
 *     now auto-created (IF NOT EXISTS) like users/trades, in case a
 *     fresh DB never ran upload_to_db.py's schema.
 *   - POST /api/predict now writes each forecast into predictions
 *     (upserted on symbol+timestamp, so re-running analysis on the same
 *     day updates the existing row instead of duplicating it) — it never
 *     fails the actual prediction response if the logging write fails.
 *   - NEW: GET /api/predictions/accuracy/:symbol — joins logged
 *     predictions against the actual close once the daily price sync
 *     has caught up to that date, returning average % error and
 *     direction accuracy (did the model correctly call up vs down).
 *
 * Everything else (auth, real per-user trading, portfolio, logging,
 * LSTM bars-based /predict, crypto removal) is unchanged from v5.
 */

const express = require('express');
const cors = require('cors');
const axios = require('axios');
const helmet = require('helmet');
const { Pool } = require('pg');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const rateLimit = require('express-rate-limit');
const winston = require('winston');
const morgan = require('morgan');
const fs = require('fs');
require('dotenv').config();

// ============================================
// STARTUP CONFIG VALIDATION
// ============================================

const REQUIRED_ENV = ['DATABASE_URL', 'JWT_SECRET'];
const missingEnv = REQUIRED_ENV.filter((key) => !process.env[key]);
if (missingEnv.length > 0) {
  console.error(`FATAL: Missing required env vars: ${missingEnv.join(', ')}`);
  process.exit(1);
}

const app = express();
const PORT = process.env.PORT || 3001;
const JWT_SECRET = process.env.JWT_SECRET;
const JWT_EXPIRES_IN = '7d';
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8000';
const FRONTEND_ORIGIN = process.env.FRONTEND_ORIGIN || 'http://localhost:3000';
const MIN_BARS_REQUIRED = 130; // 90-day sequence + 40-day indicator warmup
// Starting virtual cash for every new account. $100k gives users enough
// room to hold several positions at once without constantly running out
// of buying power — $10k was tight once someone owned 3-4 stocks.
const DEFAULT_STARTING_BALANCE = 100000;

// ============================================
// LOGGING (winston + morgan)
// ============================================

if (!fs.existsSync('logs')) fs.mkdirSync('logs');

const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  transports: [
    new winston.transports.File({ filename: 'logs/error.log', level: 'error' }),
    new winston.transports.File({ filename: 'logs/combined.log' }),
    new winston.transports.Console({
      format: winston.format.combine(winston.format.colorize(), winston.format.simple()),
    }),
  ],
});

// ============================================
// DATABASE CONNECTION
// ============================================

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false, sslmode: 'require' } : false,
});

pool.query('SELECT NOW()', (err, res) => {
  if (err) {
    logger.error('Database connection failed', { error: err.message });
  } else {
    logger.info(`Database connected at ${res.rows[0].now}`);
  }
});

// Auto-migrate: create/extend tables so upgrading between versions
// doesn't need manual SQL. Safe to run every startup — all IF NOT EXISTS.
async function ensureSchema() {
  try {
    await pool.query(`
      CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        email VARCHAR(255) UNIQUE NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        balance DECIMAL(14,2) NOT NULL DEFAULT ${DEFAULT_STARTING_BALANCE},
        initial_balance DECIMAL(14,2) NOT NULL DEFAULT ${DEFAULT_STARTING_BALANCE},
        created_at TIMESTAMP DEFAULT NOW()
      );
    `);
    // In case users table already existed from an earlier version without these columns:
    await pool.query(`ALTER TABLE users ADD COLUMN IF NOT EXISTS balance DECIMAL(14,2) NOT NULL DEFAULT ${DEFAULT_STARTING_BALANCE};`);
    await pool.query(`ALTER TABLE users ADD COLUMN IF NOT EXISTS initial_balance DECIMAL(14,2) NOT NULL DEFAULT ${DEFAULT_STARTING_BALANCE};`);

    await pool.query(`
      CREATE TABLE IF NOT EXISTS trades (
        id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(id),
        symbol VARCHAR(10),
        action VARCHAR(4),
        price DECIMAL(12,2),
        quantity DECIMAL(14,6),
        timestamp TIMESTAMP DEFAULT NOW(),
        profit_loss DECIMAL(12,2)
      );
    `);
    // In case trades table already existed from before without user_id,
    // or without a default on timestamp (the source of the "1/1/1970"
    // trade-history bug — CREATE TABLE IF NOT EXISTS alone can't fix a
    // column default on a table that already existed).
    await pool.query(`ALTER TABLE trades ADD COLUMN IF NOT EXISTS user_id INTEGER REFERENCES users(id);`);
    await pool.query(`ALTER TABLE trades ALTER COLUMN timestamp SET DEFAULT NOW();`);
    await pool.query(`CREATE INDEX IF NOT EXISTS idx_trades_user_symbol ON trades(user_id, symbol);`);

    // predictions: logs every AI forecast so it can be scored later
    // against the actual close once the daily price sync catches up.
    await pool.query(`
      CREATE TABLE IF NOT EXISTS predictions (
        id SERIAL PRIMARY KEY,
        symbol VARCHAR(10),
        timestamp TIMESTAMP,
        predicted_price DECIMAL(12,2),
        confidence DECIMAL(6,4),
        created_at TIMESTAMP DEFAULT NOW()
      );
    `);
    // One prediction per symbol per predicted day — re-running analysis
    // the same day updates the existing forecast instead of duplicating it.
    await pool.query(`
      CREATE UNIQUE INDEX IF NOT EXISTS idx_predictions_symbol_ts
      ON predictions(symbol, timestamp);
    `);

    logger.info('Schema ready (users + trades + predictions)');
  } catch (err) {
    logger.error('Failed to ensure schema', { error: err.message });
  }
}
ensureSchema();

// ============================================
// MIDDLEWARE
// ============================================

app.use(helmet());
app.use(cors({ origin: FRONTEND_ORIGIN, credentials: true }));
app.use(express.json());
app.use(
  morgan('combined', {
    stream: { write: (msg) => logger.info(msg.trim()) },
  })
);

const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 20,
  standardHeaders: true,
  legacyHeaders: false,
  message: { success: false, error: 'Too many auth attempts, please try again later.' },
});

logger.info('='.repeat(60));
logger.info('QUANTUM TRADE BACKEND SERVER STARTING');
logger.info(`Database: ${process.env.DATABASE_URL ? 'configured' : 'NOT CONFIGURED'}`);
logger.info(`ML Service: ${ML_SERVICE_URL}`);
logger.info(`Frontend origin (CORS): ${FRONTEND_ORIGIN}`);
logger.info(`Port: ${PORT}`);
logger.info('='.repeat(60));

// ============================================
// ERROR HANDLING UTILITIES
// ============================================

class AppError extends Error {
  constructor(message, statusCode, details = null) {
    super(message);
    this.statusCode = statusCode;
    this.details = details;
  }
}

function asyncHandler(fn) {
  return (req, res, next) => fn(req, res, next).catch(next);
}

// ============================================
// AUTH MIDDLEWARE
// ============================================

function requireAuth(req, res, next) {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ success: false, error: 'Missing or invalid authorization header' });
  }
  const token = authHeader.split(' ')[1];
  try {
    req.user = jwt.verify(token, JWT_SECRET);
    next();
  } catch (err) {
    return res.status(401).json({ success: false, error: 'Token invalid or expired' });
  }
}

// ============================================
// HELPER FUNCTIONS (for /api/trade signal logic — unchanged)
// ============================================

// ============================================
// HELPER FUNCTIONS (for /api/trade signal logic)
// ============================================

// calculateVolatility and calculateTrend now live in utils/marketMath.js —
// pulled out specifically so they're unit-testable without spinning up
// Express or touching Postgres. See backend/tests/marketMath.test.js.
const { calculateVolatility, calculateTrend } = require('./utils/marketMath');

// Weighted-average cost basis for a user's current holdings in a symbol,
// computed from their own trade history (BUY adds shares+cost, SELL
// reduces shares proportionally — matches standard brokerage accounting).
async function getHoldingsAndAvgCost(userId, symbol) {
  const result = await pool.query(
    `SELECT action, price, quantity FROM trades WHERE user_id = $1 AND symbol = $2 ORDER BY timestamp ASC`,
    [userId, symbol]
  );

  let shares = 0;
  let totalCost = 0;

  for (const trade of result.rows) {
    const price = parseFloat(trade.price);
    const qty = parseFloat(trade.quantity);
    if (trade.action === 'BUY') {
      totalCost += price * qty;
      shares += qty;
    } else if (trade.action === 'SELL') {
      const avgCost = shares > 0 ? totalCost / shares : 0;
      totalCost -= avgCost * qty;
      shares -= qty;
    }
  }

  const avgCost = shares > 0 ? totalCost / shares : 0;
  return { shares, avgCost };
}

// ============================================
// AUTH ROUTES
// ============================================

app.post(
  '/api/auth/register',
  authLimiter,
  asyncHandler(async (req, res) => {
    const { email, password } = req.body;

    if (!email || !password) {
      throw new AppError('Email and password are required', 400);
    }
    if (password.length < 8) {
      throw new AppError('Password must be at least 8 characters', 400);
    }

    const existing = await pool.query('SELECT id FROM users WHERE email = $1', [email]);
    if (existing.rows.length > 0) {
      throw new AppError('An account with this email already exists', 409);
    }

    const passwordHash = await bcrypt.hash(password, 10);
    const result = await pool.query(
      `INSERT INTO users (email, password_hash, balance, initial_balance)
       VALUES ($1, $2, $3, $3) RETURNING id, email, balance, initial_balance, created_at`,
      [email, passwordHash, DEFAULT_STARTING_BALANCE]
    );
    const user = result.rows[0];

    const token = jwt.sign({ userId: user.id, email: user.email }, JWT_SECRET, { expiresIn: JWT_EXPIRES_IN });

    logger.info(`New user registered: ${email}`);
    res.status(201).json({
      success: true,
      token,
      user: { id: user.id, email: user.email, balance: parseFloat(user.balance), initial_balance: parseFloat(user.initial_balance) },
    });
  })
);

app.post(
  '/api/auth/login',
  authLimiter,
  asyncHandler(async (req, res) => {
    const { email, password } = req.body;

    if (!email || !password) {
      throw new AppError('Email and password are required', 400);
    }

    const result = await pool.query('SELECT * FROM users WHERE email = $1', [email]);
    const user = result.rows[0];

    if (!user || !(await bcrypt.compare(password, user.password_hash))) {
      throw new AppError('Invalid email or password', 401);
    }

    const token = jwt.sign({ userId: user.id, email: user.email }, JWT_SECRET, { expiresIn: JWT_EXPIRES_IN });

    logger.info(`User logged in: ${email}`);
    res.json({
      success: true,
      token,
      user: { id: user.id, email: user.email, balance: parseFloat(user.balance), initial_balance: parseFloat(user.initial_balance) },
    });
  })
);

app.get(
  '/api/auth/me',
  requireAuth,
  asyncHandler(async (req, res) => {
    const result = await pool.query('SELECT id, email, balance, initial_balance FROM users WHERE id = $1', [req.user.userId]);
    if (result.rows.length === 0) throw new AppError('User not found', 404);
    const user = result.rows[0];
    res.json({
      success: true,
      user: { id: user.id, email: user.email, balance: parseFloat(user.balance), initial_balance: parseFloat(user.initial_balance) },
    });
  })
);

// ============================================
// PUBLIC ENDPOINTS
// ============================================

app.get(
  '/api/health',
  asyncHandler(async (req, res) => {
    await pool.query('SELECT 1');
    res.json({
      status: 'healthy',
      database: 'connected',
      mlService: ML_SERVICE_URL,
      timestamp: new Date().toISOString(),
    });
  })
);

app.get(
  '/api/stocks',
  asyncHandler(async (req, res) => {
    const result = await pool.query('SELECT DISTINCT symbol FROM historical_prices ORDER BY symbol');
    const symbols = result.rows.map((row) => row.symbol);
    res.json({ success: true, symbols, count: symbols.length });
  })
);

app.get(
  '/api/prices/:symbol',
  asyncHandler(async (req, res) => {
    const { symbol } = req.params;
    const { limit = 100 } = req.query;

    const result = await pool.query(
      `SELECT * FROM historical_prices WHERE symbol = $1 ORDER BY timestamp DESC LIMIT $2`,
      [symbol, parseInt(limit, 10)]
    );

    if (result.rows.length === 0) {
      throw new AppError(`No data found for ${symbol}`, 404);
    }

    res.json({ success: true, data: result.rows, symbol, count: result.rows.length });
  })
);

// ============================================
// PROTECTED ENDPOINTS — AI PREDICTION
// ============================================

app.post(
  '/api/predict',
  requireAuth,
  asyncHandler(async (req, res) => {
    const { symbol } = req.body;
    if (!symbol) throw new AppError('Symbol is required', 400);

    logger.info(`Predict requested for ${symbol} by user ${req.user.userId}`);

    const result = await pool.query(
      `SELECT open, high, low, close, volume, timestamp
       FROM historical_prices
       WHERE symbol = $1
       ORDER BY timestamp DESC
       LIMIT $2`,
      [symbol, MIN_BARS_REQUIRED]
    );

    if (result.rows.length < MIN_BARS_REQUIRED) {
      throw new AppError(
        `Insufficient data for ${symbol}. Need at least ${MIN_BARS_REQUIRED} days, have ${result.rows.length}.`,
        400
      );
    }

    const bars = result.rows
      .slice()
      .reverse()
      .map((r) => ({
        open: parseFloat(r.open),
        high: parseFloat(r.high),
        low: parseFloat(r.low),
        close: parseFloat(r.close),
        volume: parseFloat(r.volume),
      }));

    try {
      const mlResponse = await axios.post(
        `${ML_SERVICE_URL}/predict`,
        { symbol, bars },
        { timeout: 10000 }
      );

      // Log this forecast so it can be scored later against the actual
      // close once the daily price sync reaches that date. result.rows[0]
      // is the most recent bar (rows are DESC-ordered before the .reverse()
      // into `bars` above) — the prediction is for the next trading day
      // after that. Upserted on (symbol, timestamp): re-running analysis
      // for a symbol already predicted today just updates that row rather
      // than creating a duplicate. Logging failure never blocks the
      // response — the person still gets their forecast either way.
      try {
        const predictionDate = new Date(result.rows[0].timestamp);
        predictionDate.setDate(predictionDate.getDate() + 1);
        await pool.query(
          `INSERT INTO predictions (symbol, timestamp, predicted_price, confidence)
           VALUES ($1, $2, $3, $4)
           ON CONFLICT (symbol, timestamp) DO UPDATE
           SET predicted_price = EXCLUDED.predicted_price,
               confidence = EXCLUDED.confidence,
               created_at = NOW()`,
          [symbol, predictionDate, mlResponse.data.predicted_price, mlResponse.data.confidence ?? null]
        );
      } catch (logErr) {
        logger.error('Failed to log prediction', { error: logErr.message, symbol });
      }

      res.json({ success: true, prediction: mlResponse.data, symbol });
    } catch (mlError) {
      if (mlError.code === 'ECONNREFUSED') {
        throw new AppError('ML service is not reachable', 503, {
          hint: 'Start ML service with: cd ml-service && python main.py',
        });
      }
      if (mlError.response) {
        throw new AppError(
          mlError.response.data?.detail || 'ML service rejected the request',
          mlError.response.status
        );
      }
      throw mlError;
    }
  })
);

app.post(
  '/api/trade',
  requireAuth,
  asyncHandler(async (req, res) => {
    const { symbol, balance, shares } = req.body;
    if (!symbol) throw new AppError('Symbol is required', 400);

    const result = await pool.query(
      `SELECT close, timestamp FROM historical_prices WHERE symbol = $1 ORDER BY timestamp DESC LIMIT $2`,
      [symbol, 90]
    );

    if (result.rows.length === 0) {
      throw new AppError(`No data found for ${symbol}`, 404);
    }

    const current_price = parseFloat(result.rows[0].close);
    const prices = result.rows.map((r) => parseFloat(r.close));
    const price_change = prices.length > 1 ? ((current_price - prices[1]) / prices[1]) * 100 : 0;
    const volatility = calculateVolatility(prices);
    const trend = calculateTrend(prices);

    try {
      const mlResponse = await axios.post(
        `${ML_SERVICE_URL}/trade_decision`,
        { balance: balance || DEFAULT_STARTING_BALANCE, shares: shares || 0, current_price, price_change, volatility, trend, symbol },
        { timeout: 10000 }
      );

      logger.info(`Trade decision for ${symbol}: ${mlResponse.data.action} (user ${req.user.userId})`);

      res.json({
        success: true,
        decision: mlResponse.data,
        current_price,
        market_context: {
          price_change: price_change.toFixed(2),
          volatility: volatility.toFixed(2),
          trend: trend > 0 ? 'BULLISH' : 'BEARISH',
        },
      });
    } catch (mlError) {
      if (mlError.code === 'ECONNREFUSED') {
        throw new AppError('ML service is not reachable', 503, {
          hint: 'Start ML service with: cd ml-service && python main.py',
        });
      }
      throw mlError;
    }
  })
);

// Predicted-vs-actual track record for a symbol. Only predictions whose
// target date already has an actual close in historical_prices show up
// here — a prediction made for tomorrow won't appear until the daily
// sync inserts tomorrow's real close.
app.get(
  '/api/predictions/accuracy/:symbol',
  requireAuth,
  asyncHandler(async (req, res) => {
    const { symbol } = req.params;

    const result = await pool.query(
      `SELECT p.timestamp, p.predicted_price, p.confidence, h.close AS actual_close,
              prev.close AS prev_close
       FROM predictions p
       JOIN historical_prices h
         ON h.symbol = p.symbol AND h.timestamp = p.timestamp
       LEFT JOIN LATERAL (
         SELECT close FROM historical_prices
         WHERE symbol = p.symbol AND timestamp < p.timestamp
         ORDER BY timestamp DESC LIMIT 1
       ) prev ON true
       WHERE p.symbol = $1
       ORDER BY p.timestamp DESC
       LIMIT 60`,
      [symbol]
    );

    const rows = result.rows.map((r) => {
      const predicted = parseFloat(r.predicted_price);
      const actual = parseFloat(r.actual_close);
      const prevClose = r.prev_close != null ? parseFloat(r.prev_close) : null;
      const pctError = actual ? ((predicted - actual) / actual) * 100 : null;
      const predictedDir = prevClose != null ? (predicted >= prevClose ? 'UP' : 'DOWN') : null;
      const actualDir = prevClose != null ? (actual >= prevClose ? 'UP' : 'DOWN') : null;
      return {
        date: r.timestamp,
        predicted,
        actual,
        pctError: pctError != null ? parseFloat(pctError.toFixed(2)) : null,
        directionCorrect: predictedDir && actualDir ? predictedDir === actualDir : null,
      };
    });

    const withDirection = rows.filter((r) => r.directionCorrect !== null);
    const avgAbsPctError = rows.length
      ? rows.reduce((sum, r) => sum + Math.abs(r.pctError || 0), 0) / rows.length
      : null;
    const directionAccuracy = withDirection.length
      ? (withDirection.filter((r) => r.directionCorrect).length / withDirection.length) * 100
      : null;

    res.json({
      success: true,
      symbol,
      count: rows.length,
      avgAbsPctError: avgAbsPctError != null ? parseFloat(avgAbsPctError.toFixed(2)) : null,
      directionAccuracy: directionAccuracy != null ? parseFloat(directionAccuracy.toFixed(1)) : null,
      history: rows,
    });
  })
);

// ============================================
// PROTECTED ENDPOINTS — REAL TRADING (persisted per-user)
// ============================================

// The actual buy/sell endpoint. Price comes from the DB, never the
// client — a client-supplied price would let someone fake profits.
app.post(
  '/api/trade/execute',
  requireAuth,
  asyncHandler(async (req, res) => {
    const { symbol, action, quantity } = req.body;
    const userId = req.user.userId;

    if (!symbol || !action || !quantity) {
      throw new AppError('symbol, action, and quantity are required', 400);
    }
    if (!['BUY', 'SELL'].includes(action)) {
      throw new AppError("action must be 'BUY' or 'SELL'", 400);
    }
    const qty = parseFloat(quantity);
    if (isNaN(qty) || qty <= 0) {
      throw new AppError('quantity must be a positive number', 400);
    }

    const priceResult = await pool.query(
      `SELECT close FROM historical_prices WHERE symbol = $1 ORDER BY timestamp DESC LIMIT 1`,
      [symbol]
    );
    if (priceResult.rows.length === 0) {
      throw new AppError(`No price data for ${symbol}`, 404);
    }
    const currentPrice = parseFloat(priceResult.rows[0].close);

    const userResult = await pool.query('SELECT balance FROM users WHERE id = $1', [userId]);
    const currentBalance = parseFloat(userResult.rows[0].balance);

    const { shares: currentShares, avgCost } = await getHoldingsAndAvgCost(userId, symbol);

    let profitLoss = null;
    let newBalance = currentBalance;

    if (action === 'BUY') {
      const cost = qty * currentPrice;
      if (cost > currentBalance) {
        throw new AppError(
          `Insufficient funds: need $${cost.toFixed(2)}, have $${currentBalance.toFixed(2)}`,
          400
        );
      }
      newBalance = currentBalance - cost;
    } else {
      // SELL
      if (qty > currentShares) {
        throw new AppError(
          `Insufficient shares: trying to sell ${qty}, hold ${currentShares.toFixed(4)}`,
          400
        );
      }
      const revenue = qty * currentPrice;
      profitLoss = (currentPrice - avgCost) * qty;
      newBalance = currentBalance + revenue;
    }

    // Update balance and insert the trade in one transaction so they
    // can't drift apart if something fails mid-way.
    const client = await pool.connect();
    try {
      await client.query('BEGIN');
      await client.query('UPDATE users SET balance = $1 WHERE id = $2', [newBalance, userId]);
      const insertResult = await client.query(
        `INSERT INTO trades (user_id, symbol, action, price, quantity, profit_loss)
         VALUES ($1, $2, $3, $4, $5, $6) RETURNING *`,
        [userId, symbol, action, currentPrice, qty, profitLoss]
      );
      await client.query('COMMIT');

      logger.info(`Trade executed: user ${userId} ${action} ${qty} ${symbol} @ $${currentPrice}`);

      res.json({
        success: true,
        trade: insertResult.rows[0],
        newBalance,
        currentPrice,
      });
    } catch (err) {
      await client.query('ROLLBACK');
      throw err;
    } finally {
      client.release();
    }
  })
);

// All trades for the logged-in user, across every symbol — used to
// populate the History tab.
app.get(
  '/api/trades',
  requireAuth,
  asyncHandler(async (req, res) => {
    const result = await pool.query(
      'SELECT * FROM trades WHERE user_id = $1 ORDER BY timestamp DESC LIMIT 100',
      [req.user.userId]
    );
    res.json({ success: true, data: result.rows, count: result.rows.length });
  })
);

app.get(
  '/api/trades/:symbol',
  requireAuth,
  asyncHandler(async (req, res) => {
    const { symbol } = req.params;
    const result = await pool.query(
      'SELECT * FROM trades WHERE user_id = $1 AND symbol = $2 ORDER BY timestamp DESC LIMIT 50',
      [req.user.userId, symbol]
    );
    res.json({ success: true, data: result.rows, count: result.rows.length });
  })
);

// Full portfolio snapshot for the logged-in user: cash balance, every
// held symbol with live value and P/L, and the total account value.
app.get(
  '/api/portfolio',
  requireAuth,
  asyncHandler(async (req, res) => {
    const userId = req.user.userId;

    const userResult = await pool.query('SELECT balance, initial_balance FROM users WHERE id = $1', [userId]);
    if (userResult.rows.length === 0) throw new AppError('User not found', 404);
    const balance = parseFloat(userResult.rows[0].balance);
    const initialBalance = parseFloat(userResult.rows[0].initial_balance);

    const symbolsResult = await pool.query(
      'SELECT DISTINCT symbol FROM trades WHERE user_id = $1',
      [userId]
    );

    const portfolio = {};
    let totalHoldingsValue = 0;

    for (const row of symbolsResult.rows) {
      const symbol = row.symbol;
      const { shares, avgCost } = await getHoldingsAndAvgCost(userId, symbol);

      if (shares > 0.0000001) {
        const priceResult = await pool.query(
          'SELECT close FROM historical_prices WHERE symbol = $1 ORDER BY timestamp DESC LIMIT 1',
          [symbol]
        );
        const currentPrice = priceResult.rows.length > 0 ? parseFloat(priceResult.rows[0].close) : avgCost;
        const currentValue = shares * currentPrice;
        const profitLoss = (currentPrice - avgCost) * shares;

        portfolio[symbol] = {
          shares,
          avgCost,
          currentPrice,
          currentValue,
          profitLoss,
        };
        totalHoldingsValue += currentValue;
      }
    }

    const totalValue = balance + totalHoldingsValue;

    res.json({
      success: true,
      balance,
      initialBalance,
      portfolio,
      totalHoldingsValue,
      totalValue,
      totalProfitLoss: totalValue - initialBalance,
    });
  })
);

// ============================================
// 404 HANDLER
// ============================================

app.use((req, res) => {
  res.status(404).json({
    success: false,
    error: 'Endpoint not found',
    availableEndpoints: [
      'POST /api/auth/register',
      'POST /api/auth/login',
      'GET  /api/auth/me',
      'GET  /api/health',
      'GET  /api/stocks',
      'GET  /api/prices/:symbol',
      'POST /api/predict (auth required)',
      'POST /api/trade (auth required)',
      'POST /api/trade/execute (auth required)',
      'GET  /api/trades (auth required)',
      'GET  /api/trades/:symbol (auth required)',
      'GET  /api/portfolio (auth required)',
      'GET  /api/predictions/accuracy/:symbol (auth required)',
    ],
  });
});

// ============================================
// GLOBAL ERROR HANDLER (must be registered last)
// ============================================

app.use((err, req, res, next) => {
  const statusCode = err.statusCode || 500;

  logger.error(`${req.method} ${req.path} - ${err.message}`, {
    stack: err.stack,
    statusCode,
  });

  const body = {
    success: false,
    error: statusCode === 500 ? 'Internal server error' : err.message,
  };
  if (err.details) body.hint = err.details;

  res.status(statusCode).json(body);
});

// ============================================
// SERVER STARTUP
// ============================================

// require.main === module is true when this file is run directly
// (`node server.js` — exactly what the Docker CMD and local dev both do),
// and false when it's require()'d by something else (test files via
// supertest). The previous `NODE_ENV !== 'production'` guard meant the
// server would never call listen() inside the Docker container, since
// the Dockerfile sets NODE_ENV=production — this fixes that too.
if (require.main === module) {
  app.listen(PORT, () => {
    logger.info('='.repeat(60));
    logger.info(`BACKEND SERVER RUNNING at http://localhost:${PORT}`);
    logger.info('='.repeat(60));
  });
}

module.exports = app;