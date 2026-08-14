// backend/utils/marketMath.js
//
// Pure functions extracted from server.js — no DB queries, no HTTP,
// no side effects. Kept separate specifically so they're trivially
// unit-testable without spinning up Express or Postgres.

function calculateVolatility(prices) {
  if (!prices || prices.length < 2) return 0;
  const returns = [];
  for (let i = 1; i < prices.length; i++) {
    returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
  }
  if (returns.length === 0) return 0;
  const mean = returns.reduce((sum, v) => sum + v, 0) / returns.length;
  const variance = returns.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / returns.length;
  return Math.sqrt(variance) * 100;
}

function calculateTrend(prices) {
  if (!prices || prices.length < 20) return 0;
  const recentAvg = prices.slice(0, 10).reduce((s, v) => s + v, 0) / 10;
  const olderAvg = prices.slice(10, 20).reduce((s, v) => s + v, 0) / 10;
  if (olderAvg === 0) return 0;
  return ((recentAvg - olderAvg) / olderAvg) * 100;
}

module.exports = { calculateVolatility, calculateTrend };