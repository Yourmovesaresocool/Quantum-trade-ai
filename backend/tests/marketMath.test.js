// backend/tests/marketMath.test.js
//
// Run with: npx jest tests/marketMath.test.js
// (or just `npm test` once jest is wired into package.json — see notes below)

const { calculateVolatility, calculateTrend } = require('../utils/marketMath');

describe('calculateVolatility', () => {
  test('returns 0 for fewer than 2 prices', () => {
    expect(calculateVolatility([])).toBe(0);
    expect(calculateVolatility([100])).toBe(0);
  });

  test('returns 0 for a perfectly flat price series', () => {
    expect(calculateVolatility([100, 100, 100, 100])).toBe(0);
  });

  test('returns a higher number for a more volatile series', () => {
    const stable = [100, 101, 100, 101, 100];
    const volatile = [100, 130, 80, 140, 70];
    expect(calculateVolatility(volatile)).toBeGreaterThan(calculateVolatility(stable));
  });

  test('handles null/undefined input without throwing', () => {
    expect(calculateVolatility(null)).toBe(0);
    expect(calculateVolatility(undefined)).toBe(0);
  });
});

describe('calculateTrend', () => {
  test('returns 0 for fewer than 20 prices', () => {
    expect(calculateTrend(Array(19).fill(100))).toBe(0);
  });

  test('returns a positive number when recent prices are higher than older prices', () => {
    // prices[0..9] = "recent" (higher), prices[10..19] = "older" (lower) —
    // matches the array ordering calculateTrend expects (index 0 = most recent).
    const recentHigh = Array(10).fill(120);
    const olderLow = Array(10).fill(100);
    const trend = calculateTrend([...recentHigh, ...olderLow]);
    expect(trend).toBeGreaterThan(0);
  });

  test('returns a negative number when recent prices are lower than older prices', () => {
    const recentLow = Array(10).fill(90);
    const olderHigh = Array(10).fill(100);
    const trend = calculateTrend([...recentLow, ...olderHigh]);
    expect(trend).toBeLessThan(0);
  });

  test('returns 0 rather than dividing by zero when older average is 0', () => {
    const recent = Array(10).fill(50);
    const olderAllZero = Array(10).fill(0);
    expect(calculateTrend([...recent, ...olderAllZero])).toBe(0);
  });
});