// frontend/src/utils/format.test.js
//
// Run with: npm test (CRA's react-scripts test already wires up Jest —
// no extra config needed since this project was created with create-react-app)

import { fmtMoney, fmtPct } from './format';

describe('fmtMoney', () => {
  test('formats a positive number with 2 decimal places and a $ sign', () => {
    expect(fmtMoney(1234.5)).toBe('$1,234.50');
  });

  test('formats zero correctly', () => {
    expect(fmtMoney(0)).toBe('$0.00');
  });

  test('treats null/undefined as 0 rather than throwing', () => {
    expect(fmtMoney(null)).toBe('$0.00');
    expect(fmtMoney(undefined)).toBe('$0.00');
  });

  test('adds thousands separators for large numbers', () => {
    expect(fmtMoney(1000000)).toBe('$1,000,000.00');
  });
});

describe('fmtPct', () => {
  test('prefixes a positive value with +', () => {
    expect(fmtPct(2.834)).toBe('+2.83%');
  });

  test('does not double-prefix a negative value', () => {
    expect(fmtPct(-0.96)).toBe('-0.96%');
  });

  test('treats exactly 0 as non-negative (gets a + prefix)', () => {
    expect(fmtPct(0)).toBe('+0.00%');
  });

  test('rounds to 2 decimal places', () => {
    expect(fmtPct(1.005)).toMatch(/^\+1\.0[01]%$/); // float rounding edge case, either is acceptable
  });
});