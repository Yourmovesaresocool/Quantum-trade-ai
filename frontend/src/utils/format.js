// frontend/src/utils/format.js
//
// Pure display-formatting functions extracted from App.js — no React,
// no hooks, no side effects — so they're trivially unit-testable.

export const fmtMoney = (n) =>
  `$${(n || 0).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

export const fmtPct = (n) =>
  `${(n || 0) >= 0 ? '+' : ''}${(n || 0).toFixed(2)}%`;