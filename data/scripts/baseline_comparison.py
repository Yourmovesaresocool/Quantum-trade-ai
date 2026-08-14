"""
NAIVE BASELINE COMPARISON (v2 — for RETURNS target)

CHANGE FROM v1: v1's target was price levels, so the fair baseline was
"predict tomorrow's price = today's price." Now that prepare_data.py v3
predicts RETURNS instead, the equivalent trivial baseline is "always
predict 0% change" (i.e. no movement) — NOT "predict yesterday's price",
which doesn't apply to a returns target.

Run this locally (no GPU needed) from the folder containing
lstm_X_train.npy, lstm_y_train.npy, lstm_symbols.npy:

    python baseline_comparison.py

Requires: numpy, scikit-learn
"""

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

print("Loading data...")
X = np.load('lstm_X_train.npy')
y = np.load('lstm_y_train.npy')  # returns, e.g. 0.012 = +1.2%
symbols = np.load('lstm_symbols.npy', allow_pickle=True)

# ============================================
# Recreate the exact same time-based split used in training
# ============================================
train_idx, test_idx = [], []
for symbol in np.unique(symbols):
    idx = np.where(symbols == symbol)[0]
    split_point = int(len(idx) * 0.8)
    train_idx.extend(idx[:split_point])
    test_idx.extend(idx[split_point:])

test_idx = np.array(test_idx)
y_test = y[test_idx]

print(f"Test set: {len(y_test):,} sequences\n")

# ============================================
# NAIVE BASELINE for a RETURNS target: "always predict 0% change"
# ============================================
naive_predictions = np.zeros_like(y_test)

naive_r2 = r2_score(y_test, naive_predictions)
naive_mae = mean_absolute_error(y_test, naive_predictions)

# Directional accuracy of the naive baseline is undefined (it never
# predicts a direction), so report the base rate of up-days instead —
# this is what a coin flip / majority-class guess would score.
up_day_rate = float(np.mean(y_test > 0))

print("=" * 60)
print("NAIVE BASELINE (always predict 0% change)")
print("=" * 60)
print(f"R^2:  {naive_r2:.4f} ({naive_r2*100:.2f}%)")
print(f"MAE:  {naive_mae:.5f}")
print(f"Base rate of up-days in test set: {up_day_rate*100:.2f}% (chance-level directional accuracy)")
print()
print("Compare these numbers against lstm_metadata.json's r2_score,")
print("mae_return_scale, and directional_accuracy after retraining.")
print("A meaningfully positive R^2 here (unlike the old price-level")
print("version, where naive R^2 was 0.99+) means the model is now")
print("being measured on something it actually has to work for.")