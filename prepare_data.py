"""
STEP 1: PREPARE DATA FOR GOOGLE CLOUD
Converts your Yahoo Finance data into format ready for LSTM training
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

print("=" * 70)
print("📊 PREPARING DATA FOR GOOGLE CLOUD LSTM TRAINING")
print("=" * 70)

# ============================================
# LOAD YOUR REAL DATA
# ============================================

print("\n📂 Loading real_stock_data.csv...")
try:
    df = pd.read_csv('real_stock_data.csv')
    print(f"✅ Loaded {len(df):,} records")
    print(f"📊 Stocks: {df['symbol'].nunique()}")
except FileNotFoundError:
    print("❌ real_stock_data.csv not found!")
    print("Make sure you ran download_real_data.py first")
    exit(1)

# ============================================
# PREPARE DATA FOR LSTM
# ============================================

print("\n🔧 Preparing data for LSTM...")

# Convert timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)

# We'll use closing prices for prediction
# LSTM needs sequences: [day1, day2, day3...] -> predict day N+1

def create_sequences(data, seq_length=60):
    """
    Create sequences for LSTM
    seq_length: how many days to look back (60 = ~3 months)
    """
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        target = data[i + seq_length]
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

# Process each stock separately
all_sequences = []
all_targets = []
all_symbols = []

print("  Creating sequences for each stock...")

for symbol in df['symbol'].unique():
    stock_data = df[df['symbol'] == symbol].copy()
    
    if len(stock_data) < 70:  # Need at least 70 days
        continue
    
    # Get close prices
    prices = stock_data['close'].values
    
    # Normalize prices (LSTM works better with normalized data)
    scaler = MinMaxScaler()
    prices_normalized = scaler.fit_transform(prices.reshape(-1, 1)).flatten()
    
    # Create sequences (60 days -> predict next day)
    sequences, targets = create_sequences(prices_normalized, seq_length=60)
    
    all_sequences.append(sequences)
    all_targets.append(targets)
    all_symbols.extend([symbol] * len(sequences))
    
    print(f"    {symbol}: {len(sequences)} sequences created")

# Combine all stocks
X = np.vstack(all_sequences)
y = np.hstack(all_targets)

print(f"\n✅ Data prepared!")
print(f"  Total sequences: {len(X):,}")
print(f"  Sequence shape: {X.shape}")
print(f"  Target shape: {y.shape}")

# ============================================
# SAVE DATA
# ============================================

print("\n💾 Saving data...")

# Reshape for LSTM (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Save as numpy files (efficient for Google Cloud)
np.save('lstm_X_train.npy', X)
np.save('lstm_y_train.npy', y)

# Save scaler (we'll need this for predictions)
scaler = MinMaxScaler()
scaler.fit(df['close'].values.reshape(-1, 1))
with open('price_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"✅ Saved:")
print(f"  lstm_X_train.npy - {X.nbytes / 1024 / 1024:.1f} MB")
print(f"  lstm_y_train.npy - {y.nbytes / 1024 / 1024:.1f} MB")
print(f"  price_scaler.pkl")

# ============================================
# CREATE INFO FILE
# ============================================

info = {
    'num_samples': len(X),
    'num_stocks': df['symbol'].nunique(),
    'sequence_length': 60,
    'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
    'stocks': list(df['symbol'].unique())
}

with open('training_info.txt', 'w') as f:
    for key, value in info.items():
        f.write(f"{key}: {value}\n")

print("\n" + "=" * 70)
print("🎉 DATA PREPARATION COMPLETE!")
print("=" * 70)

print("\n📝 FILES CREATED:")
print("  1. lstm_X_train.npy - Input sequences")
print("  2. lstm_y_train.npy - Target values")
print("  3. price_scaler.pkl - Price normalizer")
print("  4. training_info.txt - Data info")

print("\n📤 NEXT STEPS:")
print("  1. Upload these 4 files to Google Cloud Storage")
print("  2. Run the training script on Google Colab (FREE GPU!)")
print("  3. Download the trained model")

print("\n💡 TIP: Use Google Colab for FREE GPU training!")
print("=" * 70)