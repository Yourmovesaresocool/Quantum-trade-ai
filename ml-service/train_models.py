"""
Train LSTM and DQN models for the AI Trading Bot
Run this script to generate model files
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import pickle
from collections import deque
import random
import os
import sys

print("="*60)
print("AI TRADING BOT - MODEL TRAINING")
print("="*60)

# Check if stock_data.csv exists
if not os.path.exists('../stock_data.csv'):
    print("\n❌ stock_data.csv not found!")
    print("Please run generate_dataset.py first from the project root:")
    print("  python generate_dataset.py")
    sys.exit(1)

# Load data
print("\n1. Loading data from stock_data.csv...")
df = pd.read_csv('../stock_data.csv')
print(f"✅ Loaded {len(df)} records")

# Filter for one symbol to train on (e.g., BTC or AAPL)
symbol = 'BTC' if 'BTC' in df['symbol'].values else df['symbol'].iloc[0]
df_symbol = df[df['symbol'] == symbol].copy()
df_symbol = df_symbol.sort_values('timestamp')
print(f"✅ Using {symbol} with {len(df_symbol)} records")

# Extract closing prices
data = df_symbol['close'].values.reshape(-1, 1)

# ============================================
# PART 1: TRAIN LSTM MODEL
# ============================================
print("\n" + "="*60)
print("PART 1: Training LSTM Price Prediction Model")
print("="*60)

# Create scaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create sequences
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(scaled_data, seq_length)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Build LSTM model
print("\nBuilding LSTM model...")
lstm_model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=25),
    Dense(units=1)
])

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
print("✅ Model built")

# Train
print("\nTraining LSTM model (this may take 5-10 minutes)...")
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = lstm_model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=30,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# Evaluate
train_loss = lstm_model.evaluate(X_train, y_train, verbose=0)
test_loss = lstm_model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Training complete!")
print(f"Train Loss: {train_loss:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Save LSTM model
lstm_model.save('lstm_model.h5')
print("✅ LSTM model saved: lstm_model.h5")

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Scaler saved: scaler.pkl")

# ============================================
# PART 2: TRAIN DQN TRADING AGENT
# ============================================
print("\n" + "="*60)
print("PART 2: Training DQN Trading Agent")
print("="*60)

# Trading Environment
class TradingEnvironment:
    def __init__(self, prices, initial_balance=10000):
        self.prices = prices
        self.initial_balance = initial_balance
        self.reset()
    
    def reset(self):
        self.balance = self.initial_balance
        self.shares = 0
        self.current_step = 0
        return self._get_state()
    
    def _get_state(self):
        if self.current_step == 0:
            price_change = 0
        else:
            price_change = (self.prices[self.current_step] - self.prices[self.current_step-1]) / self.prices[self.current_step-1]
        
        return np.array([
            self.balance / self.initial_balance,
            self.shares,
            self.prices[self.current_step] / 1000,
            price_change
        ])
    
    def step(self, action):
        current_price = self.prices[self.current_step]
        
        if action == 1:  # Buy
            shares_to_buy = self.balance // current_price
            if shares_to_buy > 0:
                self.shares += shares_to_buy
                self.balance -= shares_to_buy * current_price
        
        elif action == 2:  # Sell
            if self.shares > 0:
                self.balance += self.shares * current_price
                self.shares = 0
        
        self.current_step += 1
        portfolio_value = self.balance + (self.shares * current_price)
        reward = (portfolio_value - self.initial_balance) / self.initial_balance
        
        done = self.current_step >= len(self.prices) - 1
        next_state = self._get_state() if not done else None
        
        return next_state, reward, done

# DQN Agent
class DQNAgent:
    def __init__(self, state_size=4, action_size=3):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
    
    def _build_model(self):
        model = Sequential([
            Dense(64, input_dim=self.state_size, activation='relu'),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape(1, -1), verbose=0)[0])
            
            target_f = self.model.predict(state.reshape(1, -1), verbose=0)
            target_f[0][action] = target
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Train DQN
print("\nInitializing trading environment...")
prices = df_symbol['close'].values
env = TradingEnvironment(prices)
agent = DQNAgent()

episodes = 30
print(f"Training DQN agent for {episodes} episodes...")

for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    
    while True:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        
        if done:
            print(f"Episode {episode+1}/{episodes} - Reward: {total_reward:.4f} - Epsilon: {agent.epsilon:.3f}")
            break
        
        agent.replay(batch_size=32)

print("\n✅ DQN training complete!")

# Save DQN model
agent.model.save('dqn_trading_model.h5')
print("✅ DQN model saved: dqn_trading_model.h5")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*60)
print("TRAINING COMPLETE! 🎉")
print("="*60)
print("\nGenerated files:")
print("  ✅ lstm_model.h5 (~5-10 MB)")
print("  ✅ scaler.pkl (~1 KB)")
print("  ✅ dqn_trading_model.h5 (~2-5 MB)")
print("\nThese files are now ready to be used by the FastAPI service!")
print("\nNext steps:")
print("1. Keep these files in the ml-service folder")
print("2. Start FastAPI: uvicorn main:app --reload --port 8000")
print("3. Start backend: cd ../backend && npm run dev")
print("4. Start frontend: cd ../frontend && npm start")
print("="*60)