import pandas as pd
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('backend/.env')

# Get database URL
DATABASE_URL = os.getenv('DATABASE_URL')

if not DATABASE_URL:
    print("❌ DATABASE_URL not found in backend/.env file")
    print("Please add your Railway PostgreSQL URL to backend/.env")
    exit(1)

print("Connecting to database...")
engine = create_engine(DATABASE_URL)

# Create tables
print("Creating tables...")
create_tables_query = """
-- Drop existing tables if any
DROP TABLE IF EXISTS trades CASCADE;
DROP TABLE IF EXISTS predictions CASCADE;
DROP TABLE IF EXISTS historical_prices CASCADE;

-- Create historical_prices table
CREATE TABLE historical_prices (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10),
    timestamp TIMESTAMP,
    open DECIMAL(12,2),
    high DECIMAL(12,2),
    low DECIMAL(12,2),
    close DECIMAL(12,2),
    volume BIGINT,
    UNIQUE(symbol, timestamp)
);

-- Create index for faster queries
CREATE INDEX idx_symbol_timestamp ON historical_prices(symbol, timestamp);

-- Create predictions table
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10),
    timestamp TIMESTAMP,
    predicted_price DECIMAL(12,2),
    confidence DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create trades table
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10),
    action VARCHAR(4),
    price DECIMAL(12,2),
    quantity DECIMAL(10,4),
    timestamp TIMESTAMP,
    profit_loss DECIMAL(12,2)
);
"""

try:
    with engine.connect() as conn:
        conn.execute(text(create_tables_query))
        conn.commit()
    print("✅ Tables created successfully!")
except Exception as e:
    print(f"❌ Error creating tables: {e}")
    exit(1)

# Load CSV data
print("\nLoading stock_data.csv...")
try:
    df = pd.read_csv('stock_data.csv')
    print(f"✅ Loaded {len(df)} records from CSV")
except FileNotFoundError:
    print("❌ stock_data.csv not found!")
    print("Please run generate_dataset.py first")
    exit(1)

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Upload in chunks
print("\nUploading data to database...")
chunk_size = 5000
total_chunks = (len(df) // chunk_size) + 1

for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i:i+chunk_size]
    chunk_num = (i // chunk_size) + 1
    
    try:
        chunk.to_sql('historical_prices', engine, if_exists='append', index=False)
        print(f"✅ Uploaded chunk {chunk_num}/{total_chunks} ({len(chunk)} records)")
    except Exception as e:
        print(f"❌ Error uploading chunk {chunk_num}: {e}")

# Verify upload
print("\nVerifying data...")
try:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM historical_prices"))
        count = result.fetchone()[0]
        print(f"✅ Total records in database: {count}")
        
        result = conn.execute(text("SELECT COUNT(DISTINCT symbol) FROM historical_prices"))
        symbols = result.fetchone()[0]
        print(f"✅ Total unique symbols: {symbols}")
        
        result = conn.execute(text("SELECT symbol, COUNT(*) as cnt FROM historical_prices GROUP BY symbol ORDER BY cnt DESC LIMIT 5"))
        print("\nTop 5 symbols by record count:")
        for row in result:
            print(f"  {row[0]}: {row[1]} records")
            
except Exception as e:
    print(f"❌ Error verifying data: {e}")

print("\n🎉 Database setup complete!")
print("\nNext steps:")
print("1. Start the ML service: cd ml-service && uvicorn main:app --reload --port 8000")
print("2. Start the backend: cd backend && npm run dev")
print("3. Start the frontend: cd frontend && npm start")