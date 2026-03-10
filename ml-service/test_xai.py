import requests
import json

# The URL where your FastAPI server is running
URL = "http://127.0.0.1:8000/trade_decision"

# Mock data to simulate a trade situation
test_data = {
    "balance": 10500.0,
    "shares": 10,
    "current_price": 65000.0,
    "price_change": 0.035  # Simulating a 3.5% price increase
}

def test_trading_logic():
    print("🚀 Sending test trade request to ML service...")
    try:
        response = requests.post(URL, json=test_data)
        if response.status_code == 200:
            result = response.json()
            print("\n✅ API Response Received:")
            print(f"Action: {result.get('action')}")
            print(f"Confidence: {result.get('confidence')}")
            print(f"AI Reasoning: {result.get('reason')}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Connection Failed: {e}")

if __name__ == "__main__":
    test_trading_logic()