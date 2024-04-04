from data_manager import fetch_data
from strategy import generate_signals

def main():
    data = fetch_data(symbol='ETH/USDT', timeframe='1d', limit=100)
    signals = generate_signals(data)
    print(signals.tail(10))  # Print the last 10 signals

if __name__ == "__main__":
    main()
