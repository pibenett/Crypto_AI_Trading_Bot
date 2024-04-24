from pprint import pprint
from data_manager import fetch_data
from strategy import generate_signals
from data_manager import create_dataset
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from utils import print_colored


def main():
    # To load the CSV file back into a panda dataframe
    df = pd.read_csv("data/my_dataframe.csv")
    # df = fetch_data(symbol='ETH/USDT', timeframe='1d', limit=300)

    df = generate_signals(df)
    print_colored('DATAFRAME', '42')
    # pprint(df.tail(10))  # Print the last 10 signals
    pprint(df.head(10))  # Print the first 10 signals

    # Scaling data by setting up a scaler between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    # scaled close price and volume
    scaled_data = scaler.fit_transform(
        df[["volume", "open","high", "low", "close" ]]
    )  
    print_colored('SCALED DATA', '42')
    pprint(scaled_data);

    # Analyze close prices
    target_prices_serie = df[["close"]]

    print_colored('target_prices_serie', '42')
    pprint(target_prices_serie)
    # pprint(type(target_prices))

    # Split data into samples
    time_steps = 10  # Example lookback period
    sequence, target_prices = create_dataset(pd.DataFrame(scaled_data), target_prices_serie, time_steps)
    print_colored('SEQUENCE', '42')
    pprint(sequence)

    print_colored('TARGET PRICES', '42')
    pprint(target_prices)

    # Split data into training and test sets (80% for training and 20% for testing)
    split = int(len(sequence) * 0.8)
    sequence_train, sequence_test = sequence[:split], sequence[split:]
    target_prices_train, target_prices_test = target_prices[:split], target_prices[split:]

    print_colored('sequence_train', '42')
    pprint(sequence_train)

    print_colored('sequence_test', '42')
    pprint(sequence_test)

    print_colored('target_prices_train', '42')
    pprint(target_prices_train)

    print_colored('target_prices_test', '42')
    pprint(target_prices_test)


    # Define LSTM model
    model = Sequential(
        [
            LSTM(
                50,
                return_sequences=True,
                input_shape=(sequence_train.shape[1], sequence_train.shape[2]),
            ),
            LSTM(50),
            Dense(1),
        ]
    )

    model.compile(optimizer="adam", loss="mean_squared_error")

    # Train the model
    model.fit(
        sequence_train, target_prices_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1
    )

    # You can add model evaluation and prediction steps here

    # Predicting the Test set results
    target_prices_prediction = model.predict(sequence_test)

    print_colored('target_prices_prediction', '42')
    pprint(target_prices_prediction)

    # Evaluating the model
    from sklearn.metrics import mean_squared_error

    mse = mean_squared_error(target_prices_test, target_prices_prediction)

    print_colored('MEAN SQUARED ERROR', '42')
    pprint(f"MSE: {mse}")


if __name__ == "__main__":
    main()
