import yfinance as yf
import pandas as pd
import numpy as np
import time
import threading
import os
from filelock import FileLock
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense

from flask import Flask, jsonify, send_from_directory

app = Flask(__name__)

predictions = {}

# Function to fetch data from Yahoo Finance for a single stock and append it to a CSV file
def fetch_and_append_data(ticker, csv_file):
    lock_file = csv_file + ".lock"
    scaler = MinMaxScaler(feature_range=(0, 1))

    while True:
        # Fetch data
        try:
            data = yf.download(ticker, period="1d", interval="1m")
            if data is not None and not data.empty:
                # Name columns
                data.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                
                # Append data to CSV file
                with FileLock(lock_file):
                    # Check if the file exists
                    file_exists = os.path.exists(csv_file)
                    # Append data to CSV file, write header only if the file does not exist
                    data.to_csv(csv_file, mode='a', header=not file_exists)
                print(f"Data for {ticker} appended to {csv_file}")
                
                # Prepare data for prediction
                dataset = data['Close'].values.reshape(-1, 1)
                scaled_data = scaler.fit_transform(dataset)
                
                # Predict next 5 minutes
                X, y = [], []
                for i in range(60, len(scaled_data) - 5):
                    X.append(scaled_data[i-60:i, 0])
                    y.append(scaled_data[i:i+5, 0])
                X, y = np.array(X), np.array(y)
                X = np.reshape(X, (X.shape[0], X.shape[1], 1))
                
                model = Sequential()
                model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
                model.add(LSTM(units=50))
                model.add(Dense(units=5))
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(X, y, epochs=1, batch_size=1, verbose=2)
                
                predicted_stock_price = model.predict(X[-1].reshape(1, 60, 1))
                predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
                print(f"Predicted next 5 minutes for {ticker}: {predicted_stock_price.flatten()}")
                
                # Store predictions in a global dictionary
                predictions[ticker] = predicted_stock_price.flatten().tolist()
                
            else:
                print(f"Error: Failed to fetch data for {ticker}")
                time.sleep(120)  # Wait for 5 minutes before trying again
                continue
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            time.sleep(10)  # Wait for 10 seconds before trying again
            continue
        time.sleep(120)  # Wait for 5 minutes before fetching data again

@app.route('/predictions')
def get_predictions():
    return jsonify(predictions)

@app.route('/prediction')
def serve_predictions_page():
    return send_from_directory('templates', 'prediction.html')
    

def start_flask_app():
    app.run(debug=True, use_reloader=False,host='0.0.0.0' ,port=5001)

# Function to clear the contents of the CSV file
def clear_csv_file(csv_file):
    lock_file = csv_file + ".lock"
    try:
        with FileLock(lock_file):
            with open(csv_file, 'w') as file:
                file.truncate(0)  # Truncate the file to empty its contents
        print(f"Data cleared for {csv_file}")
    except Exception as e:
        print(f"Error clearing data for {csv_file}: {e}")

# Function to clear CSV files periodically
def clear_csv_files_periodically():
    while True:
        # List of tickers to fetch data for
        tickers = ["RELIANCE.NS", "TATAMOTORS.NS", "MSFT", "AAPL"]
        for ticker in tickers:
            csv_file = f"{ticker}_data.csv"
            clear_csv_file(csv_file)
        time.sleep(180)  

# Function to preprocess data for LSTM model
def preprocess_data(data):
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

# Function to create LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to train the LSTM model
def train_lstm_model(ticker, csv_file):
    while True:
        time.sleep(150)  # Sleep for 5 minutes
        try:
            with FileLock(csv_file + ".lock"):
                if os.path.exists(csv_file):
                    data = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                    data = data['Close'].values.reshape(-1, 1)
                    if len(data) > 60:
                        X_train, y_train, scaler = preprocess_data(data)
                        
                        # Create and train the model
                        model = create_lstm_model((X_train.shape[1], 1))
                        model.fit(X_train, y_train, epochs=1, batch_size=32)
                        
                        # Save the model and scaler
                        model.save(f'{ticker}_lstm_model.h5')
                        np.save(f'{ticker}_scaler.npy', scaler)
                        print(f"LSTM model trained for {ticker}")
                    else:
                        print(f"Not enough data to train the model for {ticker}")
                else:
                    print(f"No data available for {ticker} to train the model")
        except Exception as e:
            print(f"Error training model for {ticker}: {e}")

# Function to predict the stock price for the next 5 minutes
def predict_stock_price(ticker, csv_file):
    while True:
        time.sleep(150)  # Sleep for 5 minutes
        try:
            with FileLock(csv_file + ".lock"):
                if os.path.exists(csv_file):
                    data = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                    data = data['Close'].values.reshape(-1, 1)
                    if len(data) > 60:
                        # Load the model and scaler
                        model = load_model(f'{ticker}_lstm_model.h5')
                        scaler = np.load(f'{ticker}_scaler.npy', allow_pickle=True).item()
                        
                        # Prepare the input data
                        last_60_minutes = data[-60:]
                        scaled_last_60_minutes = scaler.transform(last_60_minutes)
                        X_test = []
                        X_test.append(scaled_last_60_minutes)
                        X_test = np.array(X_test)
                        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                        
                        # Make prediction
                        predicted_price = model.predict(X_test)
                        predicted_price = scaler.inverse_transform(predicted_price)
                        predictions[ticker] = predicted_price.flatten().tolist()
                        print(f"Prediction for {ticker} after 5 minutes: {predicted_price[0][0]}")
                    else:
                        print(f"Not enough data to make a prediction for {ticker}")
                else:
                    print(f"No data available for {ticker} to make a prediction")
        except Exception as e:
            print(f"Error predicting data for {ticker}: {e}")

if __name__ == "__main__":
    # Start a thread to clear CSV files periodically
    clear_thread = threading.Thread(target=clear_csv_files_periodically)
    clear_thread.daemon = True
    clear_thread.start()

    # List of tickers to fetch data for
    tickers = ["RELIANCE.NS", "TATAMOTORS.NS", "MSFT", "AAPL"]

    # Create a CSV file for each ticker
    for ticker in tickers:
        csv_file = f"{ticker}_data.csv"
        # Start fetching and appending data for each ticker in a separate thread
        fetch_thread = threading.Thread(target=fetch_and_append_data, args=(ticker, csv_file))
        fetch_thread.daemon = True
        fetch_thread.start()

        # Start training LSTM models for each ticker in a separate thread
        train_thread = threading.Thread(target=train_lstm_model, args=(ticker, csv_file))
        train_thread.daemon = True
        train_thread.start()


        # Start predicting stock prices for each ticker in a separate thread
        predict_thread = threading.Thread(target=predict_stock_price, args=(ticker, csv_file))
        predict_thread.daemon = True
        predict_thread.start()

    # Start Flask app
    flask_thread = threading.Thread(target=start_flask_app)
    flask_thread.daemon = True
    flask_thread.start()

    # Keep the main thread running
    while True:
        time.sleep(10)
