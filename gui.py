import yfinance as yf
import mplfinance as mpf
import mpld3
import matplotlib.pyplot as plt
from threading import Thread
import tkinter as tk
import webbrowser
from flask import Flask, send_file, Response, send_from_directory
import requests
import os
import time
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mpld3")

# Flask application
app = Flask(__name__)

def fetch_prediction_page():
    app_py_url = 'http://app:5001/prediction'  # Change to app.py's URL
    response = requests.get(app_py_url)
    return response.content  # Return the content of the prediction.html file

@app.route('/')
def home():
    return send_file('templates/index.html')

@app.route('/prediction')
def serve_predictions_page():
    prediction_html_content = fetch_prediction_page()
    return prediction_html_content

@app.route('/plot')
def home2():
    return send_file('static/plot.html')

@app.route('/plot/<ticker>')
def plot(ticker):
    file_path = f'static/{ticker}_plot.html'
    if os.path.exists(file_path):
        return send_file(file_path)
    else:
        return f"Plot for {ticker} not found, please wait for it to be generated.", 404

@app.route('/stream')
def stream():
    def event_stream():
        while True:
            time.sleep(1)  # Update interval
            # Stream all available plots
            for ticker in ["RELIANCE.NS", "TATAMOTORS.NS", "MSFT", "AAPL"]:
                file_path = f'static/{ticker}_plot.html'
                if os.path.exists(file_path):
                    with open(file_path, 'r') as file:
                        yield f"data: {file.read()}\n\n"
    return Response(event_stream(), content_type='text/event-stream')

def start_flask_app():
    app.run(debug=True, use_reloader=False,host='0.0.0.0' ,port=5002)

# Function to fetch data from Yahoo Finance for a single stock
def fetch_data(ticker):
    return yf.download(ticker, period="1d", interval="1m")

# Function to plot data for a single stock with indicators
def plot_data(ticker, data):
    # Calculate Relative Strength Index (RSI)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    data['RSI'] = 100 - (100 / (1 + RS))

    # Create a candlestick plot
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})

    # Plot the candlestick chart on the first axis
    mpf.plot(data, type='candle', ax=ax1, volume=False, show_nontrading=True)
    ax1.set_title(f'{ticker} Stock Price')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')

    # Plot the RSI on the second axis
    ax2.plot(data.index, data['RSI'], label='RSI', color='red')
    ax2.axhline(70, color='gray', linestyle='--')
    ax2.axhline(30, color='gray', linestyle='--')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('RSI')
    ax2.legend()

    plt.tight_layout()

    # Convert plot to HTML using mpld3
    html_fig = mpld3.fig_to_html(fig)
    plt.close(fig)  # Close the figure to release resources
    return html_fig

# Function to update plot and convert it to HTML
def update_and_convert_plot(root, ticker):
    # Fetch data
    stock_data = fetch_data(ticker)
    # Check if data is None
    if stock_data is None:
        print(f"Error: Failed to fetch data for {ticker}")
        return
    html_plot = plot_data(ticker, stock_data)
    # Write HTML to a file
    file_path = f'static/{ticker}_plot.html'
    with open(file_path, 'w') as file:
        file.write(html_plot)
    print(f"Plot for {ticker} updated.")
    # Schedule the next update
    root.after(1000, update_and_convert_plot, root, ticker)

# Tkinter application
def start_tkinter_app():
    # Start the Tkinter application
    root = tk.Tk()
    root.title("Tkinter and Flask")

    # Function to start updating plots
    def start_plot_updates():
        for ticker in ["RELIANCE.NS", "TATAMOTORS.NS", "MSFT", "AAPL"]:
            # Start the update loop for each ticker
            update_and_convert_plot(root, ticker)

    # Start fetching data and updating plots immediately
    start_plot_updates()

    # Open the Flask plot page in the default web browser
    webbrowser.open("http://127.0.0.1:5002")

    root.mainloop()

# Run Flask and Tkinter applications in separate threads
if __name__ == '__main__':
    # Start the Flask server in a separate thread
    flask_thread = Thread(target=start_flask_app)
    flask_thread.daemon = True
    flask_thread.start()

    # Start the Tkinter application
    start_tkinter_app()
