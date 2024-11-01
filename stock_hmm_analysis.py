import numpy as np
import os
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import datetime

# Download stock price data
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data = stock_data[['Adj Close']]
    stock_data['Returns'] = stock_data['Adj Close'].pct_change()
    stock_data.dropna(inplace=True)  # Drop any NaN values
    return stock_data

# Financial Indicators
def calculate_moving_average(data, window=14):
    return data['Adj Close'].rolling(window=window).mean()

def calculate_exponential_moving_average(data, window=14):
    return data['Adj Close'].ewm(span=window, adjust=False).mean()

def calculate_rsi(data, window=14):
    delta = data['Adj Close'].diff(1)
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = data['Adj Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['Adj Close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram

def calculate_volatility(data, window=30):
    return data['Returns'].rolling(window=window).std() * np.sqrt(window)

# Prepare the HMM model
def train_hmm(data, n_components=2):
    model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000)
    model.fit(data[['Returns']])
    return model

# Predict states and visualize
def analyze_and_plot(data, model, ticker, ax1):
    hidden_states = model.predict(data[['Returns']])

    # Plot each hidden state on the main y-axis
    for i in range(model.n_components):
        state = (hidden_states == i)
        ax1.plot(data.index[state], data['Adj Close'][state], '.', label=f'State {i}')

    # Plot additional financial indicators on the main y-axis
    ax1.plot(data.index, data['MA_14'], label='14-Day MA', linestyle='--')
    ax1.plot(data.index, data['EMA_14'], label='14-Day EMA', linestyle='-.')
    ax1.plot(data.index, data['MACD_Line'], label='MACD Line', color='purple')
    ax1.plot(data.index, data['Signal_Line'], label='Signal Line', color='orange')
    ax1.plot(data.index, data['Adj Close'], label=f'{ticker} Price', alpha=0.5)

    # Set labels and title for main y-axis
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    ax1.set_title(f"{ticker} Price Analysis with HMM and Financial Indicators")

    # Annotate weekly price points
    for i in range(0, len(data), 7):  # Every 7 days (weekly)
        price_value = float(data['Adj Close'].iloc[i])  # Explicitly convert to float
        ax1.annotate(f"{price_value:.2f}",
                     (data.index[i], price_value),
                     textcoords="offset points",
                     xytext=(0, 5),
                     ha='center')

    # Create a secondary y-axis for the RSI
    ax2 = ax1.twinx()
    ax2.plot(data.index, data['RSI_14'], label='14-Day RSI', color='green')
    ax2.set_ylabel("RSI")
    ax2.set_ylim(0, 100)  # Set typical RSI scale (0-100)

    # Add legends for both axes
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

# Function to interpret HMM states based on average returns and volatility
def interpret_states(data, hidden_states):
    for i in range(len(np.unique(hidden_states))):
        state_data = data[hidden_states == i]
        avg_return = state_data['Returns'].mean()
        volatility = state_data['Returns'].std()
        print(f"State {i}:")
        print(f"  Average Return: {avg_return}")
        print(f"  Volatility: {volatility}")
        print(f"  {'Bullish' if avg_return > 0 else 'Bearish'} State with {'High' if volatility > 0.01 else 'Low'} Volatility")

# Function to save stock data to CSV
def save_data_to_csv(data, ticker, start_date, end_date):
    os.makedirs(ticker, exist_ok=True)
    csv_file_path = os.path.join(ticker, f"{ticker}_{start_date}_to_{end_date}.csv")
    data.to_csv(csv_file_path)
    print(f"Data saved to {csv_file_path}")

def validate_tickers(ticker_list):
    valid_tickers = []
    for ticker in ticker_list:
        try:
            data = yf.download(ticker, period="1d")
            if not data.empty:
                valid_tickers.append(ticker)
            else:
                print(f"Ticker '{ticker}' is invalid and will be removed.")
        except Exception as e:
            print(f"Ticker '{ticker}' is invalid and will be removed. Error: {e}")
    return valid_tickers

def main():
    # Set the start and end dates
    start_date = "2018-01-01"
    end_date = datetime.datetime.today().strftime('%Y-%m-%d')
    
    pre_tickers = []
    ticker = input("Enter the stock tickers you'd like to see analyzed (-1 to quit): ").upper()
    while ticker != "-1":
        pre_tickers.append(ticker)
        ticker = input("Enter the stock tickers you'd like to see analyzed (-1 to quit): ").upper()
    # List of stocks to analyze
    tickers = validate_tickers(pre_tickers)

    # Set up the figure and subplots
    fig, axes = plt.subplots(len(tickers), 1, figsize=(15, 10 * len(tickers)))  # Increase height for multiple subplots
    fig.subplots_adjust(hspace=0.4)

    # Perform analysis for each stock
    for idx, ticker in enumerate(tickers):
        print(f"\nAnalyzing {ticker}...\n")
        
        # Load stock data
        stock_data = get_stock_data(ticker, start_date, end_date)
        
        # Calculate additional financial indicators
        stock_data['MA_14'] = calculate_moving_average(stock_data, window=14)
        stock_data['EMA_14'] = calculate_exponential_moving_average(stock_data, window=14)
        stock_data['RSI_14'] = calculate_rsi(stock_data, window=14)
        macd_line, signal_line, _ = calculate_macd(stock_data)
        stock_data['MACD_Line'] = macd_line
        stock_data['Signal_Line'] = signal_line
        stock_data['Volatility_30'] = calculate_volatility(stock_data, window=30)

        # Train the HMM model
        hmm_model = train_hmm(stock_data)

        # Analyze and plot in the specified subplot
        ax1 = axes[idx] if len(tickers) > 1 else axes  # Handle single or multiple subplots
        analyze_and_plot(stock_data, hmm_model, ticker, ax1)

        # Save stock data with indicators to CSV
        save_data_to_csv(stock_data, ticker, start_date, end_date)

    plt.show()

if __name__ == "__main__":
    main()
