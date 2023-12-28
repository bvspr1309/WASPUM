import yfinance as yf
import pandas as pd

def get_stock_data(ticker, start_date='2000-01-01', end_date = None):
    """
    Fetches historical stock data from Yahoo Finance.

    :param ticker: Ticker symbol of the stock (e.g., 'AAPL' for Apple).
    :param start_date: Start date for historical data (default is '2000-01-01').
    :param end_date: End date for historical data (default is None, which fetches till the current date).
    :return: Pandas DataFrame containing the historical stock data.
    """
    stock_data = yf.download(ticker, start=start_date, end =end_date)
    return stock_data

def prepare_data_for_prediction(stock_data):
    """
    Prepares the stock data for prediction. This function should be modified 
    according to the requirements of our LSTM model.

    :param stock_data: DataFrame with the stock's historical data.
    :return: Processed data ready for prediction.
    """
    # Placeholder for data preprocessing steps required by your model
    # Example: selecting 'Close' prices and normalizing the data
    # processed_data = preprocessing_function(stock_data)
    
    # For now, just returning the 'Close' prices
    return stock_data['Close']

# Example usage
ticker_symbol = "AAPL"  # will be dynamically obtained from user input
historical_data = get_stock_data(ticker_symbol)

# Preparing data for prediction
data_for_prediction = prepare_data_for_prediction(historical_data)

# Here we will pass 'data_for_prediction' to our LSTM model to get the predicted prices

# The implementation of the actual prediction logic will depend on our LSTM model.