import yfinance as yf
import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

def get_sp500_tickers():
    """
    Fetches S&P 500 stock tickers from Wikipedia.
    
    :return: List of S&P 500 stock tickers.
    """
    # URL of the Wikipedia page containing the S&P 500 component list
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    # Send a GET request to the Wikipedia page
    response = requests.get(url)

    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the table containing the S&P 500 component list by its class
        table = soup.find('table', {'class': 'wikitable'})

        # Initialize an empty list to store tickers
        tickers = []

        # Iterate through the rows of the table
        for row in table.find_all('tr')[1:]:
            # Extract the ticker symbol from the first column of each row
            ticker = row.find_all('td')[0].text.strip()
            tickers.append(ticker)

        return tickers
    else:
        print("Failed to fetch S&P 500 tickers from Wikipedia.")
        return []

def update_stock_data(tickers, data_path):
    """
    Downloads historical stock data for a list of tickers and saves it as CSV files.

    :param tickers: List of stock ticker symbols.
    :param data_path: Path to save the CSV files.
    """
    # Create the data directory if it doesn't exist
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    for ticker in tickers:
        print(f"Fetching data for {ticker}")
        data = download_stock_data(ticker)
        if not data.empty:
            csv_filename = os.path.join(data_path, f"{ticker}.csv")
            data.to_csv(csv_filename, index=False)
            print(f"Data for {ticker} saved as {csv_filename}")
        else:
            print(f"No data available for {ticker}")

def download_stock_data(ticker):
    """
    Downloads historical stock data for a given ticker.

    :param ticker: Stock ticker symbol.
    :return: Pandas DataFrame containing the stock data.
    """
    stock_data = yf.download(ticker)
    return stock_data

# Define the list of S&P 500 tickers
tickers = get_sp500_tickers()

# Specify the data path
data_path = "D:/Files/Projects/Capstone/Code/Data"

# Update stock data and save as CSV files
update_stock_data(tickers, data_path)