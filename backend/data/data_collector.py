import yfinance as yf
import pandas as pd
from sqlalchemy import text, create_engine, inspect
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

def download_stock_data(ticker):
    """
    Downloads historical stock data for a given ticker.
    
    :param ticker: Stock ticker symbol.
    :return: Pandas DataFrame containing the stock data.
    """
    stock_data = yf.download(ticker)
    return stock_data

def update_stock_data(engine):
    """
    Updates the database with the latest stock data for all S&P 500 stocks if necessary.

    :param engine: SQLAlchemy engine instance.
    """
    # Get the current date
    current_date = datetime.now()

    # Calculate the start date for fetching data (5 years ago from the current date)
    start_date = current_date - timedelta(days=365 * 5)

    # Get the list of S&P 500 tickers
    tickers = get_sp500_tickers()

    # Use SQLAlchemy inspect to get table names
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()

    # Create a connection outside the loop
    connection = engine.connect()

    for ticker in tickers:
        # Check if the table exists for the stock ticker
        if ticker not in existing_tables:
            print(f"Creating table for {ticker}")
            create_table_for_ticker(engine, ticker)

        # Check if data exists for the last 5 years for this stock
        query = text("SELECT COUNT(*) FROM {} WHERE Date >= :start_date".format(ticker))
        query = query.bindparams(start_date=start_date) # Bind the parameter
        result = connection.execute(query).scalar() # # Execute the query and fetch the result

        if result == 0:
            # Data is not present, so download and save it
            print(f"Fetching data for {ticker}")
            data = download_stock_data(ticker)
            save_to_database(data, ticker, engine)
            print(f"Data for {ticker} saved in the database.")
        else:
            print(f"Data for {ticker} is up to date.")

    # Close the connection after the loop
    connection.close()

def create_table_for_ticker(engine, ticker):
    """
    Creates a table for a stock ticker if it doesn't exist.

    :param engine: SQLAlchemy engine instance.
    :param ticker: Stock ticker symbol.
    """
    # Create a connection
    connection = engine.connect()

    create_table_query = text(f"""
    CREATE TABLE IF NOT EXISTS {ticker} (
        Date DATE PRIMARY KEY,
        Open FLOAT,
        High FLOAT,
        Low FLOAT,
        Close FLOAT,
        Adj_Close FLOAT,
        Volume INT
    )
    """)

    # Execute the SQL query using the connection
    connection.execute(create_table_query)

    # Close the connection
    connection.close()

# save_to_database function to append data
def save_to_database(data, ticker, engine):
    """
    Appends the stock data to a SQL database.

    :param data: Pandas DataFrame containing the stock data.
    :param ticker: Stock ticker symbol.
    :param engine: SQLAlchemy engine instance.
    """
    data.columns = [col.replace(' ', '_') for col in data.columns] #Replace spaces with underscores in column names
    data.to_sql(name=ticker, con=engine, if_exists='append', index=False)

# Database connection setup
database_uri = "mysql+pymysql://DBadmin:root13Musashi@localhost:3306/waspum"
engine = create_engine(database_uri)

# Updating stock data
update_stock_data(engine)