import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
import mysql.connector
from sqlalchemy import create_engine
import yfinance as yf
import requests
from bs4 import BeautifulSoup

# Define the database connection parameters
db_params = {
    'user': 'DBadmin',
    'password': 'root13Musashi',
    'host': 'localhost',
    'database': 'waspum',
    'port': 3306,
}

# Automatically fetch S&P 500 tickers from Wikipedia
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'wikitable'})
        tickers = [row.find_all('td')[0].text.strip() for row in table.find_all('tr')[1:]]
        return tickers
    else:
        print("Failed to fetch S&P 500 tickers from Wikipedia.")
        return []

# Define the list of S&P 500 tickers
ticker_list = get_sp500_tickers()

# Defining the number of previous days' data to consider for prediction
look_back = 1000

# Create empty lists to store data
all_data = []

# Connect to the database
db_conn = mysql.connector.connect(**db_params)

# Retrieve data from the database for all tickers
engine = create_engine(f"mysql+mysqlconnector://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}")
for ticker in ticker_list:
    query = f"SELECT * FROM {ticker}"
    data_df = pd.read_sql(query, engine)
    
    # Convert the 'Date' column to datetime
    data_df['Date'] = pd.to_datetime(data_df['Date'])
    
    # Sort the data by date
    data_df.sort_values(by='Date', inplace=True)
    data_df.set_index('Date', inplace=True)
    
    # Extract the 'Close' prices as the target variable
    target_col = 'Close'
    target_data = data_df[target_col].values.reshape(-1, 1)
    
    # Normalize the target variable to values between 0 and 1
    scaler = MinMaxScaler()
    target_data = scaler.fit_transform(target_data)
    
    # Create sequences of data for training
    data_sequences = []
    target_sequences = []
    
    for i in range(len(target_data) - look_back):
        data_sequences.append(target_data[i:i + look_back])
        target_sequences.append(target_data[i + look_back])
    
    data_sequences = np.array(data_sequences)
    target_sequences = np.array(target_sequences)
    
    # Add the ticker identifier column
    ticker_column = np.full((len(data_sequences), 1), ticker)
    data_sequences = np.hstack((ticker_column, data_sequences))
    
    # Append the data for this ticker to the all_data list
    all_data.append(data_sequences)

# Combine data for all tickers into a single dataset
all_data = np.vstack(all_data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(all_data[:, 1:], all_data[:, 0], test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64)

# Save the model
model.save("lstm_model.h5")

print("Model saved as 'lstm_model.h5'")
