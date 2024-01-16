import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import mysql.connector
from sqlalchemy import create_engine

# Define the database connection parameters
db_params = {
    'user': 'DBadmin',
    'password': 'root13Musashi',
    'host': 'localhost',
    'database': 'waspum',
    'port': 3306,
}

# Define the table name
table_name = "SnP_500"

# Defining the number of previous days' data to consider for prediction (should match the value used in lstm_model.py)
look_back = 180

# Load the trained LSTM model
model = load_model("lstm_model.h5")

# Connect to the database
db_conn = mysql.connector.connect(**db_params)

# Retrieve the most recent data from the database
engine = create_engine(f"mysql+mysqlconnector://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}")
query = f"SELECT * FROM {table_name} ORDER BY Date DESC LIMIT {look_back}"
recent_data_df = pd.read_sql(query, engine)

# Close the database connection
db_conn.close()

# Convert the 'Date' column to datetime
recent_data_df['Date'] = pd.to_datetime(recent_data_df['Date'])

# Sort the data by date
recent_data_df.sort_values(by='Date', inplace=True)
recent_data_df.set_index('Date', inplace=True)

# Extract the most recent 'Close' prices
recent_close_prices = recent_data_df['Close'].values.reshape(-1, 1)

# Normalize the recent data
scaler = MinMaxScaler()
recent_close_prices = scaler.fit_transform(recent_close_prices)

# Create a sequence of data for prediction
input_data = recent_close_prices[-look_back:].reshape(1, look_back, 1)

# Make a prediction using the loaded model
predicted_price = model.predict(input_data)

# Inverse transform the predicted price to the original scale
predicted_price = scaler.inverse_transform(predicted_price)

print(f"Predicted Closing Price for the Next Day: ${predicted_price[0][0]:.2f}")
