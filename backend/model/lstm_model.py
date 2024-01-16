import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import mysql.connector
from sqlalchemy import create_engine
import yfinance as yf

# Define the database connection parameters
db_params = {
    'user': 'DBadmin',
    'password': 'root13Musashi',
    'host': 'localhost',
    'database': 'waspum',
    'port': 3306,
}

# Define the table name
table_name = "snp_500"

# Defining the number of previous days' data to consider for prediction
look_back = 180

# Connect to the database
db_conn = mysql.connector.connect(**db_params)

# Retrieve data from the database
engine = create_engine(f"mysql+mysqlconnector://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}")
query = f"SELECT * FROM {table_name}"
data_df = pd.read_sql(query, engine)

# Close the database connection
db_conn.close()

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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_sequences, target_sequences, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# Save the model
model.save("lstm_model.h5")

print("Model saved as 'lstm_model.h5'")
