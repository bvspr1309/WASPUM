import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Load historical stock data from the database
def load_data(engine, ticker):
    query = f"SELECT Date, Close FROM {ticker} ORDER BY Date ASC"
    data = pd.read_sql(query, engine)
    return data

# Prepare the data for LSTM
def prepare_data(data):
    scaler = MinMaxScaler()
    data['Close'] = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    sequence_length = 30  # Adjust this as needed
    X, y = [], []

    for i in range(len(data) - sequence_length):
        X.append(data['Close'].iloc[i:i+sequence_length].values)
        y.append(data['Close'].iloc[i+sequence_length])

    X, y = np.array(X), np.array(y)

    # Reshape data for LSTM (samples, sequence_length, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    return X, y

# Build the LSTM model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

# Train the LSTM model
def train_model(model, X_train, y_train, epochs):
    model.fit(X_train, y_train, batch_size=32, epochs=epochs)

# Save the trained model
def save_model(model, filename):
    model.save(filename)

if __name__ == '__main__':
    # Define the stock ticker symbol
    ticker = 'AAPL'  # Replace with the desired stock symbol

    # Load historical stock data
    engine = create_engine("mysql+pymysql://DBadmin:root@13Musashi@localhost:3306/waspum")
    data = load_data(engine, ticker)

    # Prepare the data
    X, y = prepare_data(data)

    # Build the model
    input_shape = (X.shape[1], 1)
    model = build_model(input_shape)

    # Train the model
    train_model(model, X, y, epochs=50)

    # Save the trained model
    save_model(model, f'{ticker}_lstm_model.h5')