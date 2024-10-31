import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from feature_engine.outliers import Winsorizer
from sqlalchemy import create_engine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Streamlit title
st.title("Stock Price Prediction using LSTM")

# MySQL database connection
user = "root"  # User
pw = "965877"  # Password
db = "stock_db"  # Database
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    # Read data
    data = pd.read_csv(uploaded_file)
    st.write("Data Overview", data.head())

    # Data preprocessing
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data.sort_values(by=['date'], inplace=True)
    
    # Drop irrelevant columns
    data = data.drop(columns=['Name'], errors='ignore')
    
    # Fill missing values before Winsorization
    data[['open', 'high', 'low', 'close', 'volume']] = data[['open', 'high', 'low', 'close', 'volume']].fillna(method='ffill')

    # Handle outliers using Winsorizer
    winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5,
                        variables=['open', 'high', 'low', 'close', 'volume'])
    
    # Apply Winsorizer
    data[['open', 'high', 'low', 'close', 'volume']] = winsor.fit_transform(data[['open', 'high', 'low', 'close', 'volume']])

    # Fill any remaining NaN values after Winsorization
    data.fillna(method='ffill', inplace=True)

    # Check for NaN values after processing
    if data.isnull().sum().sum() > 0:
        st.warning("There are still NaN values in the dataset after processing. Please check the data.")
    else:
        st.success("No NaN values detected in the dataset.")

        # Save data to MySQL database
        data.to_sql("stock", con=engine, if_exists="replace", index=False)

        # Data preparation
        data_close = data[['close']].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data_close)

        # Define training and test data
        train_size = int(len(scaled_data) * 0.8)  # 80% for training, 20% for testing
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]

        # Create sequences of data for LSTM input
        def create_sequences(data, sequence_length=60):
            X, y = [], []
            for i in range(len(data) - sequence_length):
                X.append(data[i:i + sequence_length, 0])
                y.append(data[i + sequence_length, 0])
            return np.array(X), np.array(y)

        # Generate sequences
        sequence_length = 60
        X_train, y_train = create_sequences(train_data, sequence_length)
        X_test, y_test = create_sequences(test_data, sequence_length)

        # Reshape input data for LSTM [samples, time steps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Build the LSTM Model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=25))
        model.add(Dense(units=1))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train, y_train, batch_size=32, epochs=10)

        # Predictions
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)  # Unscale predictions

        # Unscale actual values
        y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Calculate RMSE and MAPE
        rmse = np.sqrt(mean_squared_error(y_test_unscaled, predictions))
        mape = mean_absolute_percentage_error(y_test_unscaled, predictions)

        # Display metrics
        st.write(f"RMSE: {rmse:.2f}")
        st.write(f"MAPE: {mape:.2f}%")

        # Visualization of results
        train = data_close[:train_size]
        valid = data_close[train_size:]
        valid = np.concatenate((valid[:sequence_length], predictions))

        plt.figure(figsize=(14, 6))
        plt.plot(train, label="Training Data")
        plt.plot(range(train_size, train_size + len(predictions)), valid, label="Predictions")
        plt.title("Stock Price Prediction")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.legend()
        st.pyplot(plt)

