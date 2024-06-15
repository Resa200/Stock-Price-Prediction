import streamlit as st
import pickle
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import datetime

# Function to prepare XGBoost data
def prepare_xgb_data(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# Function to load XGBoost models
def load_xgb_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Function to download data
def download_data(tickers, start, end):
    data = {ticker: yf.download(ticker, start=start, end=end) for ticker in tickers}
    combined_data = pd.concat(data, axis=1)
    combined_data.columns = [f'{ticker}_{col}' for ticker in tickers for col in data[ticker].columns]
    return combined_data

# Main Streamlit app function
def main():
    st.title('Stock Price Prediction and Visualization with XGBoost Models')

    st.sidebar.title('Input Parameters')
    ticker = st.sidebar.selectbox('Select Ticker', ['AAPL', 'MSFT', 'GOOGL', 'AMZN'])

    # Download historical data for selected ticker
    start_date = st.sidebar.date_input("Start Date", datetime.datetime(2010, 1, 1))
    end_date_test = st.sidebar.date_input("End Date", datetime.datetime(2024, 6, 14))  
    test_data = download_data([ticker], start_date, end_date_test)

    # Load XGBoost model for the selected ticker
    model_path = f'src/models/{ticker}_xgb_model.pkl'  # Adjust path as per your saved models
    model = load_xgb_model(model_path)

    # Prepare data for XGBoost prediction
    seq_length = 60
    X_test, _ = prepare_xgb_data(test_data[f'{ticker}_Close'].values, seq_length)

    # Display last ten days' actual and predicted prices
    st.subheader(f'Last Ten Days Prediction for {ticker}')
    predicted_prices = model.predict(X_test)
    last_ten_days_data = test_data.iloc[-10:]
    plt.figure(figsize=(12, 6))
    plt.plot(last_ten_days_data.index, last_ten_days_data[f'{ticker}_Close'], label='Actual')
    plt.plot(last_ten_days_data.index, predicted_prices[-10:], label='Predicted')
    plt.title(f'Last Ten Days Prediction for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(plt)

    # Optionally, allow users to input custom parameters for prediction
    # Example: Input date range, model parameters, etc.
    # Add more inputs and processing as needed

if __name__ == '__main__':
    main()
