import yfinance as yf
import pandas as pd

# Define the list of tickers and date ranges
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
start_date = '2010-01-01'
end_date_train = '2022-12-31'
end_date_valid = '2023-06-30'
end_date_test = '2023-12-31'

# Function to download data for a given date range
def download_data(tickers, start, end):
    data = {ticker: yf.download(ticker, start=start, end=end) for ticker in tickers}
    combined_data = pd.concat(data, axis=1)
    combined_data.columns = [f'{ticker}_{col}' for ticker in tickers for col in data[ticker].columns]
    return combined_data

# Download historical data for training (2010-2022)
train_data = download_data(tickers, start_date, end_date_train)

# Download data for validation (first half of 2023)
valid_data = download_data(tickers, '2023-01-01', end_date_valid)

# Download data for testing (second half of 2023)
test_data = download_data(tickers, '2023-07-01', end_date_test)

# Export data for train, validation and test
train_data.to_csv('Stock-price-data-train-2010-2022.csv')
valid_data.to_csv('Stock-price-data-validation-2023.csv')
test_data.to_csv('Stock-price-data-test-2023.csv')