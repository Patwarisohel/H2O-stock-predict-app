import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import yfinance as yf

# Initialize H2O cluster
h2o.init()

# Fetch stock data from Yahoo Finance
stock_data = yf.download("AAPL", start="2022-01-01", end="2023-01-01")
stock_data['Date'] = stock_data.index
stock_data = stock_data.reset_index(drop=True)

# Prepare data for H2O
h2o_data = h2o.H2OFrame(stock_data)
h2o_data['Date'] = h2o_data['Date'].as_date('%Y-%m-%d')

# Define features and target
features = ['Open', 'High', 'Low', 'Close', 'Volume']
target = 'Close'

# Split data into train and test
train, test = h2o_data.split_frame(ratios=[0.8], seed=1234)

# Initialize and run H2O AutoML
aml = H2OAutoML(max_models=20, seed=1)
aml.train(x=features, y=target, training_frame=train)

# Save the best model
model_path = h2o.save_model(model=aml.leader, path="./best_model", force=True)
print(f"Model saved to: {model_path}")

import streamlit as st
import h2o
import yfinance as yf
import pandas as pd
from h2o.automl import H2OAutoML

# Initialize H2O cluster
h2o.init()

# Load the saved model
model_path = "./best_model/StackedEnsemble_AllModels_AutoML_1_2023"
model = h2o.load_model(model_path)

# Streamlit UI
st.title("Stock Prediction App")
st.write("Predict future stock prices using a pre-trained model.")

# User input for stock ticker
stock_ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")

# Fetch stock data from Yahoo Finance
def fetch_stock_data(ticker):
    data = yf.download(ticker, start="2023-01-01", end="2024-01-01")
    data['Date'] = data.index
    return data.reset_index(drop=True)

# Fetch new data and make predictions
if st.button("Predict"):
    stock_data = fetch_stock_data(stock_ticker)
    h2o_data = h2o.H2OFrame(stock_data)
    
    # Define features and target
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Make predictions
    predictions = model.predict(h2o_data[features])
    stock_data['Predicted_Close'] = h2o.as_list(predictions)['predict']
    
    # Display results
    st.write(stock_data[['Date', 'Close', 'Predicted_Close']])
    st.line_chart(stock_data.set_index('Date')[['Close', 'Predicted_Close']])

# Shutdown H2O cluster on exit
def shutdown_h2o():
    h2o.cluster().shutdown()
atexit.register(shutdown_h2o)
