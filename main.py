import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Alpha Vantage API configuration
API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")


# Page configuration
st.set_page_config(page_title="Stock Price Prediction", layout="wide")
st.title("Stock Price Prediction App")
st.markdown("Select a stock ticker to view historical data and price predictions")


# Function to fetch data from Alpha Vantage
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_stock_data(ticker):
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "outputsize": "full",
        "apikey": API_KEY,
        "datatype": "json"
    }
    
    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        
        if "Error Message" in data:
            st.error(f"Error fetching data: {data['Error Message']}")
            return None
            
        # Parse time series data
        time_series = data.get("Time Series (Daily)", {})
        
        if not time_series:
            st.warning("No data available for this ticker")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Convert columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
            
        # Rename columns
        df.columns = ["open", "high", "low", "close", "volume"]
        
        return df
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# Function to train ARIMA model and make predictions
def predict_prices(data, days_to_predict=30):
    # Use adjusted close price for prediction
    train_data = data['close'].values
    
    # Fit ARIMA model (p,d,q) - these parameters can be optimized using auto_arima
    model = ARIMA(train_data, order=(5, 1, 0))
    model_fit = model.fit()
    
    # Forecast future prices
    forecast = model_fit.forecast(steps=days_to_predict)
    
    # Create forecast DataFrame
    last_date = data.index[-1]
    forecast_dates = [last_date + timedelta(days=i+1) for i in range(days_to_predict)]
    forecast_df = pd.DataFrame({
        'date': forecast_dates,
        'predicted_close': forecast
    })
    forecast_df.set_index('date', inplace=True)
    
    # Calculate metrics on the last 30 days (if available)
    if len(train_data) > 30:
        train_set = train_data[:-30]
        test_set = train_data[-30:]
        
        # Fit model on training set
        validation_model = ARIMA(train_set, order=(5, 1, 0))
        validation_model_fit = validation_model.fit()
        
        # Predict on test set
        validation_predictions = validation_model_fit.forecast(steps=30)
        
        # Calculate error metrics
        mae = mean_absolute_error(test_set, validation_predictions)
        mse = mean_squared_error(test_set, validation_predictions)
        rmse = np.sqrt(mse)
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse
        }
    else:
        metrics = None
    
    return forecast_df, model_fit, metrics

# Sidebar for user inputs
st.sidebar.header("Settings")

# Stock ticker selection (with popular defaults)
default_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
user_ticker = st.sidebar.text_input("Enter Stock Ticker Symbol", "AAPL")
ticker_selector = st.sidebar.selectbox(
    "Or select from popular stocks:", 
    options=["Custom"] + default_tickers,
    index=1
)

if ticker_selector != "Custom":
    selected_ticker = ticker_selector
else:
    selected_ticker = user_ticker

# Prediction period
prediction_days = st.sidebar.slider("Days to Predict", min_value=7, max_value=90, value=30)

# Display period
display_period = st.sidebar.selectbox(
    "Historical Data to Display",
    options=["1 Month", "3 Months", "6 Months", "1 Year", "5 Years", "All"],
    index=3
)

# Button to trigger analysis
analyze_button = st.sidebar.button("Analyze Stock")

# Main content
if analyze_button:
    with st.spinner(f"Fetching and analyzing data for {selected_ticker}..."):
        # Fetch data
        df = fetch_stock_data(selected_ticker)
        
        if df is not None and not df.empty:
            # Filter data based on selected period
            if display_period == "1 Month":
                df_display = df.last('30D')
            elif display_period == "3 Months":
                df_display = df.last('90D')
            elif display_period == "6 Months":
                df_display = df.last('180D')
            elif display_period == "1 Year":
                df_display = df.last('365D')
            elif display_period == "5 Years":
                df_display = df.last('1825D')
            else:
                df_display = df
            
            # Get predictions
            forecast_df, model, metrics = predict_prices(df, days_to_predict=prediction_days)
            
            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["Price Prediction", "Historical Data", "Model Details"])
            
            with tab1:
                st.subheader(f"{selected_ticker} Price Prediction")
                
                # Create a plotly figure for actual and predicted prices
                fig = make_subplots(rows=1, cols=1)
                
                # Add historical close prices
                fig.add_trace(
                    go.Scatter(
                        x=df_display.index,
                        y=df_display['close'],
                        mode='lines',
                        name='Historical Close Price',
                        line=dict(color='blue')
                    )
                )
                
                # Add predicted prices
                fig.add_trace(
                    go.Scatter(
                        x=forecast_df.index,
                        y=forecast_df['predicted_close'],
                        mode='lines',
                        name='Predicted Close Price',
                        line=dict(color='red', dash='dash')
                    )
                )
                
                # Layout updates
                fig.update_layout(
                    title=f"{selected_ticker} Stock Price Prediction",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display metrics if available
                if metrics:
                    st.subheader("Model Performance Metrics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Mean Absolute Error (MAE)", f"${metrics['MAE']:.2f}")
                    col2.metric("Mean Squared Error (MSE)", f"${metrics['MSE']:.2f}")
                    col3.metric("Root Mean Squared Error (RMSE)", f"${metrics['RMSE']:.2f}")
                
            with tab2:
                st.subheader(f"{selected_ticker} Historical Data")
                
                # Display historical stock price chart
                fig2 = go.Figure()
                
                # Candlestick chart
                fig2.add_trace(
                    go.Candlestick(
                        x=df_display.index,
                        open=df_display['open'],
                        high=df_display['high'],
                        low=df_display['low'],
                        close=df_display['close'],
                        name="OHLC"
                    )
                )
                
                # Add volume as bar chart on secondary y-axis
                fig2.add_trace(
                    go.Bar(
                        x=df_display.index,
                        y=df_display['volume'],
                        name="Volume",
                        marker_color='rgba(0,0,255,0.3)',
                        yaxis="y2"
                    )
                )
                
                # Layout updates
                fig2.update_layout(
                    title=f"{selected_ticker} Historical Price and Volume",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    yaxis2=dict(
                        title="Volume",
                        overlaying="y",
                        side="right",
                        showgrid=False
                    ),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=600
                )
                
                st.plotly_chart(fig2, use_container_width=True)
                
                # Display summary statistics
                st.subheader("Summary Statistics")
                stats_df = df_display.describe()
                st.dataframe(stats_df)
                
                # Display recent data
                st.subheader("Recent Data")
                st.dataframe(df_display.tail(10))
                
            with tab3:
                st.subheader("Model Details")
                
                # Display model summary
                st.write("ARIMA Model Summary")
                model_summary = model.summary().tables[1].as_html()
                st.write(pd.read_html(model_summary, header=0, index_col=0)[0])
                
                # Display forecast details
                st.subheader("Price Forecast Details")
                st.dataframe(forecast_df)
                
                # Download forecast as CSV
                csv = forecast_df.to_csv().encode('utf-8')
                st.download_button(
                    label="Download Forecast as CSV",
                    data=csv,
                    file_name=f"{selected_ticker}_forecast.csv",
                    mime="text/csv"
                )
        else:
            st.error(f"No data available for {selected_ticker}. Please check the ticker symbol and try again.")

# Display instructions if no analysis has been run yet
if not analyze_button:
    st.info("👈 Enter a stock ticker and click 'Analyze Stock' to start")
    
    # Display sample images or placeholders
    st.subheader("Sample Prediction Visualization")
    
    # Create a placeholder image using plotly
    dates = pd.date_range(start='2023-01-01', periods=180)
    sample_price = 100 + np.cumsum(np.random.normal(0, 1, 180)) / 3
    pred_dates = pd.date_range(start=dates[-1] + timedelta(days=1), periods=30)
    pred_price = sample_price[-1] + np.cumsum(np.random.normal(0, 1, 30)) / 2
    
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=sample_price,
            mode='lines',
            name='Sample Historical Data',
            line=dict(color='blue')
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pred_dates,
            y=pred_price,
            mode='lines',
            name='Sample Prediction',
            line=dict(color='red', dash='dash')
        )
    )
    fig.update_layout(
        title="Sample Stock Price Prediction",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Note: This is a simplified prediction model for educational purposes. Real trading decisions should be based on more comprehensive analysis.")
st.caption("Data provided by Alpha Vantage API. You need to add your own API key to use this application.")