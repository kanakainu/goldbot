import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import requests
import time
import os
from typing import Tuple, Optional, Dict, Any, Union

# Define gold ticker symbol for YFinance API
GOLD_TICKER = "GC=F"  # Gold futures ticker
GOLD_MT5_SYMBOL = "XAUUSD"  # Gold symbol in MT5

# Import MT5 connector if available
try:
    from mt5_connector import MT5Connector, create_mt5_connection
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

# Global MT5 connector instance
mt5_connector = None

def initialize_mt5_connector():
    """
    Initialize MT5 connector if credentials are available
    
    Returns:
        bool: True if initialized successfully, False otherwise
    """
    global mt5_connector
    
    # Check if MT5 is available and credentials exist
    if not MT5_AVAILABLE:
        print("MT5 connector module not available")
        return False
    
    # Get MT5 credentials from environment variables
    mt5_api_url = os.environ.get("MT5_API_URL")
    mt5_api_key = os.environ.get("MT5_API_KEY")
    mt5_account = os.environ.get("MT5_ACCOUNT")
    
    if not mt5_api_url or not mt5_api_key:
        print("MT5 credentials not found in environment variables")
        return False
    
    try:
        # Create MT5 connector
        mt5_connector = create_mt5_connection(mt5_api_url, mt5_api_key, mt5_account)
        
        if mt5_connector.is_connected:
            print("Connected to MetaTrader 5 successfully")
            return True
        else:
            print(f"Failed to connect to MetaTrader 5: {mt5_connector.get_last_error()}")
            return False
            
    except Exception as e:
        print(f"Error initializing MT5 connector: {str(e)}")
        return False

def fetch_realtime_gold_price():
    """
    Fetch real-time gold price from MT5 (if available) or YFinance API
    
    Returns:
        tuple: (current_price, timestamp) - Current gold price and timestamp
    """
    # Try to get price from MT5 if available
    if MT5_AVAILABLE and mt5_connector and mt5_connector.is_connected:
        try:
            price, timestamp = mt5_connector.get_current_price(GOLD_MT5_SYMBOL)
            if price is not None:
                print(f"Gold price fetched from MT5: ${price:.2f}")
                return price, timestamp
            # Fall through to YFinance if MT5 returns None
        except Exception as e:
            print(f"Error fetching gold price from MT5: {str(e)}")
            # Continue to YFinance if MT5 fails
    
    # Fall back to YFinance API
    try:
        # Fetch the latest data for gold
        gold_data = yf.Ticker(GOLD_TICKER)
        current_data = gold_data.history(period="1d")
        
        if current_data.empty:
            raise ValueError("No current gold price data available")
        
        # Get the latest price and timestamp
        current_price = current_data['Close'].iloc[-1]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"Gold price fetched from YFinance: ${current_price:.2f}")
        return current_price, timestamp
    except Exception as e:
        print(f"Error fetching real-time gold price from YFinance: {str(e)}")
        # If YFinance fails, attempt to use a fallback source
        try:
            return fetch_fallback_gold_price()
        except:
            # If all fail, return a None value with current timestamp
            return None, datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def fetch_fallback_gold_price():
    """
    Fallback method to fetch gold price if YFinance fails
    
    Returns:
        tuple: (current_price, timestamp) - Current gold price and timestamp
    """
    # Use a different public API that provides gold price data
    response = requests.get('https://metals-api.com/api/latest?access_key=dummy_key&base=USD&symbols=XAU')
    
    if response.status_code == 200:
        data = response.json()
        # Extract gold price from response (XAU is the code for gold)
        price = 1 / data['rates']['XAU']  # Convert to USD/oz
        timestamp = datetime.fromtimestamp(data['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
        return price, timestamp
    else:
        raise Exception(f"Fallback API failed with status code: {response.status_code}")

def fetch_historical_gold_data(start_date, end_date):
    """
    Fetch historical gold price data from MT5 (if available) or YFinance API
    
    Args:
        start_date (datetime.date): Start date for historical data
        end_date (datetime.date): End date for historical data
        
    Returns:
        pandas.DataFrame: Historical gold price data
    """
    # Try to get historical data from MT5 if available
    if MT5_AVAILABLE and mt5_connector and mt5_connector.is_connected:
        try:
            # Convert dates to datetime objects
            start_datetime = datetime.combine(start_date, datetime.min.time())
            end_datetime = datetime.combine(end_date, datetime.max.time())
            
            # Fetch historical data from MT5
            mt5_data = mt5_connector.get_historical_data(
                symbol=GOLD_MT5_SYMBOL,
                timeframe="D1",  # Daily timeframe
                start_time=start_datetime,
                end_time=end_datetime
            )
            
            if not mt5_data.empty:
                print(f"Historical gold data fetched from MT5 ({len(mt5_data)} records)")
                
                # Rename columns to match YFinance format if needed
                if 'open' in mt5_data.columns and 'Open' not in mt5_data.columns:
                    mt5_data = mt5_data.rename(columns={
                        'open': 'Open',
                        'high': 'High',
                        'low': 'Low',
                        'close': 'Close',
                        'volume': 'Volume'
                    })
                
                # Process the data (similar to YFinance processing)
                gold_data = mt5_data
                gold_data = gold_data.rename_axis('Date')
                
                # Calculate daily returns
                gold_data['Daily_Return'] = gold_data['Close'].pct_change()
                
                # Calculate features that might be useful for analysis
                gold_data['MA_5'] = gold_data['Close'].rolling(window=5).mean()
                gold_data['MA_20'] = gold_data['Close'].rolling(window=20).mean()
                gold_data['MA_50'] = gold_data['Close'].rolling(window=50).mean()
                
                # Calculate volatility (standard deviation of returns over rolling 20-day window)
                gold_data['Volatility'] = gold_data['Daily_Return'].rolling(window=20).std()
                
                # Calculate RSI (Relative Strength Index)
                delta = gold_data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).fillna(0)
                loss = (-delta.where(delta < 0, 0)).fillna(0)
                
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                
                rs = avg_gain / avg_loss
                gold_data['RSI'] = 100 - (100 / (1 + rs))
                
                return gold_data
            # Fall through to YFinance if MT5 returns empty data
        except Exception as e:
            print(f"Error fetching historical gold data from MT5: {str(e)}")
            # Continue to YFinance if MT5 fails
    
    # Fall back to YFinance API
    try:
        # Convert dates to string format expected by yfinance
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = (end_date + timedelta(days=1)).strftime('%Y-%m-%d')  # Add one day to include end_date
        
        # Fetch historical data
        gold_data = yf.download(GOLD_TICKER, start=start_str, end=end_str)
        
        if gold_data.empty:
            print("No historical gold data available from YFinance for the specified date range")
            return None
        
        print(f"Historical gold data fetched from YFinance ({len(gold_data)} records)")
        
        # Clean and process data
        gold_data = gold_data.rename_axis('Date')
        
        # Calculate daily returns
        gold_data['Daily_Return'] = gold_data['Close'].pct_change()
        
        # Calculate features that might be useful for analysis
        gold_data['MA_5'] = gold_data['Close'].rolling(window=5).mean()
        gold_data['MA_20'] = gold_data['Close'].rolling(window=20).mean()
        gold_data['MA_50'] = gold_data['Close'].rolling(window=50).mean()
        
        # Calculate volatility (standard deviation of returns over rolling 20-day window)
        gold_data['Volatility'] = gold_data['Daily_Return'].rolling(window=20).std()
        
        # Calculate RSI (Relative Strength Index)
        delta = gold_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        gold_data['RSI'] = 100 - (100 / (1 + rs))
        
        return gold_data
    
    except Exception as e:
        print(f"Error fetching historical gold data from YFinance: {str(e)}")
        return None

def update_gold_data(existing_data=None, days=30):
    """
    Update existing gold price data or fetch new data for a specified period
    
    Args:
        existing_data (pandas.DataFrame, optional): Existing gold price data
        days (int, optional): Number of days to fetch if existing_data is None
        
    Returns:
        pandas.DataFrame: Updated gold price data
    """
    try:
        if existing_data is None:
            # If no existing data, fetch data for the specified number of days
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)
            return fetch_historical_gold_data(start_date, end_date)
        else:
            # Get the latest date in the existing data
            latest_date = existing_data.index.max().date()
            
            # If the latest date is today, no need to update
            if latest_date >= datetime.now().date():
                return existing_data
            
            # Otherwise, fetch data from the latest date to today
            start_date = latest_date + timedelta(days=1)
            end_date = datetime.now().date()
            
            if start_date <= end_date:
                # Fetch new data
                new_data = fetch_historical_gold_data(start_date, end_date)
                
                if new_data is not None and not new_data.empty:
                    # Combine existing and new data
                    updated_data = pd.concat([existing_data, new_data])
                    # Remove duplicates if any
                    updated_data = updated_data[~updated_data.index.duplicated(keep='last')]
                    return updated_data
            
            return existing_data
    
    except Exception as e:
        print(f"Error updating gold data: {str(e)}")
        return existing_data
