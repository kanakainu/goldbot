import pandas as pd
import numpy as np
import time
import threading
import datetime
from datetime import timedelta
import json
import os

from data_fetcher import fetch_realtime_gold_price, fetch_historical_gold_data, update_gold_data, initialize_mt5_connector

# Check if MT5 module is available
try:
    from mt5_connector import MT5Connector, create_mt5_connection, MT5_AVAILABLE
except ImportError:
    MT5_AVAILABLE = False

# Global MT5 connector instance
mt5_connector = None
from trading_strategy import SignalGenerator
from model import GoldPricePredictor
from utils import process_data_for_model, generate_trading_signal
import database as db

class AutoTrader:
    """
    Automated trading system for executing gold trades based on AI signals
    """
    
    def __init__(self, initial_capital=10000, strategy_type='trend_following', signal_threshold=0.01, 
                 stop_loss=0.02, take_profit=0.03, feature_window=14, prediction_days=7, use_mt5=False):
        """
        Initialize the auto trader
        
        Args:
            initial_capital (float): Initial capital for trading
            strategy_type (str): Type of trading strategy ('trend_following', 'mean_reversion', 'ml_based')
            signal_threshold (float): Signal threshold for trades
            stop_loss (float): Stop loss percentage
            take_profit (float): Take profit percentage
            feature_window (int): Window size for ML features
            prediction_days (int): Days ahead to predict for ML model
            use_mt5 (bool): Whether to use MetaTrader 5 for real trading (if available)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.strategy_type = strategy_type
        self.signal_threshold = signal_threshold
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.feature_window = feature_window
        self.prediction_days = prediction_days
        self.use_mt5 = use_mt5 and MT5_AVAILABLE
        
        # Initialize MT5 connection if needed
        self.mt5_connected = False
        if self.use_mt5:
            self.mt5_connected = initialize_mt5_connector()
            if not self.mt5_connected:
                print("Warning: MT5 connection failed. Will use simulation mode.")
        
        # Trading state
        self.is_trading = False
        self.trading_thread = None
        self.current_position = 0  # 0: no position, >0: long position size
        self.current_position_value = 0
        self.entry_price = 0
        self.last_trade_time = None
        self.mt5_position_id = None  # Track MT5 position ID if using real trading
        self.trade_history = []
        self.price_history = []
        
        # Initialize trading components
        self.signal_generator = SignalGenerator(
            threshold=signal_threshold,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy=strategy_type
        )
        
        # Model for ML-based strategy
        if strategy_type == 'ml_based':
            self.model = GoldPricePredictor()
            self.model_trained = False
        else:
            self.model = None
            self.model_trained = True
        
        # Trading session parameters
        self.session_id = None
        
        # Load trading state from database
        self._load_trading_state()
    
    def start_trading(self, interval_seconds=300):
        """
        Start automated trading
        
        Args:
            interval_seconds (int): Time interval between trading checks in seconds
        """
        if self.is_trading:
            return "Trading is already running"
        
        self.is_trading = True
        
        # Create a trading session in the database
        parameters = {
            'signal_threshold': self.signal_threshold,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'feature_window': self.feature_window,
            'prediction_days': self.prediction_days
        }
        self.session_id = db.start_trading_session(self.strategy_type, self.initial_capital, parameters)
        
        # Start trading in a separate thread
        self.trading_thread = threading.Thread(target=self._trading_loop, args=(interval_seconds,))
        self.trading_thread.daemon = True  # Thread will exit when main program exits
        self.trading_thread.start()
        
        return "Automated trading started"
    
    def stop_trading(self):
        """
        Stop automated trading
        """
        if not self.is_trading:
            return "Trading is not running"
        
        self.is_trading = False
        if self.trading_thread:
            self.trading_thread.join(timeout=10)  # Wait for thread to terminate
        
        # End the trading session in the database
        if self.session_id:
            db.end_trading_session(self.session_id)
            self.session_id = None
        
        return "Automated trading stopped"
    
    def get_trading_status(self):
        """
        Get current trading status
        
        Returns:
            dict: Trading status information
        """
        return {
            "is_trading": self.is_trading,
            "current_capital": self.current_capital,
            "current_position": self.current_position,
            "current_position_value": self.current_position_value,
            "entry_price": self.entry_price if self.current_position > 0 else 0,
            "last_trade_time": self.last_trade_time.strftime("%Y-%m-%d %H:%M:%S") if self.last_trade_time else None,
            "total_trades": len(self.trade_history),
            "strategy_type": self.strategy_type,
            "profit_loss": self.current_capital - self.initial_capital
        }
    
    def get_trade_history(self, limit=100):
        """
        Get trading history from both in-memory cache and database
        
        Args:
            limit (int, optional): Maximum number of trades to return
        
        Returns:
            pandas.DataFrame: Trading history
        """
        # Get any recent trades from memory
        memory_trades = pd.DataFrame(self.trade_history) if self.trade_history else pd.DataFrame()
        
        try:
            # Get trades from database
            db_trades = db.get_trades(limit=limit)
            
            # Combine and sort by timestamp (most recent first)
            if not memory_trades.empty and not db_trades.empty:
                # Combine only if both have data
                combined_trades = pd.concat([memory_trades, db_trades])
                combined_trades = combined_trades.drop_duplicates(subset=['timestamp', 'action', 'price'])
                combined_trades = combined_trades.sort_values('timestamp', ascending=False)
                return combined_trades.head(limit)
            elif not db_trades.empty:
                # Return database trades if memory is empty
                return db_trades
            elif not memory_trades.empty:
                # Return memory trades if database is empty
                return memory_trades.sort_values('timestamp', ascending=False).head(limit)
            else:
                # Return empty DataFrame if both are empty
                return pd.DataFrame(columns=['timestamp', 'action', 'price', 'quantity', 'value', 'capital_after'])
        except Exception as e:
            print(f"Error fetching trade history from database: {str(e)}")
            # Fall back to in-memory trades if database query fails
            if not memory_trades.empty:
                return memory_trades.sort_values('timestamp', ascending=False).head(limit)
            return pd.DataFrame(columns=['timestamp', 'action', 'price', 'quantity', 'value', 'capital_after'])
    
    def get_recent_prices(self, days=7):
        """
        Get recent price data from both in-memory cache and database
        
        Args:
            days (int): Number of days to retrieve
            
        Returns:
            pandas.DataFrame: Recent price data
        """
        # Get in-memory prices
        memory_prices = pd.DataFrame(self.price_history) if self.price_history else pd.DataFrame()
        
        try:
            # Get prices from database for the last 'days' days
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=days)
            db_prices = db.get_gold_prices(start_date=start_date, end_date=end_date)
            
            # Format db_prices to match memory_prices format if needed
            if not db_prices.empty:
                db_prices_formatted = pd.DataFrame({
                    'timestamp': db_prices.index,
                    'price': db_prices['Close']
                }).reset_index(drop=True)
                
                # Combine and deduplicate
                if not memory_prices.empty:
                    combined_prices = pd.concat([memory_prices, db_prices_formatted])
                    combined_prices = combined_prices.drop_duplicates(subset=['timestamp'])
                    combined_prices = combined_prices.sort_values('timestamp', ascending=False)
                    return combined_prices
                else:
                    return db_prices_formatted.sort_values('timestamp', ascending=False)
            
            # Return memory prices if db is empty
            if not memory_prices.empty:
                return memory_prices.sort_values('timestamp', ascending=False)
                
            # If both are empty, return empty DataFrame
            return pd.DataFrame(columns=['timestamp', 'price'])
            
        except Exception as e:
            print(f"Error fetching price history from database: {str(e)}")
            # Fall back to in-memory prices if database query fails
            if not memory_prices.empty:
                return memory_prices.sort_values('timestamp', ascending=False)
            return pd.DataFrame(columns=['timestamp', 'price'])
    
    def _trading_loop(self, interval_seconds):
        """
        Main trading loop
        
        Args:
            interval_seconds (int): Time interval between trading checks in seconds
        """
        while self.is_trading:
            try:
                # Check for new trading signals
                self._check_for_signals()
                
                # Save current state
                self._save_trading_state()
                
                # Sleep for the specified interval
                time.sleep(interval_seconds)
            except Exception as e:
                print(f"Error in trading loop: {str(e)}")
                time.sleep(60)  # Sleep for a minute if there's an error
    
    def _check_for_signals(self):
        """
        Check for trading signals and execute trades if necessary
        """
        try:
            # Get current price
            current_price, timestamp = fetch_realtime_gold_price()
            if current_price is None:
                print("Failed to fetch current gold price")
                return
            
            # Record price history
            self._record_price(current_price, timestamp)
            
            # Get historical data for signal generation
            end_date = datetime.datetime.now().date()
            start_date = end_date - datetime.timedelta(days=60)  # Get 60 days of data
            historical_data = fetch_historical_gold_data(start_date, end_date)
            
            if historical_data is None or historical_data.empty:
                print("Failed to fetch historical data for signal generation")
                return
            
            # Generate trading signals based on strategy
            if self.strategy_type == 'ml_based':
                signal = self._generate_ml_signal(historical_data, current_price)
            else:
                signal = self._generate_technical_signal(historical_data)
            
            # Execute trade based on signal
            if signal != 0:  # If there's a buy or sell signal
                self._execute_trade(signal, current_price, timestamp)
            
            # Check stop loss and take profit for existing position
            if self.current_position > 0:
                self._check_position_management(current_price, timestamp)
        
        except Exception as e:
            print(f"Error checking for signals: {str(e)}")
    
    def _generate_technical_signal(self, historical_data):
        """
        Generate trading signal based on technical analysis
        
        Args:
            historical_data (pandas.DataFrame): Historical price data
            
        Returns:
            int: Trading signal (1=buy, -1=sell, 0=hold)
        """
        # Use the SignalGenerator to generate signals
        signals = self.signal_generator.generate_signals(historical_data)
        
        # Get the most recent signal
        if not signals.empty:
            latest_signal = signals.iloc[-1]['signal']
            
            # Save signal to database if it's not a hold signal (0)
            if latest_signal != 0:
                try:
                    # Get the latest row data
                    latest_row = signals.iloc[-1]
                    latest_date = latest_row.name if isinstance(latest_row.name, datetime.datetime) else datetime.datetime.now()
                    price = latest_row.get('Close', 0)
                    
                    # Create signal data dictionary
                    signal_data = {
                        'timestamp': latest_date,
                        'signal_type': int(latest_signal),
                        'price': float(price),
                        'strategy': self.strategy_type,
                        'threshold': self.signal_threshold,
                        'indicators': {
                            'rsi': float(latest_row.get('RSI', 0)),
                            'macd': float(latest_row.get('MACD', 0)),
                            'ema': float(latest_row.get('EMA', 0)) if 'EMA' in latest_row else 0
                        }
                    }
                    
                    # Save to database
                    db.save_trading_signal(signal_data)
                except Exception as e:
                    print(f"Error saving trading signal to database: {str(e)}")
            
            return latest_signal
        
        return 0  # No signal if data is empty
    
    def _generate_ml_signal(self, historical_data, current_price):
        """
        Generate trading signal based on machine learning prediction
        
        Args:
            historical_data (pandas.DataFrame): Historical price data
            current_price (float): Current gold price
            
        Returns:
            int: Trading signal (1=buy, -1=sell, 0=hold)
        """
        # Train the model if not trained
        if not self.model_trained:
            self._train_ml_model(historical_data)
        
        # Process data for prediction
        X_train, y_train, X_test, y_test, _ = process_data_for_model(
            historical_data, 
            self.feature_window, 
            self.prediction_days
        )
        
        if X_train is None or y_train is None:
            print("Not enough data to generate ML signal")
            return 0
        
        # Retrain model periodically with latest data
        if len(X_train) > 0 and len(y_train) > 0:
            self.model.train(X_train, y_train)
            self.model_trained = True
        
        # Make prediction for last data point
        if len(X_test) > 0:
            prediction = self.model.predict(X_test)
            predicted_price = prediction[-1]
            
            # Save ML prediction to database
            try:
                # Create prediction data dictionary
                prediction_data = {
                    'timestamp': datetime.datetime.now(),
                    'model_type': 'ml_based',
                    'predicted_price': float(predicted_price),
                    'actual_price': float(current_price),
                    'prediction_horizon': self.prediction_days,
                    'features_used': {
                        'feature_window': self.feature_window,
                        'model_type': self.model.model_type
                    }
                }
                
                # Save to database
                db.save_model_prediction(prediction_data)
            except Exception as e:
                print(f"Error saving model prediction to database: {str(e)}")
            
            # Generate trading signal based on prediction
            signal = generate_trading_signal(
                current_price, 
                predicted_price, 
                self.signal_threshold
            )
            
            # Save signal to database if it's not a hold signal (0)
            if signal != 0:
                try:
                    # Create signal data dictionary
                    signal_data = {
                        'timestamp': datetime.datetime.now(),
                        'signal_type': int(signal),
                        'price': float(current_price),
                        'strategy': 'ml_based',
                        'threshold': self.signal_threshold,
                        'indicators': {
                            'predicted_price': float(predicted_price),
                            'current_price': float(current_price),
                            'prediction_days': self.prediction_days
                        }
                    }
                    
                    # Save to database
                    db.save_trading_signal(signal_data)
                except Exception as e:
                    print(f"Error saving ML signal to database: {str(e)}")
            
            return signal
        
        return 0  # No signal if prediction fails
    
    def _execute_trade(self, signal, current_price, timestamp):
        """
        Execute a trade based on signal - either through MT5 (if connected) or simulated
        
        Args:
            signal (int): Trading signal (1=buy, -1=sell)
            current_price (float): Current gold price
            timestamp (str): Current timestamp
        """
        # Convert timestamp string to datetime if needed
        if isinstance(timestamp, str):
            timestamp = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        
        # Check if we can trade (avoid trading too frequently)
        if self.last_trade_time and (timestamp - self.last_trade_time) < timedelta(minutes=5):
            return  # Don't trade if last trade was less than 5 minutes ago
        
        # Calculate common values
        trade_capital = self.current_capital * 0.9  # Use 90% of capital, keep 10% as reserve
        quantity = trade_capital / current_price
        mt5_success = False
        
        # Handle buy signal
        if signal == 1 and self.current_position == 0:
            # Try to execute on MT5 if connected
            if self.use_mt5 and self.mt5_connected and mt5_connector:
                try:
                    # Convert quantity to lots (MT5 standard is 100 oz per lot for XAUUSD)
                    lot_size = quantity / 100
                    
                    # Calculate stop loss and take profit levels
                    sl_price = current_price * (1 - self.stop_loss)
                    tp_price = current_price * (1 + self.take_profit)
                    
                    # Execute trade on MT5
                    result = mt5_connector.execute_trade(
                        symbol="XAUUSD",
                        order_type="BUY",
                        volume=lot_size,
                        price=None,  # Market order
                        sl=sl_price,
                        tp=tp_price,
                        comment=f"AI Gold Bot - {self.strategy_type}"
                    )
                    
                    if result and result.get("success"):
                        self.mt5_position_id = result.get("order_id", 0)
                        mt5_success = True
                        print(f"MT5 BUY order executed: Position ID {self.mt5_position_id}")
                    else:
                        print(f"MT5 trade execution failed: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    print(f"Error executing MT5, falling back to simulation: {str(e)}")
            
            # Execute in simulation mode if MT5 failed or not connected
            if not mt5_success:
                # Execute buy (simulation)
                self.current_position = quantity
                self.entry_price = current_price
                self.current_position_value = quantity * current_price
                print(f"Simulated BUY executed at ${current_price:.2f}, quantity: {quantity:.6f}")
            
            self.last_trade_time = timestamp
            
            # Record the trade (both for MT5 and simulation)
            trade = {
                'timestamp': timestamp,
                'action': 'BUY',
                'price': current_price,
                'quantity': quantity,
                'value': quantity * current_price,
                'capital_after': self.current_capital,
                'strategy': self.strategy_type,
                'execution': 'MT5' if mt5_success else 'Simulation'
            }
            self.trade_history.append(trade)
            
            # Save to database
            db.save_trade(trade)
        
        # Handle sell signal
        elif signal == -1 and self.current_position > 0:
            # Try to execute on MT5 if connected and we have an open position
            if self.use_mt5 and self.mt5_connected and mt5_connector and self.mt5_position_id:
                try:
                    # Close the position on MT5
                    close_result = mt5_connector.close_position(self.mt5_position_id)
                    
                    if close_result:
                        mt5_success = True
                        print(f"MT5 position {self.mt5_position_id} closed successfully")
                        self.mt5_position_id = None
                    else:
                        print(f"MT5 position close failed: {mt5_connector.get_last_error()}")
                except Exception as e:
                    print(f"Error closing MT5 position, falling back to simulation: {str(e)}")
            
            # Calculate sell value and profit/loss
            sell_value = self.current_position * current_price
            profit_loss = sell_value - self.current_position_value
            
            # Update capital (only in simulation or if MT5 trade was successful)
            if not self.use_mt5 or not self.mt5_connected or mt5_success:
                self.current_capital += profit_loss
            
            # Record the trade
            trade = {
                'timestamp': timestamp,
                'action': 'SELL',
                'price': current_price,
                'quantity': self.current_position,
                'value': sell_value,
                'pnl': profit_loss,
                'pnl_percent': (profit_loss / self.current_position_value) * 100,
                'capital_after': self.current_capital,
                'strategy': self.strategy_type,
                'execution': 'MT5' if mt5_success else 'Simulation'
            }
            self.trade_history.append(trade)
            
            # Save to database
            db.save_trade(trade)
            
            # Reset position
            self.current_position = 0
            self.current_position_value = 0
            self.entry_price = 0
            self.last_trade_time = timestamp
            
            print(f"SELL executed at ${current_price:.2f}, P&L: ${profit_loss:.2f} ({(profit_loss / sell_value) * 100:.2f}%)")
    
    def _check_position_management(self, current_price, timestamp):
        """
        Check if current position needs to be managed (stop loss, take profit)
        Note: For MT5 trades, stop loss and take profit are handled by MT5 platform
        This method only applies to simulation trading
        
        Args:
            current_price (float): Current gold price
            timestamp (str): Current timestamp
        """
        # If using MT5 and connected, check position status
        if self.use_mt5 and self.mt5_connected and mt5_connector and self.mt5_position_id:
            try:
                # Get open positions
                open_positions = mt5_connector.get_open_positions()
                
                # Check if our position is still open
                position_open = False
                for pos in open_positions:
                    if pos.get('ticket') == self.mt5_position_id:
                        position_open = True
                        break
                
                # If position is closed in MT5 but we still think it's open
                if not position_open and self.current_position > 0:
                    # Get the latest price data to determine approximate P&L
                    current_value = self.current_position * current_price
                    profit_loss = current_value - self.current_position_value
                    profit_loss_pct = (current_price / self.entry_price) - 1
                    
                    # Create a trade record (we don't know if it was stop loss or take profit)
                    action = 'TAKE_PROFIT' if profit_loss > 0 else 'STOP_LOSS'
                    
                    # Record the trade
                    trade = {
                        'timestamp': timestamp,
                        'action': action,
                        'price': current_price,
                        'quantity': self.current_position,
                        'value': current_value,
                        'pnl': profit_loss,
                        'pnl_percent': profit_loss_pct * 100,
                        'capital_after': self.current_capital + profit_loss,
                        'strategy': self.strategy_type,
                        'execution': 'MT5'
                    }
                    self.trade_history.append(trade)
                    
                    # Save to database
                    db.save_trade(trade)
                    
                    # Update capital
                    self.current_capital += profit_loss
                    
                    # Reset position
                    self.current_position = 0
                    self.current_position_value = 0
                    self.entry_price = 0
                    self.mt5_position_id = None
                    self.last_trade_time = timestamp
                    
                    print(f"MT5 position closed externally. P&L: ${profit_loss:.2f} ({profit_loss_pct * 100:.2f}%)")
                    return
            except Exception as e:
                print(f"Error checking MT5 position status: {str(e)}")
        
        # For simulation or if MT5 check failed
        if self.current_position <= 0:
            return  # No position to manage
        
        # Convert timestamp string to datetime if needed
        if isinstance(timestamp, str):
            timestamp = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        
        # Calculate current profit/loss percentage
        current_value = self.current_position * current_price
        profit_loss_pct = (current_price / self.entry_price) - 1
        
        # Check stop loss
        if profit_loss_pct <= -self.stop_loss:
            # Execute stop loss
            profit_loss = current_value - self.current_position_value
            self.current_capital += profit_loss
            
            # Try to close MT5 position if we have one
            mt5_success = False
            if self.use_mt5 and self.mt5_connected and mt5_connector and self.mt5_position_id:
                try:
                    close_result = mt5_connector.close_position(self.mt5_position_id)
                    if close_result:
                        mt5_success = True
                        print(f"MT5 position {self.mt5_position_id} closed by stop loss")
                    self.mt5_position_id = None
                except Exception as e:
                    print(f"Error closing MT5 position on stop loss: {str(e)}")
            
            # Record the trade
            trade = {
                'timestamp': timestamp,
                'action': 'STOP_LOSS',
                'price': current_price,
                'quantity': self.current_position,
                'value': current_value,
                'pnl': profit_loss,
                'pnl_percent': profit_loss_pct * 100,
                'capital_after': self.current_capital,
                'strategy': self.strategy_type,
                'execution': 'MT5' if mt5_success else 'Simulation'
            }
            self.trade_history.append(trade)
            
            # Save to database
            db.save_trade(trade)
            
            # Reset position
            self.current_position = 0
            self.current_position_value = 0
            self.entry_price = 0
            self.last_trade_time = timestamp
            
            print(f"STOP LOSS executed at ${current_price:.2f}, P&L: ${profit_loss:.2f} ({profit_loss_pct * 100:.2f}%)")
        
        # Check take profit
        elif profit_loss_pct >= self.take_profit:
            # Execute take profit
            profit_loss = current_value - self.current_position_value
            self.current_capital += profit_loss
            
            # Try to close MT5 position if we have one
            mt5_success = False
            if self.use_mt5 and self.mt5_connected and mt5_connector and self.mt5_position_id:
                try:
                    close_result = mt5_connector.close_position(self.mt5_position_id)
                    if close_result:
                        mt5_success = True
                        print(f"MT5 position {self.mt5_position_id} closed by take profit")
                    self.mt5_position_id = None
                except Exception as e:
                    print(f"Error closing MT5 position on take profit: {str(e)}")
            
            # Record the trade
            trade = {
                'timestamp': timestamp,
                'action': 'TAKE_PROFIT',
                'price': current_price,
                'quantity': self.current_position,
                'value': current_value,
                'pnl': profit_loss,
                'pnl_percent': profit_loss_pct * 100,
                'capital_after': self.current_capital,
                'strategy': self.strategy_type,
                'execution': 'MT5' if mt5_success else 'Simulation'
            }
            self.trade_history.append(trade)
            
            # Save to database
            db.save_trade(trade)
            
            # Reset position
            self.current_position = 0
            self.current_position_value = 0
            self.entry_price = 0
            self.last_trade_time = timestamp
            
            print(f"TAKE PROFIT executed at ${current_price:.2f}, P&L: ${profit_loss:.2f} ({profit_loss_pct * 100:.2f}%)")
    
    def _record_price(self, price, timestamp):
        """
        Record price history
        
        Args:
            price (float): Current price
            timestamp (str): Current timestamp
        """
        # Convert timestamp string to datetime if needed
        if isinstance(timestamp, str):
            timestamp = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        
        # Add to price history
        self.price_history.append({
            'timestamp': timestamp,
            'price': price
        })
        
        # Keep only the last 1000 price points to avoid memory issues
        if len(self.price_history) > 1000:
            self.price_history = self.price_history[-1000:]
        
        # Save to database
        price_data = {
            'timestamp': timestamp,
            'close': price,  # Using price as close since it's the current price
            'open': price,   # Set open to same as close for single price point
            'high': price,   # Set high to same as close for single price point
            'low': price,    # Set low to same as close for single price point
            'volume': None   # Volume not available for real-time data
        }
        
        try:
            db.save_gold_price(price_data)
        except Exception as e:
            print(f"Error saving price to database: {str(e)}")
    
    def _train_ml_model(self, historical_data):
        """
        Train the ML model for trading
        
        Args:
            historical_data (pandas.DataFrame): Historical price data
        """
        # Process data for model
        X_train, y_train, _, _, _ = process_data_for_model(
            historical_data, 
            self.feature_window, 
            self.prediction_days
        )
        
        # Train model if we have enough data
        if X_train is not None and y_train is not None and len(X_train) > 0:
            self.model.train(X_train, y_train)
            self.model_trained = True
            print("ML model trained successfully")
        else:
            print("Not enough data to train the ML model")
    
    def _save_trading_state(self):
        """
        Save trading state to file
        """
        try:
            # Create state directory if it doesn't exist
            if not os.path.exists('trading_state'):
                os.makedirs('trading_state')
            
            # Save trading state
            state = {
                'current_capital': self.current_capital,
                'current_position': self.current_position,
                'current_position_value': self.current_position_value,
                'entry_price': self.entry_price,
                'last_trade_time': self.last_trade_time.isoformat() if self.last_trade_time else None,
                'is_trading': self.is_trading,
                'initial_capital': self.initial_capital,
                'strategy_type': self.strategy_type,
                'signal_threshold': self.signal_threshold,
                'stop_loss': self.stop_loss,
                'take_profit': self.take_profit
            }
            
            with open('trading_state/state.json', 'w') as f:
                json.dump(state, f)
            
            # Save trade history (convert timestamps to strings)
            trade_history_save = []
            for trade in self.trade_history:
                trade_copy = trade.copy()
                # Convert timestamp if it's a datetime
                if isinstance(trade_copy['timestamp'], datetime.datetime):
                    trade_copy['timestamp'] = trade_copy['timestamp'].isoformat()
                trade_history_save.append(trade_copy)
            
            with open('trading_state/trade_history.json', 'w') as f:
                json.dump(trade_history_save, f)
            
            # Save price history (convert timestamps to strings)
            price_history_save = []
            for price_point in self.price_history:
                price_copy = price_point.copy()
                # Convert timestamp if it's a datetime
                if isinstance(price_copy['timestamp'], datetime.datetime):
                    price_copy['timestamp'] = price_copy['timestamp'].isoformat()
                price_history_save.append(price_copy)
            
            with open('trading_state/price_history.json', 'w') as f:
                json.dump(price_history_save, f)
        
        except Exception as e:
            print(f"Error saving trading state: {str(e)}")
    
    def _load_trading_state(self):
        """
        Load trading state from file
        """
        try:
            # Check if state files exist
            if not os.path.exists('trading_state/state.json'):
                return
            
            # Load state
            with open('trading_state/state.json', 'r') as f:
                state = json.load(f)
            
            self.current_capital = state.get('current_capital', self.initial_capital)
            self.current_position = state.get('current_position', 0)
            self.current_position_value = state.get('current_position_value', 0)
            self.entry_price = state.get('entry_price', 0)
            
            last_trade_time = state.get('last_trade_time')
            if last_trade_time:
                self.last_trade_time = datetime.datetime.fromisoformat(last_trade_time)
            
            # Load trade history if exists
            if os.path.exists('trading_state/trade_history.json'):
                with open('trading_state/trade_history.json', 'r') as f:
                    trade_history = json.load(f)
                
                # Convert string timestamps back to datetime
                for trade in trade_history:
                    if 'timestamp' in trade and isinstance(trade['timestamp'], str):
                        trade['timestamp'] = datetime.datetime.fromisoformat(trade['timestamp'])
                
                self.trade_history = trade_history
            
            # Load price history if exists
            if os.path.exists('trading_state/price_history.json'):
                with open('trading_state/price_history.json', 'r') as f:
                    price_history = json.load(f)
                
                # Convert string timestamps back to datetime
                for price_point in price_history:
                    if 'timestamp' in price_point and isinstance(price_point['timestamp'], str):
                        price_point['timestamp'] = datetime.datetime.fromisoformat(price_point['timestamp'])
                
                self.price_history = price_history
            
            print("Trading state loaded successfully")
        
        except Exception as e:
            print(f"Error loading trading state: {str(e)}")