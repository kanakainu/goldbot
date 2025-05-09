import pandas as pd
import numpy as np
from scipy import stats
import talib as ta

class SignalGenerator:
    """
    Class for generating trading signals based on various strategies
    """
    
    def __init__(self, threshold=0.01, stop_loss=0.02, take_profit=0.03, strategy='trend_following'):
        """
        Initialize the signal generator
        
        Args:
            threshold (float): Signal threshold (as decimal)
            stop_loss (float): Stop loss percentage (as decimal)
            take_profit (float): Take profit percentage (as decimal)
            strategy (str): Strategy type - 'trend_following', 'mean_reversion', 'ml_based'
        """
        self.threshold = threshold
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.strategy = strategy
    
    def generate_signals(self, data):
        """
        Generate trading signals based on the selected strategy
        
        Args:
            data (pandas.DataFrame): Price data with technical indicators
            
        Returns:
            pandas.DataFrame: DataFrame with signals (1=buy, -1=sell, 0=hold)
        """
        if self.strategy == 'trend_following':
            return self._trend_following_strategy(data)
        elif self.strategy == 'mean_reversion':
            return self._mean_reversion_strategy(data)
        elif self.strategy == 'ml_based':
            return self._ml_based_strategy(data)
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")
    
    def _trend_following_strategy(self, data):
        """
        Generate signals based on trend following indicators
        
        Args:
            data (pandas.DataFrame): Price data
            
        Returns:
            pandas.DataFrame: DataFrame with signals
        """
        # Create a copy of the dataframe to avoid modifying the original
        df = data.copy()
        
        try:
            # If the dataframe doesn't have moving averages, calculate them
            if 'MA_5' not in df.columns:
                df['MA_5'] = df['Close'].rolling(window=5).mean()
            if 'MA_20' not in df.columns:
                df['MA_20'] = df['Close'].rolling(window=20).mean()
            if 'MA_50' not in df.columns:
                df['MA_50'] = df['Close'].rolling(window=50).mean()
            
            # Calculate additional technical indicators
            # RSI (Relative Strength Index)
            df['RSI'] = self._calculate_rsi(df['Close'], 14)
            
            # MACD (Moving Average Convergence Divergence)
            df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = self._calculate_macd(df['Close'])
            
            # ATR (Average True Range) for volatility
            df['ATR'] = self._calculate_atr(df['High'], df['Low'], df['Close'], 14)
            
            # Initialize signal column
            df['signal'] = 0
            
            # Generate signals based on multiple conditions
            for i in range(50, len(df)):  # Start from index 50 to have enough data for indicators
                # Skip dates where we have NaN values for indicators
                if (pd.isna(df['MA_5'].iloc[i]) or pd.isna(df['MA_20'].iloc[i]) or
                        pd.isna(df['RSI'].iloc[i]) or pd.isna(df['MACD'].iloc[i])):
                    continue
                
                # Previous position (0 if no position)
                prev_position = df['signal'].iloc[:i].sum()
                
                # Buy conditions (enter long)
                buy_condition = (
                    (df['MA_5'].iloc[i] > df['MA_20'].iloc[i]) and  # Short MA above long MA
                    (df['MA_5'].iloc[i-1] <= df['MA_20'].iloc[i-1]) and  # Crossover just occurred
                    (df['RSI'].iloc[i] > 30) and  # RSI not in oversold territory
                    (df['MACD'].iloc[i] > df['MACD_Signal'].iloc[i])  # MACD above signal line
                )
                
                # Sell conditions (exit long)
                sell_condition = (
                    (df['MA_5'].iloc[i] < df['MA_20'].iloc[i]) and  # Short MA below long MA
                    (df['MA_5'].iloc[i-1] >= df['MA_20'].iloc[i-1]) and  # Crossover just occurred
                    (df['RSI'].iloc[i] < 70) and  # RSI not in overbought territory
                    (df['MACD'].iloc[i] < df['MACD_Signal'].iloc[i])  # MACD below signal line
                )
                
                # Stop loss and take profit conditions
                if prev_position > 0:  # If in a long position
                    # Get entry price (approximate as the price when signal was 1)
                    entry_indices = df.index[df['signal'] == 1]
                    if len(entry_indices) > 0:
                        latest_entry_idx = df.index.get_loc(entry_indices[-1])
                        entry_price = df['Close'].iloc[latest_entry_idx]
                        current_price = df['Close'].iloc[i]
                        
                        # Calculate profit/loss percentage
                        pnl_pct = (current_price / entry_price) - 1
                        
                        # Stop loss
                        if pnl_pct <= -self.stop_loss:
                            sell_condition = True
                        
                        # Take profit
                        if pnl_pct >= self.take_profit:
                            sell_condition = True
                
                # Apply the signal based on conditions and previous position
                if buy_condition and prev_position <= 0:  # Buy if condition met and not already long
                    df.iloc[i, df.columns.get_indexer(['signal'])[0]] = 1
                elif sell_condition and prev_position > 0:  # Sell if condition met and in long position
                    df.iloc[i, df.columns.get_indexer(['signal'])[0]] = -1
                else:
                    df.iloc[i, df.columns.get_indexer(['signal'])[0]] = 0
            
            return df[['signal']]
        
        except Exception as e:
            print(f"Error generating trend following signals: {str(e)}")
            # Return empty signals in case of error
            return pd.DataFrame(0, index=data.index, columns=['signal'])
    
    def _mean_reversion_strategy(self, data):
        """
        Generate signals based on mean reversion indicators
        
        Args:
            data (pandas.DataFrame): Price data
            
        Returns:
            pandas.DataFrame: DataFrame with signals
        """
        # Create a copy of the dataframe to avoid modifying the original
        df = data.copy()
        
        try:
            # Calculate Bollinger Bands
            if 'Close' in df.columns:
                df['SMA_20'] = df['Close'].rolling(window=20).mean()
                df['STD_20'] = df['Close'].rolling(window=20).std()
                df['Upper_Band'] = df['SMA_20'] + (df['STD_20'] * 2)
                df['Lower_Band'] = df['SMA_20'] - (df['STD_20'] * 2)
                df['Bandwidth'] = (df['Upper_Band'] - df['Lower_Band']) / df['SMA_20']
                
                # Calculate Z-score for mean reversion
                df['Z_Score'] = (df['Close'] - df['SMA_20']) / df['STD_20']
                
                # RSI for overbought/oversold conditions
                df['RSI'] = self._calculate_rsi(df['Close'], 14)
                
                # Initialize signal column
                df['signal'] = 0
                
                # Generate signals based on multiple conditions
                for i in range(20, len(df)):  # Start from index 20 to have enough data for indicators
                    # Skip dates where we have NaN values for indicators
                    if (pd.isna(df['SMA_20'].iloc[i]) or pd.isna(df['Upper_Band'].iloc[i]) or
                            pd.isna(df['Lower_Band'].iloc[i]) or pd.isna(df['RSI'].iloc[i])):
                        continue
                    
                    # Previous position (0 if no position)
                    prev_position = df['signal'].iloc[:i].sum()
                    
                    # Buy conditions (enter long when price is below lower band and oversold)
                    buy_condition = (
                        (df['Close'].iloc[i] < df['Lower_Band'].iloc[i]) and  # Price below lower band
                        (df['RSI'].iloc[i] < 30) and  # RSI in oversold territory
                        (df['Z_Score'].iloc[i] < -2)  # Price significantly below mean
                    )
                    
                    # Sell conditions (exit long when price is above upper band and overbought)
                    sell_condition = (
                        (df['Close'].iloc[i] > df['Upper_Band'].iloc[i]) and  # Price above upper band
                        (df['RSI'].iloc[i] > 70) and  # RSI in overbought territory
                        (df['Z_Score'].iloc[i] > 2)  # Price significantly above mean
                    )
                    
                    # Stop loss and take profit conditions
                    if prev_position > 0:  # If in a long position
                        # Get entry price (approximate as the price when signal was 1)
                        entry_indices = df.index[df['signal'] == 1]
                        if len(entry_indices) > 0:
                            latest_entry_idx = df.index.get_loc(entry_indices[-1])
                            entry_price = df['Close'].iloc[latest_entry_idx]
                            current_price = df['Close'].iloc[i]
                            
                            # Calculate profit/loss percentage
                            pnl_pct = (current_price / entry_price) - 1
                            
                            # Stop loss
                            if pnl_pct <= -self.stop_loss:
                                sell_condition = True
                            
                            # Take profit
                            if pnl_pct >= self.take_profit:
                                sell_condition = True
                    
                    # Apply the signal based on conditions and previous position
                    if buy_condition and prev_position <= 0:  # Buy if condition met and not already long
                        df.iloc[i, df.columns.get_indexer(['signal'])[0]] = 1
                    elif sell_condition and prev_position > 0:  # Sell if condition met and in long position
                        df.iloc[i, df.columns.get_indexer(['signal'])[0]] = -1
                    else:
                        df.iloc[i, df.columns.get_indexer(['signal'])[0]] = 0
                
                return df[['signal']]
            else:
                print("Error: 'Close' column not found in data")
                return pd.DataFrame(0, index=data.index, columns=['signal'])
        
        except Exception as e:
            print(f"Error generating mean reversion signals: {str(e)}")
            # Return empty signals in case of error
            return pd.DataFrame(0, index=data.index, columns=['signal'])
    
    def _ml_based_strategy(self, data):
        """
        Generate signals based on machine learning predictions
        
        Args:
            data (pandas.DataFrame): Price data with prediction column
            
        Returns:
            pandas.DataFrame: DataFrame with signals
        """
        # This method expects that the data already has a 'prediction' column
        # from a machine learning model
        
        # Create a copy of the dataframe to avoid modifying the original
        df = data.copy()
        
        try:
            # If prediction column doesn't exist, we can't generate signals
            if 'prediction' not in df.columns:
                print("Error: 'prediction' column not found in data")
                return pd.DataFrame(0, index=data.index, columns=['signal'])
            
            # Initialize signal column
            df['signal'] = 0
            
            # Generate signals based on predictions
            for i in range(1, len(df)):
                # Skip dates where we have NaN values for predictions
                if pd.isna(df['prediction'].iloc[i]):
                    continue
                
                # Previous position (0 if no position)
                prev_position = df['signal'].iloc[:i].sum()
                
                # Calculate predicted return
                predicted_return = (df['prediction'].iloc[i] / df['Close'].iloc[i-1]) - 1
                
                # Buy condition: predicted return exceeds threshold
                buy_condition = predicted_return > self.threshold
                
                # Sell condition: predicted return below negative threshold
                sell_condition = predicted_return < -self.threshold
                
                # Stop loss and take profit conditions
                if prev_position > 0:  # If in a long position
                    # Get entry price (approximate as the price when signal was 1)
                    entry_indices = df.index[df['signal'] == 1]
                    if len(entry_indices) > 0:
                        latest_entry_idx = df.index.get_loc(entry_indices[-1])
                        entry_price = df['Close'].iloc[latest_entry_idx]
                        current_price = df['Close'].iloc[i]
                        
                        # Calculate profit/loss percentage
                        pnl_pct = (current_price / entry_price) - 1
                        
                        # Stop loss
                        if pnl_pct <= -self.stop_loss:
                            sell_condition = True
                        
                        # Take profit
                        if pnl_pct >= self.take_profit:
                            sell_condition = True
                
                # Apply the signal based on conditions and previous position
                if buy_condition and prev_position <= 0:  # Buy if condition met and not already long
                    df.iloc[i, df.columns.get_indexer(['signal'])[0]] = 1
                elif sell_condition and prev_position > 0:  # Sell if condition met and in long position
                    df.iloc[i, df.columns.get_indexer(['signal'])[0]] = -1
                else:
                    df.iloc[i, df.columns.get_indexer(['signal'])[0]] = 0
            
            return df[['signal']]
        
        except Exception as e:
            print(f"Error generating ML-based signals: {str(e)}")
            # Return empty signals in case of error
            return pd.DataFrame(0, index=data.index, columns=['signal'])
    
    def _calculate_rsi(self, prices, period=14):
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            prices (pandas.Series): Price series
            period (int): RSI period
            
        Returns:
            pandas.Series: RSI values
        """
        try:
            # Calculate price changes
            delta = prices.diff()
            
            # Separate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # Calculate average gains and losses
            avg_gain = gains.rolling(window=period).mean()
            avg_loss = losses.rolling(window=period).mean()
            
            # Calculate relative strength
            rs = avg_gain / avg_loss
            
            # Calculate RSI
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except:
            # If there's an error, try to use TA-Lib if available
            try:
                import talib
                return talib.RSI(prices.values, timeperiod=period)
            except:
                # If TA-Lib is not available, return a series of NaN
                return pd.Series(np.nan, index=prices.index)
    
    def _calculate_macd(self, prices, fast_period=12, slow_period=26, signal_period=9):
        """
        Calculate Moving Average Convergence Divergence (MACD)
        
        Args:
            prices (pandas.Series): Price series
            fast_period (int): Fast EMA period
            slow_period (int): Slow EMA period
            signal_period (int): Signal line period
            
        Returns:
            tuple: (MACD, Signal, Histogram)
        """
        try:
            # Calculate exponential moving averages
            ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
            ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
            
            # Calculate MACD line
            macd_line = ema_fast - ema_slow
            
            # Calculate signal line
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            return macd_line, signal_line, histogram
        except:
            # If there's an error, try to use TA-Lib if available
            try:
                import talib
                macd, signal, hist = talib.MACD(
                    prices.values, 
                    fastperiod=fast_period, 
                    slowperiod=slow_period, 
                    signalperiod=signal_period
                )
                return pd.Series(macd, index=prices.index), pd.Series(signal, index=prices.index), pd.Series(hist, index=prices.index)
            except:
                # If TA-Lib is not available, return series of NaN
                return pd.Series(np.nan, index=prices.index), pd.Series(np.nan, index=prices.index), pd.Series(np.nan, index=prices.index)
    
    def _calculate_atr(self, high, low, close, period=14):
        """
        Calculate Average True Range (ATR)
        
        Args:
            high (pandas.Series): High prices
            low (pandas.Series): Low prices
            close (pandas.Series): Close prices
            period (int): ATR period
            
        Returns:
            pandas.Series: ATR values
        """
        try:
            # Calculate true range
            tr1 = high - low  # Current high - current low
            tr2 = abs(high - close.shift())  # Current high - previous close
            tr3 = abs(low - close.shift())  # Current low - previous close
            
            # True range is the maximum of the three values
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate average true range
            atr = tr.rolling(window=period).mean()
            
            return atr
        except:
            # If there's an error, try to use TA-Lib if available
            try:
                import talib
                return pd.Series(talib.ATR(high.values, low.values, close.values, timeperiod=period), index=high.index)
            except:
                # If TA-Lib is not available, return a series of NaN
                return pd.Series(np.nan, index=high.index)
