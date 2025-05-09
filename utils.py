import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def calculate_metrics(data):
    """
    Calculate key performance metrics from historical price data
    
    Args:
        data (pandas.DataFrame): Historical price data
        
    Returns:
        dict: Dictionary of performance metrics
    """
    # Calculate daily returns if not already present
    if 'Daily_Return' not in data.columns:
        daily_returns = data['Close'].pct_change().dropna()
    else:
        daily_returns = data['Daily_Return'].dropna()
    
    # Convert to percentage
    daily_returns_pct = daily_returns * 100
    
    # Average daily return
    avg_return = daily_returns_pct.mean()
    
    # Daily volatility
    volatility = daily_returns_pct.std()
    
    # Annualized volatility (assuming 252 trading days in a year)
    annual_volatility = volatility * np.sqrt(252)
    
    # Calculate drawdowns
    cumulative_returns = (1 + daily_returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min() * 100  # Convert to percentage
    
    # Calculate Sharpe ratio (assuming risk-free rate of 0)
    annualized_return = avg_return * 252
    sharpe_ratio = annualized_return / annual_volatility if annual_volatility != 0 else 0
    
    return {
        'avg_return': avg_return,
        'volatility': annual_volatility,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio
    }

def process_data_for_model(data, feature_window=14, prediction_days=7):
    """
    Process historical data into features and target for machine learning model
    
    Args:
        data (pandas.DataFrame): Historical price data
        feature_window (int): Number of days to use for features
        prediction_days (int): Number of days ahead to predict
        
    Returns:
        tuple: (X_train, y_train, X_test, y_test, test_dates)
    """
    if data is None or len(data) < feature_window + prediction_days:
        print("Not enough data for model training")
        return None, None, None, None, None
    
    # Create features from price and technical indicators
    feature_columns = ['Close', 'Daily_Return', 'Volatility', 'RSI', 'MA_5', 'MA_20', 'MA_50']
    
    # Check available columns
    available_features = [col for col in feature_columns if col in data.columns]
    
    if not available_features:
        print("No feature columns available in data")
        return None, None, None, None, None
    
    # Select features
    features = data[available_features].values
    
    # Create target: price 'prediction_days' days in the future
    target = data['Close'].shift(-prediction_days).values
    
    # Remove NaN values
    valid_indices = ~np.isnan(target)
    features = features[valid_indices]
    target = target[valid_indices]
    
    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features)
    
    # Create sequences for time series prediction
    X, y = [], []
    for i in range(feature_window, len(features_scaled)):
        X.append(features_scaled[i-feature_window:i, :])
        y.append(target[i-feature_window])
    
    X, y = np.array(X), np.array(y)
    
    # Reshape X for neural network models if needed
    # X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    
    # Split into train and test sets (use the last 20% for testing)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Get dates for test set for plotting
    test_dates_idx = np.arange(len(features) - len(X_test), len(features))
    test_dates = data.index[test_dates_idx]
    
    # Reshape for non-neural network models
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    return X_train, y_train, X_test, y_test, test_dates

def generate_trading_signal(current_price, predicted_price, threshold=0.01):
    """
    Generate a trading signal based on price prediction
    
    Args:
        current_price (float): Current gold price
        predicted_price (float): Predicted gold price
        threshold (float): Signal threshold (as decimal)
        
    Returns:
        int: 1 for buy, -1 for sell, 0 for hold
    """
    # Calculate predicted return
    predicted_return = (predicted_price / current_price) - 1
    
    # Generate signal based on predicted return
    if predicted_return > threshold:
        return 1  # Buy signal
    elif predicted_return < -threshold:
        return -1  # Sell signal
    else:
        return 0  # Hold

def format_currency(value):
    """
    Format a value as USD currency
    
    Args:
        value (float): Value to format
        
    Returns:
        str: Formatted currency string
    """
    return f"${value:,.2f}"

def format_percentage(value):
    """
    Format a value as percentage
    
    Args:
        value (float): Value to format
        
    Returns:
        str: Formatted percentage string
    """
    return f"{value:.2f}%"

def create_summary_report(metrics, trade_list):
    """
    Create a summary report of trading performance
    
    Args:
        metrics (dict): Performance metrics
        trade_list (pandas.DataFrame): List of trades
        
    Returns:
        str: Summary report as markdown
    """
    report = "# Trading Strategy Performance Summary\n\n"
    
    # Overall metrics
    report += "## Overall Performance\n\n"
    report += f"* Initial Capital: {format_currency(10000)}\n"
    report += f"* Final Portfolio Value: {format_currency(metrics['final_value'])}\n"
    report += f"* Total Return: {format_percentage(metrics['total_return'])}\n"
    report += f"* Annualized Return: {format_percentage(metrics['annualized_return'])}\n"
    report += f"* Sharpe Ratio: {metrics['sharpe_ratio']:.3f}\n"
    report += f"* Max Drawdown: {format_percentage(metrics['max_drawdown'])}\n\n"
    
    # Trading metrics
    report += "## Trading Statistics\n\n"
    report += f"* Total Trades: {metrics['total_trades']}\n"
    report += f"* Win Rate: {format_percentage(metrics['win_rate'])}\n"
    report += f"* Profit Factor: {metrics['profit_factor']:.3f}\n"
    report += f"* Average Win: {format_currency(metrics['avg_win'])}\n"
    report += f"* Average Loss: {format_currency(metrics['avg_loss'])}\n\n"
    
    # Recent trades
    if not trade_list.empty:
        report += "## Recent Trades\n\n"
        report += "| Entry Date | Exit Date | Entry Price | Exit Price | P&L | P&L % | Outcome |\n"
        report += "|------------|-----------|-------------|------------|-----|-------|---------|\n"
        
        for _, trade in trade_list.tail(5).iterrows():
            outcome = "🟢 Win" if trade['pnl'] > 0 else "🔴 Loss"
            report += f"| {trade['entry_date'].date()} | {trade['exit_date'].date()} | "
            report += f"{format_currency(trade['entry_price'])} | {format_currency(trade['exit_price'])} | "
            report += f"{format_currency(trade['pnl'])} | {format_percentage(trade['pnl_pct'])} | {outcome} |\n"
    
    return report
