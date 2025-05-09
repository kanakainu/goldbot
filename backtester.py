import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

class Backtester:
    """
    Class for backtesting trading strategies on historical gold price data
    """
    
    def __init__(self, initial_capital=10000, signal_data=None):
        """
        Initialize the backtester with initial capital
        
        Args:
            initial_capital (float): Initial capital for the portfolio
            signal_data (pandas.DataFrame, optional): Pre-generated signal data
        """
        self.initial_capital = initial_capital
        self.signal_data = signal_data
        self.backtest_results = None
        self.trade_list = pd.DataFrame()
    
    def run_backtest(self, price_data):
        """
        Run a backtest using price data and pre-generated signals
        
        Args:
            price_data (pandas.DataFrame): Historical price data
            
        Returns:
            pandas.DataFrame: Backtest results with portfolio values
        """
        if price_data is None or price_data.empty:
            raise ValueError("Price data is empty or None")
        
        # Ensure the price_data has Close prices
        if 'Close' not in price_data.columns:
            raise ValueError("Price data must include 'Close' column")
        
        # Merge signals with price data if signals were provided
        if self.signal_data is not None:
            # Make sure data indexes match
            common_index = price_data.index.intersection(self.signal_data.index)
            price_data = price_data.loc[common_index]
            signals = self.signal_data.loc[common_index]
        else:
            # If no signals provided, create a buy and hold strategy
            signals = pd.DataFrame(index=price_data.index)
            signals['signal'] = 0
            signals.iloc[0, signals.columns.get_indexer(['signal'])[0]] = 1  # Buy signal on first day
        
        # Initialize portfolio metrics
        portfolio = pd.DataFrame(index=price_data.index)
        portfolio['price'] = price_data['Close']
        portfolio['signal'] = signals['signal']
        portfolio['position'] = portfolio['signal'].cumsum()  # Current position (number of units)
        
        # Calculate daily returns of price
        portfolio['price_return'] = portfolio['price'].pct_change()
        
        # Calculate strategy returns
        portfolio['strategy_return'] = portfolio['position'].shift(1) * portfolio['price_return']
        
        # Calculate equity curve
        portfolio['portfolio_value'] = (1 + portfolio['strategy_return']).cumprod() * self.initial_capital
        
        # Calculate portfolio daily returns
        portfolio['daily_returns'] = portfolio['portfolio_value'].pct_change()
        
        # Calculate drawdowns
        portfolio['peak'] = portfolio['portfolio_value'].cummax()
        portfolio['drawdown'] = (portfolio['portfolio_value'] - portfolio['peak']) / portfolio['peak']
        
        # Generate trade list
        self._generate_trade_list(portfolio)
        
        # Store the backtest results
        self.backtest_results = portfolio
        
        return portfolio
    
    def run_buy_and_hold(self, price_data):
        """
        Run a buy and hold backtest for comparison
        
        Args:
            price_data (pandas.DataFrame): Historical price data
            
        Returns:
            pandas.DataFrame: Backtest results for buy and hold strategy
        """
        if price_data is None or price_data.empty:
            raise ValueError("Price data is empty or None")
        
        # Create a dataframe for the buy and hold strategy
        bh_portfolio = pd.DataFrame(index=price_data.index)
        bh_portfolio['price'] = price_data['Close']
        
        # Calculate returns
        bh_portfolio['price_return'] = bh_portfolio['price'].pct_change()
        
        # For buy and hold, strategy return equals price return
        bh_portfolio['strategy_return'] = bh_portfolio['price_return']
        
        # Calculate equity curve
        bh_portfolio['portfolio_value'] = (1 + bh_portfolio['strategy_return']).cumprod() * self.initial_capital
        
        # Calculate portfolio daily returns
        bh_portfolio['daily_returns'] = bh_portfolio['portfolio_value'].pct_change()
        
        # Calculate drawdowns
        bh_portfolio['peak'] = bh_portfolio['portfolio_value'].cummax()
        bh_portfolio['drawdown'] = (bh_portfolio['portfolio_value'] - bh_portfolio['peak']) / bh_portfolio['peak']
        
        return bh_portfolio
    
    def _generate_trade_list(self, portfolio):
        """
        Generate a list of trades from the backtest results
        
        Args:
            portfolio (pandas.DataFrame): Backtest results
        """
        trades = []
        in_position = False
        entry_price = 0
        entry_date = None
        
        for date, row in portfolio.iterrows():
            # Check for entry signal
            if not in_position and row['signal'] == 1:
                in_position = True
                entry_price = row['price']
                entry_date = date
            
            # Check for exit signal
            elif in_position and row['signal'] == -1:
                exit_price = row['price']
                pnl = exit_price - entry_price
                pnl_pct = (exit_price / entry_price - 1) * 100
                
                # Record the trade
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'duration': (date - entry_date).days
                })
                
                in_position = False
                entry_price = 0
                entry_date = None
        
        # Handle open position at the end of the backtest
        if in_position:
            exit_price = portfolio['price'].iloc[-1]
            pnl = exit_price - entry_price
            pnl_pct = (exit_price / entry_price - 1) * 100
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': portfolio.index[-1],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'duration': (portfolio.index[-1] - entry_date).days,
                'status': 'open'
            })
        
        # Create a DataFrame from the trade list
        if trades:
            self.trade_list = pd.DataFrame(trades)
            
            # Mark winning and losing trades
            self.trade_list['outcome'] = ['win' if pnl > 0 else 'loss' for pnl in self.trade_list['pnl']]
        else:
            # Empty DataFrame if no trades
            self.trade_list = pd.DataFrame(columns=[
                'entry_date', 'exit_date', 'entry_price', 'exit_price',
                'pnl', 'pnl_pct', 'duration', 'outcome'
            ])
    
    def calculate_performance_metrics(self):
        """
        Calculate performance metrics from backtest results
        
        Returns:
            dict: Performance metrics
        """
        if self.backtest_results is None or self.backtest_results.empty:
            raise ValueError("No backtest results available")
        
        # Get key data from backtest results
        portfolio_value = self.backtest_results['portfolio_value']
        daily_returns = self.backtest_results['daily_returns'].dropna()
        drawdowns = self.backtest_results['drawdown']
        
        # Calculate performance metrics
        total_return = (portfolio_value.iloc[-1] / self.initial_capital - 1) * 100
        
        # Annualized return
        start_date = self.backtest_results.index[0]
        end_date = self.backtest_results.index[-1]
        years = (end_date - start_date).days / 365.25
        annualized_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100 if years > 0 else 0
        
        # Risk metrics
        volatility = daily_returns.std() * np.sqrt(252) * 100  # Annualized volatility
        max_drawdown = drawdowns.min() * 100
        
        # Calculate average drawdown
        negative_drawdowns = drawdowns[drawdowns < 0]
        avg_drawdown = negative_drawdowns.mean() * 100 if len(negative_drawdowns) > 0 else 0
        
        # Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Trade metrics
        if not self.trade_list.empty:
            win_trades = self.trade_list[self.trade_list['pnl'] > 0]
            loss_trades = self.trade_list[self.trade_list['pnl'] < 0]
            
            win_rate = len(win_trades) / len(self.trade_list) * 100 if len(self.trade_list) > 0 else 0
            
            avg_win = win_trades['pnl'].mean() if len(win_trades) > 0 else 0
            avg_loss = loss_trades['pnl'].mean() if len(loss_trades) > 0 else 0
            
            total_wins = win_trades['pnl'].sum() if len(win_trades) > 0 else 0
            total_losses = abs(loss_trades['pnl'].sum()) if len(loss_trades) > 0 else 0
            
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        return {
            'final_value': portfolio_value.iloc[-1],
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'win_rate': win_rate,
            'total_trades': len(self.trade_list),
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
    
    def get_trade_list(self):
        """
        Get the list of trades from the backtest
        
        Returns:
            pandas.DataFrame: List of trades
        """
        return self.trade_list
    
    def get_top_drawdowns(self, n=5):
        """
        Get the top N drawdown periods
        
        Args:
            n (int): Number of drawdown periods to return
            
        Returns:
            pandas.DataFrame: Top drawdown periods
        """
        if self.backtest_results is None or self.backtest_results.empty:
            raise ValueError("No backtest results available")
        
        # Get drawdown series
        drawdowns = self.backtest_results['drawdown']
        
        # Find drawdown periods
        # A drawdown period starts when drawdown becomes negative and ends when it returns to 0
        drawdown_starts = []
        drawdown_ends = []
        drawdown_depths = []
        
        in_drawdown = False
        start_idx = 0
        max_drawdown = 0
        
        for i, (date, drawdown) in enumerate(drawdowns.items()):
            if not in_drawdown and drawdown < 0:
                # Start of a drawdown period
                in_drawdown = True
                start_idx = i
                max_drawdown = drawdown
            elif in_drawdown:
                if drawdown < max_drawdown:
                    # Update maximum drawdown depth
                    max_drawdown = drawdown
                
                if drawdown >= 0:
                    # End of a drawdown period
                    drawdown_starts.append(drawdowns.index[start_idx])
                    drawdown_ends.append(date)
                    drawdown_depths.append(max_drawdown * 100)  # Convert to percentage
                    
                    in_drawdown = False
        
        # If still in drawdown at the end of the series
        if in_drawdown:
            drawdown_starts.append(drawdowns.index[start_idx])
            drawdown_ends.append(drawdowns.index[-1])
            drawdown_depths.append(max_drawdown * 100)  # Convert to percentage
        
        # Create a DataFrame of drawdown periods
        if drawdown_starts:
            drawdown_periods = pd.DataFrame({
                'start_date': drawdown_starts,
                'end_date': drawdown_ends,
                'max_drawdown': drawdown_depths,
                'duration_days': [(end - start).days for start, end in zip(drawdown_starts, drawdown_ends)]
            })
            
            # Sort by maximum drawdown depth and get top N
            drawdown_periods = drawdown_periods.sort_values('max_drawdown').head(n).reset_index(drop=True)
            
            return drawdown_periods
        else:
            return pd.DataFrame(columns=['start_date', 'end_date', 'max_drawdown', 'duration_days'])
    
    def calculate_beta(self, price_data):
        """
        Calculate beta of the strategy relative to gold prices
        
        Args:
            price_data (pandas.DataFrame): Historical price data
            
        Returns:
            float: Beta coefficient
        """
        if self.backtest_results is None or self.backtest_results.empty:
            raise ValueError("No backtest results available")
        
        # Market returns (gold price returns)
        market_returns = price_data['Close'].pct_change().dropna()
        
        # Strategy returns
        strategy_returns = self.backtest_results['daily_returns'].dropna()
        
        # Ensure the indexes match
        common_index = market_returns.index.intersection(strategy_returns.index)
        market_returns = market_returns.loc[common_index]
        strategy_returns = strategy_returns.loc[common_index]
        
        # Calculate covariance and variance
        covariance = strategy_returns.cov(market_returns)
        variance = market_returns.var()
        
        # Calculate beta
        beta = covariance / variance if variance > 0 else 0
        
        return beta
    
    def calculate_sortino_ratio(self, risk_free_rate=0):
        """
        Calculate Sortino ratio (using only downside deviation)
        
        Args:
            risk_free_rate (float): Annualized risk-free rate
            
        Returns:
            float: Sortino ratio
        """
        if self.backtest_results is None or self.backtest_results.empty:
            raise ValueError("No backtest results available")
        
        # Get daily returns
        daily_returns = self.backtest_results['daily_returns'].dropna()
        
        # Calculate annualized return
        annualized_return = self.calculate_performance_metrics()['annualized_return']
        
        # Calculate downside deviation (standard deviation of negative returns only)
        negative_returns = daily_returns[daily_returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252) * 100  # Annualized
        
        # Calculate Sortino ratio
        sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        return sortino_ratio
