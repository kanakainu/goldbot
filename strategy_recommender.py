"""
AI-Powered Trading Strategy Recommendation Engine

This module analyzes market conditions, historical performance, and price patterns
to recommend optimal trading strategies, parameters, and settings for gold trading.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Any, Tuple, Optional

# ML-related imports
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

# Local imports
from utils import process_data_for_model, calculate_metrics
import database as db

class StrategyRecommender:
    """
    AI-powered trading strategy recommendation engine that analyzes market conditions
    and historical performance to suggest optimal trading strategies and parameters
    """
    
    def __init__(self):
        # Supported strategies
        self.strategies = [
            'trend_following', 
            'mean_reversion', 
            'breakout', 
            'momentum', 
            'ml_based'
        ]
        
        # Parameter ranges for optimization
        self.param_ranges = {
            'signal_threshold': {
                'min': 0.005,  # 0.5%
                'max': 0.03,   # 3%
                'step': 0.005
            },
            'stop_loss': {
                'min': 0.01,   # 1%
                'max': 0.05,   # 5%
                'step': 0.005
            },
            'take_profit': {
                'min': 0.01,   # 1%
                'max': 0.05,   # 5%
                'step': 0.005
            },
            'feature_window': {
                'min': 7,
                'max': 30,
                'step': 1
            },
            'prediction_days': {
                'min': 1,
                'max': 14,
                'step': 1
            }
        }
        
        # Strategy classifiers
        self.market_condition_classifier = None
        self.strategy_performance_model = None
        
        # Market condition features
        self.market_conditions = {
            'trending_up': False,
            'trending_down': False,
            'sideways': False,
            'volatile': False,
            'low_volatility': False,
            'overbought': False,
            'oversold': False
        }
        
        # Performance tracking
        self.strategy_performance = {}
        
        # Load historical performances if available
        self._load_strategy_performance()
    
    def analyze_market_conditions(self, price_data: pd.DataFrame) -> Dict[str, bool]:
        """
        Analyze current market conditions based on price data
        
        Args:
            price_data (pd.DataFrame): Historical price data
            
        Returns:
            dict: Market condition indicators
        """
        if price_data is None or len(price_data) < 30:
            # Need sufficient data for analysis
            return self.market_conditions
        
        try:
            # Copy to avoid modifying original
            df = price_data.copy()
            
            # Ensure we have the required columns
            required_columns = ['Close', 'MA_20', 'Volatility', 'RSI']
            if not all(col in df.columns for col in required_columns):
                # Calculate necessary indicators if missing
                if 'MA_20' not in df.columns:
                    df['MA_20'] = df['Close'].rolling(window=20).mean()
                
                if 'Volatility' not in df.columns:
                    df['Daily_Return'] = df['Close'].pct_change()
                    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
                
                if 'RSI' not in df.columns:
                    # Calculate RSI
                    delta = df['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).fillna(0)
                    loss = (-delta.where(delta < 0, 0)).fillna(0)
                    
                    avg_gain = gain.rolling(window=14).mean()
                    avg_loss = loss.rolling(window=14).mean()
                    
                    rs = avg_gain / avg_loss
                    df['RSI'] = 100 - (100 / (1 + rs))
            
            # Drop NaN values
            df = df.dropna()
            
            if len(df) < 20:
                return self.market_conditions
            
            # Get recent data points for analysis
            recent = df.iloc[-20:]
            
            # Calculate price trends (using linear regression slope)
            x = np.arange(len(recent))
            y = recent['Close'].values
            slope = np.polyfit(x, y, 1)[0]
            
            # Calculate recent highs and lows
            high_point = recent['Close'].max()
            low_point = recent['Close'].min()
            latest_price = recent['Close'].iloc[-1]
            
            # Determine if there's a strong trend
            trend_strength = slope / recent['Close'].mean()
            strong_trend = abs(trend_strength) > 0.001  # Adjusted threshold
            
            # Analyze volatility
            avg_volatility = recent['Volatility'].mean()
            current_volatility = recent['Volatility'].iloc[-1]
            high_volatility = current_volatility > (1.5 * avg_volatility)
            low_volatility = current_volatility < (0.5 * avg_volatility)
            
            # Determine if market is sideways (no significant trend)
            price_range = (high_point - low_point) / low_point
            sideways = price_range < 0.03 and not strong_trend
            
            # Check RSI for overbought/oversold conditions
            current_rsi = recent['RSI'].iloc[-1]
            overbought = current_rsi > 70
            oversold = current_rsi < 30
            
            # Update market conditions
            self.market_conditions = {
                'trending_up': slope > 0 and strong_trend,
                'trending_down': slope < 0 and strong_trend,
                'sideways': sideways,
                'volatile': high_volatility,
                'low_volatility': low_volatility,
                'overbought': overbought,
                'oversold': oversold
            }
            
            return self.market_conditions
            
        except Exception as e:
            print(f"Error analyzing market conditions: {str(e)}")
            return self.market_conditions
    
    def get_recommended_strategies(self, price_data: pd.DataFrame, top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Get recommended trading strategies based on current market conditions
        
        Args:
            price_data (pd.DataFrame): Historical price data
            top_n (int): Number of top strategies to recommend
            
        Returns:
            list: List of recommended strategy configurations
        """
        # Analyze market conditions
        market_conditions = self.analyze_market_conditions(price_data)
        
        # Strategy recommendations based on market conditions
        recommendations = []
        
        # Rule-based recommendations based on market conditions
        if market_conditions['trending_up']:
            # In uptrend, trend following and momentum work well
            recommendations.append({
                'strategy': 'trend_following',
                'signal_threshold': 0.01,
                'stop_loss': 0.02,
                'take_profit': 0.03,
                'confidence': 0.85,
                'reason': 'Market is in an uptrend, trend following should be effective'
            })
            
            recommendations.append({
                'strategy': 'momentum',
                'signal_threshold': 0.015,
                'stop_loss': 0.025,
                'take_profit': 0.035,
                'confidence': 0.8,
                'reason': 'Strong upward momentum detected'
            })
            
            recommendations.append({
                'strategy': 'ml_based',
                'signal_threshold': 0.01,
                'stop_loss': 0.02,
                'take_profit': 0.03,
                'feature_window': 14,
                'prediction_days': 5,
                'confidence': 0.75,
                'reason': 'ML can capture complex patterns in the current trend'
            })
            
        elif market_conditions['trending_down']:
            # In downtrend, careful trend following and mean reversion at support
            recommendations.append({
                'strategy': 'trend_following',
                'signal_threshold': 0.015,
                'stop_loss': 0.015,
                'take_profit': 0.025,
                'confidence': 0.8,
                'reason': 'Market is in a downtrend, trend following with tighter stop loss'
            })
            
            recommendations.append({
                'strategy': 'ml_based',
                'signal_threshold': 0.015,
                'stop_loss': 0.02,
                'take_profit': 0.025,
                'feature_window': 21,
                'prediction_days': 3,
                'confidence': 0.75,
                'reason': 'ML can identify reversal points in downtrend'
            })
            
            recommendations.append({
                'strategy': 'mean_reversion',
                'signal_threshold': 0.02,
                'stop_loss': 0.015,
                'take_profit': 0.025,
                'confidence': 0.7,
                'reason': 'Look for oversold conditions in the downtrend for mean reversion'
            })
            
        elif market_conditions['sideways']:
            # In sideways market, mean reversion and breakout work well
            recommendations.append({
                'strategy': 'mean_reversion',
                'signal_threshold': 0.01,
                'stop_loss': 0.015,
                'take_profit': 0.02,
                'confidence': 0.85,
                'reason': 'Sideways market ideal for mean reversion strategies'
            })
            
            recommendations.append({
                'strategy': 'breakout',
                'signal_threshold': 0.02,
                'stop_loss': 0.015,
                'take_profit': 0.03,
                'confidence': 0.75,
                'reason': 'Watch for breakouts from the sideways range'
            })
            
            recommendations.append({
                'strategy': 'ml_based',
                'signal_threshold': 0.01,
                'stop_loss': 0.015,
                'take_profit': 0.02,
                'feature_window': 10,
                'prediction_days': 2,
                'confidence': 0.65,
                'reason': 'ML can detect subtle patterns in ranging markets'
            })
            
        elif market_conditions['volatile']:
            # In volatile markets, more conservative approaches
            recommendations.append({
                'strategy': 'ml_based',
                'signal_threshold': 0.02,
                'stop_loss': 0.025,
                'take_profit': 0.035,
                'feature_window': 21,
                'prediction_days': 3,
                'confidence': 0.75,
                'reason': 'ML can adapt to volatile conditions'
            })
            
            recommendations.append({
                'strategy': 'breakout',
                'signal_threshold': 0.025,
                'stop_loss': 0.02,
                'take_profit': 0.04,
                'confidence': 0.7,
                'reason': 'Capture volatile breakout moves with wider take profit'
            })
            
            recommendations.append({
                'strategy': 'trend_following',
                'signal_threshold': 0.025,
                'stop_loss': 0.03,
                'take_profit': 0.045,
                'confidence': 0.6,
                'reason': 'Follow established trends with wider stops in volatile markets'
            })
            
        elif market_conditions['low_volatility']:
            # In low volatility, look for breakouts and use tighter parameters
            recommendations.append({
                'strategy': 'breakout',
                'signal_threshold': 0.01,
                'stop_loss': 0.01,
                'take_profit': 0.025,
                'confidence': 0.85,
                'reason': 'Low volatility often precedes breakouts, use tight parameters'
            })
            
            recommendations.append({
                'strategy': 'mean_reversion',
                'signal_threshold': 0.01,
                'stop_loss': 0.01,
                'take_profit': 0.015,
                'confidence': 0.75,
                'reason': 'Mean reversion works well in low volatility environments'
            })
            
            recommendations.append({
                'strategy': 'ml_based',
                'signal_threshold': 0.01,
                'stop_loss': 0.015,
                'take_profit': 0.02,
                'feature_window': 14,
                'prediction_days': 3,
                'confidence': 0.7,
                'reason': 'ML can identify subtle patterns before volatility increases'
            })
            
        elif market_conditions['overbought']:
            # In overbought conditions, look for reversals
            recommendations.append({
                'strategy': 'mean_reversion',
                'signal_threshold': 0.015,
                'stop_loss': 0.015,
                'take_profit': 0.025,
                'confidence': 0.8,
                'reason': 'Market is overbought, mean reversion likely'
            })
            
            recommendations.append({
                'strategy': 'ml_based',
                'signal_threshold': 0.015,
                'stop_loss': 0.02,
                'take_profit': 0.025,
                'feature_window': 14,
                'prediction_days': 3,
                'confidence': 0.75,
                'reason': 'ML can predict when overbought conditions will lead to reversals'
            })
            
            recommendations.append({
                'strategy': 'trend_following',
                'signal_threshold': 0.02,
                'stop_loss': 0.015,
                'take_profit': 0.02,
                'confidence': 0.6,
                'reason': 'Continue following trend but with tighter parameters due to overbought conditions'
            })
            
        elif market_conditions['oversold']:
            # In oversold conditions, look for reversals
            recommendations.append({
                'strategy': 'mean_reversion',
                'signal_threshold': 0.015,
                'stop_loss': 0.015,
                'take_profit': 0.025,
                'confidence': 0.8,
                'reason': 'Market is oversold, mean reversion likely'
            })
            
            recommendations.append({
                'strategy': 'ml_based',
                'signal_threshold': 0.015,
                'stop_loss': 0.02,
                'take_profit': 0.025,
                'feature_window': 14,
                'prediction_days': 3,
                'confidence': 0.75,
                'reason': 'ML can predict when oversold conditions will lead to reversals'
            })
            
            recommendations.append({
                'strategy': 'momentum',
                'signal_threshold': 0.015,
                'stop_loss': 0.015,
                'take_profit': 0.025,
                'confidence': 0.7,
                'reason': 'Look for momentum shift from oversold conditions'
            })
            
        else:
            # Default recommendations if no specific conditions detected
            recommendations.append({
                'strategy': 'ml_based',
                'signal_threshold': 0.015,
                'stop_loss': 0.02,
                'take_profit': 0.025,
                'feature_window': 14,
                'prediction_days': 5,
                'confidence': 0.7,
                'reason': 'Machine learning adapts to current market conditions'
            })
            
            recommendations.append({
                'strategy': 'trend_following',
                'signal_threshold': 0.015,
                'stop_loss': 0.02,
                'take_profit': 0.03,
                'confidence': 0.65,
                'reason': 'Standard trend following parameters as baseline'
            })
            
            recommendations.append({
                'strategy': 'mean_reversion',
                'signal_threshold': 0.015,
                'stop_loss': 0.02,
                'take_profit': 0.025,
                'confidence': 0.6,
                'reason': 'Standard mean reversion parameters as alternative'
            })
        
        # Adjust recommendations based on historical performance if available
        self._adjust_recommendations_by_performance(recommendations)
        
        # Sort by confidence and return top N
        sorted_recommendations = sorted(recommendations, key=lambda x: x['confidence'], reverse=True)
        return sorted_recommendations[:top_n]
    
    def backtest_recommendations(self, price_data: pd.DataFrame, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run backtests on recommended strategies to validate and refine them
        
        Args:
            price_data (pd.DataFrame): Historical price data
            recommendations (list): List of strategy recommendations
            
        Returns:
            list: Updated recommendations with backtest results
        """
        from backtester import Backtester
        from trading_strategy import SignalGenerator
        
        if price_data is None or len(price_data) < 30:
            # Not enough data for meaningful backtesting
            return recommendations
        
        updated_recommendations = []
        
        for rec in recommendations:
            try:
                # Create signal generator for the strategy
                signal_generator = SignalGenerator(
                    threshold=rec['signal_threshold'],
                    stop_loss=rec['stop_loss'],
                    take_profit=rec['take_profit'],
                    strategy=rec['strategy']
                )
                
                # Generate signals
                signals = signal_generator.generate_signals(price_data)
                
                # Run backtest
                backtester = Backtester(initial_capital=10000, signal_data=signals)
                results = backtester.run_backtest(price_data)
                
                # Calculate performance metrics
                metrics = backtester.calculate_performance_metrics()
                
                # Update recommendation with backtest results
                rec_copy = rec.copy()
                rec_copy['backtest_results'] = {
                    'final_value': metrics.get('final_value', 0),
                    'total_return': metrics.get('total_return', 0),
                    'annualized_return': metrics.get('annualized_return', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0),
                    'win_rate': metrics.get('win_rate', 0),
                    'profit_factor': metrics.get('profit_factor', 0),
                    'trade_count': metrics.get('trade_count', 0)
                }
                
                # Adjust confidence based on backtest performance
                score = (
                    0.4 * (metrics.get('total_return', 0) / 10)  # Scale return to 0-1 range (assuming 10% is good)
                    + 0.2 * min(1, metrics.get('sharpe_ratio', 0) / 2)  # Scale Sharpe (2 or higher is good)
                    + 0.2 * (metrics.get('win_rate', 0) / 100)  # Win rate already 0-1
                    + 0.2 * (1 - min(1, metrics.get('max_drawdown', 0) / 20))  # Inverse of drawdown (20% max)
                )
                
                # Blend original confidence with backtest score
                rec_copy['confidence'] = 0.3 * rec['confidence'] + 0.7 * min(1, max(0, score))
                
                # Add reason based on backtest
                rec_copy['backtest_reason'] = f"Backtest: {metrics.get('total_return', 0):.2f}% return, {metrics.get('win_rate', 0):.2f}% win rate, Sharpe: {metrics.get('sharpe_ratio', 0):.2f}"
                
                updated_recommendations.append(rec_copy)
                
                # Update strategy performance history
                self._update_strategy_performance(rec['strategy'], metrics)
                
            except Exception as e:
                print(f"Error backtesting recommendation: {str(e)}")
                updated_recommendations.append(rec)  # Keep original if backtest fails
        
        # Sort by confidence again after backtesting
        return sorted(updated_recommendations, key=lambda x: x['confidence'], reverse=True)
    
    def train_strategy_selector(self, price_data: pd.DataFrame, performance_history: pd.DataFrame = None) -> None:
        """
        Train a machine learning model to select strategies based on market conditions
        
        Args:
            price_data (pd.DataFrame): Historical price data with market conditions
            performance_history (pd.DataFrame): Historical strategy performance (optional)
        """
        try:
            if price_data is None or len(price_data) < 60:
                print("Not enough data to train strategy selector")
                return
            
            # Generate market condition features for each day
            feature_data = []
            
            for i in range(30, len(price_data)):
                window = price_data.iloc[i-30:i]
                conditions = self.analyze_market_conditions(window)
                
                # Add price features
                row = {
                    'date': price_data.index[i],
                    'close': price_data['Close'].iloc[i],
                    'return_1d': price_data['Close'].iloc[i] / price_data['Close'].iloc[i-1] - 1,
                    'return_5d': price_data['Close'].iloc[i] / price_data['Close'].iloc[i-5] - 1 if i >= 5 else 0,
                    'volatility': price_data['Volatility'].iloc[i] if 'Volatility' in price_data.columns else 0,
                    'rsi': price_data['RSI'].iloc[i] if 'RSI' in price_data.columns else 50,
                }
                
                # Add market condition features
                for condition, value in conditions.items():
                    row[condition] = 1 if value else 0
                
                feature_data.append(row)
            
            features_df = pd.DataFrame(feature_data)
            
            # If we have performance history, use it to train a strategy selector
            if performance_history is not None and not performance_history.empty:
                # Join features with strategy performance
                training_data = pd.merge(
                    features_df,
                    performance_history,
                    on='date',
                    how='inner'
                )
                
                if not training_data.empty:
                    # Prepare X (features) and y (best strategy) for training
                    X = training_data.drop(['date', 'best_strategy', 'best_strategy_return'], axis=1)
                    y = training_data['best_strategy']
                    
                    # Train a classifier to predict the best strategy
                    self.market_condition_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
                    self.market_condition_classifier.fit(X, y)
                    
                    print("Strategy selector model trained successfully")
            
        except Exception as e:
            print(f"Error training strategy selector: {str(e)}")
    
    def _adjust_recommendations_by_performance(self, recommendations: List[Dict[str, Any]]) -> None:
        """
        Adjust strategy recommendations based on historical performance
        
        Args:
            recommendations (list): List of strategy recommendations to adjust
        """
        if not self.strategy_performance:
            return  # No performance data available
        
        # Get strategy rankings based on average performance
        rankings = {}
        for strategy, metrics in self.strategy_performance.items():
            if not metrics:
                continue
                
            # Calculate average metrics
            avg_return = sum(m.get('total_return', 0) for m in metrics) / len(metrics)
            avg_sharpe = sum(m.get('sharpe_ratio', 0) for m in metrics) / len(metrics)
            avg_win_rate = sum(m.get('win_rate', 0) for m in metrics) / len(metrics)
            
            # Composite score
            rankings[strategy] = (avg_return * 0.5) + (avg_sharpe * 0.3) + (avg_win_rate * 0.2)
        
        # Sort strategies by performance score
        sorted_strategies = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
        
        # Adjust confidence based on historical performance
        for rec in recommendations:
            strategy = rec['strategy']
            
            # Find strategy rank
            rank = next((i for i, (s, _) in enumerate(sorted_strategies) if s == strategy), None)
            
            if rank is not None:
                # Adjust confidence based on historical rank (top strategies get a boost)
                rank_factor = 1 - (rank / max(1, len(sorted_strategies)))
                rec['confidence'] = 0.7 * rec['confidence'] + 0.3 * rank_factor
    
    def _update_strategy_performance(self, strategy: str, metrics: Dict[str, float]) -> None:
        """
        Update strategy performance history
        
        Args:
            strategy (str): Strategy name
            metrics (dict): Performance metrics
        """
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = []
        
        # Add date to metrics
        metrics_with_date = metrics.copy()
        metrics_with_date['date'] = datetime.now().strftime("%Y-%m-%d")
        
        # Add to performance history
        self.strategy_performance[strategy].append(metrics_with_date)
        
        # Keep only recent history (last 10 entries)
        if len(self.strategy_performance[strategy]) > 10:
            self.strategy_performance[strategy] = self.strategy_performance[strategy][-10:]
        
        # Save to file
        self._save_strategy_performance()
    
    def _save_strategy_performance(self) -> None:
        """
        Save strategy performance history to file
        """
        try:
            # Convert datetime objects to strings for JSON serialization
            serializable_performance = {}
            for strategy, metrics_list in self.strategy_performance.items():
                serializable_performance[strategy] = []
                for metrics in metrics_list:
                    metrics_copy = metrics.copy()
                    # Convert any non-serializable objects
                    for key, value in metrics_copy.items():
                        if isinstance(value, (datetime, np.float32, np.float64, np.int32, np.int64)):
                            metrics_copy[key] = float(value)
                    serializable_performance[strategy].append(metrics_copy)
            
            with open('strategy_performance.json', 'w') as f:
                json.dump(serializable_performance, f)
        except Exception as e:
            print(f"Error saving strategy performance: {str(e)}")
    
    def _load_strategy_performance(self) -> None:
        """
        Load strategy performance history from file
        """
        try:
            if os.path.exists('strategy_performance.json'):
                with open('strategy_performance.json', 'r') as f:
                    self.strategy_performance = json.load(f)
        except Exception as e:
            print(f"Error loading strategy performance: {str(e)}")
            self.strategy_performance = {}
    
    def get_market_condition_summary(self) -> Dict[str, Any]:
        """
        Get a human-readable summary of current market conditions
        
        Returns:
            dict: Market condition summary with explanations
        """
        conditions_map = {
            'trending_up': {
                'label': 'Uptrend',
                'explanation': 'The market is in a sustained upward trend, indicating bullish momentum.',
                'recommended_strategies': ['trend_following', 'momentum']
            },
            'trending_down': {
                'label': 'Downtrend',
                'explanation': 'The market is in a sustained downward trend, indicating bearish pressure.',
                'recommended_strategies': ['trend_following (short)', 'mean_reversion at support']
            },
            'sideways': {
                'label': 'Sideways/Range-bound',
                'explanation': 'The market is moving sideways within a defined range, lacking clear direction.',
                'recommended_strategies': ['mean_reversion', 'breakout']
            },
            'volatile': {
                'label': 'Highly Volatile',
                'explanation': 'The market is experiencing higher than normal price fluctuations.',
                'recommended_strategies': ['breakout with wider stops', 'ml_based']
            },
            'low_volatility': {
                'label': 'Low Volatility',
                'explanation': 'The market is experiencing unusually low price fluctuations, often preceding a breakout.',
                'recommended_strategies': ['breakout with tight stops', 'mean_reversion']
            },
            'overbought': {
                'label': 'Overbought',
                'explanation': 'Technical indicators suggest the market has risen too far too quickly and may be due for a correction.',
                'recommended_strategies': ['mean_reversion', 'careful trend following']
            },
            'oversold': {
                'label': 'Oversold',
                'explanation': 'Technical indicators suggest the market has fallen too far too quickly and may be due for a bounce.',
                'recommended_strategies': ['mean_reversion', 'momentum on reversal']
            }
        }
        
        # Build summary based on active conditions
        active_conditions = [
            conditions_map[condition] 
            for condition, is_active in self.market_conditions.items() 
            if is_active
        ]
        
        if not active_conditions:
            return {
                'summary': 'No distinct market conditions detected.',
                'explanations': ['The market is not showing any strong characteristics at the moment.'],
                'recommended_strategy_types': ['ml_based', 'trend_following']
            }
        
        # Create summary
        summary = ' and '.join([cond['label'] for cond in active_conditions])
        explanations = [cond['explanation'] for cond in active_conditions]
        
        # Get unique recommended strategies
        recommended_strategies = []
        for cond in active_conditions:
            recommended_strategies.extend(cond['recommended_strategies'])
        recommended_strategies = list(set(recommended_strategies))
        
        return {
            'summary': summary,
            'explanations': explanations,
            'recommended_strategy_types': recommended_strategies
        }


# Helper function to get recommendations
def get_strategy_recommendations(price_data, top_n=3, run_backtest=True):
    """
    Get AI-powered strategy recommendations for the current market conditions
    
    Args:
        price_data (pd.DataFrame): Historical gold price data
        top_n (int): Number of recommendations to return
        run_backtest (bool): Whether to run backtest on recommendations
        
    Returns:
        list: Recommended trading strategies with parameters
    """
    recommender = StrategyRecommender()
    recommendations = recommender.get_recommended_strategies(price_data, top_n=top_n)
    
    if run_backtest:
        recommendations = recommender.backtest_recommendations(price_data, recommendations)
    
    # Get market condition summary
    market_summary = recommender.get_market_condition_summary()
    
    return {
        'recommendations': recommendations,
        'market_conditions': recommender.market_conditions,
        'market_summary': market_summary
    }