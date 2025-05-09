"""
OpenAI-powered Gold Trading Strategy Advisor

This module uses OpenAI to generate trading strategy recommendations and market analysis
based on current gold price data and market conditions.
"""

import json
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
from openai import OpenAI

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# Initialize OpenAI client only if API key is available
openai = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

class OpenAIAdvisor:
    """
    Uses OpenAI to provide trading strategy recommendations and market insights
    """
    
    def __init__(self):
        self.openai_client = openai if OPENAI_API_KEY else None
        self.model = "gpt-4o"  # Use the latest model
        
        # Supported strategies
        self.strategies = [
            'trend_following', 
            'mean_reversion', 
            'breakout', 
            'momentum', 
            'ml_based'
        ]
        
        # Parameter ranges
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
            }
        }
    
    def is_available(self) -> bool:
        """
        Check if OpenAI API is available
        
        Returns:
            bool: True if OpenAI API is available, False otherwise
        """
        return self.openai_client is not None
    
    def get_market_analysis(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a detailed market analysis for gold using OpenAI
        
        Args:
            price_data (pd.DataFrame): Historical gold price data
            
        Returns:
            dict: Market analysis with insights and recommendations
        """
        if not self.is_available():
            return {
                "error": "OpenAI API key not available. Please set the OPENAI_API_KEY environment variable."
            }
        
        try:
            # Prepare recent price data for analysis
            recent_data = price_data.tail(30).copy()  # Last 30 days
            
            # Calculate daily returns
            recent_data['daily_return'] = recent_data['Close'].pct_change() * 100
            
            # Calculate key metrics
            current_price = recent_data['Close'].iloc[-1]
            prev_day_price = recent_data['Close'].iloc[-2]
            daily_change = ((current_price / prev_day_price) - 1) * 100
            
            # Calculate 7-day and 30-day changes
            price_7d_ago = recent_data['Close'].iloc[-8] if len(recent_data) >= 8 else recent_data['Close'].iloc[0]
            price_30d_ago = recent_data['Close'].iloc[0]
            
            change_7d = ((current_price / price_7d_ago) - 1) * 100
            change_30d = ((current_price / price_30d_ago) - 1) * 100
            
            # Calculate volatility
            volatility = recent_data['daily_return'].std()
            
            # Get price range
            price_high = recent_data['Close'].max()
            price_low = recent_data['Close'].min()
            
            # Calculate moving averages if available
            ma20 = recent_data['MA_20'].iloc[-1] if 'MA_20' in recent_data.columns else None
            ma50 = recent_data['MA_50'].iloc[-1] if 'MA_50' in recent_data.columns else None
            
            # Calculate RSI if available
            rsi = recent_data['RSI'].iloc[-1] if 'RSI' in recent_data.columns else None
            
            # Prepare the prompt for OpenAI
            prompt = f"""
            You are an expert gold trading analyst with decades of experience. Analyze the following gold price data and provide detailed market insights.

            Current Gold Price: ${current_price:.2f}
            Daily Change: {daily_change:.2f}%
            7-Day Change: {change_7d:.2f}%
            30-Day Change: {change_30d:.2f}%
            30-Day Volatility: {volatility:.2f}%
            30-Day High: ${price_high:.2f}
            30-Day Low: ${price_low:.2f}
            """
            
            if ma20 is not None and ma50 is not None:
                prompt += f"""
                20-Day Moving Average: ${ma20:.2f}
                50-Day Moving Average: ${ma50:.2f}
                Price relative to 20-MA: {((current_price/ma20)-1)*100:.2f}%
                Price relative to 50-MA: {((current_price/ma50)-1)*100:.2f}%
                """
            
            if rsi is not None:
                prompt += f"RSI (14): {rsi:.2f}\n"
            
            prompt += """
            Based on this data, provide a detailed analysis with the following sections:
            1. Market Summary: Brief overview of current gold market conditions.
            2. Technical Analysis: Key technical indicators and what they suggest.
            3. Market Sentiment: Current market sentiment for gold.
            4. Potential Catalysts: Key events or factors that could impact gold prices.
            5. Trading Strategy Recommendations: Recommended strategies and parameters for current market conditions.
            
            For the Trading Strategy Recommendations section, suggest the top 3 most suitable strategies from this list:
            - Trend Following
            - Mean Reversion
            - Breakout
            - Momentum
            - ML-Based

            For each strategy, recommend specific parameter values:
            - Signal Threshold (0.5% to 3%)
            - Stop Loss (1% to 5%)
            - Take Profit (1% to 5%)
            
            Output your analysis in JSON format with the following structure:
            {
                "market_summary": "Brief summary of current gold market conditions",
                "technical_analysis": "Detailed technical analysis with key indicators",
                "market_sentiment": "Analysis of current market sentiment",
                "potential_catalysts": "Key events or factors that could impact gold prices",
                "strategy_recommendations": [
                    {
                        "strategy": "Strategy name",
                        "signal_threshold": 0.01,
                        "stop_loss": 0.02,
                        "take_profit": 0.03,
                        "confidence": 0.85,
                        "reasoning": "Explanation for why this strategy is recommended"
                    }
                ]
            }
            """
            
            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert gold trading analyst and advisor."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            analysis = json.loads(response.choices[0].message.content)
            
            # Add timestamp
            analysis['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            analysis['current_price'] = current_price
            
            return analysis
            
        except Exception as e:
            return {
                "error": f"Error generating market analysis: {str(e)}"
            }
    
    def get_strategy_explanation(self, strategy: str) -> str:
        """
        Get a detailed explanation of a trading strategy from OpenAI
        
        Args:
            strategy (str): Trading strategy name
            
        Returns:
            str: Detailed explanation of the strategy
        """
        if not self.is_available():
            return "OpenAI API key not available. Please set the OPENAI_API_KEY environment variable."
        
        try:
            # Prepare the prompt
            prompt = f"""
            Provide a detailed explanation of the '{strategy}' trading strategy for gold trading. Include:
            
            1. How the strategy works
            2. When it's most effective (market conditions)
            3. Key parameters and how to optimize them
            4. Typical risk/reward characteristics
            5. Strengths and weaknesses
            6. Example scenario where this strategy would be effective
            
            Make your explanation informative, concise, and practical.
            """
            
            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert gold trading strategist and educator."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Return the explanation
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating strategy explanation: {str(e)}"
    
    def get_parameter_optimization_advice(self, strategy: str, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get advice on optimizing parameters for a specific strategy based on current market conditions
        
        Args:
            strategy (str): Trading strategy name
            price_data (pd.DataFrame): Historical gold price data
            
        Returns:
            dict: Parameter optimization advice
        """
        if not self.is_available():
            return {
                "error": "OpenAI API key not available. Please set the OPENAI_API_KEY environment variable."
            }
        
        try:
            # Prepare recent price data for analysis
            recent_data = price_data.tail(30).copy()  # Last 30 days
            
            # Calculate key metrics
            current_price = recent_data['Close'].iloc[-1]
            avg_daily_change = recent_data['Close'].pct_change().abs().mean() * 100
            
            # Calculate volatility
            volatility = recent_data['Close'].pct_change().std() * 100
            
            # Get RSI if available
            rsi = recent_data['RSI'].iloc[-1] if 'RSI' in recent_data.columns else None
            
            # Trend strength (using linear regression slope)
            import numpy as np
            x = np.arange(len(recent_data))
            y = recent_data['Close'].values
            slope = np.polyfit(x, y, 1)[0]
            trend_strength = slope / recent_data['Close'].mean() * 100  # Normalized
            
            # Prepare the prompt
            prompt = f"""
            You are optimizing parameters for a '{strategy}' gold trading strategy. Current market conditions:
            
            Current Gold Price: ${current_price:.2f}
            Average Daily Change: {avg_daily_change:.2f}%
            30-Day Volatility: {volatility:.2f}%
            Trend Strength: {trend_strength:.4f}% per day
            """
            
            if rsi is not None:
                prompt += f"RSI (14): {rsi:.2f}\n"
            
            prompt += f"""
            Based on these conditions, recommend optimal parameter values for the {strategy} strategy.
            Parameter ranges to consider:
            - Signal Threshold: 0.5% to 3%
            - Stop Loss: 1% to 5%
            - Take Profit: 1% to 5%
            
            Provide your recommendations and explanations in JSON format with the following structure:
            {{
                "recommended_parameters": {{
                    "signal_threshold": 0.015,
                    "stop_loss": 0.02,
                    "take_profit": 0.03
                }},
                "explanations": {{
                    "signal_threshold": "Explanation for signal threshold recommendation",
                    "stop_loss": "Explanation for stop loss recommendation",
                    "take_profit": "Explanation for take profit recommendation"
                }},
                "market_condition_summary": "Brief summary of current market conditions",
                "optimization_strategy": "Overall approach to parameter optimization for this market"
            }}
            """
            
            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert trading strategy optimizer."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            advice = json.loads(response.choices[0].message.content)
            
            # Add timestamp
            advice['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            advice['strategy'] = strategy
            
            return advice
            
        except Exception as e:
            return {
                "error": f"Error generating parameter optimization advice: {str(e)}"
            }

# Helper function to get OpenAI-powered recommendations
def get_openai_recommendations(price_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Get OpenAI-powered trading strategy recommendations
    
    Args:
        price_data (pd.DataFrame): Historical gold price data
        
    Returns:
        dict: Market analysis and strategy recommendations
    """
    advisor = OpenAIAdvisor()
    
    if not advisor.is_available():
        return {
            "error": "OpenAI API key not available. Please set the OPENAI_API_KEY environment variable.",
            "is_available": False
        }
    
    analysis = advisor.get_market_analysis(price_data)
    analysis['is_available'] = True
    
    return analysis

# Helper function to get strategy explanation
def get_strategy_explanation(strategy: str) -> str:
    """
    Get a detailed explanation of a trading strategy
    
    Args:
        strategy (str): Trading strategy name
        
    Returns:
        str: Detailed explanation of the strategy
    """
    advisor = OpenAIAdvisor()
    return advisor.get_strategy_explanation(strategy)

# Helper function to get parameter optimization advice
def get_parameter_optimization_advice(strategy: str, price_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Get advice on optimizing parameters for a specific strategy
    
    Args:
        strategy (str): Trading strategy name
        price_data (pd.DataFrame): Historical gold price data
        
    Returns:
        dict: Parameter optimization advice
    """
    advisor = OpenAIAdvisor()
    return advisor.get_parameter_optimization_advice(strategy, price_data)