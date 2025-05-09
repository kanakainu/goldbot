import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
from datetime import timedelta
import time
import os
import random

from data_fetcher import fetch_realtime_gold_price, fetch_historical_gold_data, initialize_mt5_connector
from model import GoldPricePredictor
from backtester import Backtester
from trading_strategy import SignalGenerator
from auto_trader import AutoTrader
from visualization import (
    plot_price_chart, plot_returns, plot_prediction_vs_actual, plot_signals, plot_performance_metrics,
    create_animated_performance_dashboard, create_animated_performance_chart, create_pl_distribution_chart, 
    create_trade_breakdown_chart, create_monthly_performance_heatmap, create_key_metrics_indicators
)
from utils import calculate_metrics, process_data_for_model, format_currency, format_percentage
import database as db
from strategy_recommender import get_strategy_recommendations
from openai_advisor import get_openai_recommendations, get_strategy_explanation, get_parameter_optimization_advice

# Check if MT5 module is available
try:
    from mt5_connector import MT5_AVAILABLE
except ImportError:
    MT5_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Gold Trading Bot",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("AI-Powered Gold Trading Bot")
st.markdown("""
This application analyzes gold market data, generates trading signals using machine learning, 
and provides performance analytics to assist in gold trading decisions.
""")

# Sidebar for controls
st.sidebar.header("Trading Bot Controls")

# Date range selector
st.sidebar.subheader("Historical Data Range")
end_date = datetime.datetime.now().date()
start_date = end_date - datetime.timedelta(days=365)  # Default to 1 year

start_date_input = st.sidebar.date_input("Start Date", start_date)
end_date_input = st.sidebar.date_input("End Date", end_date)

if start_date_input >= end_date_input:
    st.sidebar.error("The end date must be after the start date.")

# Model parameters
st.sidebar.subheader("Model Parameters")
prediction_days = st.sidebar.slider("Prediction Window (Days)", 1, 30, 7)
feature_days = st.sidebar.slider("Feature Window (Days)", 5, 60, 14)

# Trading strategy parameters
st.sidebar.subheader("Trading Strategy Parameters")
signal_threshold = st.sidebar.slider("Signal Threshold (%)", 0.5, 5.0, 1.0) / 100
stop_loss = st.sidebar.slider("Stop Loss (%)", 1.0, 10.0, 2.0) / 100
take_profit = st.sidebar.slider("Take Profit (%)", 1.0, 10.0, 3.0) / 100

# MetaTrader 5 integration
st.sidebar.subheader("MetaTrader 5 Connection")
mt5_section_expanded = st.sidebar.checkbox("Configure MT5 Connection", value=False)

if mt5_section_expanded:
    # Get MT5 credentials from environment variables or session state
    mt5_api_url = os.environ.get("MT5_API_URL", "")
    mt5_api_key = os.environ.get("MT5_API_KEY", "")
    mt5_account = os.environ.get("MT5_ACCOUNT", "")
    
    # Set up form for MT5 credentials
    with st.sidebar.form("mt5_connection_form"):
        st.write("Enter MetaTrader 5 Connection Details")
        new_mt5_api_url = st.text_input("MT5 API URL", value=mt5_api_url)
        new_mt5_api_key = st.text_input("MT5 API Key", value=mt5_api_key, type="password")
        new_mt5_account = st.text_input("MT5 Account Number (optional)", value=mt5_account)
        
        # Submit button
        submit_button = st.form_submit_button("Connect to MT5")
        
        if submit_button:
            # Save credentials to environment variables
            os.environ["MT5_API_URL"] = new_mt5_api_url
            os.environ["MT5_API_KEY"] = new_mt5_api_key
            if new_mt5_account:
                os.environ["MT5_ACCOUNT"] = new_mt5_account
            
            # Initialize MT5 connector
            if initialize_mt5_connector():
                st.success("Successfully connected to MetaTrader 5!")
            else:
                st.error("Failed to connect to MetaTrader 5. Please check your credentials.")
    
    # Show MT5 connection status
    if "mt5_connected" not in st.session_state:
        st.session_state.mt5_connected = False
    
    if MT5_AVAILABLE:
        conn_status = initialize_mt5_connector()
        st.session_state.mt5_connected = conn_status
        
        if st.session_state.mt5_connected:
            st.sidebar.success("MT5 Status: Connected")
        else:
            st.sidebar.error("MT5 Status: Disconnected")
    else:
        st.sidebar.warning("MT5 Integration: Not Available")
        st.sidebar.info("MetaTrader 5 module is not installed or available in this environment.")

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Live Dashboard", "Historical Analysis", "Backtesting", "Performance Metrics", "Database Analytics", "Auto Trading", "Strategy Advisor", "Performance Dashboard"])

with tab1:
    st.header("Live Gold Trading Dashboard")
    
    # Fetch real-time gold price
    try:
        current_price, timestamp = fetch_realtime_gold_price()
        
        # Display current gold price
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Gold Price (USD/oz)", f"${current_price:.2f}")
        with col2:
            st.write(f"Last Updated: {timestamp}")
            
        # Get historical data for the chart
        short_term_data = fetch_historical_gold_data(
            (datetime.datetime.now() - datetime.timedelta(days=30)).date(),
            datetime.datetime.now().date()
        )
        
        if short_term_data is not None:
            # Create price chart
            st.subheader("Gold Price - Last 30 Days")
            fig = plot_price_chart(short_term_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Generate trading signals
            signal_gen = SignalGenerator(signal_threshold, stop_loss, take_profit)
            signals = signal_gen.generate_signals(short_term_data)
            
            # Show recent signals
            st.subheader("Recent Trading Signals")
            
            # Try to get signals from database first
            try:
                db_signals = db.get_trading_signals(limit=5)
                has_db_signals = not db_signals.empty
            except Exception as e:
                print(f"Error getting signals from database: {str(e)}")
                has_db_signals = False
            
            if has_db_signals:
                # Show signals from database
                for _, row in db_signals.iterrows():
                    signal_type = row.get('signal_type', 0)
                    timestamp = row.get('timestamp')
                    price = row.get('price', 0)
                    strategy = row.get('strategy', 'unknown')
                    
                    if signal_type == 1:
                        st.success(f"BUY Signal on {timestamp.strftime('%Y-%m-%d %H:%M')} at ${price:.2f} ({strategy})")
                    elif signal_type == -1:
                        st.error(f"SELL Signal on {timestamp.strftime('%Y-%m-%d %H:%M')} at ${price:.2f} ({strategy})")
                    else:
                        st.info(f"HOLD on {timestamp.strftime('%Y-%m-%d %H:%M')} at ${price:.2f} ({strategy})")
            else:
                # Fall back to generated signals if no database signals
                recent_signals = signals.tail(5).reset_index()
                if not recent_signals.empty:
                    for _, row in recent_signals.iterrows():
                        if row['signal'] == 1:
                            st.success(f"BUY Signal on {row['Date']} at ${row['Close']:.2f}")
                        elif row['signal'] == -1:
                            st.error(f"SELL Signal on {row['Date']} at ${row['Close']:.2f}")
                        else:
                            st.info(f"HOLD on {row['Date']} at ${row['Close']:.2f}")
                else:
                    st.info("No recent signals generated")
            
            # Show signal chart
            st.subheader("Trading Signals - Last 30 Days")
            signal_fig = plot_signals(short_term_data, signals)
            st.plotly_chart(signal_fig, use_container_width=True)
        else:
            st.error("Failed to fetch historical data for live dashboard")
    
    except Exception as e:
        st.error(f"Error in Live Dashboard: {str(e)}")
        st.info("Please check your internet connection or try again later.")

with tab2:
    st.header("Historical Gold Price Analysis")
    
    try:
        # Fetch historical data
        if st.button("Fetch Historical Data"):
            with st.spinner("Fetching historical gold price data..."):
                historical_data = fetch_historical_gold_data(start_date_input, end_date_input)
                
                if historical_data is not None:
                    # Show data
                    st.subheader("Historical Gold Price Data")
                    st.dataframe(historical_data.head())
                    
                    # Plot historical price
                    st.subheader("Historical Gold Price Chart")
                    fig = plot_price_chart(historical_data)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate and display key metrics
                    metrics = calculate_metrics(historical_data)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Avg. Daily Return", f"{metrics['avg_return']:.3f}%")
                    with col2:
                        st.metric("Volatility", f"{metrics['volatility']:.3f}%")
                    with col3:
                        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
                    with col4:
                        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}")
                    
                    # Plot returns distribution
                    st.subheader("Returns Distribution")
                    returns_fig = plot_returns(historical_data)
                    st.plotly_chart(returns_fig, use_container_width=True)
                    
                    # AI Price Prediction
                    st.subheader("AI Price Prediction Model")
                    
                    with st.spinner("Training AI prediction model..."):
                        # Process data for model
                        X_train, y_train, X_test, y_test, test_dates = process_data_for_model(
                            historical_data, feature_days, prediction_days
                        )
                        
                        if X_train is not None:
                            # Initialize and train the model
                            predictor = GoldPricePredictor()
                            predictor.train(X_train, y_train)
                            
                            # Make predictions
                            predictions = predictor.predict(X_test)
                            
                            # Plot predictions vs actual
                            pred_fig = plot_prediction_vs_actual(test_dates, y_test, predictions)
                            st.plotly_chart(pred_fig, use_container_width=True)
                            
                            # Model performance metrics
                            mse, mae, r2 = predictor.evaluate(y_test, predictions)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Mean Squared Error", f"{mse:.4f}")
                            with col2:
                                st.metric("Mean Absolute Error", f"{mae:.4f}")
                            with col3:
                                st.metric("R² Score", f"{r2:.4f}")
                        else:
                            st.warning("Not enough data to train the prediction model")
                else:
                    st.error("Failed to fetch historical data")
    
    except Exception as e:
        st.error(f"Error in Historical Analysis: {str(e)}")

with tab3:
    st.header("Strategy Backtesting")
    
    st.markdown("""
    Backtest your trading strategy against historical gold price data to evaluate its performance.
    Adjust the parameters in the sidebar to optimize your strategy.
    """)
    
    try:
        if st.button("Run Backtest"):
            with st.spinner("Running backtest..."):
                # Fetch data for backtesting
                backtest_data = fetch_historical_gold_data(start_date_input, end_date_input)
                
                if backtest_data is not None and len(backtest_data) > 0:
                    # Generate signals for backtesting
                    signal_gen = SignalGenerator(signal_threshold, stop_loss, take_profit)
                    signals = signal_gen.generate_signals(backtest_data)
                    
                    # Initialize backtester
                    initial_capital = 10000  # $10,000 initial capital
                    backtester = Backtester(initial_capital, signals)
                    
                    # Run backtest
                    backtest_results = backtester.run_backtest(backtest_data)
                    
                    # Display backtest results
                    st.subheader("Backtest Results")
                    
                    # Performance metrics
                    metrics = backtester.calculate_performance_metrics()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Final Portfolio Value", f"${metrics['final_value']:.2f}", 
                                f"{((metrics['final_value'] / initial_capital) - 1) * 100:.2f}%")
                    with col2:
                        st.metric("Total Return", f"{metrics['total_return']:.2f}%")
                    with col3:
                        st.metric("Annualized Return", f"{metrics['annualized_return']:.2f}%")
                    with col4:
                        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
                    with col2:
                        st.metric("Win Rate", f"{metrics['win_rate']:.2f}%")
                    with col3:
                        st.metric("Total Trades", f"{metrics['total_trades']}")
                    with col4:
                        st.metric("Profit Factor", f"{metrics['profit_factor']:.3f}")
                    
                    # Plot portfolio value over time
                    st.subheader("Portfolio Value Over Time")
                    portfolio_fig = plot_performance_metrics(backtest_results)
                    st.plotly_chart(portfolio_fig, use_container_width=True)
                    
                    # Display signals on price chart
                    st.subheader("Trading Signals")
                    signal_fig = plot_signals(backtest_data, signals)
                    st.plotly_chart(signal_fig, use_container_width=True)
                    
                    # Show trade list
                    st.subheader("Trade List")
                    trade_list = backtester.get_trade_list()
                    if not trade_list.empty:
                        st.dataframe(trade_list)
                    else:
                        st.info("No trades executed during the backtest period")
                else:
                    st.error("Failed to fetch data for backtesting")
    
    except Exception as e:
        st.error(f"Error in Backtesting: {str(e)}")

with tab4:
    st.header("Performance Analytics")
    
    st.markdown("""
    This section provides detailed performance metrics and analytics for your gold trading strategy.
    """)
    
    try:
        # Choose comparison options
        strategy_option = st.selectbox(
            "Select Performance View", 
            ["Strategy vs. Buy & Hold", "Risk Analysis", "Drawdown Analysis"]
        )
        
        # Add auto trading strategy selection in sidebar
        st.sidebar.subheader("Auto Trading Settings")
        strategy_type = st.sidebar.selectbox(
            "Trading Strategy Type",
            ["trend_following", "mean_reversion", "ml_based"],
            index=0
        )
        
        if st.button("Generate Performance Report"):
            with st.spinner("Generating performance analytics..."):
                # Fetch data
                performance_data = fetch_historical_gold_data(start_date_input, end_date_input)
                
                if performance_data is not None and len(performance_data) > 0:
                    # Generate signals
                    signal_gen = SignalGenerator(signal_threshold, stop_loss, take_profit)
                    signals = signal_gen.generate_signals(performance_data)
                    
                    # Run backtest for strategy
                    initial_capital = 10000
                    backtester = Backtester(initial_capital, signals)
                    backtest_results = backtester.run_backtest(performance_data)
                    strategy_metrics = backtester.calculate_performance_metrics()
                    
                    # Calculate buy & hold metrics
                    buy_hold_backtester = Backtester(initial_capital)
                    buy_hold_results = buy_hold_backtester.run_buy_and_hold(performance_data)
                    buy_hold_metrics = buy_hold_backtester.calculate_performance_metrics()
                    
                    if strategy_option == "Strategy vs. Buy & Hold":
                        # Compare strategy vs buy & hold
                        st.subheader("Strategy vs. Buy & Hold Performance")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Strategy Total Return", f"{strategy_metrics['total_return']:.2f}%")
                            st.metric("Strategy Annualized Return", f"{strategy_metrics['annualized_return']:.2f}%")
                            st.metric("Strategy Sharpe Ratio", f"{strategy_metrics['sharpe_ratio']:.3f}")
                            st.metric("Strategy Max Drawdown", f"{strategy_metrics['max_drawdown']:.2f}%")
                        
                        with col2:
                            st.metric("Buy & Hold Total Return", f"{buy_hold_metrics['total_return']:.2f}%")
                            st.metric("Buy & Hold Annualized Return", f"{buy_hold_metrics['annualized_return']:.2f}%")
                            st.metric("Buy & Hold Sharpe Ratio", f"{buy_hold_metrics['sharpe_ratio']:.3f}")
                            st.metric("Buy & Hold Max Drawdown", f"{buy_hold_metrics['max_drawdown']:.2f}%")
                        
                        # Plot comparison
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=backtest_results.index, 
                            y=backtest_results['portfolio_value'],
                            mode='lines',
                            name='Trading Strategy'
                        ))
                        fig.add_trace(go.Scatter(
                            x=buy_hold_results.index, 
                            y=buy_hold_results['portfolio_value'],
                            mode='lines',
                            name='Buy & Hold'
                        ))
                        fig.update_layout(
                            title='Performance Comparison',
                            xaxis_title='Date',
                            yaxis_title='Portfolio Value ($)',
                            height=500
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif strategy_option == "Risk Analysis":
                        # Risk metrics
                        st.subheader("Risk Analysis")
                        
                        # Calculate monthly returns
                        monthly_returns = backtest_results['portfolio_value'].resample('M').last().pct_change().dropna()
                        monthly_returns_bh = buy_hold_results['portfolio_value'].resample('M').last().pct_change().dropna()
                        
                        # Volatility by month
                        monthly_vol = monthly_returns.std() * 100
                        monthly_vol_bh = monthly_returns_bh.std() * 100
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Strategy Volatility", f"{strategy_metrics['volatility']:.2f}%")
                            st.metric("Strategy Beta", f"{backtester.calculate_beta(performance_data):.3f}")
                        with col2:
                            st.metric("Buy & Hold Volatility", f"{buy_hold_metrics['volatility']:.2f}%")
                            st.metric("Strategy Sortino Ratio", f"{backtester.calculate_sortino_ratio():.3f}")
                        with col3:
                            st.metric("Strategy Monthly Volatility", f"{monthly_vol:.2f}%")
                            st.metric("Buy & Hold Monthly Volatility", f"{monthly_vol_bh:.2f}%")
                        
                        # Plot risk/return scatter
                        risk_return_fig = go.Figure()
                        risk_return_fig.add_trace(go.Scatter(
                            x=[strategy_metrics['volatility']],
                            y=[strategy_metrics['annualized_return']],
                            mode='markers',
                            name='Trading Strategy',
                            marker=dict(size=15, color='blue')
                        ))
                        risk_return_fig.add_trace(go.Scatter(
                            x=[buy_hold_metrics['volatility']],
                            y=[buy_hold_metrics['annualized_return']],
                            mode='markers',
                            name='Buy & Hold',
                            marker=dict(size=15, color='red')
                        ))
                        risk_return_fig.update_layout(
                            title='Risk-Return Profile',
                            xaxis_title='Volatility (%)',
                            yaxis_title='Annualized Return (%)',
                            height=500
                        )
                        st.plotly_chart(risk_return_fig, use_container_width=True)
                        
                        # Plot returns distribution
                        returns_fig = go.Figure()
                        returns_fig.add_trace(go.Histogram(
                            x=backtest_results['daily_returns'].dropna() * 100,
                            name='Strategy Returns',
                            opacity=0.7,
                            nbinsx=30
                        ))
                        returns_fig.add_trace(go.Histogram(
                            x=buy_hold_results['daily_returns'].dropna() * 100,
                            name='Buy & Hold Returns',
                            opacity=0.7,
                            nbinsx=30
                        ))
                        returns_fig.update_layout(
                            title='Daily Returns Distribution',
                            xaxis_title='Daily Return (%)',
                            yaxis_title='Frequency',
                            barmode='overlay',
                            height=500
                        )
                        st.plotly_chart(returns_fig, use_container_width=True)
                    
                    elif strategy_option == "Drawdown Analysis":
                        # Drawdown analysis
                        st.subheader("Drawdown Analysis")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Strategy Max Drawdown", f"{strategy_metrics['max_drawdown']:.2f}%")
                            st.metric("Strategy Avg. Drawdown", f"{strategy_metrics['avg_drawdown']:.2f}%")
                        with col2:
                            st.metric("Buy & Hold Max Drawdown", f"{buy_hold_metrics['max_drawdown']:.2f}%")
                            st.metric("Buy & Hold Avg. Drawdown", f"{buy_hold_metrics['avg_drawdown']:.2f}%")
                        
                        # Plot drawdowns
                        drawdown_fig = go.Figure()
                        drawdown_fig.add_trace(go.Scatter(
                            x=backtest_results.index,
                            y=backtest_results['drawdown'] * 100,
                            mode='lines',
                            name='Strategy Drawdown',
                            line=dict(color='red')
                        ))
                        drawdown_fig.add_trace(go.Scatter(
                            x=buy_hold_results.index,
                            y=buy_hold_results['drawdown'] * 100,
                            mode='lines',
                            name='Buy & Hold Drawdown',
                            line=dict(color='blue')
                        ))
                        drawdown_fig.update_layout(
                            title='Drawdown Over Time',
                            xaxis_title='Date',
                            yaxis_title='Drawdown (%)',
                            yaxis=dict(autorange="reversed"),  # Invert y-axis for better visualization
                            height=500
                        )
                        st.plotly_chart(drawdown_fig, use_container_width=True)
                        
                        # Top drawdowns table
                        st.subheader("Top 5 Drawdown Periods")
                        top_drawdowns = backtester.get_top_drawdowns(5)
                        if not top_drawdowns.empty:
                            st.dataframe(top_drawdowns)
                        else:
                            st.info("No significant drawdowns detected")
                else:
                    st.error("Failed to fetch data for performance analytics")
    
    except Exception as e:
        st.error(f"Error in Performance Analytics: {str(e)}")

# Auto Trading tab implementation
with tab5:
    st.header("Database Analytics")
    
    st.markdown("""
    This tab provides analytics based on data stored in the database, including price history, 
    trading signals, trade executions, and machine learning predictions.
    """)
    
    try:
        # Create a timeframe selector
        timeframe = st.selectbox(
            "Select Timeframe",
            ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "All Time"],
            index=1
        )
        
        # Set date range based on timeframe
        end_date = datetime.datetime.now()
        if timeframe == "Last 24 Hours":
            start_date = end_date - datetime.timedelta(days=1)
        elif timeframe == "Last 7 Days":
            start_date = end_date - datetime.timedelta(days=7)
        elif timeframe == "Last 30 Days":
            start_date = end_date - datetime.timedelta(days=30)
        else:  # All Time
            start_date = end_date - datetime.timedelta(days=365)  # Default to 1 year if no data older than that
        
        # Create tabs within the Database Analytics tab
        db_tab1, db_tab2, db_tab3 = st.tabs(["Price History", "Trading Signals", "Trading Performance"])
        
        with db_tab1:
            st.subheader("Gold Price History")
            
            try:
                # Fetch price data from database
                price_data = db.get_gold_prices(start_date=start_date, end_date=end_date)
                
                if not price_data.empty:
                    # Display price data statistics
                    st.write(f"**Data Points**: {len(price_data)}")
                    
                    # Calculate statistics
                    price_stats = {
                        "Latest Price": price_data['Close'].iloc[-1],
                        "Highest Price": price_data['Close'].max(),
                        "Lowest Price": price_data['Close'].min(),
                        "Average Price": price_data['Close'].mean(),
                        "Price Volatility": price_data['Close'].std()
                    }
                    
                    # Display statistics in columns
                    cols = st.columns(5)
                    cols[0].metric("Latest Price", f"${price_stats['Latest Price']:.2f}")
                    cols[1].metric("Highest", f"${price_stats['Highest Price']:.2f}")
                    cols[2].metric("Lowest", f"${price_stats['Lowest Price']:.2f}")
                    cols[3].metric("Average", f"${price_stats['Average Price']:.2f}")
                    cols[4].metric("Volatility", f"${price_stats['Price Volatility']:.2f}")
                    
                    # Create price chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=price_data.index,
                        y=price_data['Close'],
                        mode='lines',
                        name='Gold Price',
                        line=dict(color='gold', width=2)
                    ))
                    
                    # Add range slider
                    fig.update_layout(
                        title='Gold Price History',
                        xaxis_title='Date',
                        yaxis_title='Price (USD/oz)',
                        height=500,
                        xaxis=dict(
                            rangeselector=dict(
                                buttons=list([
                                    dict(count=1, label="1d", step="day", stepmode="backward"),
                                    dict(count=7, label="1w", step="day", stepmode="backward"),
                                    dict(count=1, label="1m", step="month", stepmode="backward"),
                                    dict(step="all")
                                ])
                            ),
                            rangeslider=dict(visible=True),
                            type="date"
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Allow downloading the data
                    csv = price_data.to_csv().encode('utf-8')
                    st.download_button(
                        label="Download Price Data as CSV",
                        data=csv,
                        file_name=f'gold_price_data_{start_date.strftime("%Y%m%d")}_to_{end_date.strftime("%Y%m%d")}.csv',
                        mime='text/csv',
                    )
                else:
                    st.info(f"No price data available for the selected timeframe ({timeframe})")
            
            except Exception as e:
                st.error(f"Error fetching price data: {str(e)}")
        
        with db_tab2:
            st.subheader("Trading Signals History")
            
            try:
                # Fetch signal data from database
                signals = db.get_trading_signals(start_date=start_date, end_date=end_date)
                
                if not signals.empty:
                    # Display signal data statistics
                    st.write(f"**Total Signals**: {len(signals)}")
                    
                    # Count signals by type
                    buy_signals = len(signals[signals['signal_type'] == 1])
                    sell_signals = len(signals[signals['signal_type'] == -1])
                    hold_signals = len(signals[signals['signal_type'] == 0])
                    
                    # Display signal counts in columns
                    cols = st.columns(3)
                    cols[0].metric("Buy Signals", buy_signals)
                    cols[1].metric("Sell Signals", sell_signals)
                    cols[2].metric("Hold Signals", hold_signals)
                    
                    # Group signals by strategy
                    strategy_signals = signals.groupby('strategy').size().reset_index(name='count')
                    
                    # Create pie chart of signals by strategy
                    if len(strategy_signals) > 0:
                        fig = go.Figure(data=[go.Pie(
                            labels=strategy_signals['strategy'],
                            values=strategy_signals['count'],
                            hole=.3,
                            marker_colors=['#2E86C1', '#28B463', '#E74C3C']
                        )])
                        fig.update_layout(
                            title_text='Trading Signals by Strategy',
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display recent signals table
                    st.subheader("Recent Trading Signals")
                    recent_signals = signals.head(10)  # Show most recent 10 signals
                    
                    # Create a more readable dataframe for display
                    display_signals = pd.DataFrame({
                        'Timestamp': recent_signals['timestamp'],
                        'Signal': recent_signals['signal_type'].apply(lambda x: "BUY" if x == 1 else "SELL" if x == -1 else "HOLD"),
                        'Price': recent_signals['price'].apply(lambda x: f"${x:.2f}"),
                        'Strategy': recent_signals['strategy'],
                        'Threshold': recent_signals['threshold'].apply(lambda x: f"{x*100:.2f}%")
                    })
                    
                    st.dataframe(display_signals)
                    
                    # Allow downloading the signals data
                    csv = signals.to_csv().encode('utf-8')
                    st.download_button(
                        label="Download Signal Data as CSV",
                        data=csv,
                        file_name=f'trading_signals_{start_date.strftime("%Y%m%d")}_to_{end_date.strftime("%Y%m%d")}.csv',
                        mime='text/csv',
                    )
                else:
                    st.info(f"No trading signals available for the selected timeframe ({timeframe})")
            
            except Exception as e:
                st.error(f"Error fetching trading signals: {str(e)}")
        
        with db_tab3:
            st.subheader("Trading Performance Analytics")
            
            try:
                # Fetch trades from database
                trades = db.get_trades(limit=None)  # Get all trades
                
                if not trades.empty:
                    # Filter by date range
                    trades = trades[(trades['timestamp'] >= start_date) & (trades['timestamp'] <= end_date)]
                    
                    if not trades.empty:
                        # Display trade data statistics
                        st.write(f"**Total Trades**: {len(trades)}")
                        
                        # Count trades by action
                        buy_trades = len(trades[trades['action'] == 'BUY'])
                        sell_trades = len(trades[trades['action'] == 'SELL'])
                        stop_loss_trades = len(trades[trades['action'] == 'STOP_LOSS'])
                        take_profit_trades = len(trades[trades['action'] == 'TAKE_PROFIT'])
                        
                        # Display trade counts in columns
                        cols = st.columns(4)
                        cols[0].metric("Buy Trades", buy_trades)
                        cols[1].metric("Sell Trades", sell_trades)
                        cols[2].metric("Stop Loss", stop_loss_trades)
                        cols[3].metric("Take Profit", take_profit_trades)
                        
                        # Calculate profit/loss for closed trades
                        closed_trades = trades[(trades['action'] == 'SELL') | 
                                              (trades['action'] == 'STOP_LOSS') | 
                                              (trades['action'] == 'TAKE_PROFIT')]
                        
                        if not closed_trades.empty:
                            total_pnl = closed_trades['pnl'].sum()
                            avg_pnl = closed_trades['pnl'].mean()
                            win_trades = closed_trades[closed_trades['pnl'] > 0]
                            lose_trades = closed_trades[closed_trades['pnl'] <= 0]
                            win_rate = len(win_trades) / len(closed_trades) * 100 if len(closed_trades) > 0 else 0
                            
                            # Display P&L metrics
                            cols = st.columns(4)
                            cols[0].metric("Total P&L", f"${total_pnl:.2f}")
                            cols[1].metric("Average P&L per Trade", f"${avg_pnl:.2f}")
                            cols[2].metric("Win Rate", f"{win_rate:.2f}%")
                            cols[3].metric("Profit Factor", f"{abs(win_trades['pnl'].sum() / lose_trades['pnl'].sum()):.2f}" if len(lose_trades) > 0 and lose_trades['pnl'].sum() != 0 else "∞")
                            
                            # Create performance chart
                            fig = go.Figure()
                            
                            # Use capital_after for portfolio value over time
                            if 'capital_after' in trades.columns:
                                fig.add_trace(go.Scatter(
                                    x=trades['timestamp'],
                                    y=trades['capital_after'],
                                    mode='lines+markers',
                                    name='Portfolio Value',
                                    line=dict(color='blue', width=2)
                                ))
                                
                                # Add reference line for initial capital
                                initial_capital = 10000  # Assuming standard initial capital
                                fig.add_shape(
                                    type="line",
                                    x0=trades['timestamp'].min(),
                                    y0=initial_capital,
                                    x1=trades['timestamp'].max(),
                                    y1=initial_capital,
                                    line=dict(color="red", width=2, dash="dash"),
                                )
                                
                                fig.update_layout(
                                    title='Portfolio Value Over Time',
                                    xaxis_title='Date',
                                    yaxis_title='Portfolio Value ($)',
                                    height=500,
                                    legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="right",
                                        x=1
                                    )
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Create trade distribution chart
                            if 'pnl' in closed_trades.columns:
                                fig = go.Figure()
                                fig.add_trace(go.Histogram(
                                    x=closed_trades['pnl'],
                                    marker_color='blue',
                                    opacity=0.75,
                                    name='P&L Distribution'
                                ))
                                
                                fig.update_layout(
                                    title='Profit/Loss Distribution',
                                    xaxis_title='Profit/Loss ($)',
                                    yaxis_title='Number of Trades',
                                    height=400,
                                    bargap=0.1
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Display recent trades table
                        st.subheader("Recent Trades")
                        recent_trades = trades.head(10)  # Show most recent 10 trades
                        
                        # Create a more readable dataframe for display
                        display_trades = pd.DataFrame({
                            'Timestamp': recent_trades['timestamp'],
                            'Action': recent_trades['action'],
                            'Price': recent_trades['price'].apply(lambda x: f"${x:.2f}"),
                            'Quantity (oz)': recent_trades['quantity'].apply(lambda x: f"{x:.6f}"),
                            'Value ($)': recent_trades['value'].apply(lambda x: f"${x:.2f}"),
                            'P&L': recent_trades['pnl'].apply(lambda x: f"${x:.2f}" if not pd.isna(x) else "N/A"),
                            'Strategy': recent_trades['strategy']
                        })
                        
                        st.dataframe(display_trades)
                        
                        # Allow downloading the trade data
                        csv = trades.to_csv().encode('utf-8')
                        st.download_button(
                            label="Download Trade Data as CSV",
                            data=csv,
                            file_name=f'trade_history_{start_date.strftime("%Y%m%d")}_to_{end_date.strftime("%Y%m%d")}.csv',
                            mime='text/csv',
                        )
                    else:
                        st.info(f"No trades available for the selected timeframe ({timeframe})")
                else:
                    st.info("No trade history available in the database")
            
            except Exception as e:
                st.error(f"Error fetching trade data: {str(e)}")
    
    except Exception as e:
        st.error(f"Error in Database Analytics tab: {str(e)}")

with tab6:
    st.header("Automated Gold Trading")
    
    st.markdown("""
    This tab allows you to set up and monitor automated trading for gold using AI-generated signals.
    The bot will automatically execute trades based on the selected strategy and parameters.
    You can also connect to MetaTrader 5 for real trading capabilities.
    """)
    
    try:
        # Create trading directory if it doesn't exist
        if not os.path.exists('trading_state'):
            os.makedirs('trading_state')
        
        # MT5 connection options
        st.subheader("MetaTrader 5 Integration")
        
        use_mt5 = False
        if MT5_AVAILABLE:
            # Check if we already have MT5 credentials stored
            mt5_api_url = os.environ.get("MT5_API_URL", "")
            mt5_api_key = os.environ.get("MT5_API_KEY", "")
            mt5_credentials_available = bool(mt5_api_url and mt5_api_key)
            
            # MT5 trading toggle
            use_mt5 = st.checkbox("Use MetaTrader 5 for Real Trading", value=False, 
                                 help="When enabled, trades will be executed through MetaTrader 5 if connected")
            
            # Display connection status
            if use_mt5:
                if mt5_credentials_available:
                    # Try to initialize the connection
                    mt5_connected = initialize_mt5_connector()
                    if mt5_connected:
                        st.success("✅ Connected to MetaTrader 5")
                    else:
                        st.error("❌ Failed to connect to MetaTrader 5. Please check your credentials in the sidebar.")
                        st.info("Trading will run in simulation mode until a connection is established.")
                else:
                    st.warning("MetaTrader 5 credentials not configured. Please set them up in the sidebar.")
                    st.info("Trading will run in simulation mode until credentials are provided.")
        else:
            st.info("MetaTrader 5 integration is not available in this environment.")
            
        st.subheader("Trading Configuration")
        
        # Initialize or load the auto trader
        auto_trader = AutoTrader(
            initial_capital=10000,
            strategy_type=strategy_type,
            signal_threshold=signal_threshold,
            stop_loss=stop_loss,
            take_profit=take_profit,
            feature_window=feature_days,
            prediction_days=prediction_days,
            use_mt5=use_mt5
        )
        
        # Get trading status
        trading_status = auto_trader.get_trading_status()
        
        # Create dashboard layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Trading controls
            st.subheader("Trading Controls")
            
            # Trading status display
            status_col1, status_col2 = st.columns(2)
            with status_col1:
                st.metric("Trading Status", "Active" if trading_status["is_trading"] else "Inactive")
                st.metric("Current Capital", format_currency(trading_status["current_capital"]))
                
            with status_col2:
                profit_loss = trading_status["current_capital"] - 10000  # Initial capital is 10000
                st.metric("Profit/Loss", 
                          format_currency(profit_loss), 
                          f"{profit_loss/10000*100:.2f}%")
                
                if trading_status["current_position"] > 0:
                    st.metric("Current Position", 
                            f"{trading_status['current_position']:.6f} oz", 
                            f"Value: {format_currency(trading_status['current_position_value'])}")
            
            # Start/Stop trading buttons
            if not trading_status["is_trading"]:
                if st.button("Start Automated Trading"):
                    result = auto_trader.start_trading(interval_seconds=300)  # Check every 5 minutes
                    st.success(result)
                    # Force refresh
                    st.rerun()
            else:
                if st.button("Stop Automated Trading"):
                    result = auto_trader.stop_trading()
                    st.success(result)
                    # Force refresh
                    st.rerun()
            
            # Trading strategy info
            st.subheader("Current Trading Strategy")
            st.write(f"**Strategy Type:** {strategy_type.replace('_', ' ').title()}")
            st.write(f"**Signal Threshold:** {signal_threshold*100:.2f}%")
            st.write(f"**Stop Loss:** {stop_loss*100:.2f}%")
            st.write(f"**Take Profit:** {take_profit*100:.2f}%")
            
            # Recent trades
            st.subheader("Recent Trades")
            trade_history = auto_trader.get_trade_history(limit=10)  # Fetch up to 10 recent trades from database
            
            if not trade_history.empty:
                # Display the most recent trades
                for _, trade in trade_history.iterrows():
                    trade_color = "success" if trade.get('action') in ['BUY', 'TAKE_PROFIT'] else "error"
                    
                    # Check if trade has execution info (MT5 or Simulation)
                    execution_type = trade.get('execution', 'Simulation')
                    exec_badge = "🔄" if execution_type == 'Simulation' else "🌐"
                    
                    if trade.get('action') == 'BUY':
                        message = f"🛒 **BUY** {trade.get('quantity', 0):.6f} oz at {format_currency(trade.get('price', 0))}"
                    elif trade.get('action') == 'SELL':
                        pnl = trade.get('pnl', 0)
                        pnl_pct = trade.get('pnl_percent', 0)
                        message = f"💰 **SELL** {trade.get('quantity', 0):.6f} oz at {format_currency(trade.get('price', 0))} - P&L: {format_currency(pnl)} ({pnl_pct:.2f}%)"
                    elif trade.get('action') == 'STOP_LOSS':
                        pnl = trade.get('pnl', 0)
                        pnl_pct = trade.get('pnl_percent', 0)
                        message = f"🛑 **STOP LOSS** {trade.get('quantity', 0):.6f} oz at {format_currency(trade.get('price', 0))} - P&L: {format_currency(pnl)} ({pnl_pct:.2f}%)"
                    elif trade.get('action') == 'TAKE_PROFIT':
                        pnl = trade.get('pnl', 0)
                        pnl_pct = trade.get('pnl_percent', 0)
                        message = f"✅ **TAKE PROFIT** {trade.get('quantity', 0):.6f} oz at {format_currency(trade.get('price', 0))} - P&L: {format_currency(pnl)} ({pnl_pct:.2f}%)"
                    else:
                        message = f"**{trade.get('action', 'UNKNOWN')}** at {format_currency(trade.get('price', 0))}"
                    
                    getattr(st, trade_color)(f"{message} - {trade.get('timestamp').strftime('%Y-%m-%d %H:%M:%S')} {exec_badge} {execution_type}")
            else:
                st.info("No trades executed yet")
        
        with col2:
            # Display current price
            try:
                current_price, timestamp = fetch_realtime_gold_price()
                
                st.subheader("Current Gold Price")
                st.metric("Gold Price (USD/oz)", 
                        format_currency(current_price),
                        f"Updated: {timestamp}")
                
                # Display entry price and potential profit/loss if in position
                if trading_status["current_position"] > 0:
                    entry_price = trading_status["entry_price"]
                    price_change = ((current_price / entry_price) - 1) * 100
                    
                    st.metric("Entry Price", 
                            format_currency(entry_price),
                            f"{price_change:.2f}%")
                    
                    # Show stop loss and take profit levels
                    sl_price = entry_price * (1 - stop_loss)
                    tp_price = entry_price * (1 + take_profit)
                    
                    st.metric("Stop Loss Level", format_currency(sl_price))
                    st.metric("Take Profit Level", format_currency(tp_price))
            except:
                st.error("Unable to fetch current gold price")
            
            # Display trade stats if available
            if not trade_history.empty:
                st.subheader("Trading Statistics")
                
                # Calculate win rate
                win_trades = trade_history[
                    (trade_history['action'] == 'SELL') | 
                    (trade_history['action'] == 'TAKE_PROFIT')
                ]
                win_trades = win_trades[win_trades.get('pnl', 0) > 0]
                
                lose_trades = trade_history[
                    (trade_history['action'] == 'SELL') | 
                    (trade_history['action'] == 'STOP_LOSS') |
                    (trade_history['action'] == 'TAKE_PROFIT')
                ]
                lose_trades = lose_trades[lose_trades.get('pnl', 0) <= 0]
                
                total_closed_trades = len(win_trades) + len(lose_trades)
                
                if total_closed_trades > 0:
                    win_rate = len(win_trades) / total_closed_trades * 100
                    st.metric("Win Rate", f"{win_rate:.2f}%")
                    
                    # Calculate average win and loss
                    if len(win_trades) > 0:
                        avg_win = win_trades['pnl'].mean()
                        st.metric("Average Win", format_currency(avg_win))
                    
                    if len(lose_trades) > 0:
                        avg_loss = lose_trades['pnl'].mean()
                        st.metric("Average Loss", format_currency(avg_loss))
    
        # Trading performance chart
        st.subheader("Trading Performance")
        if not trade_history.empty and 'timestamp' in trade_history.columns and 'capital_after' in trade_history.columns:
            # Create performance chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=trade_history['timestamp'],
                y=trade_history['capital_after'],
                mode='lines+markers',
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ))
            
            # Add horizontal line for initial capital
            fig.add_shape(
                type="line",
                x0=trade_history['timestamp'].iloc[0],
                y0=10000,
                x1=trade_history['timestamp'].iloc[-1],
                y1=10000,
                line=dict(color="red", width=2, dash="dash"),
            )
            
            fig.update_layout(
                title='Portfolio Value Over Time',
                xaxis_title='Date',
                yaxis_title='Portfolio Value ($)',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Complete trade history table
            with st.expander("Complete Trade History"):
                st.dataframe(trade_history)
        else:
            st.info("Not enough trading data to display performance chart")
    
    except Exception as e:
        st.error(f"Error in Auto Trading: {str(e)}")
        st.info("Try refreshing the page or check the console for more information.")

# Strategy Advisor tab
with tab7:
    st.header("AI-Powered Strategy Advisor")
    
    st.markdown("""
    This section provides AI-powered trading strategy recommendations based on current market conditions.
    The system analyzes price patterns, volatility, and trend strength to suggest optimal trading strategies and parameters.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Fetch data for analysis
        try:
            # OpenAI toggle
            use_openai = st.checkbox("Use OpenAI for advanced analysis (requires API key)", 
                                   value=st.session_state.get('use_openai', False),
                                   help="When enabled, the system will use OpenAI to provide more detailed market analysis and strategy explanations.")
            
            # Save the OpenAI preference to session state
            st.session_state.use_openai = use_openai
            
            # Use the last 90 days of data for strategy analysis
            analysis_period = st.slider("Analysis Period (Days)", 30, 180, 90)
            
            analysis_start_date = datetime.datetime.now().date() - datetime.timedelta(days=analysis_period)
            analysis_end_date = datetime.datetime.now().date()
            
            if st.button("Generate Strategy Recommendations"):
                with st.spinner("Analyzing market conditions and generating recommendations..."):
                    # Fetch historical data
                    price_data = fetch_historical_gold_data(analysis_start_date, analysis_end_date)
                    
                    if price_data is not None and len(price_data) > 0:
                        # Calculate indicators if they don't exist
                        if 'MA_20' not in price_data.columns:
                            price_data['MA_20'] = price_data['Close'].rolling(window=20).mean()
                        
                        if 'MA_50' not in price_data.columns:
                            price_data['MA_50'] = price_data['Close'].rolling(window=50).mean()
                        
                        if 'Volatility' not in price_data.columns:
                            price_data['Daily_Return'] = price_data['Close'].pct_change()
                            price_data['Volatility'] = price_data['Daily_Return'].rolling(window=20).std()
                        
                        if 'RSI' not in price_data.columns:
                            # Calculate RSI
                            delta = price_data['Close'].diff()
                            gain = (delta.where(delta > 0, 0)).fillna(0)
                            loss = (-delta.where(delta < 0, 0)).fillna(0)
                            
                            avg_gain = gain.rolling(window=14).mean()
                            avg_loss = loss.rolling(window=14).mean()
                            
                            rs = avg_gain / avg_loss
                            price_data['RSI'] = 100 - (100 / (1 + rs))
                        
                        # Get strategy recommendations
                        recommendations = get_strategy_recommendations(price_data, top_n=3, run_backtest=True)
                        
                        # Save to session state
                        st.session_state.strategy_recommendations = recommendations
                        
                        # Check if user wants OpenAI analysis
                        use_openai = st.session_state.get('use_openai', False)
                        openai_available = os.environ.get("OPENAI_API_KEY") is not None
                        
                        if use_openai and not openai_available:
                            st.warning("OpenAI API key not available. You can continue with the rule-based recommendations or set up an API key.")
                        elif use_openai and openai_available:
                            # Get OpenAI recommendations if available
                            with st.spinner("Generating advanced AI analysis..."):
                                openai_analysis = get_openai_recommendations(price_data)
                                st.session_state.openai_analysis = openai_analysis
                    else:
                        st.error("Failed to fetch data for strategy analysis")
        
        except Exception as e:
            st.error(f"Error generating strategy recommendations: {str(e)}")
            st.info("Try adjusting the analysis period or check your internet connection.")
        
        # Display strategy recommendations if available
        if hasattr(st.session_state, 'strategy_recommendations'):
            recommendations = st.session_state.strategy_recommendations
            
            # Market conditions summary
            st.subheader("Current Market Conditions")
            
            market_conditions = recommendations.get('market_conditions', {})
            market_summary = recommendations.get('market_summary', {})
            
            # Create a more user-friendly display of market conditions
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Summary:** {market_summary.get('summary', 'No distinct market conditions detected.')}")
                
                # Display active market conditions with checkmarks
                active_conditions = []
                for condition, is_active in market_conditions.items():
                    if is_active:
                        active_conditions.append(f"✅ {condition.replace('_', ' ').title()}")
                
                if active_conditions:
                    st.markdown("\n".join(active_conditions))
            
            with col2:
                for explanation in market_summary.get('explanations', []):
                    st.info(explanation)
            
            # Strategy recommendations
            st.subheader("Recommended Trading Strategies")
            
            for i, rec in enumerate(recommendations.get('recommendations', [])):
                with st.expander(f"Strategy {i+1}: {rec['strategy'].replace('_', ' ').title()} (Confidence: {rec['confidence']*100:.1f}%)"):
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        st.markdown(f"**Why this strategy:** {rec['reason']}")
                        
                        if 'backtest_reason' in rec:
                            st.markdown(f"**Backtest results:** {rec['backtest_reason']}")
                    
                    with col2:
                        # Parameters table
                        params_df = pd.DataFrame({
                            'Parameter': ['Signal Threshold', 'Stop Loss', 'Take Profit'],
                            'Value': [
                                f"{rec['signal_threshold']*100:.1f}%",
                                f"{rec['stop_loss']*100:.1f}%",
                                f"{rec['take_profit']*100:.1f}%"
                            ]
                        })
                        
                        if 'feature_window' in rec:
                            params_df = pd.concat([params_df, pd.DataFrame({
                                'Parameter': ['Feature Window'],
                                'Value': [f"{rec['feature_window']} days"]
                            })], ignore_index=True)
                            
                        if 'prediction_days' in rec:
                            params_df = pd.concat([params_df, pd.DataFrame({
                                'Parameter': ['Prediction Days'],
                                'Value': [f"{rec['prediction_days']} days"]
                            })], ignore_index=True)
                        
                        st.table(params_df)
                    
                    # Backtest results if available
                    if 'backtest_results' in rec:
                        backtest = rec['backtest_results']
                        
                        st.markdown("##### Backtest Performance")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Return", f"{backtest.get('total_return', 0):.2f}%")
                        with col2:
                            st.metric("Sharpe Ratio", f"{backtest.get('sharpe_ratio', 0):.2f}")
                        with col3:
                            st.metric("Win Rate", f"{backtest.get('win_rate', 0):.2f}%")
                        with col4:
                            st.metric("Max Drawdown", f"{backtest.get('max_drawdown', 0):.2f}%")
            
            # Apply selected strategy button
            st.subheader("Apply a Recommended Strategy")
            selected_strategy = st.selectbox(
                "Select a strategy to apply to the trading bot",
                [f"{i+1}: {rec['strategy'].replace('_', ' ').title()}" for i, rec in enumerate(recommendations.get('recommendations', []))]
            )
            
            if st.button("Apply Selected Strategy"):
                # Get the selected strategy index
                strategy_index = int(selected_strategy.split(":")[0]) - 1
                selected_rec = recommendations['recommendations'][strategy_index]
                
                # Update sidebar parameters
                with st.spinner("Applying strategy parameters..."):
                    # Get the current Streamlit session state and update it
                    st.session_state.signal_threshold = selected_rec['signal_threshold'] * 100
                    st.session_state.stop_loss = selected_rec['stop_loss'] * 100
                    st.session_state.take_profit = selected_rec['take_profit'] * 100
                    
                    if 'feature_window' in selected_rec:
                        st.session_state.feature_days = selected_rec['feature_window']
                    
                    if 'prediction_days' in selected_rec:
                        st.session_state.prediction_days = selected_rec['prediction_days']
                    
                    st.session_state.strategy_type = selected_rec['strategy']
                    
                    st.success(f"Strategy '{selected_rec['strategy'].replace('_', ' ').title()}' applied! Parameters have been updated in the sidebar.")
        
    with col2:
        st.subheader("OpenAI Strategy Advisor")
        
        # Check if OpenAI key is available
        openai_available = os.environ.get("OPENAI_API_KEY") is not None
        
        if not openai_available:
            st.info("To enable the OpenAI Strategy Advisor, you need to provide an OpenAI API key.")
            
            if st.button("Set OpenAI API Key"):
                st.session_state.show_openai_key_input = True
            
            if hasattr(st.session_state, 'show_openai_key_input') and st.session_state.show_openai_key_input:
                api_key = st.text_input("Enter your OpenAI API Key", type="password")
                
                if api_key and st.button("Save API Key"):
                    os.environ["OPENAI_API_KEY"] = api_key
                    st.success("OpenAI API key set successfully!")
                    st.session_state.show_openai_key_input = False
                    st.rerun()
        else:
            st.success("OpenAI API connected!")
            
            if hasattr(st.session_state, 'openai_analysis'):
                analysis = st.session_state.openai_analysis
                
                if 'error' in analysis:
                    st.error(analysis['error'])
                elif not analysis.get('is_available', False):
                    st.warning("OpenAI API is not available. Please check your API key.")
                else:
                    st.markdown(f"**Market Summary:**\n{analysis.get('market_summary', 'No summary available.')}")
                    
                    with st.expander("Technical Analysis"):
                        st.markdown(analysis.get('technical_analysis', 'No technical analysis available.'))
                    
                    with st.expander("Market Sentiment"):
                        st.markdown(analysis.get('market_sentiment', 'No market sentiment analysis available.'))
                    
                    with st.expander("Potential Catalysts"):
                        st.markdown(analysis.get('potential_catalysts', 'No catalyst information available.'))
                    
                    # OpenAI strategy recommendations
                    st.subheader("AI-Recommended Strategies")
                    
                    for i, rec in enumerate(analysis.get('strategy_recommendations', [])):
                        with st.expander(f"{rec['strategy']} (Confidence: {rec['confidence']*100:.1f}%)"):
                            st.markdown(f"**Parameters:**")
                            st.markdown(f"- Signal Threshold: {rec['signal_threshold']*100:.1f}%")
                            st.markdown(f"- Stop Loss: {rec['stop_loss']*100:.1f}%")
                            st.markdown(f"- Take Profit: {rec['take_profit']*100:.1f}%")
                            
                            st.markdown(f"**Reasoning:**\n{rec['reasoning']}")
                            
                            # Apply button for this specific strategy
                            if st.button(f"Apply Strategy {i+1}"):
                                # Update sidebar parameters
                                st.session_state.signal_threshold = rec['signal_threshold'] * 100
                                st.session_state.stop_loss = rec['stop_loss'] * 100
                                st.session_state.take_profit = rec['take_profit'] * 100
                                
                                # Map strategy names to our internal strategy types
                                strategy_mapping = {
                                    'Trend Following': 'trend_following',
                                    'Mean Reversion': 'mean_reversion',
                                    'Breakout': 'breakout',
                                    'Momentum': 'momentum',
                                    'ML-Based': 'ml_based'
                                }
                                
                                # Try to match strategy name, fallback to trend_following
                                for key, value in strategy_mapping.items():
                                    if key.lower() in rec['strategy'].lower():
                                        st.session_state.strategy_type = value
                                        break
                                else:
                                    st.session_state.strategy_type = 'trend_following'
                                
                                st.success(f"Strategy '{rec['strategy']}' applied! Parameters have been updated in the sidebar.")
            else:
                if st.button("Get OpenAI Strategy Analysis"):
                    st.warning("Please generate strategy recommendations first to access OpenAI analysis.")
            
        # Add links to strategy explanations
        st.subheader("Learn About Strategies")
        
        strategy_explanations = {
            "trend_following": "Strategy that follows the direction of the market trend",
            "mean_reversion": "Strategy based on the theory that prices will revert to their mean",
            "breakout": "Strategy that enters trades when price breaks through support/resistance",
            "momentum": "Strategy that follows acceleration in price movement",
            "ml_based": "Strategy using machine learning models to predict price movements"
        }
        
        for strategy, description in strategy_explanations.items():
            if st.button(f"About {strategy.replace('_', ' ').title()}", key=f"btn_{strategy}"):
                if openai_available:
                    with st.spinner(f"Generating detailed explanation of {strategy.replace('_', ' ').title()} strategy..."):
                        explanation = get_strategy_explanation(strategy)
                        st.markdown(f"## {strategy.replace('_', ' ').title()} Strategy")
                        st.markdown(explanation)
                else:
                    st.info(f"{strategy.replace('_', ' ').title()}: {description}")
                    st.warning("For detailed explanations, please set your OpenAI API key.")

with tab8:
    st.header("Animated Trading Performance Dashboard")
    
    st.markdown("""
    This interactive dashboard provides a dynamic visualization of your trading performance with animated charts,
    metrics, and analysis tools. Watch your trading strategy performance unfold over time.
    """)
    
    # Add dashboard controls
    st.subheader("Dashboard Controls")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        data_source = st.radio(
            "Data Source",
            ["Trading History", "Backtest Results"],
            index=0
        )
    
    with col2:
        view_period = st.selectbox(
            "View Period",
            ["Last 30 Days", "Last 90 Days", "Last 6 Months", "Last Year", "All Time"],
            index=0
        )
    
    with col3:
        animation_speed = st.slider(
            "Animation Speed",
            min_value=100,
            max_value=1000,
            value=300,
            step=100,
            help="Control the speed of the animations (in milliseconds)"
        )
    
    # Load trading history data
    try:
        # Get trading history from database
        trade_history = None
        with st.spinner("Loading trading history data..."):
            try:
                # First try to get from database
                trade_history = db.get_trading_history(limit=1000)
                
                # If no data in DB, try to load from auto_trader
                if trade_history is None or trade_history.empty:
                    auto_trader = AutoTrader()
                    trade_history = auto_trader.get_trade_history(limit=1000)
            except Exception as e:
                st.warning(f"Could not fetch trade history from database: {str(e)}")
                
        # If backtest results are selected, use those instead
        if data_source == "Backtest Results" and "backtest_results" in st.session_state:
            # Generate trade history from backtest results
            if hasattr(st.session_state, "backtester"):
                trade_history = st.session_state.backtester.get_trade_list()
                
        # Process data based on selected period
        if trade_history is not None and not trade_history.empty:
            # Convert period to days for filtering
            period_days = {
                "Last 30 Days": 30,
                "Last 90 Days": 90,
                "Last 6 Months": 180,
                "Last Year": 365,
                "All Time": 9999
            }
            
            days = period_days.get(view_period, 30)
            
            if 'timestamp' in trade_history.columns:
                # Filter by timestamp
                cutoff_date = (datetime.datetime.now() - datetime.timedelta(days=days)).date()
                trade_history = trade_history[
                    pd.to_datetime(trade_history['timestamp']).dt.date >= cutoff_date
                ]
            
            # Create the interactive dashboard
            if not trade_history.empty:
                # Generate animated dashboard
                with st.spinner("Generating animated performance dashboard..."):
                    dashboard_figures = create_animated_performance_dashboard(trade_history)
                    
                    if 'error' in dashboard_figures:
                        st.error(dashboard_figures['error'])
                    else:
                        # Display main performance chart
                        if 'performance' in dashboard_figures:
                            st.subheader("📈 Trading Performance Over Time")
                            st.plotly_chart(dashboard_figures['performance'], use_container_width=True)
                            
                        # Display metrics in two columns
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Display P&L distribution chart
                            if 'pl_distribution' in dashboard_figures:
                                st.subheader("💰 P&L Distribution")
                                st.plotly_chart(dashboard_figures['pl_distribution'], use_container_width=True)
                            
                            # Display trade breakdown chart
                            if 'trade_breakdown' in dashboard_figures:
                                st.subheader("🔄 Trade Breakdown")
                                st.plotly_chart(dashboard_figures['trade_breakdown'], use_container_width=True)
                        
                        with col2:
                            # Display key metrics indicators
                            if 'metrics' in dashboard_figures:
                                st.subheader("📊 Key Metrics")
                                st.plotly_chart(dashboard_figures['metrics'], use_container_width=True)
                            
                            # Display monthly performance heatmap
                            if 'monthly_heatmap' in dashboard_figures:
                                st.subheader("📅 Monthly Performance")
                                st.plotly_chart(dashboard_figures['monthly_heatmap'], use_container_width=True)
                        
                        # Add trade details table
                        with st.expander("View Detailed Trade History"):
                            st.dataframe(trade_history)
            else:
                st.warning("No trading data available for the selected period.")
        else:
            st.info("No trading history available. Run some trades or a backtest first.")
            
            # Create sample data for demonstration
            with st.expander("Generate Sample Dashboard"):
                if st.button("Generate Sample Dashboard"):
                    # Generate sample trade history
                    with st.spinner("Creating sample data for demonstration..."):
                        try:
                            # Create synthetic data directly without relying on database
                            # This is safe for demonstration purposes since we're not presenting it as real data
                            
                            # Generate dates for the past 180 days
                            end_date = datetime.datetime.now()
                            dates = [end_date - datetime.timedelta(days=i) for i in range(180)]
                            dates.reverse()  # Chronological order
                            
                            # Generate simulated trades and performance data
                            trade_data = []
                            cumulative_pl = 0
                            
                            # Create a mix of buy and sell trades with some randomization
                            for i, date in enumerate(dates):
                                # Only create trades on some days (about 30% of days)
                                if i % 3 == 0 or i % 7 == 0:  
                                    # Simulate price around $1800 with some random walk
                                    base_price = 1800 + (i * 3) + (random.random() * 20 - 10)
                                    
                                    # Alternate between buy and sell with some randomness
                                    trade_type = 'buy' if (i % 2 == 0 or random.random() > 0.4) else 'sell'
                                    
                                    # Random profit/loss between -$200 and $300
                                    profit_loss = (random.random() * 500 - 200) 
                                    
                                    # Make it more likely to profit to show interesting data
                                    if random.random() > 0.4:
                                        profit_loss = abs(profit_loss)
                                    
                                    # Update cumulative P&L
                                    cumulative_pl += profit_loss
                                    
                                    # Create trade entry
                                    trade_data.append({
                                        'timestamp': date,
                                        'date': date.date(),
                                        'type': trade_type,
                                        'price': base_price,
                                        'quantity': random.randint(1, 5),
                                        'profit_loss': profit_loss,
                                        'cumulative_pl': cumulative_pl
                                    })
                            
                            # Create DataFrame
                            if trade_data:
                                sample_trade_history = pd.DataFrame(trade_data)
                                
                                # Generate animated dashboard from sample data
                                dashboard_figures = create_animated_performance_dashboard(sample_trade_history)
                                
                                if 'error' in dashboard_figures:
                                    st.error(f"Error generating dashboard: {dashboard_figures['error']}")
                                else:
                                    st.success("Sample dashboard generated successfully!")
                                    
                                    # Display main performance chart
                                    if 'performance' in dashboard_figures:
                                        st.subheader("📈 Sample Trading Performance")
                                        st.plotly_chart(dashboard_figures['performance'], use_container_width=True)
                                    
                                    # Display metrics in two columns
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        # Display P&L distribution chart
                                        if 'pl_distribution' in dashboard_figures:
                                            st.subheader("💰 P&L Distribution")
                                            st.plotly_chart(dashboard_figures['pl_distribution'], use_container_width=True)
                                        
                                        # Display trade breakdown chart
                                        if 'trade_breakdown' in dashboard_figures:
                                            st.subheader("🔄 Trade Breakdown")
                                            st.plotly_chart(dashboard_figures['trade_breakdown'], use_container_width=True)
                                    
                                    with col2:
                                        # Display key metrics indicators
                                        if 'metrics' in dashboard_figures:
                                            st.subheader("📊 Key Metrics")
                                            st.plotly_chart(dashboard_figures['metrics'], use_container_width=True)
                                        
                                        # Display monthly performance heatmap
                                        if 'monthly_heatmap' in dashboard_figures:
                                            st.subheader("📅 Monthly Performance")
                                            st.plotly_chart(dashboard_figures['monthly_heatmap'], use_container_width=True)
                                    
                                    # Add disclaimer that this is sample data
                                    st.info("Note: This is synthetic sample data for demonstration purposes only.")
                            else:
                                st.error("Failed to generate sample trade data.")
                                
                        except Exception as e:
                            st.error(f"Error generating sample data: {str(e)}")
                            st.info("Please try again. If the issue persists, check your connection.")
    except Exception as e:
        st.error(f"Error in Performance Dashboard: {str(e)}")
        st.info("Please try again or check your data source.")

# Show app footer
st.markdown("---")
st.markdown("### About the Gold Trading Bot")
st.markdown("""
This AI-powered gold trading bot uses machine learning algorithms to analyze 
historical gold price data and generate trading signals based on predicted price movements.

**Key Features:**
* Real-time gold price monitoring
* Historical price analysis
* AI-based price prediction
* Trading signal generation
* Strategy backtesting and optimization
* Performance analytics
* Automated trading execution
* AI-powered strategy recommendations

**Note:** This tool is for educational and informational purposes only. It should not be 
considered financial advice or a recommendation to trade gold.
""")
