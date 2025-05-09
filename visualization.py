import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import math
import random

def plot_price_chart(data):
    """
    Create a candlestick chart of price data
    
    Args:
        data (pandas.DataFrame): Price data with OHLC columns
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Check if we have OHLC data or just Close
    has_ohlc = all(col in data.columns for col in ['Open', 'High', 'Low', 'Close'])
    
    if has_ohlc:
        # Add candlestick trace
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Gold Price'
        ))
    else:
        # If we only have Close prices, use a line chart
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name='Gold Price'
        ))
    
    # Add moving averages if available
    if 'MA_5' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MA_5'],
            mode='lines',
            line=dict(color='blue', width=1),
            name='5-day MA'
        ))
    
    if 'MA_20' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MA_20'],
            mode='lines',
            line=dict(color='orange', width=1),
            name='20-day MA'
        ))
    
    if 'MA_50' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MA_50'],
            mode='lines',
            line=dict(color='green', width=1),
            name='50-day MA'
        ))
    
    # Update layout
    fig.update_layout(
        title='Gold Price Chart',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        xaxis_rangeslider_visible=False,
        height=500,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update y-axis for better visualization
    if has_ohlc:
        # Calculate price range and add some margin
        price_range = data['High'].max() - data['Low'].min()
        y_min = data['Low'].min() - price_range * 0.05
        y_max = data['High'].max() + price_range * 0.05
        
        fig.update_yaxes(
            range=[y_min, y_max],
            tickformat='.2f'
        )
    
    return fig

def plot_returns(data):
    """
    Create a histogram and KDE plot of daily returns
    
    Args:
        data (pandas.DataFrame): Price data with Daily_Return column
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    if 'Daily_Return' not in data.columns:
        # Calculate daily returns if not already present
        returns = data['Close'].pct_change().dropna()
    else:
        returns = data['Daily_Return'].dropna()
    
    # Convert to percentage
    returns_pct = returns * 100
    
    # Create figure
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=returns_pct,
        name='Daily Returns',
        opacity=0.7,
        nbinsx=30,
        marker_color='blue',
        histnorm='probability density'
    ))
    
    # Calculate statistics
    mean = returns_pct.mean()
    std = returns_pct.std()
    
    # Add normal distribution curve
    x_range = np.linspace(mean - 4*std, mean + 4*std, 100)
    y_range = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean) / std) ** 2)
    
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_range,
        mode='lines',
        name='Normal Distribution',
        line=dict(color='red', width=2)
    ))
    
    # Add vertical line at mean
    fig.add_shape(
        type='line',
        x0=mean, x1=mean,
        y0=0, y1=y_range.max() * 1.1,
        line=dict(color='green', width=2, dash='dash')
    )
    
    # Annotate mean and std
    fig.add_annotation(
        x=mean, y=y_range.max() * 1.1,
        text=f"Mean: {mean:.3f}%",
        showarrow=True,
        arrowhead=1,
        ax=50, ay=-40
    )
    
    fig.add_annotation(
        x=mean + 2*std, y=y_range.max() / 2,
        text=f"Std Dev: {std:.3f}%",
        showarrow=True,
        arrowhead=1,
        ax=50, ay=0
    )
    
    # Update layout
    fig.update_layout(
        title='Distribution of Daily Returns',
        xaxis_title='Daily Return (%)',
        yaxis_title='Probability Density',
        height=500,
        template='plotly_white'
    )
    
    return fig

def plot_prediction_vs_actual(dates, actual, predicted):
    """
    Create a line chart comparing actual vs predicted prices
    
    Args:
        dates (array-like): Dates for x-axis
        actual (array-like): Actual prices
        predicted (array-like): Predicted prices
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Add actual prices
    fig.add_trace(go.Scatter(
        x=dates,
        y=actual,
        mode='lines+markers',
        name='Actual',
        line=dict(color='blue', width=2)
    ))
    
    # Add predicted prices
    fig.add_trace(go.Scatter(
        x=dates,
        y=predicted,
        mode='lines+markers',
        name='Predicted',
        line=dict(color='red', width=2)
    ))
    
    # Calculate error metrics for the title
    mse = np.mean((actual - predicted) ** 2)
    mae = np.mean(np.abs(actual - predicted))
    
    # Update layout
    fig.update_layout(
        title=f'Actual vs Predicted Gold Prices (MSE: {mse:.2f}, MAE: {mae:.2f})',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        height=500,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_signals(data, signals):
    """
    Create a price chart with buy/sell signals
    
    Args:
        data (pandas.DataFrame): Price data
        signals (pandas.DataFrame): DataFrame with signal column
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Gold Price',
        line=dict(color='blue', width=2)
    ))
    
    # Add moving averages if available
    if 'MA_20' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MA_20'],
            mode='lines',
            line=dict(color='orange', width=1),
            name='20-day MA'
        ))
    
    # Find buy signals (signal=1)
    buy_indices = signals.index[signals['signal'] == 1]
    buy_prices = data.loc[buy_indices, 'Close']
    
    # Find sell signals (signal=-1)
    sell_indices = signals.index[signals['signal'] == -1]
    sell_prices = data.loc[sell_indices, 'Close']
    
    # Add buy signals
    fig.add_trace(go.Scatter(
        x=buy_indices,
        y=buy_prices,
        mode='markers',
        name='Buy Signal',
        marker=dict(
            color='green',
            size=10,
            symbol='triangle-up',
            line=dict(color='green', width=2)
        )
    ))
    
    # Add sell signals
    fig.add_trace(go.Scatter(
        x=sell_indices,
        y=sell_prices,
        mode='markers',
        name='Sell Signal',
        marker=dict(
            color='red',
            size=10,
            symbol='triangle-down',
            line=dict(color='red', width=2)
        )
    ))
    
    # Update layout
    fig.update_layout(
        title='Gold Price with Trading Signals',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        height=500,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_performance_metrics(backtest_results):
    """
    Create a chart of portfolio value over time
    
    Args:
        backtest_results (pandas.DataFrame): Backtest results with portfolio_value
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Create subplot figure with 2 rows
    fig = make_subplots(
        rows=2, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Portfolio Value', 'Drawdown'),
        row_heights=[0.7, 0.3]
    )
    
    # Add portfolio value line
    fig.add_trace(
        go.Scatter(
            x=backtest_results.index,
            y=backtest_results['portfolio_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Add drawdown chart
    fig.add_trace(
        go.Scatter(
            x=backtest_results.index,
            y=backtest_results['drawdown'] * 100,  # Convert to percentage
            mode='lines',
            name='Drawdown',
            line=dict(color='red', width=2),
            fill='tozeroy'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='Portfolio Performance',
        height=700,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis2_title='Date',
    )
    
    # Update y-axis titles
    fig.update_yaxes(title_text='Value (USD)', row=1, col=1)
    fig.update_yaxes(title_text='Drawdown (%)', row=2, col=1)
    
    # Invert y-axis for drawdown
    fig.update_yaxes(autorange="reversed", row=2, col=1)
    
    return fig

def create_animated_performance_dashboard(trade_history, price_data=None):
    """
    Create an animated and interactive trading performance dashboard
    
    Args:
        trade_history (pandas.DataFrame): Trading history with timestamps, trade types, prices
        price_data (pandas.DataFrame, optional): Historical price data
        
    Returns:
        dict: Dictionary of plotly figures for the dashboard
    """
    figures = {}
    
    # Validate input data
    if trade_history is None or trade_history.empty:
        # Return empty figures if no data
        return {"error": "No trade history data available"}
    
    # Ensure we have the necessary columns
    required_cols = ['timestamp', 'type', 'price', 'profit_loss']
    if not all(col in trade_history.columns for col in required_cols):
        # Create dummy columns if needed for demonstration
        if 'timestamp' not in trade_history.columns and 'date' in trade_history.columns:
            trade_history['timestamp'] = pd.to_datetime(trade_history['date'])
        if 'type' not in trade_history.columns and 'signal' in trade_history.columns:
            trade_history['type'] = trade_history['signal'].apply(lambda x: 'buy' if x == 1 else 'sell' if x == -1 else 'hold')
    
    # Sort by timestamp
    trade_history = trade_history.sort_values('timestamp')
    
    # 1. Create main performance chart with animation
    # -----------------------------------------------
    # Calculate cumulative P&L if not present
    if 'cumulative_pl' not in trade_history.columns:
        if 'profit_loss' in trade_history.columns:
            trade_history['cumulative_pl'] = trade_history['profit_loss'].cumsum()
        else:
            # Simulate cumulative P&L if not available
            trade_history['cumulative_pl'] = trade_history.index
    
    # Create animated performance chart
    performance_fig = create_animated_performance_chart(trade_history)
    figures['performance'] = performance_fig
    
    # 2. Create P&L distribution chart
    # --------------------------------
    if 'profit_loss' in trade_history.columns:
        pl_distribution_fig = create_pl_distribution_chart(trade_history)
        figures['pl_distribution'] = pl_distribution_fig
    
    # 3. Create trade type breakdown chart
    # ------------------------------------
    if 'type' in trade_history.columns:
        trade_breakdown_fig = create_trade_breakdown_chart(trade_history)
        figures['trade_breakdown'] = trade_breakdown_fig
    
    # 4. Create monthly performance heatmap
    # -------------------------------------
    monthly_heatmap_fig = create_monthly_performance_heatmap(trade_history)
    figures['monthly_heatmap'] = monthly_heatmap_fig
    
    # 5. Create key metrics indicators
    # --------------------------------
    metrics_fig = create_key_metrics_indicators(trade_history)
    figures['metrics'] = metrics_fig
    
    return figures

def create_animated_performance_chart(trade_history):
    """
    Create an animated chart showing performance over time
    
    Args:
        trade_history (pandas.DataFrame): Trading history with timestamps and profit/loss
        
    Returns:
        plotly.graph_objects.Figure: Animated Plotly figure
    """
    # Create figure
    fig = go.Figure()
    
    # Process data for animation
    trade_history = trade_history.sort_values('timestamp')
    
    # Extract date component for cleaner display
    trade_history['date'] = pd.to_datetime(trade_history['timestamp']).dt.date
    
    # Get unique dates for frames
    unique_dates = sorted(trade_history['date'].unique())
    
    if not unique_dates:
        # Return empty figure if no dates
        return fig
    
    # Initial data (first date)
    initial_data = trade_history[trade_history['date'] <= unique_dates[0]]
    
    # Add trace for cumulative P&L
    fig.add_trace(
        go.Scatter(
            x=initial_data['timestamp'],
            y=initial_data['cumulative_pl'],
            mode='lines+markers',
            name='Cumulative P&L',
            line=dict(width=3, color='rgba(0, 128, 255, 0.8)'),
            marker=dict(
                size=8,
                color='rgba(0, 128, 255, 0.8)',
                line=dict(width=1, color='rgba(0, 64, 128, 1)')
            ),
            fill='tozeroy',
            fillcolor='rgba(0, 128, 255, 0.2)'
        )
    )
    
    # Add markers for buy/sell points if available
    if 'type' in trade_history.columns:
        # Buy points
        buy_points = initial_data[initial_data['type'] == 'buy']
        if not buy_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_points['timestamp'],
                    y=buy_points['cumulative_pl'],
                    mode='markers',
                    name='Buy',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='green',
                        line=dict(width=1, color='darkgreen')
                    )
                )
            )
        
        # Sell points
        sell_points = initial_data[initial_data['type'] == 'sell']
        if not sell_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_points['timestamp'],
                    y=sell_points['cumulative_pl'],
                    mode='markers',
                    name='Sell',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color='red',
                        line=dict(width=1, color='darkred')
                    )
                )
            )
    
    # Create frames for animation
    frames = []
    for i, date in enumerate(unique_dates):
        frame_data = trade_history[trade_history['date'] <= date]
        
        frame = go.Frame(
            data=[
                go.Scatter(
                    x=frame_data['timestamp'],
                    y=frame_data['cumulative_pl'],
                    mode='lines+markers',
                    line=dict(width=3, color='rgba(0, 128, 255, 0.8)'),
                    marker=dict(
                        size=8,
                        color='rgba(0, 128, 255, 0.8)',
                        line=dict(width=1, color='rgba(0, 64, 128, 1)')
                    ),
                    fill='tozeroy',
                    fillcolor='rgba(0, 128, 255, 0.2)'
                )
            ],
            name=str(date)
        )
        
        # Add buy/sell points to frame if available
        if 'type' in trade_history.columns:
            buy_points = frame_data[frame_data['type'] == 'buy']
            if not buy_points.empty:
                frame.data = frame.data + (
                    go.Scatter(
                        x=buy_points['timestamp'],
                        y=buy_points['cumulative_pl'],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-up',
                            size=12,
                            color='green',
                            line=dict(width=1, color='darkgreen')
                        )
                    ),
                )
            
            sell_points = frame_data[frame_data['type'] == 'sell']
            if not sell_points.empty:
                frame.data = frame.data + (
                    go.Scatter(
                        x=sell_points['timestamp'],
                        y=sell_points['cumulative_pl'],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-down',
                            size=12,
                            color='red',
                            line=dict(width=1, color='darkred')
                        )
                    ),
                )
        
        frames.append(frame)
    
    fig.frames = frames
    
    # Add slider and play button
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None, 
                            dict(
                                frame=dict(duration=300, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=300, easing="cubic-in-out")
                            )
                        ]
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None], 
                            dict(
                                frame=dict(duration=0, redraw=True),
                                mode="immediate",
                                transition=dict(duration=0)
                            )
                        ]
                    )
                ],
                direction="left",
                pad=dict(r=10, t=10),
                x=0.1,
                y=0,
                xanchor="right",
                yanchor="top"
            )
        ],
        sliders=[
            dict(
                active=0,
                yanchor="top",
                xanchor="left",
                currentvalue=dict(
                    font=dict(size=16),
                    prefix="Date: ",
                    visible=True,
                    xanchor="right"
                ),
                transition=dict(duration=300, easing="cubic-in-out"),
                pad=dict(b=10, t=50),
                len=0.9,
                x=0.1,
                y=0,
                steps=[
                    dict(
                        method="animate",
                        args=[
                            [str(date)],
                            dict(
                                frame=dict(duration=300, redraw=True),
                                mode="immediate",
                                transition=dict(duration=300, easing="cubic-in-out")
                            )
                        ],
                        label=date.strftime("%Y-%m-%d")
                    )
                    for date in unique_dates
                ]
            )
        ]
    )
    
    # Update layout - using plotly_white template for better compatibility
    fig.update_layout(
        title="Trading Performance Over Time (Animated)",
        xaxis_title="Date",
        yaxis_title="Cumulative P&L ($)",
        height=600,
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=120),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12)
        )
    )
    
    return fig

def create_pl_distribution_chart(trade_history):
    """
    Create an interactive P&L distribution chart
    
    Args:
        trade_history (pandas.DataFrame): Trading history with profit/loss data
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    # Ensure we have profit_loss column
    if 'profit_loss' not in trade_history.columns:
        return None
    
    # Create figure
    fig = go.Figure()
    
    # Add histogram
    hist_values = trade_history['profit_loss'].dropna()
    
    # Add distribution with kde
    fig.add_trace(
        go.Histogram(
            x=hist_values,
            name='P&L Distribution',
            autobinx=True,
            marker_color='rgba(75, 192, 192, 0.7)',
            opacity=0.8,
            marker_line=dict(color='rgba(75, 192, 192, 1)', width=1)
        )
    )
    
    # Calculate statistics
    mean_pl = hist_values.mean()
    median_pl = hist_values.median()
    std_pl = hist_values.std()
    
    # Add vertical lines for mean and median
    fig.add_vline(
        x=mean_pl,
        line_width=2,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Mean: ${mean_pl:.2f}",
        annotation_position="top right"
    )
    
    fig.add_vline(
        x=median_pl,
        line_width=2,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Median: ${median_pl:.2f}",
        annotation_position="top left"
    )
    
    # Update layout
    fig.update_layout(
        title="P&L Distribution",
        xaxis_title="Profit/Loss ($)",
        yaxis_title="Frequency",
        height=400,
        template="plotly_white",  # Changed to white theme for consistency
        bargap=0.1,
        margin=dict(l=50, r=50, t=80, b=50),
        annotations=[
            dict(
                x=0.5,
                y=1.1,
                xref="paper",
                yref="paper",
                text=f"Avg: ${mean_pl:.2f} | Std Dev: ${std_pl:.2f}",
                showarrow=False,
                font=dict(size=14)
            )
        ]
    )
    
    return fig

def create_trade_breakdown_chart(trade_history):
    """
    Create an interactive trade breakdown chart
    
    Args:
        trade_history (pandas.DataFrame): Trading history with trade types
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    # Ensure we have type column
    if 'type' not in trade_history.columns:
        return None
    
    # Count trades by type
    trade_counts = trade_history['type'].value_counts()
    
    # Set colors
    colors = {
        'buy': 'rgba(75, 192, 192, 0.8)',
        'sell': 'rgba(255, 99, 132, 0.8)',
        'hold': 'rgba(54, 162, 235, 0.8)'
    }
    
    # Create trace colors
    trace_colors = [colors.get(t, 'rgba(128, 128, 128, 0.8)') for t in trade_counts.index]
    
    # Create pie chart
    fig = go.Figure(data=[
        go.Pie(
            labels=trade_counts.index,
            values=trade_counts.values,
            textinfo='label+percent',
            insidetextorientation='radial',
            textposition='inside',
            marker=dict(colors=trace_colors, line=dict(color='#000000', width=1)),
            hole=0.5,
            sort=False
        )
    ])
    
    # Calculate total trades
    total_trades = trade_counts.sum()
    
    # Update layout - updated to match other charts
    fig.update_layout(
        title="Trade Type Breakdown",
        height=400,
        template="plotly_white",  # Changed to white theme
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5
        ),
        annotations=[
            dict(
                text=f"Total Trades<br>{total_trades}",
                x=0.5,
                y=0.5,
                font_size=16,
                showarrow=False
            )
        ]
    )
    
    return fig

def create_monthly_performance_heatmap(trade_history):
    """
    Create a monthly performance heatmap
    
    Args:
        trade_history (pandas.DataFrame): Trading history with timestamps and profit/loss
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    # Ensure we have timestamp and profit_loss columns
    if 'timestamp' not in trade_history.columns or 'profit_loss' not in trade_history.columns:
        return None
    
    # Convert timestamp to datetime if not already
    trade_history['timestamp'] = pd.to_datetime(trade_history['timestamp'])
    
    # Extract month and year
    trade_history['month'] = trade_history['timestamp'].dt.month
    trade_history['year'] = trade_history['timestamp'].dt.year
    
    # Group by month and year, and calculate sum of profit/loss
    monthly_pl = trade_history.groupby(['year', 'month'])['profit_loss'].sum().reset_index()
    
    # Create a pivot table for the heatmap
    pivot_data = monthly_pl.pivot_table(index='month', columns='year', values='profit_loss')
    
    if pivot_data.empty:
        # Return empty figure if no data
        return go.Figure()
    
    # Get month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=[month_names[i-1] for i in pivot_data.index],
        colorscale=[
            [0.0, 'rgba(255, 50, 50, 0.8)'],
            [0.5, 'rgba(255, 255, 255, 0.0)'],
            [1.0, 'rgba(50, 255, 50, 0.8)']
        ],
        colorbar=dict(
            title=dict(text="P&L ($)", side="right")
        ),
        text=[[f"${val:.2f}" for val in row] for row in pivot_data.values],
        hoverinfo="text",
        texttemplate="%{text}",
        textfont={"size": 12}
    ))
    
    # Update layout - updated to match other charts
    fig.update_layout(
        title="Monthly Performance Heatmap",
        height=400,
        template="plotly_white",  # Changed to white theme
        margin=dict(l=50, r=50, t=80, b=50),
        xaxis=dict(
            title="Year"
        ),
        yaxis=dict(
            title="Month"
        )
    )
    
    return fig

def create_key_metrics_indicators(trade_history):
    """
    Create a dashboard of key performance metrics
    
    Args:
        trade_history (pandas.DataFrame): Trading history with profit/loss data
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure with indicators
    """
    # Ensure we have profit_loss column
    if 'profit_loss' not in trade_history.columns:
        return None
    
    # Calculate key metrics
    total_profit = trade_history['profit_loss'].sum()
    avg_profit = trade_history['profit_loss'].mean()
    win_rate = (trade_history['profit_loss'] > 0).mean() * 100
    
    # Handle case where there are no negative trades
    negative_sum = abs(trade_history.loc[trade_history['profit_loss'] < 0, 'profit_loss'].sum())
    positive_sum = abs(trade_history.loc[trade_history['profit_loss'] > 0, 'profit_loss'].sum())
    
    # Calculate profit factor (avoid division by zero)
    if negative_sum > 0:
        profit_factor = positive_sum / negative_sum
    else:
        profit_factor = positive_sum if positive_sum > 0 else 1.0
    
    # Create metrics dashboard
    fig = go.Figure()
    
    # Total Profit/Loss
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=total_profit,
        number={"prefix": "$", "valueformat": ",.2f"},
        delta={"position": "top", "reference": 0, "valueformat": ",.2f"},
        title={"text": "Total P&L"},
        domain={"row": 0, "column": 0}
    ))
    
    # Average Profit/Loss
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=avg_profit,
        number={"prefix": "$", "valueformat": ",.2f"},
        delta={"position": "top", "reference": 0, "valueformat": ",.2f"},
        title={"text": "Avg. P&L per Trade"},
        domain={"row": 0, "column": 1}
    ))
    
    # Win Rate - Simplified gauge to avoid potential issues
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=win_rate,
        number={"suffix": "%", "valueformat": ".1f"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "rgba(75, 192, 192, 0.8)"},
            "steps": [
                {"range": [0, 40], "color": "rgba(255, 50, 50, 0.5)"},
                {"range": [40, 60], "color": "rgba(255, 255, 50, 0.5)"},
                {"range": [60, 100], "color": "rgba(50, 255, 50, 0.5)"}
            ]
        },
        title={"text": "Win Rate"},
        domain={"row": 1, "column": 0}
    ))
    
    # Profit Factor - Simplified gauge to avoid potential issues
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=min(profit_factor, 3.0),  # Cap at 3.0 to avoid scale issues
        number={"valueformat": ".2f"},
        gauge={
            "axis": {"range": [0, 3]},
            "bar": {"color": "rgba(54, 162, 235, 0.8)"},
            "steps": [
                {"range": [0, 1], "color": "rgba(255, 50, 50, 0.5)"},
                {"range": [1, 2], "color": "rgba(255, 255, 50, 0.5)"},
                {"range": [2, 3], "color": "rgba(50, 255, 50, 0.5)"}
            ]
        },
        title={"text": "Profit Factor"},
        domain={"row": 1, "column": 1}
    ))
    
    # Update layout
    fig.update_layout(
        grid={"rows": 2, "columns": 2, "pattern": "independent"},
        title="Key Trading Metrics",
        height=400,
        template="plotly_white",  # Changed to white template
        margin=dict(l=50, r=50, t=80, b=20)
    )
    
    return fig
