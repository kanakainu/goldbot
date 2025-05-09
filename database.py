import os
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import json

# Get database URL from environment variables
DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/postgres')

# Create SQLAlchemy engine
print(f"Connecting to database with URL: {DATABASE_URL}")
engine = create_engine(DATABASE_URL)

# Create a base class for models
Base = declarative_base()

# Define models
class GoldPrice(Base):
    """Model for storing gold price data"""
    __tablename__ = 'gold_prices'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    close = Column(Float, nullable=False)
    open = Column(Float, nullable=True)
    high = Column(Float, nullable=True)
    low = Column(Float, nullable=True)
    volume = Column(Float, nullable=True)
    daily_return = Column(Float, nullable=True)
    
    def __repr__(self):
        return f"<GoldPrice(timestamp='{self.timestamp}', close='{self.close}')>"

class TradingSignal(Base):
    """Model for storing trading signals"""
    __tablename__ = 'trading_signals'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    signal_type = Column(Integer, nullable=False)  # 1 = buy, -1 = sell, 0 = hold
    price = Column(Float, nullable=False)
    strategy = Column(String(50), nullable=False)
    threshold = Column(Float, nullable=False)
    indicators = Column(JSON, nullable=True)  # Store indicator values as JSON
    
    def __repr__(self):
        return f"<TradingSignal(timestamp='{self.timestamp}', signal_type='{self.signal_type}')>"

class Trade(Base):
    """Model for storing trades"""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    action = Column(String(20), nullable=False)  # 'BUY', 'SELL', 'STOP_LOSS', 'TAKE_PROFIT'
    price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    value = Column(Float, nullable=False)
    pnl = Column(Float, nullable=True)
    pnl_percent = Column(Float, nullable=True)
    capital_after = Column(Float, nullable=False)
    strategy = Column(String(50), nullable=True)
    
    def __repr__(self):
        return f"<Trade(timestamp='{self.timestamp}', action='{self.action}', price='{self.price}')>"

class TradingSession(Base):
    """Model for storing trading session data"""
    __tablename__ = 'trading_sessions'
    
    id = Column(Integer, primary_key=True)
    start_time = Column(DateTime, nullable=False, index=True)
    end_time = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    strategy = Column(String(50), nullable=False)
    initial_capital = Column(Float, nullable=False)
    final_capital = Column(Float, nullable=True)
    profit_loss = Column(Float, nullable=True)
    profit_loss_percent = Column(Float, nullable=True)
    num_trades = Column(Integer, default=0)
    win_rate = Column(Float, nullable=True)
    parameters = Column(JSON, nullable=True)  # Store strategy parameters as JSON
    
    def __repr__(self):
        return f"<TradingSession(id='{self.id}', strategy='{self.strategy}')>"

class ModelPrediction(Base):
    """Model for storing ML model predictions"""
    __tablename__ = 'model_predictions'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    model_type = Column(String(50), nullable=False)
    predicted_price = Column(Float, nullable=False)
    actual_price = Column(Float, nullable=True)  # Can be filled later when actual price is known
    prediction_horizon = Column(Integer, nullable=False)  # Days ahead
    features_used = Column(JSON, nullable=True)  # Store features used for prediction
    error = Column(Float, nullable=True)  # Error once actual price is known
    
    def __repr__(self):
        return f"<ModelPrediction(timestamp='{self.timestamp}', predicted_price='{self.predicted_price}')>"

# Create the tables
Base.metadata.create_all(engine)

# Create a session factory
Session = sessionmaker(bind=engine)

# Database operations
def get_session():
    """Create and return a new database session"""
    return Session()

def save_gold_price(price_data):
    """
    Save gold price data to the database
    
    Args:
        price_data (dict or pandas.DataFrame): Gold price data to save
    """
    session = get_session()
    
    try:
        if isinstance(price_data, pd.DataFrame):
            # Handle DataFrame input
            for index, row in price_data.iterrows():
                # Check if this timestamp already exists
                existing = session.query(GoldPrice).filter_by(timestamp=index).first()
                if existing:
                    # Update existing record
                    existing.close = row.get('Close', row.get('close', existing.close))
                    existing.open = row.get('Open', row.get('open', existing.open))
                    existing.high = row.get('High', row.get('high', existing.high))
                    existing.low = row.get('Low', row.get('low', existing.low))
                    existing.volume = row.get('Volume', row.get('volume', existing.volume))
                    existing.daily_return = row.get('Daily_Return', row.get('daily_return', existing.daily_return))
                else:
                    # Create new record
                    gold_price = GoldPrice(
                        timestamp=index,
                        close=row.get('Close', row.get('close', 0)),
                        open=row.get('Open', row.get('open', None)),
                        high=row.get('High', row.get('high', None)),
                        low=row.get('Low', row.get('low', None)),
                        volume=row.get('Volume', row.get('volume', None)),
                        daily_return=row.get('Daily_Return', row.get('daily_return', None))
                    )
                    session.add(gold_price)
        else:
            # Handle dictionary input (single record)
            timestamp = price_data.get('timestamp')
            if isinstance(timestamp, str):
                timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            
            # Check if this timestamp already exists
            existing = session.query(GoldPrice).filter_by(timestamp=timestamp).first()
            if existing:
                # Update existing record
                existing.close = price_data.get('close', price_data.get('Close', existing.close))
                existing.open = price_data.get('open', price_data.get('Open', existing.open))
                existing.high = price_data.get('high', price_data.get('High', existing.high))
                existing.low = price_data.get('low', price_data.get('Low', existing.low))
                existing.volume = price_data.get('volume', price_data.get('Volume', existing.volume))
                existing.daily_return = price_data.get('daily_return', price_data.get('Daily_Return', existing.daily_return))
            else:
                # Create new record
                gold_price = GoldPrice(
                    timestamp=timestamp,
                    close=price_data.get('close', price_data.get('Close', 0)),
                    open=price_data.get('open', price_data.get('Open', None)),
                    high=price_data.get('high', price_data.get('High', None)),
                    low=price_data.get('low', price_data.get('Low', None)),
                    volume=price_data.get('volume', price_data.get('Volume', None)),
                    daily_return=price_data.get('daily_return', price_data.get('Daily_Return', None))
                )
                session.add(gold_price)
        
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"Error saving gold price data: {str(e)}")
    finally:
        session.close()

def get_gold_prices(start_date=None, end_date=None):
    """
    Get gold prices from the database for a specified date range
    
    Args:
        start_date (datetime): Start date for the range
        end_date (datetime): End date for the range
        
    Returns:
        pandas.DataFrame: Gold price data as a DataFrame
    """
    session = get_session()
    
    try:
        query = session.query(GoldPrice)
        
        if start_date:
            query = query.filter(GoldPrice.timestamp >= start_date)
        if end_date:
            query = query.filter(GoldPrice.timestamp <= end_date)
        
        # Order by timestamp
        query = query.order_by(GoldPrice.timestamp)
        
        # Convert to DataFrame
        results = query.all()
        if not results:
            return pd.DataFrame()
        
        # Create DataFrame
        data = []
        for r in results:
            data.append({
                'timestamp': r.timestamp,
                'Close': r.close,
                'Open': r.open if r.open is not None else r.close,
                'High': r.high if r.high is not None else r.close,
                'Low': r.low if r.low is not None else r.close,
                'Volume': r.volume,
                'Daily_Return': r.daily_return
            })
        
        df = pd.DataFrame(data)
        if not df.empty:
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Calculate missing values
            if 'Daily_Return' not in df.columns or df['Daily_Return'].isnull().any():
                df['Daily_Return'] = df['Close'].pct_change()
        
        return df
    
    except Exception as e:
        print(f"Error getting gold prices: {str(e)}")
        return pd.DataFrame()
    finally:
        session.close()

def save_trade(trade_data):
    """
    Save a trade to the database
    
    Args:
        trade_data (dict): Trade data to save
    """
    session = get_session()
    
    try:
        # Convert timestamp if it's a string
        timestamp = trade_data.get('timestamp')
        if isinstance(timestamp, str):
            timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        
        # Create trade object
        trade = Trade(
            timestamp=timestamp,
            action=trade_data.get('action', ''),
            price=trade_data.get('price', 0),
            quantity=trade_data.get('quantity', 0),
            value=trade_data.get('value', 0),
            pnl=trade_data.get('pnl', None),
            pnl_percent=trade_data.get('pnl_percent', None),
            capital_after=trade_data.get('capital_after', 0),
            strategy=trade_data.get('strategy', None)
        )
        
        session.add(trade)
        session.commit()
        
        # Update the trading session if one is active
        active_session = session.query(TradingSession).filter_by(is_active=True).first()
        if active_session:
            active_session.num_trades += 1
            active_session.final_capital = trade_data.get('capital_after', active_session.final_capital)
            active_session.profit_loss = active_session.final_capital - active_session.initial_capital
            active_session.profit_loss_percent = (active_session.profit_loss / active_session.initial_capital) * 100
            
            # Calculate win rate
            win_trades = session.query(Trade).filter(
                Trade.action.in_(['SELL', 'TAKE_PROFIT']),
                Trade.pnl > 0
            ).count()
            
            total_closed_trades = session.query(Trade).filter(
                Trade.action.in_(['SELL', 'STOP_LOSS', 'TAKE_PROFIT'])
            ).count()
            
            if total_closed_trades > 0:
                active_session.win_rate = (win_trades / total_closed_trades) * 100
            
            session.commit()
        
        return True
    
    except Exception as e:
        session.rollback()
        print(f"Error saving trade: {str(e)}")
        return False
    finally:
        session.close()

def get_trades(limit=None):
    """
    Get trades from the database
    
    Args:
        limit (int, optional): Maximum number of trades to return
        
    Returns:
        pandas.DataFrame: Trade data as a DataFrame
    """
    session = get_session()
    
    try:
        query = session.query(Trade).order_by(Trade.timestamp.desc())
        
        if limit:
            query = query.limit(limit)
        
        # Convert to DataFrame
        results = query.all()
        if not results:
            return pd.DataFrame()
        
        # Create DataFrame
        data = []
        for r in results:
            data.append({
                'timestamp': r.timestamp,
                'action': r.action,
                'price': r.price,
                'quantity': r.quantity,
                'value': r.value,
                'pnl': r.pnl,
                'pnl_percent': r.pnl_percent,
                'capital_after': r.capital_after,
                'strategy': r.strategy
            })
        
        return pd.DataFrame(data)
    
    except Exception as e:
        print(f"Error getting trades: {str(e)}")
        return pd.DataFrame()
    finally:
        session.close()

def get_trading_history(start_date=None, end_date=None, limit=None):
    """
    Get trading history from the database, formatted for the performance dashboard
    
    Args:
        start_date (datetime, optional): Start date for the range
        end_date (datetime, optional): End date for the range
        limit (int, optional): Maximum number of trades to return
        
    Returns:
        pandas.DataFrame: Trading history data as a DataFrame
    """
    session = get_session()
    
    try:
        query = session.query(Trade).order_by(Trade.timestamp.asc())
        
        if start_date:
            query = query.filter(Trade.timestamp >= start_date)
        if end_date:
            query = query.filter(Trade.timestamp <= end_date)
        if limit:
            query = query.limit(limit)
        
        # Convert to DataFrame
        results = query.all()
        if not results:
            return pd.DataFrame()
        
        # Create DataFrame
        data = []
        cumulative_pl = 0
        
        for r in results:
            # Calculate profit/loss
            profit_loss = r.pnl if r.pnl is not None else 0
            
            # Update cumulative P&L
            cumulative_pl += profit_loss
            
            # Map action to type for the dashboard
            trade_type = 'buy' if r.action == 'BUY' else 'sell' if r.action in ['SELL', 'STOP_LOSS', 'TAKE_PROFIT'] else 'hold'
            
            data.append({
                'timestamp': r.timestamp,
                'date': r.timestamp.date(),
                'type': trade_type,
                'price': r.price,
                'quantity': r.quantity,
                'value': r.value,
                'profit_loss': profit_loss,
                'profit_loss_percent': r.pnl_percent,
                'cumulative_pl': cumulative_pl,
                'strategy': r.strategy
            })
        
        df = pd.DataFrame(data)
        return df
    
    except Exception as e:
        print(f"Error getting trading history: {str(e)}")
        return pd.DataFrame()
    finally:
        session.close()

def start_trading_session(strategy, initial_capital, parameters=None):
    """
    Start a new trading session
    
    Args:
        strategy (str): Trading strategy name
        initial_capital (float): Initial capital for the session
        parameters (dict, optional): Strategy parameters
        
    Returns:
        int: Session ID
    """
    session = get_session()
    
    try:
        # Check if there's already an active session
        active_session = session.query(TradingSession).filter_by(is_active=True).first()
        if active_session:
            # End the current session
            end_trading_session(active_session.id)
        
        # Create new session
        trading_session = TradingSession(
            start_time=datetime.now(),
            strategy=strategy,
            initial_capital=initial_capital,
            final_capital=initial_capital,
            parameters=parameters
        )
        
        session.add(trading_session)
        session.commit()
        
        return trading_session.id
    
    except Exception as e:
        session.rollback()
        print(f"Error starting trading session: {str(e)}")
        return None
    finally:
        session.close()

def end_trading_session(session_id=None):
    """
    End a trading session
    
    Args:
        session_id (int, optional): Session ID to end. If None, ends the active session.
        
    Returns:
        bool: Success
    """
    session = get_session()
    
    try:
        if session_id:
            trading_session = session.query(TradingSession).filter_by(id=session_id).first()
        else:
            trading_session = session.query(TradingSession).filter_by(is_active=True).first()
        
        if trading_session:
            trading_session.end_time = datetime.now()
            trading_session.is_active = False
            session.commit()
            return True
        
        return False
    
    except Exception as e:
        session.rollback()
        print(f"Error ending trading session: {str(e)}")
        return False
    finally:
        session.close()

def get_active_session():
    """
    Get the active trading session
    
    Returns:
        dict: Active session data
    """
    session = get_session()
    
    try:
        active_session = session.query(TradingSession).filter_by(is_active=True).first()
        
        if active_session:
            return {
                'id': active_session.id,
                'start_time': active_session.start_time,
                'strategy': active_session.strategy,
                'initial_capital': active_session.initial_capital,
                'current_capital': active_session.final_capital,
                'profit_loss': active_session.profit_loss,
                'profit_loss_percent': active_session.profit_loss_percent,
                'num_trades': active_session.num_trades,
                'win_rate': active_session.win_rate,
                'parameters': active_session.parameters
            }
        
        return None
    
    except Exception as e:
        print(f"Error getting active session: {str(e)}")
        return None
    finally:
        session.close()

def save_trading_signal(signal_data):
    """
    Save a trading signal to the database
    
    Args:
        signal_data (dict): Signal data to save
    """
    session = get_session()
    
    try:
        # Convert timestamp if it's a string
        timestamp = signal_data.get('timestamp')
        if isinstance(timestamp, str):
            timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        
        # Create signal object
        signal = TradingSignal(
            timestamp=timestamp,
            signal_type=signal_data.get('signal_type', 0),
            price=signal_data.get('price', 0),
            strategy=signal_data.get('strategy', ''),
            threshold=signal_data.get('threshold', 0),
            indicators=signal_data.get('indicators', {})
        )
        
        session.add(signal)
        session.commit()
        
        return True
    
    except Exception as e:
        session.rollback()
        print(f"Error saving trading signal: {str(e)}")
        return False
    finally:
        session.close()

def get_trading_signals(start_date=None, end_date=None, limit=None):
    """
    Get trading signals from the database
    
    Args:
        start_date (datetime): Start date for the range
        end_date (datetime): End date for the range
        limit (int, optional): Maximum number of signals to return
        
    Returns:
        pandas.DataFrame: Signal data as a DataFrame
    """
    session = get_session()
    
    try:
        query = session.query(TradingSignal)
        
        if start_date:
            query = query.filter(TradingSignal.timestamp >= start_date)
        if end_date:
            query = query.filter(TradingSignal.timestamp <= end_date)
        
        # Order by timestamp
        query = query.order_by(TradingSignal.timestamp.desc())
        
        if limit:
            query = query.limit(limit)
        
        # Convert to DataFrame
        results = query.all()
        if not results:
            return pd.DataFrame()
        
        # Create DataFrame
        data = []
        for r in results:
            data.append({
                'timestamp': r.timestamp,
                'signal_type': r.signal_type,
                'price': r.price,
                'strategy': r.strategy,
                'threshold': r.threshold,
                'indicators': r.indicators
            })
        
        return pd.DataFrame(data)
    
    except Exception as e:
        print(f"Error getting trading signals: {str(e)}")
        return pd.DataFrame()
    finally:
        session.close()

def save_model_prediction(prediction_data):
    """
    Save a model prediction to the database
    
    Args:
        prediction_data (dict): Prediction data to save
    """
    session = get_session()
    
    try:
        # Convert timestamp if it's a string
        timestamp = prediction_data.get('timestamp')
        if isinstance(timestamp, str):
            timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        
        # Create prediction object
        prediction = ModelPrediction(
            timestamp=timestamp,
            model_type=prediction_data.get('model_type', ''),
            predicted_price=prediction_data.get('predicted_price', 0),
            actual_price=prediction_data.get('actual_price', None),
            prediction_horizon=prediction_data.get('prediction_horizon', 1),
            features_used=prediction_data.get('features_used', {}),
            error=prediction_data.get('error', None)
        )
        
        session.add(prediction)
        session.commit()
        
        return True
    
    except Exception as e:
        session.rollback()
        print(f"Error saving model prediction: {str(e)}")
        return False
    finally:
        session.close()

def get_model_predictions(model_type=None, limit=None):
    """
    Get model predictions from the database
    
    Args:
        model_type (str, optional): Filter by model type
        limit (int, optional): Maximum number of predictions to return
        
    Returns:
        pandas.DataFrame: Prediction data as a DataFrame
    """
    session = get_session()
    
    try:
        query = session.query(ModelPrediction)
        
        if model_type:
            query = query.filter_by(model_type=model_type)
        
        # Order by timestamp
        query = query.order_by(ModelPrediction.timestamp.desc())
        
        if limit:
            query = query.limit(limit)
        
        # Convert to DataFrame
        results = query.all()
        if not results:
            return pd.DataFrame()
        
        # Create DataFrame
        data = []
        for r in results:
            data.append({
                'timestamp': r.timestamp,
                'model_type': r.model_type,
                'predicted_price': r.predicted_price,
                'actual_price': r.actual_price,
                'prediction_horizon': r.prediction_horizon,
                'features_used': r.features_used,
                'error': r.error
            })
        
        return pd.DataFrame(data)
    
    except Exception as e:
        print(f"Error getting model predictions: {str(e)}")
        return pd.DataFrame()
    finally:
        session.close()

def update_model_prediction_actual(prediction_id, actual_price):
    """
    Update a model prediction with the actual price
    
    Args:
        prediction_id (int): Prediction ID
        actual_price (float): Actual price
    """
    session = get_session()
    
    try:
        prediction = session.query(ModelPrediction).filter_by(id=prediction_id).first()
        
        if prediction:
            prediction.actual_price = actual_price
            prediction.error = abs(prediction.predicted_price - actual_price)
            session.commit()
            return True
        
        return False
    
    except Exception as e:
        session.rollback()
        print(f"Error updating model prediction: {str(e)}")
        return False
    finally:
        session.close()