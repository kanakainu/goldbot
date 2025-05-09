"""
MetaTrader 5 Connector Module

This module provides integration with MetaTrader 5 via REST API or direct socket connection.
It enables retrieving real-time data and executing trades through a MetaTrader 5 terminal.
"""

import requests
import json
import pandas as pd
import time
import datetime
from typing import Dict, List, Tuple, Optional, Union, Any

class MT5Connector:
    """
    Class for connecting to and interacting with MetaTrader 5
    """
    
    def __init__(self, api_url: str = None, api_key: str = None, account_number: str = None):
        """
        Initialize the MT5 connector
        
        Args:
            api_url (str, optional): URL of the MetaTrader 5 HTTP REST API bridge
            api_key (str, optional): API key for authentication
            account_number (str, optional): MT5 account number
        """
        self.api_url = api_url
        self.api_key = api_key
        self.account_number = account_number
        self.is_connected = False
        self.last_error = None
        self.symbols_info = {}
        
    def connect(self) -> bool:
        """
        Connect to MetaTrader 5 via API
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        if not self.api_url or not self.api_key:
            self.last_error = "API URL and API key are required for connection"
            return False
        
        try:
            # Test connection with a simple request
            response = self._send_request("ping", {})
            
            if response and response.get("status") == "success":
                self.is_connected = True
                
                # Get available symbols
                self._fetch_symbols()
                
                return True
            else:
                self.last_error = "Failed to connect to MetaTrader 5 API"
                return False
                
        except Exception as e:
            self.last_error = f"Connection error: {str(e)}"
            return False
            
    def disconnect(self) -> bool:
        """
        Disconnect from MetaTrader 5
        
        Returns:
            bool: True if disconnection successful, False otherwise
        """
        self.is_connected = False
        return True
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information
        
        Returns:
            dict: Account information
        """
        if not self.is_connected:
            self.last_error = "Not connected to MetaTrader 5"
            return {}
            
        try:
            response = self._send_request("account_info", {
                "account": self.account_number
            })
            
            if response and response.get("status") == "success":
                return response.get("data", {})
            else:
                self.last_error = f"Failed to get account info: {response.get('message', '')}"
                return {}
                
        except Exception as e:
            self.last_error = f"Error getting account info: {str(e)}"
            return {}
    
    def get_current_price(self, symbol: str) -> Tuple[Optional[float], Optional[str]]:
        """
        Get current price for a symbol
        
        Args:
            symbol (str): Trading symbol (e.g., "XAUUSD" for gold)
            
        Returns:
            tuple: (current_price, timestamp) or (None, None) if failed
        """
        if not self.is_connected:
            self.last_error = "Not connected to MetaTrader 5"
            return None, None
            
        try:
            response = self._send_request("get_symbol_price", {
                "symbol": symbol
            })
            
            if response and response.get("status") == "success":
                data = response.get("data", {})
                price = (data.get("bid", 0) + data.get("ask", 0)) / 2
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                return price, timestamp
            else:
                self.last_error = f"Failed to get price: {response.get('message', '')}"
                return None, None
                
        except Exception as e:
            self.last_error = f"Error getting price: {str(e)}"
            return None, None
    
    def get_historical_data(self, symbol: str, timeframe: str, start_time: datetime.datetime, 
                          end_time: datetime.datetime) -> pd.DataFrame:
        """
        Get historical price data
        
        Args:
            symbol (str): Trading symbol (e.g., "XAUUSD" for gold)
            timeframe (str): Timeframe (e.g., "M1", "M5", "H1", "D1")
            start_time (datetime): Start time
            end_time (datetime): End time
            
        Returns:
            pd.DataFrame: Historical data
        """
        if not self.is_connected:
            self.last_error = "Not connected to MetaTrader 5"
            return pd.DataFrame()
            
        try:
            response = self._send_request("get_historical_data", {
                "symbol": symbol,
                "timeframe": timeframe,
                "start_time": start_time.timestamp(),
                "end_time": end_time.timestamp()
            })
            
            if response and response.get("status") == "success":
                data = response.get("data", [])
                if data:
                    df = pd.DataFrame(data)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df.set_index('time', inplace=True)
                    return df
                else:
                    return pd.DataFrame()
            else:
                self.last_error = f"Failed to get historical data: {response.get('message', '')}"
                return pd.DataFrame()
                
        except Exception as e:
            self.last_error = f"Error getting historical data: {str(e)}"
            return pd.DataFrame()
    
    def execute_trade(self, symbol: str, order_type: str, volume: float, 
                     price: Optional[float] = None, sl: Optional[float] = None, 
                     tp: Optional[float] = None, comment: str = "") -> Dict[str, Any]:
        """
        Execute a trade
        
        Args:
            symbol (str): Trading symbol (e.g., "XAUUSD" for gold)
            order_type (str): Order type ("BUY", "SELL", "BUY_LIMIT", "SELL_LIMIT", etc.)
            volume (float): Trade volume in lots
            price (float, optional): Price for pending orders
            sl (float, optional): Stop loss price
            tp (float, optional): Take profit price
            comment (str, optional): Trade comment
            
        Returns:
            dict: Order result information
        """
        if not self.is_connected:
            self.last_error = "Not connected to MetaTrader 5"
            return {"success": False, "error": self.last_error}
            
        try:
            request_data = {
                "symbol": symbol,
                "order_type": order_type,
                "volume": volume,
                "comment": comment
            }
            
            # Add optional parameters if provided
            if price is not None:
                request_data["price"] = price
            if sl is not None:
                request_data["sl"] = sl
            if tp is not None:
                request_data["tp"] = tp
            
            response = self._send_request("execute_trade", request_data)
            
            if response and response.get("status") == "success":
                return {
                    "success": True,
                    "order_id": response.get("data", {}).get("order", 0),
                    "message": "Order executed successfully"
                }
            else:
                self.last_error = f"Failed to execute trade: {response.get('message', '')}"
                return {"success": False, "error": self.last_error}
                
        except Exception as e:
            self.last_error = f"Error executing trade: {str(e)}"
            return {"success": False, "error": self.last_error}
    
    def close_position(self, position_id: int) -> bool:
        """
        Close an open position
        
        Args:
            position_id (int): Position identifier
            
        Returns:
            bool: True if position closed successfully, False otherwise
        """
        if not self.is_connected:
            self.last_error = "Not connected to MetaTrader 5"
            return False
            
        try:
            response = self._send_request("close_position", {
                "position_id": position_id
            })
            
            if response and response.get("status") == "success":
                return True
            else:
                self.last_error = f"Failed to close position: {response.get('message', '')}"
                return False
                
        except Exception as e:
            self.last_error = f"Error closing position: {str(e)}"
            return False
    
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """
        Get all open positions
        
        Returns:
            list: List of open positions
        """
        if not self.is_connected:
            self.last_error = "Not connected to MetaTrader 5"
            return []
            
        try:
            response = self._send_request("get_open_positions", {})
            
            if response and response.get("status") == "success":
                return response.get("data", [])
            else:
                self.last_error = f"Failed to get open positions: {response.get('message', '')}"
                return []
                
        except Exception as e:
            self.last_error = f"Error getting open positions: {str(e)}"
            return []
    
    def _send_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a request to the MT5 API
        
        Args:
            endpoint (str): API endpoint
            data (dict): Request data
            
        Returns:
            dict: Response data
        """
        if not self.api_url:
            return {"status": "error", "message": "API URL not specified"}
        
        url = f"{self.api_url}/{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        response = requests.post(url, json=data, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "status": "error",
                "message": f"HTTP Error {response.status_code}: {response.text}"
            }
    
    def _fetch_symbols(self) -> None:
        """
        Fetch available symbols from MT5
        """
        try:
            response = self._send_request("get_symbols", {})
            
            if response and response.get("status") == "success":
                symbols_data = response.get("data", [])
                self.symbols_info = {s["name"]: s for s in symbols_data}
            
        except Exception as e:
            self.last_error = f"Error fetching symbols: {str(e)}"
            
    def get_last_error(self) -> str:
        """
        Get the last error message
        
        Returns:
            str: Last error message
        """
        return self.last_error or "No error"


# Factory function to create a connection
def create_mt5_connection(api_url: str, api_key: str, account_number: str = None) -> MT5Connector:
    """
    Create and initialize a connection to MetaTrader 5
    
    Args:
        api_url (str): URL of the MetaTrader 5 HTTP REST API bridge
        api_key (str): API key for authentication
        account_number (str, optional): MT5 account number
        
    Returns:
        MT5Connector: Initialized MT5 connector
    """
    connector = MT5Connector(api_url, api_key, account_number)
    connector.connect()
    return connector