import os
import time
import logging
import hmac
import hashlib
from urllib.parse import urlencode
import requests
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from config import ExchangeConfig


class BinanceInterface:
    """
    Wrapper for the Binance API to handle all exchange-related operations.
    Implements both REST API and websocket connections for data retrieval and trading.
    """
    
    def __init__(self, config=None, testnet=False):
        """
        Initialize the Binance API interface
        
        Args:
            config: Configuration object for exchange settings
            testnet: Whether to use the testnet (default: False)
        """
        self.config = config or ExchangeConfig()
        self.testnet = testnet
        self.logger = self._setup_logger()
        
        # Set up API base URLs
        if testnet:
            self.base_url = 'https://testnet.binance.vision/api'
            self.base_wss = 'wss://testnet.binance.vision/ws'
        else:
            self.base_url = 'https://api.binance.com/api'
            self.base_wss = 'wss://stream.binance.com:9443/ws'
            
        # Load API credentials
        self.api_key = self._get_api_key()
        self.api_secret = self._get_api_secret()
        
        # Session for HTTP requests
        self.session = requests.Session()
        self.session.headers.update({
            'X-MBX-APIKEY': self.api_key
        })
        
        # Active websocket connections
        self.ws_connections = {}
        
    def _setup_logger(self):
        """
        Set up logging for the Binance interface
        
        Returns:
            Logger instance
        """
        logger = logging.getLogger('BinanceInterface')
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # Create file handler if log path exists
        if hasattr(self.config, 'LOG_PATH') and self.config.LOG_PATH:
            os.makedirs(os.path.dirname(self.config.LOG_PATH), exist_ok=True)
            fh = logging.FileHandler(self.config.LOG_PATH)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            
        return logger
    
    def _get_api_key(self):
        """
        Get API key from environment variable or config
        
        Returns:
            API key as string
        """
        # Try environment variable first
        api_key = os.environ.get('BINANCE_API_KEY')
        
        # Fall back to config
        if not api_key and hasattr(self.config, 'API_KEY'):
            api_key = self.config.API_KEY
            
        if not api_key:
            self.logger.warning("No API key found. Some endpoints will not be accessible.")
            
        return api_key
        
    def _get_api_secret(self):
        """
        Get API secret from environment variable or config
        
        Returns:
            API secret as string
        """
        # Try environment variable first
        api_secret = os.environ.get('BINANCE_API_SECRET')
        
        # Fall back to config
        if not api_secret and hasattr(self.config, 'API_SECRET'):
            api_secret = self.config.API_SECRET
            
        if not api_secret:
            self.logger.warning("No API secret found. Some endpoints will not be accessible.")
            
        return api_secret
    
    def _generate_signature(self, params: Dict) -> str:
        """
        Generate signature for authenticated requests
        
        Args:
            params: Request parameters
            
        Returns:
            HMAC SHA256 signature
        """
        if not self.api_secret:
            raise ValueError("API secret is required for authenticated requests")
            
        query_string = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
        
    def _request(self, method: str, endpoint: str, signed: bool = False, 
                params: Dict = None) -> Dict:
        """
        Make a request to the Binance API
        
        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint
            signed: Whether the request requires a signature
            params: Request parameters
            
        Returns:
            Response JSON as dictionary
        """
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        
        # Add timestamp for signed requests
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._generate_signature(params)
        
        try:
            response = None
            
            if method == 'GET':
                response = self.session.get(url, params=params)
            elif method == 'POST':
                response = self.session.post(url, params=params)
            elif method == 'DELETE':
                response = self.session.delete(url, params=params)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
                
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request error: {e}")
            
            if response is not None:
                self.logger.error(f"Response: {response.text}")
                
            raise
    
    # ===== PUBLIC API ENDPOINTS =====
    
    def get_exchange_info(self) -> Dict:
        """
        Get exchange information
        
        Returns:
            Exchange information as dictionary
        """
        return self._request('GET', '/v3/exchangeInfo')
    
    def get_server_time(self) -> int:
        """
        Get server time
        
        Returns:
            Server time in milliseconds
        """
        response = self._request('GET', '/v3/time')
        return response['serverTime']
    
    def get_ticker(self, symbol: str) -> Dict:
        """
        Get ticker price for a symbol
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            
        Returns:
            Ticker information
        """
        return self._request('GET', '/v3/ticker/price', params={'symbol': symbol})
    
    def get_klines(self, symbol: str, interval: str, limit: int = 500, 
                  start_time: int = None, end_time: int = None) -> List[List]:
        """
        Get candlestick data
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            limit: Number of candles to return (max 1000)
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            
        Returns:
            List of candlestick data
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = start_time
            
        if end_time:
            params['endTime'] = end_time
            
        return self._request('GET', '/v3/klines', params=params)
    
    def get_klines_df(self, symbol: str, interval: str, limit: int = 500, 
                     start_time: int = None, end_time: int = None) -> pd.DataFrame:
        """
        Get candlestick data as DataFrame
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            limit: Number of candles to return (max 1000)
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            
        Returns:
            DataFrame with candlestick data
        """
        klines = self.get_klines(symbol, interval, limit, start_time, end_time)
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert types
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                          'quote_asset_volume', 'taker_buy_base_asset_volume', 
                          'taker_buy_quote_asset_volume']
                         
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])
            
        # Convert timestamps to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """
        Get order book for a symbol
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            limit: Depth of the order book (max 5000)
            
        Returns:
            Order book data
        """
        return self._request('GET', '/v3/depth', params={
            'symbol': symbol,
            'limit': limit
        })
    
    # ===== AUTHENTICATED API ENDPOINTS =====
    
    def get_account_info(self) -> Dict:
        """
        Get account information (requires API key and signature)
        
        Returns:
            Account information
        """
        return self._request('GET', '/v3/account', signed=True)
    
    def get_balances(self) -> List[Dict]:
        """
        Get all asset balances
        
        Returns:
            List of balance information for all assets
        """
        account_info = self.get_account_info()
        return account_info['balances']
    
    def get_asset_balance(self, asset: str) -> Dict:
        """
        Get balance for a specific asset
        
        Args:
            asset: Asset symbol (e.g., 'BTC', 'USDT')
            
        Returns:
            Balance information for the asset
        """
        balances = self.get_balances()
        
        for balance in balances:
            if balance['asset'] == asset:
                return {
                    'asset': balance['asset'],
                    'free': float(balance['free']),
                    'locked': float(balance['locked'])
                }
                
        return None
    
    def create_order(self, symbol: str, side: str, order_type: str, 
                    quantity: float = None, price: float = None, 
                    time_in_force: str = 'GTC', **kwargs) -> Dict:
        """
        Create a new order
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            side: Order side (BUY or SELL)
            order_type: Order type (LIMIT, MARKET, STOP_LOSS, etc.)
            quantity: Order quantity
            price: Order price (required for limit orders)
            time_in_force: Time in force (GTC, IOC, FOK)
            **kwargs: Additional parameters
            
        Returns:
            Order information
        """
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
        }
        
        # Add required parameters based on order type
        if order_type == 'LIMIT':
            if price is None:
                raise ValueError("Price is required for LIMIT orders")
                
            params['timeInForce'] = time_in_force
            params['price'] = self._format_price(symbol, price)
            
        if quantity is not None:
            params['quantity'] = self._format_quantity(symbol, quantity)
            
        # Add optional parameters
        params.update(kwargs)
        
        return self._request('POST', '/v3/order', signed=True, params=params)
    
    def create_market_order(self, symbol: str, side: str, quantity: float) -> Dict:
        """
        Create a market order
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            side: Order side (BUY or SELL)
            quantity: Order quantity
            
        Returns:
            Order information
        """
        return self.create_order(
            symbol=symbol,
            side=side,
            order_type='MARKET',
            quantity=quantity
        )
    
    def create_limit_order(self, symbol: str, side: str, quantity: float, 
                          price: float, time_in_force: str = 'GTC') -> Dict:
        """
        Create a limit order
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            side: Order side (BUY or SELL)
            quantity: Order quantity
            price: Order price
            time_in_force: Time in force (GTC, IOC, FOK)
            
        Returns:
            Order information
        """
        return self.create_order(
            symbol=symbol,
            side=side,
            order_type='LIMIT',
            quantity=quantity,
            price=price,
            time_in_force=time_in_force
        )
    
    def cancel_order(self, symbol: str, order_id: int = None, 
                    client_order_id: str = None) -> Dict:
        """
        Cancel an existing order
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            order_id: Order ID
            client_order_id: Client order ID
            
        Returns:
            Cancellation information
        """
        params = {'symbol': symbol}
        
        if order_id:
            params['orderId'] = order_id
        elif client_order_id:
            params['origClientOrderId'] = client_order_id
        else:
            raise ValueError("Either order_id or client_order_id must be provided")
            
        return self._request('DELETE', '/v3/order', signed=True, params=params)
    
    def get_order(self, symbol: str, order_id: int = None, 
                 client_order_id: str = None) -> Dict:
        """
        Get order status
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            order_id: Order ID
            client_order_id: Client order ID
            
        Returns:
            Order information
        """
        params = {'symbol': symbol}
        
        if order_id:
            params['orderId'] = order_id
        elif client_order_id:
            params['origClientOrderId'] = client_order_id
        else:
            raise ValueError("Either order_id or client_order_id must be provided")
            
        return self._request('GET', '/v3/order', signed=True, params=params)
    
    def get_open_orders(self, symbol: str = None) -> List[Dict]:
        """
        Get all open orders
        
        Args:
            symbol: Trading pair symbol (optional)
            
        Returns:
            List of open orders
        """
        params = {}
        if symbol:
            params['symbol'] = symbol
            
        return self._request('GET', '/v3/openOrders', signed=True, params=params)
    
    def get_order_history(self, symbol: str, limit: int = 500, 
                         from_id: int = None) -> List[Dict]:
        """
        Get order history
        
        Args:
            symbol: Trading pair symbol
            limit: Number of orders to return (max 1000)
            from_id: Order ID to start from
            
        Returns:
            List of historical orders
        """
        params = {
            'symbol': symbol,
            'limit': limit
        }
        
        if from_id:
            params['orderId'] = from_id
            
        return self._request('GET', '/v3/allOrders', signed=True, params=params)
    
    # ===== HELPER METHODS =====
    
    def _format_price(self, symbol: str, price: float) -> str:
        """
        Format price according to symbol's tick size rules
        
        Args:
            symbol: Trading pair symbol
            price: Price value
            
        Returns:
            Formatted price as string
        """
        # For simplicity, we'll use a fixed precision for now
        # In a production system, you should get the tickSize from the exchange info
        return f"{price:.8f}".rstrip('0').rstrip('.')
    
    def _format_quantity(self, symbol: str, quantity: float) -> str:
        """
        Format quantity according to symbol's lot size rules
        
        Args:
            symbol: Trading pair symbol
            quantity: Quantity value
            
        Returns:
            Formatted quantity as string
        """
        # For simplicity, we'll use a fixed precision for now
        # In a production system, you should get the stepSize from the exchange info
        return f"{quantity:.8f}".rstrip('0').rstrip('.')
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """
        Get trading rules for a symbol
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Symbol information
        """
        exchange_info = self.get_exchange_info()
        
        for sym_info in exchange_info['symbols']:
            if sym_info['symbol'] == symbol:
                return sym_info
                
        return None


# Example usage:
if __name__ == "__main__":
    # Sample code to test the Binance interface
    binance = BinanceInterface(testnet=True)
    
    # Get server time
    server_time = binance.get_server_time()
    print(f"Server time: {datetime.fromtimestamp(server_time/1000)}")
    
    # Get BTC/USDT ticker
    ticker = binance.get_ticker("BTCUSDT")
    print(f"BTC/USDT price: {ticker['price']}")
    
    # Get recent klines
    klines = binance.get_klines_df("BTCUSDT", "1h", limit=10)
    print(klines)
    
    # If API keys are configured, test authenticated endpoints
    if binance.api_key and binance.api_secret:
        try:
            # Get account information
            account = binance.get_account_info()
            print(f"Account status: {account['accountType']}")
            
            # Get BTC balance
            btc_balance = binance.get_asset_balance("BTC")
            print(f"BTC balance: {btc_balance}")
            
            # Get open orders
            open_orders = binance.get_open_orders("BTCUSDT")
            print(f"Open orders: {len(open_orders)}")
            
        except Exception as e:
            print(f"Error testing authenticated endpoints: {e}")