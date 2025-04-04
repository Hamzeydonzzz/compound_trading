import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ccxt
import logging
from typing import Tuple, List, Dict, Union, Optional
from config import DataConfig
from feature_engineering import FeatureEngineer

class DataHandler:
    """
    Handles data acquisition, storage, preprocessing, and feature engineering
    for the trading bot.
    """
    
    def __init__(self, config=None):
        """
        Initialize the data handler
        
        Args:
            config: Configuration object for data handling
        """
        self.config = config or DataConfig()
        self.feature_engineer = FeatureEngineer()
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """
        Set up logging for the data handler
        
        Returns:
            Logger instance
        """
        logger = logging.getLogger('DataHandler')
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
    
    def download_historical_data(self, 
                               symbol: str = 'BTC/USDT', 
                               timeframe: str = '1h',
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Download historical market data from exchange
        
        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            start_date: Start date for data download
            end_date: End date for data download
            
        Returns:
            DataFrame with historical data
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=self.config.DEFAULT_DAYS_HISTORY)
            
        self.logger.info(f"Downloading {symbol} {timeframe} data from {start_date} to {end_date}")
        
        # Initialize exchange
        exchange_id = self.config.EXCHANGE.lower()
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            'enableRateLimit': True,
        })
        
        # Convert start and end dates to milliseconds timestamp
        since = int(start_date.timestamp() * 1000)
        until = int(end_date.timestamp() * 1000)
        
        # Download data
        all_candles = []
        current_since = since
        
        while current_since < until:
            try:
                self.logger.debug(f"Fetching from {datetime.fromtimestamp(current_since/1000)}")
                candles = exchange.fetch_ohlcv(symbol, timeframe, current_since, self.config.DOWNLOAD_LIMIT)
                
                if not candles or len(candles) == 0:
                    break
                    
                all_candles.extend(candles)
                
                # Update since time for next iteration
                current_since = candles[-1][0] + 1
                
                # Add delay to avoid rate limits
                exchange.sleep(exchange.rateLimit)
                
            except Exception as e:
                self.logger.error(f"Error downloading data: {e}")
                break
        
        # Convert to DataFrame
        if not all_candles:
            self.logger.error("No data was downloaded")
            return pd.DataFrame()
            
        df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        self.logger.info(f"Downloaded {len(df)} rows of historical data")
        
        # Save to file if specified
        if self.config.SAVE_DATA:
            self._save_data(df, symbol, timeframe)
            
        return df
    
    def _save_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """
        Save downloaded data to CSV file
        
        Args:
            df: DataFrame to save
            symbol: Trading pair symbol
            timeframe: Candle timeframe
        """
        # Create directory if it doesn't exist
        os.makedirs(self.config.DATA_PATH, exist_ok=True)
        
        # Create filename
        symbol_name = symbol.replace('/', '_')
        filename = f"{symbol_name}_{timeframe}_{df.index.min().strftime('%Y%m%d')}_{df.index.max().strftime('%Y%m%d')}.csv"
        filepath = os.path.join(self.config.DATA_PATH, filename)
        
        # Save to CSV
        df.to_csv(filepath)
        self.logger.info(f"Saved data to {filepath}")
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            DataFrame with loaded data
        """
        self.logger.info(f"Loading data from {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            
            # Convert timestamp to datetime if it exists
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess raw data
        
        Args:
            df: Raw data DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        self.logger.info("Preprocessing data")
        
        # Make a copy to avoid modifying original
        processed_df = df.copy()
        
        # Handle missing values
        processed_df = self._handle_missing_values(processed_df)
        
        # Remove duplicates
        processed_df = processed_df[~processed_df.index.duplicated(keep='first')]
        
        # Sort by timestamp
        processed_df.sort_index(inplace=True)
        
        # Calculate returns
        processed_df['return'] = processed_df['close'].pct_change()
        
        # Calculate target variables if enabled
        if self.config.CALCULATE_TARGETS:
            processed_df = self._calculate_targets(processed_df)
            
        return processed_df
        
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the data
        
        Args:
            df: DataFrame with possibly missing values
            
        Returns:
            DataFrame with handled missing values
        """
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            self.logger.warning(f"Found missing values: {missing_values[missing_values > 0]}")
            
            # Forward fill for most columns
            df.fillna(method='ffill', inplace=True)
            
            # If still have missing values, backward fill
            if df.isnull().sum().sum() > 0:
                df.fillna(method='bfill', inplace=True)
                
            # If still have missing values after both fills, drop rows
            if df.isnull().sum().sum() > 0:
                df.dropna(inplace=True)
                self.logger.warning(f"Dropped rows with missing values. Remaining rows: {len(df)}")
                
        return df
    
    def _calculate_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate target variables for supervised learning
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            DataFrame with target variables
        """
        # Calculate future price change
        for horizon in self.config.PREDICTION_HORIZONS:
            # Future return
            df[f'future_return_{horizon}'] = df['close'].pct_change(horizon).shift(-horizon)
            
            # Direction labels for classification
            df[f'direction_{horizon}'] = 0  # Initialize as neutral
            df.loc[df[f'future_return_{horizon}'] > self.config.UP_THRESHOLD, f'direction_{horizon}'] = 1  # Up
            df.loc[df[f'future_return_{horizon}'] < self.config.DOWN_THRESHOLD, f'direction_{horizon}'] = -1  # Down
            
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features for the model
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            DataFrame with added features
        """
        self.logger.info("Engineering features")
        
        # Use FeatureEngineer to calculate technical indicators
        feature_df = self.feature_engineer.calculate_features(df)
        
        return feature_df
    
    def prepare_model_data(self, df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for model training
        
        Args:
            df: DataFrame with features
            target_col: Name of the target column
            
        Returns:
            X: Feature tensor for model input
            y: Target tensor for model output
        """
        self.logger.info(f"Preparing model data with target: {target_col}")
        
        # Drop rows with NaN values
        df.dropna(inplace=True)
        
        # Select features based on config
        if hasattr(self.config, 'FEATURE_COLUMNS') and self.config.FEATURE_COLUMNS:
            # Use specific features from config
            features = self.config.FEATURE_COLUMNS
        else:
            # Use all columns except targets and non-feature columns
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'return']
            exclude_cols.extend([col for col in df.columns if col.startswith('future_') or col.startswith('direction_')])
            features = [col for col in df.columns if col not in exclude_cols]
        
        # Create sequences
        X, y = self._create_sequences(df, features, target_col)
        
        return X, y
    
    def _create_sequences(self, df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series model
        
        Args:
            df: DataFrame with features
            feature_cols: List of feature column names
            target_col: Target column name
            
        Returns:
            X: Array of sequences [samples, time_steps, features]
            y: Array of targets
        """
        sequence_length = self.config.SEQUENCE_LENGTH
        
        # Extract features and target
        features = df[feature_cols].values
        target = df[target_col].values
        
        # Normalize features if enabled
        if self.config.NORMALIZE_FEATURES:
            features = self._normalize_features(features)
            
        # Create sequences
        X, y = [], []
        
        for i in range(len(df) - sequence_length):
            X.append(features[i:i+sequence_length])
            y.append(target[i+sequence_length])
            
        return np.array(X), np.array(y)
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features using min-max scaling
        
        Args:
            features: Feature array
            
        Returns:
            Normalized features
        """
        # Min-max scaling to [0, 1]
        min_vals = np.min(features, axis=0)
        max_vals = np.max(features, axis=0)
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        
        normalized_features = (features - min_vals) / range_vals
        
        return normalized_features
    
    def split_train_test(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets
        
        Args:
            X: Feature tensor
            y: Target tensor
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        test_size = self.config.TEST_SIZE
        
        # Calculate split index
        split_idx = int(len(X) * (1 - test_size))
        
        # Time-based split (not random)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        self.logger.info(f"Data split: train={X_train.shape}, test={X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def get_latest_data(self, symbol: str = 'BTC/USDC', timeframe: str = '1h', 
                       limit: int = None) -> pd.DataFrame:
        """
        Get the latest market data for making predictions
        
        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with the latest market data
        """
        if limit is None:
            # Fetch enough data for feature calculation and sequence
            limit = self.config.SEQUENCE_LENGTH + 100  # Extra for feature calculation
            
        self.logger.info(f"Fetching latest {limit} candles for {symbol} {timeframe}")
        
        # Initialize exchange
        exchange_id = self.config.EXCHANGE.lower()
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class({
            'enableRateLimit': True,
        })
        
        try:
            # Fetch latest candles
            candles = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching latest data: {e}")
            return pd.DataFrame()


# Example usage:
if __name__ == "__main__":
    # Sample code to test the data handler
    data_handler = DataHandler()
    
    # Download historical data
    df = data_handler.download_historical_data(symbol='BTC/USDT', timeframe='1h')
    
    # Preprocess data
    processed_df = data_handler.preprocess_data(df)
    
    # Engineer features
    feature_df = data_handler.engineer_features(processed_df)
    
    # Prepare model data (for example, predicting direction 24 hours ahead)
    X, y = data_handler.prepare_model_data(feature_df, 'direction_24')
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = data_handler.split_train_test(X, y)
    
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")