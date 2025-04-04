"""
Configuration module for Compound Trading Bot.

This module centralizes all configuration settings for the trading bot, including:
- Paths for data, logs, and models
- Exchange API settings
- Model parameters
- Trading parameters
- Feature engineering settings
"""

import os
from pathlib import Path
import datetime as dt
import logging
from typing import Dict, List, Any, Union, Optional

# Base project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Directory paths
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LOGS_DIR = PROJECT_DIR / "logs"
MODELS_DIR = PROJECT_DIR / "models"

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Logging configuration
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(module)s | %(message)s"
LOG_FILE = LOGS_DIR / f"trading_bot_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Exchange settings
EXCHANGE = "binance"
EXCHANGE_MODE = "test"  # "test" or "live"
EXCHANGE_TIMEOUT = 30000  # in milliseconds
API_KEY_ENV_VAR = "BINANCE_API_KEY"
API_SECRET_ENV_VAR = "BINANCE_API_SECRET"

# Trading settings
SYMBOL = "BTC/USDC"
TIMEFRAME = "1h"
QUOTE_CURRENCY = "USDC"
BASE_CURRENCY = "BTC"
INITIAL_CAPITAL = 100.0  # USDC
MAX_POSITION_SIZE = 1.0  # 100% of capital (full balance)
RISK_PER_TRADE = 0.004  # 0.4% risk per trade (stop loss)
USE_STOP_LOSS = True
STOP_LOSS_PCT = 0.004  # 0.4% stop loss
TAKE_PROFIT_PCT = 0.04  # 4% max take profit
TARGET_PROFIT_PCT = 0.005  # 0.5% minimum target profit
MAX_HOLDING_PERIOD = 24  # hours
TRADING_FEE = 0.001  # 0.1% fee

class ExchangeConfig:
    """Configuration for exchange interactions"""
    
    # API Configuration
    API_KEY = "YW4aICT8FpQvGIwGrcRBDbYY4Q9tXDP6Abki4eM16U3v3zcw4Bzu4sR0WX1VK7fF"            # Default empty, should be set via environment variable
    API_SECRET = "4mA9OazrDbQBCiVLeyBRjMEjIEr6D4uE3IQkvsayrn6UDwbGtwiT1hQbWFbjZsIS"         # Default empty, should be set via environment variable
    LOG_PATH = 'logs/exchange.log'
    
    # Trading parameters
    TRADING_PAIR = 'BTC/USDT'  # Primary trading pair
    TRADE_TIMEFRAME = '1h'      # Timeframe for trading decisions
    
    # Order parameters
    ORDER_TYPE = 'MARKET'       # Default order type (MARKET or LIMIT)
    DEFAULT_QUANTITY = 0.001    # Default quantity for BTC (0.001 BTC)
    BASE_ORDER_SIZE = 10        # Base order size in USDT
    MAX_ORDER_SIZE = 100        # Maximum order size in USDT
    
    # Risk management
    STOP_LOSS_PCT = 0.02        # Stop loss percentage (2%)
    TAKE_PROFIT_PCT = 0.05      # Take profit percentage (5%)
    MAX_OPEN_TRADES = 3         # Maximum number of concurrent open trades
    
    # Position sizing
    POSITION_SIZE_TYPE = 'fixed'    # 'fixed', 'percentage', 'risk_based'
    POSITION_SIZE_VALUE = 0.1       # 10% of available balance if percentage
    RISK_PER_TRADE = 0.01           # Risk 1% of account per trade if risk_based
    
    # Rate limiting and connection
    REQUEST_TIMEOUT = 10           # API request timeout in seconds
    RETRY_COUNT = 3                # Number of retries for failed requests
    RETRY_DELAY = 1                # Delay between retries in seconds

class ModelConfig:
    """Configuration for the transformer model"""
    
    # Model Architecture
    SEQUENCE_LENGTH = 60  # Number of time steps to include (e.g., 60 for hourly data = 2.5 days)
    NUM_FEATURES = 32     # Number of features after engineering
    TRANSFORMER_LAYERS = 4  # Number of transformer blocks
    HEAD_SIZE = 64        # Size of attention heads
    NUM_HEADS = 8         # Number of attention heads
    FF_DIM = 256          # Feed-forward network hidden dimension
    DROPOUT = 0.1         # Dropout rate
    
    # Task Configuration
    TASK_TYPE = 'classification'  # 'classification' or 'regression'
    NUM_CLASSES = 3       # For classification: Up, Down, Neutral
    PREDICTION_HORIZON = 1  # For regression: how many steps ahead to predict
    
    # Training Parameters
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 0.001
    VAL_SPLIT = 0.2       # Validation split ratio
    
    # Callbacks Configuration
    CHECKPOINT_PATH = 'models/transformer/best_model.h5'
    EARLY_STOPPING = True
    PATIENCE = 20         # Early stopping patience
    LR_PATIENCE = 10      # Learning rate reduction patience
    
    # Inference
    THRESHOLD_UP = 0.6    # Probability threshold for UP prediction
    THRESHOLD_DOWN = 0.6  # Probability threshold for DOWN prediction

# Data settings
HISTORICAL_DATA_DAYS = 365
TEST_DATA_DAYS = 30
VALIDATION_SPLIT = 0.2
LOOKBACK_WINDOW = 48  # hours of data to look back for prediction

# Feature engineering settings
TECHNICAL_INDICATORS = [
    # Trend indicators
    "sma_20", "sma_50", "sma_200", "ema_12", "ema_26", 
    # Momentum indicators
    "rsi_14", "macd", "macd_signal", "macd_hist", "stoch_k", "stoch_d",
    # Volatility indicators
    "bbands_upper", "bbands_middle", "bbands_lower", "atr_14",
    # Volume indicators
    "obv", "vwap"
]

# Derived timeframes for multi-timeframe analysis
DERIVED_TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]

# Model settings
MODEL_SETTINGS = {
    "model_type": "transformer",
    "input_size": len(TECHNICAL_INDICATORS),
    "d_model": 128,
    "nhead": 8,
    "num_encoder_layers": 6,
    "dim_feedforward": 512,
    "dropout": 0.1,
    "activation": "gelu",
    "learning_rate": 0.0001,
    "batch_size": 64,
    "epochs": 100,
    "patience": 10,  # Early stopping patience
    "validation_frequency": 5,  # Validate every N epochs
    "prediction_horizon": 12,  # Predict price movement 12 hours ahead
    "short_term_horizon": 4,   # Short-term prediction for exit signals
    "peak_prediction": True,   # Enable peak prediction
    "use_exit_model": True,    # Use separate model for exit decisions
}

# Backtesting settings
BACKTEST_SETTINGS = {
    "start_date": (dt.datetime.now() - dt.timedelta(days=HISTORICAL_DATA_DAYS)).strftime("%Y-%m-%d"),
    "end_date": dt.datetime.now().strftime("%Y-%m-%d"),
    "initial_capital": INITIAL_CAPITAL,
    "commission": TRADING_FEE,
    "slippage": 0.0005,  # 0.05% slippage
}

# Inference settings
INFERENCE_SETTINGS = {
    "confidence_threshold": 0.65,  # Minimum confidence to make a trade
    "update_frequency": 1,  # Update model every N hours
    "max_positions": 1,  # Only one position at a time (using 100% of balance)
    "exit_threshold": 0.60,  # Threshold for exiting positions
    "peak_detection_window": 3,  # Hours to look for peak (local maximum)
    "enable_dynamic_exit": True,  # Allow dynamic exit based on predicted price movement
}

class DataConfig:
    """Configuration for data handling"""
    
    # Data Sources
    EXCHANGE = 'Binance'  # Exchange to use
    DATA_PATH = 'data/'   # Path to store data files
    LOG_PATH = 'logs/data_handler.log'  # Path for logs
    
    # Download Parameters
    DEFAULT_DAYS_HISTORY = 365  # Default days of history to download
    DOWNLOAD_LIMIT = 1000       # Maximum candles per request
    SAVE_DATA = True            # Whether to save downloaded data to disk
    
    # Preprocessing
    CALCULATE_TARGETS = True    # Whether to calculate target variables
    PREDICTION_HORIZONS = [1, 6, 24, 72]  # Time horizons for prediction (e.g., 1h, 6h, 24h, 72h)
    UP_THRESHOLD = 0.005        # Threshold for up move (0.5%)
    DOWN_THRESHOLD = -0.005     # Threshold for down move (-0.5%)
    
    # Feature Engineering
    NORMALIZE_FEATURES = True   # Whether to normalize features
    
    # Sequence Parameters
    SEQUENCE_LENGTH = 60        # Length of input sequences (time steps)
    
    # Train-Test Split
    TEST_SIZE = 0.2             # Proportion of data for testing
    
    # Optional: specific feature columns to use (if empty, all available features are used)
    FEATURE_COLUMNS = []

def get_config() -> Dict[str, Any]:
    """
    Returns a dictionary containing all configuration settings.
    Useful for serializing the config or passing it around.
    
    Returns:
        Dict[str, Any]: Dictionary with all configuration variables
    """
    return {k: v for k, v in globals().items() 
            if k.isupper() and not k.startswith("_")}

class DataConfig:
    """Configuration for data handling"""
    
    # Data Sources
    EXCHANGE = 'Binance'  # Exchange to use
    DATA_PATH = 'data/'   # Path to store data files
    LOG_PATH = 'logs/data_handler.log'  # Path for logs
    
    # Download Parameters
    DEFAULT_DAYS_HISTORY = 365  # Default days of history to download
    DOWNLOAD_LIMIT = 1000       # Maximum candles per request
    SAVE_DATA = True            # Whether to save downloaded data to disk
    
    # Preprocessing
    CALCULATE_TARGETS = True    # Whether to calculate target variables
    PREDICTION_HORIZONS = [1, 6, 24, 72]  # Time horizons for prediction (e.g., 1h, 6h, 24h, 72h)
    UP_THRESHOLD = 0.005        # Threshold for up move (0.5%)
    DOWN_THRESHOLD = -0.005     # Threshold for down move (-0.5%)
    
    # Feature Engineering
    NORMALIZE_FEATURES = True   # Whether to normalize features
    
    # Sequence Parameters
    SEQUENCE_LENGTH = 60        # Length of input sequences (time steps)
    
    # Train-Test Split
    TEST_SIZE = 0.2             # Proportion of data for testing
    
    # Optional: specific feature columns to use (if empty, all available features are used)
    FEATURE_COLUMNS = []

def update_config(config_updates: Dict[str, Any]) -> None:
    """
    Updates configuration variables with values from config_updates.
    
    Args:
        config_updates (Dict[str, Any]): Dictionary with new config values
    """
    for key, value in config_updates.items():
        if key.isupper() and key in globals():
            globals()[key] = value

def load_config_from_file(config_file: Union[str, Path]) -> None:
    """
    Loads configuration from a Python file.
    
    Args:
        config_file (Union[str, Path]): Path to config file
    """
    import importlib.util
    
    config_file = Path(config_file)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
        
    spec = importlib.util.spec_from_file_location("dynamic_config", config_file)
    dynamic_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dynamic_config)
    
    for key in dir(dynamic_config):
        if key.isupper() and key in globals():
            globals()[key] = getattr(dynamic_config, key)

if __name__ == "__main__":
    # Print current config for debugging
    import json
    
    print("Current Configuration:")
    config_dict = get_config()
    
    # Convert Path objects to strings for JSON serialization
    for key, value in config_dict.items():
        if isinstance(value, Path):
            config_dict[key] = str(value)
    
    print(json.dumps(config_dict, indent=4, default=str))