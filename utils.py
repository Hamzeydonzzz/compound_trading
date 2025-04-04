import os
import json
import logging
import time
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional, Any


# ===== LOGGING FUNCTIONS =====

def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with file and console handlers
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if a log file is specified
    if log_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class TradeLogger:
    """Class for logging trading activities"""
    
    def __init__(self, log_dir: str = 'logs'):
        """
        Initialize the trade logger
        
        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up loggers
        self.trade_logger = setup_logger('trade', os.path.join(log_dir, 'trades.log'))
        self.signal_logger = setup_logger('signal', os.path.join(log_dir, 'signals.log'))
        self.error_logger = setup_logger('error', os.path.join(log_dir, 'errors.log'), level=logging.ERROR)
        self.performance_logger = setup_logger('performance', os.path.join(log_dir, 'performance.log'))
        
        # Trade history
        self.trade_history_file = os.path.join(log_dir, 'trade_history.json')
        self.trade_history = self._load_trade_history()
        
    def _load_trade_history(self) -> List[Dict]:
        """
        Load trade history from file
        
        Returns:
            List of trade records
        """
        if os.path.exists(self.trade_history_file):
            try:
                with open(self.trade_history_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                self.error_logger.error(f"Failed to load trade history from {self.trade_history_file}")
                return []
        return []
    
    def _save_trade_history(self):
        """Save trade history to file"""
        try:
            with open(self.trade_history_file, 'w') as f:
                json.dump(self.trade_history, f, indent=2)
        except Exception as e:
            self.error_logger.error(f"Failed to save trade history: {e}")
    
    def log_signal(self, symbol: str, timeframe: str, direction: str, 
                 confidence: float, features: Dict = None):
        """
        Log a trading signal
        
        Args:
            symbol: Trading pair symbol
            timeframe: Signal timeframe
            direction: Signal direction (BUY, SELL, NEUTRAL)
            confidence: Signal confidence
            features: Feature values used for the signal
        """
        timestamp = datetime.now().isoformat()
        
        # Log to file
        signal_record = {
            'timestamp': timestamp,
            'symbol': symbol,
            'timeframe': timeframe,
            'direction': direction,
            'confidence': confidence
        }
        
        if features:
            feature_str = ', '.join([f"{k}={v:.4f}" for k, v in features.items()])
            signal_record['features'] = features
        else:
            feature_str = "None"
            
        self.signal_logger.info(
            f"SIGNAL: {symbol} {timeframe} {direction} (conf: {confidence:.2f}) - Features: {feature_str}"
        )
        
        return signal_record
    
    def log_trade(self, trade_type: str, symbol: str, quantity: float, price: float, 
                 order_id: Union[int, str] = None, reason: str = None):
        """
        Log a trade execution
        
        Args:
            trade_type: Type of trade (BUY, SELL)
            symbol: Trading pair symbol
            quantity: Trade quantity
            price: Trade price
            order_id: Exchange order ID
            reason: Reason for the trade
        """
        timestamp = datetime.now().isoformat()
        
        # Calculate trade value
        value = quantity * price
        
        # Create trade record
        trade_record = {
            'timestamp': timestamp,
            'type': trade_type,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'value': value,
            'order_id': order_id,
            'reason': reason
        }
        
        # Log to file
        self.trade_logger.info(
            f"TRADE: {trade_type} {symbol} {quantity} @ {price} = {value:.2f} USD | " +
            f"Order ID: {order_id} | Reason: {reason}"
        )
        
        # Add to trade history
        self.trade_history.append(trade_record)
        self._save_trade_history()
        
        return trade_record
    
    def log_error(self, error_type: str, message: str, details: Any = None):
        """
        Log an error
        
        Args:
            error_type: Type of error
            message: Error message
            details: Additional error details
        """
        self.error_logger.error(f"ERROR: [{error_type}] {message}", exc_info=details is not None)
        
        if details:
            self.error_logger.error(f"DETAILS: {details}")
    
    def log_performance(self, metrics: Dict):
        """
        Log performance metrics
        
        Args:
            metrics: Dictionary of performance metrics
        """
        timestamp = datetime.now().isoformat()
        
        # Format metrics for logging
        metrics_str = ', '.join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                               for k, v in metrics.items()])
        
        self.performance_logger.info(f"PERFORMANCE: {metrics_str}")
        
        # Save detailed metrics to file
        metrics_file = os.path.join(self.log_dir, 'metrics.json')
        
        try:
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    all_metrics = json.load(f)
            else:
                all_metrics = []
                
            metrics['timestamp'] = timestamp
            all_metrics.append(metrics)
            
            with open(metrics_file, 'w') as f:
                json.dump(all_metrics, f, indent=2)
                
        except Exception as e:
            self.error_logger.error(f"Failed to save performance metrics: {e}")


# ===== VISUALIZATION FUNCTIONS =====

def setup_plot_style():
    """Set up matplotlib and seaborn styles for plots"""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_context("talk")
    
    # Increase default figure size
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    
    # Set color palette
    sns.set_palette("viridis")

def plot_price_with_signals(df: pd.DataFrame, signals: pd.DataFrame = None, 
                           title: str = "Price Chart with Trading Signals",
                           save_path: str = None) -> plt.Figure:
    """
    Plot price chart with buy/sell signals
    
    Args:
        df: DataFrame with OHLCV data
        signals: DataFrame with signals data
        title: Plot title
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    
    fig, ax = plt.subplots()
    
    # Plot close price
    ax.plot(df.index, df['close'], label='Close Price', color='blue', linewidth=2)
    
    # Add signals if provided
    if signals is not None and not signals.empty:
        # Buy signals
        buy_signals = signals[signals['direction'] == 'BUY']
        if not buy_signals.empty:
            ax.scatter(buy_signals.index, df.loc[buy_signals.index, 'close'], 
                      color='green', s=100, marker='^', label='Buy Signal')
            
        # Sell signals
        sell_signals = signals[signals['direction'] == 'SELL']
        if not sell_signals.empty:
            ax.scatter(sell_signals.index, df.loc[sell_signals.index, 'close'], 
                      color='red', s=100, marker='v', label='Sell Signal')
    
    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title(title)
    
    # Format x-axis as dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


# ===== HELPER FUNCTIONS =====

def calculate_metrics(equity_curve: pd.Series, risk_free_rate: float = 0.0) -> Dict[str, float]:
    """
    Calculate performance metrics from equity curve
    
    Args:
        equity_curve: Series of equity values
        risk_free_rate: Annual risk-free rate (default: 0.0)
        
    Returns:
        Dictionary with performance metrics
    """
    # Calculate returns
    returns = equity_curve.pct_change().dropna()
    
    # Convert annual risk-free rate to daily
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    
    # Calculate metrics
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    daily_returns = returns.mean()
    annual_return = ((1 + daily_returns) ** 252) - 1
    daily_volatility = returns.std()
    annual_volatility = daily_volatility * (252 ** 0.5)
    
    # Sharpe ratio
    excess_returns = returns - daily_rf
    sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * (252 ** 0.5)
    
    # Drawdown
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve / rolling_max) - 1
    max_drawdown = drawdown.min()
    
    # Calculate win rate and average gain/loss
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]
    win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
    avg_gain = positive_returns.mean() if len(positive_returns) > 0 else 0
    avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
    
    # Profit factor
    profit_factor = (positive_returns.sum() / -negative_returns.sum()) if negative_returns.sum() < 0 else float('inf')
    
    # Recovery factor
    recovery_factor = total_return / abs(max_drawdown) if max_drawdown < 0 else float('inf')
    
    # Calmar ratio
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else float('inf')
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'avg_gain': avg_gain,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'recovery_factor': recovery_factor,
        'calmar_ratio': calmar_ratio
    }

def format_timestamp(timestamp: Union[int, float, datetime], format_str: str = '%Y-%m-%d %H:%M:%S') -> str:
    """
    Format timestamp to string
    
    Args:
        timestamp: Timestamp as int, float or datetime
        format_str: Format string
        
    Returns:
        Formatted timestamp string
    """
    if isinstance(timestamp, (int, float)):
        if timestamp > 1e10:  # Likely milliseconds
            timestamp = timestamp / 1000
        dt = datetime.fromtimestamp(timestamp)
    elif isinstance(timestamp, datetime):
        dt = timestamp
    else:
        raise ValueError(f"Unsupported timestamp type: {type(timestamp)}")
        
    return dt.strftime(format_str)

def position_sizing(account_balance: float, risk_per_trade: float, 
                   entry_price: float, stop_loss_price: float) -> float:
    """
    Calculate position size based on risk
    
    Args:
        account_balance: Account balance
        risk_per_trade: Risk per trade as percentage (0.01 = 1%)
        entry_price: Entry price
        stop_loss_price: Stop loss price
        
    Returns:
        Position size
    """
    risk_amount = account_balance * risk_per_trade
    price_risk = abs(entry_price - stop_loss_price)
    
    if price_risk == 0:
        return 0
        
    position_size = risk_amount / price_risk
    
    return position_size

def calculate_stop_loss(entry_price: float, position_type: str, 
                       stop_percent: float) -> float:
    """
    Calculate stop loss price
    
    Args:
        entry_price: Entry price
        position_type: Position type ('long' or 'short')
        stop_percent: Stop loss percentage (0.02 = 2%)
        
    Returns:
        Stop loss price
    """
    if position_type.lower() == 'long':
        return entry_price * (1 - stop_percent)
    elif position_type.lower() == 'short':
        return entry_price * (1 + stop_percent)
    else:
        raise ValueError(f"Invalid position type: {position_type}")

def calculate_take_profit(entry_price: float, position_type: str, 
                         profit_percent: float) -> float:
    """
    Calculate take profit price
    
    Args:
        entry_price: Entry price
        position_type: Position type ('long' or 'short')
        profit_percent: Take profit percentage (0.05 = 5%)
        
    Returns:
        Take profit price
    """
    if position_type.lower() == 'long':
        return entry_price * (1 + profit_percent)
    elif position_type.lower() == 'short':
        return entry_price * (1 - profit_percent)
    else:
        raise ValueError(f"Invalid position type: {position_type}")

def load_json(file_path: str) -> Dict:
    """
    Load JSON file
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded JSON as dictionary
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load JSON file {file_path}: {e}")
        return {}

def save_json(data: Dict, file_path: str) -> bool:
    """
    Save data to JSON file
    
    Args:
        data: Data to save
        file_path: Path to JSON file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logging.error(f"Failed to save JSON file {file_path}: {e}")
        return False

def retry_function(func, max_retries: int = 3, delay: float = 1.0, 
                  backoff_factor: float = 2.0, exceptions: Tuple = (Exception,),
                  logger: logging.Logger = None):
    """
    Decorator for retrying a function
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        delay: Initial delay between retries
        backoff_factor: Factor to increase delay for each retry
        exceptions: Exceptions to catch
        logger: Logger for logging retries
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        mtries, mdelay = max_retries, delay
        
        while mtries > 0:
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                msg = f"{func.__name__} failed: {str(e)}. Retrying in {mdelay} seconds..."
                if logger:
                    logger.warning(msg)
                else:
                    print(msg)
                    
                time.sleep(mdelay)
                mtries -= 1
                mdelay *= backoff_factor
                
        return func(*args, **kwargs)  # Last try without catching exceptions
        
    return wrapper


# Example usage:
if __name__ == "__main__":
    # Test logging
    logger = setup_logger("test", "logs/test.log")
    logger.info("This is a test log message")
    
    # Test trade logger
    trade_logger = TradeLogger()
    trade_logger.log_signal("BTC/USDT", "1h", "BUY", 0.85, {"rsi": 30.5, "macd": 0.02})
    trade_logger.log_trade("BUY", "BTC/USDT", 0.1, 50000, 12345, "Strong buy signal")
    
    # Test visualization
    setup_plot_style()
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    prices = np.random.normal(loc=100, scale=10, size=100).cumsum() + 10000
    df = pd.DataFrame({'close': prices}, index=dates)
    
    # Test price chart
    signals = pd.DataFrame({
        'direction': ['BUY', 'SELL', 'BUY', 'SELL'],
        'confidence': [0.8, 0.7, 0.9, 0.85]
    }, index=[dates[20], dates[40], dates[60], dates[80]])
    
    fig = plot_price_with_signals(df, signals, "Sample Price Chart", "logs/price_chart.png")
    
    # Test metrics calculation
    equity = pd.Series(prices, index=dates)
    metrics = calculate_metrics(equity)
    print("Performance Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


def plot_equity_curve(df: pd.DataFrame, title: str = "Equity Curve", 
                     save_path: str = None) -> plt.Figure:
    """
    Plot equity curve for performance analysis
    
    Args:
        df: DataFrame with equity data
        title: Plot title
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    
    fig, ax = plt.subplots()
    
    # Plot equity curve
    ax.plot(df.index, df['equity'], label='Equity', color='blue', linewidth=2)
    
    # Plot drawdown
    if 'drawdown' in df.columns:
        ax_dd = ax.twinx()
        ax_dd.fill_between(df.index, 0, df['drawdown'], alpha=0.3, color='red', label='Drawdown')
        ax_dd.set_ylabel('Drawdown (%)')
        ax_dd.set_ylim(bottom=0)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_dd.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    else:
        ax.legend()
    
    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Equity')
    ax.set_title(title)
    
    # Format x-axis as dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_feature_importance(feature_names: List[str], importance_values: List[float], 
                           title: str = "Feature Importance", 
                           save_path: str = None) -> plt.Figure:
    """
    Plot feature importance
    
    Args:
        feature_names: List of feature names
        importance_values: List of importance values
        title: Plot title
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    
    # Sort features by importance
    indices = np.argsort(importance_values)
    sorted_names = [feature_names[i] for i in indices]
    sorted_values = [importance_values[i] for i in indices]
    
    fig, ax = plt.subplots()
    
    # Create horizontal bar plot
    ax.barh(range(len(sorted_names)), sorted_values, align='center')
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names)
    
    # Set labels and title
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title(title)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                         title: str = "Confusion Matrix", 
                         save_path: str = None) -> plt.Figure:
    """
    Plot confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        title: Plot title
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    
    fig, ax = plt.subplots()
    
    # Plot confusion matrix
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Set ticks
    ax.set(xticks=np.arange(cm.shape[1]),
          yticks=np.arange(cm.shape[0]),
          xticklabels=class_names, yticklabels=class_names,
          title=title,
          ylabel='True label',
          xlabel='Predicted label')
    
    # Rotate x tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                  ha="center", va="center",
                  color="white" if cm[i, j] > thresh else "black")
    
    # Tight layout
    fig.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_correlation_matrix(df: pd.DataFrame, title: str = "Feature Correlation Matrix", 
                           save_path: str = None) -> plt.Figure:
    """
    Plot correlation matrix for features
    
    Args:
        df: DataFrame with features
        title: Plot title
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    setup_plot_style()
    
    # Calculate correlation matrix
    corr = df.corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Generate a custom colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Draw the heatmap
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
               square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    
    # Set title
    ax.set_title(title)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig 