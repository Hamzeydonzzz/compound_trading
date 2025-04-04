"""
trading_logic.py - Advanced trading strategy implementation for Project Compound

This module contains the trading logic and strategy implementations, including:
- Signal generation and filtering
- Entry and exit rule definitions
- Position sizing and risk management
- Stop-loss and take-profit calculations
- Market regime detection
- Top detection and reversal anticipation algorithms
- Performance metrics and trade statistics

The strategies focus on capturing >0.4% price movements while detecting tops
before reversals to maximize compound growth.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from enum import Enum
import logging
from scipy.signal import argrelextrema
import statsmodels.api as sm
from datetime import datetime, timedelta
import time
import itertools

# Import other project modules
from config import Config
from feature_engineering import FeatureEngineering
from utils import setup_logger

# Set up logger
logger = setup_logger(__name__)


class Position(Enum):
    """Enum to represent position states"""
    FLAT = 0
    LONG = 1
    SHORT = 2  # For future implementation


class TradeStatus(Enum):
    """Enum to represent the status of a trade"""
    OPEN = 0
    CLOSED = 1
    CANCELLED = 2


class MarketRegime(Enum):
    """Enum to represent different market regimes"""
    TRENDING_UP = 0
    TRENDING_DOWN = 1
    RANGING = 2
    VOLATILE = 3
    UNKNOWN = 4


class Trade:
    """Class to represent a single trade"""
    
    def __init__(
        self,
        symbol: str,
        entry_price: float,
        entry_time: datetime,
        position_size: float,
        stop_loss: float = None,
        take_profit: float = None,
        max_holding_time: int = 48,  # Hours
        trade_id: str = None,
    ):
        """
        Initialize a new trade
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            entry_price: Price at which the trade was entered
            entry_time: Time when the trade was entered
            position_size: Size of the position in quote currency
            stop_loss: Stop loss price level (optional)
            take_profit: Take profit price level (optional)
            max_holding_time: Maximum time to hold the position in hours
            trade_id: Unique identifier for the trade (generated if not provided)
        """
        self.symbol = symbol
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_holding_time = max_holding_time
        
        # Generate trade ID if not provided
        self.trade_id = trade_id or f"{symbol}_{int(entry_time.timestamp())}"
        
        # Initialize other trade properties
        self.exit_price = None
        self.exit_time = None
        self.status = TradeStatus.OPEN
        self.pnl = 0.0
        self.pnl_percent = 0.0
        self.exit_reason = None
        self.quantity = position_size / entry_price
        
        # Tracking for dynamic take profit adjustments
        self.peak_price = entry_price
        self.dynamic_take_profit = take_profit
        
        # Initialize trade notes for post-analysis
        self.notes = []
        
    def close_trade(
        self,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str
    ) -> None:
        """
        Close the trade
        
        Args:
            exit_price: Price at which the trade was exited
            exit_time: Time when the trade was exited
            exit_reason: Reason for exiting the trade
        """
        if self.status != TradeStatus.OPEN:
            logger.warning(f"Attempting to close already closed trade {self.trade_id}")
            return
            
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.status = TradeStatus.CLOSED
        self.exit_reason = exit_reason
        
        # Calculate P&L
        self.pnl = (exit_price - self.entry_price) * self.quantity
        self.pnl_percent = ((exit_price / self.entry_price) - 1) * 100
        
        logger.info(f"Closed trade {self.trade_id}: {self.exit_reason}, P&L: {self.pnl_percent:.2f}%")
        
    def add_note(self, note: str) -> None:
        """Add a note to the trade for post-analysis"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.notes.append(f"[{timestamp}] {note}")
        
    def time_in_trade(self) -> timedelta:
        """Return the time elapsed since entry"""
        current_time = self.exit_time if self.exit_time else datetime.now()
        return current_time - self.entry_time
        
    def should_exit_by_time(self) -> bool:
        """Check if the trade should exit based on max holding time"""
        return self.time_in_trade() > timedelta(hours=self.max_holding_time)
        
    def update_with_new_price(self, current_price: float) -> None:
        """
        Update trade metrics with new price information
        
        Args:
            current_price: Current market price
        """
        # Update peak price if current price is higher
        if current_price > self.peak_price:
            self.peak_price = current_price
            # Update dynamic take profit based on new peak
            self._update_dynamic_take_profit(current_price)
            
    def _update_dynamic_take_profit(self, current_price: float) -> None:
        """
        Update dynamic take profit based on current price movement
        
        Args:
            current_price: Current market price
        """
        # Calculate price movement since entry as percentage
        price_movement_pct = ((current_price / self.entry_price) - 1) * 100
        
        # If price has moved significantly, adjust take profit to lock in gains
        if price_movement_pct > 1.0:  # Over 1% move
            # Calculate trailing stop as a percentage of the move
            trailing_pct = min(0.6, 0.3 + (price_movement_pct - 1.0) * 0.1)
            
            # Calculate new dynamic take profit price
            new_tp = self.peak_price * (1 - trailing_pct / 100)
            
            # Only update if the new TP is better than the current one
            if self.dynamic_take_profit is None or new_tp > self.dynamic_take_profit:
                self.dynamic_take_profit = new_tp
                self.add_note(
                    f"Updated dynamic TP to {new_tp:.2f} "
                    f"({trailing_pct:.1f}% trail from peak {self.peak_price:.2f})"
                )
                
    def to_dict(self) -> Dict:
        """Convert trade to dictionary for serialization"""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat(),
            'exit_price': self.exit_price,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'position_size': self.position_size,
            'quantity': self.quantity,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'dynamic_take_profit': self.dynamic_take_profit,
            'status': self.status.name,
            'pnl': self.pnl,
            'pnl_percent': self.pnl_percent,
            'exit_reason': self.exit_reason,
            'peak_price': self.peak_price,
            'notes': self.notes,
            'time_in_trade': str(self.time_in_trade())
        }
        
    @classmethod
    def from_dict(cls, trade_dict: Dict) -> 'Trade':
        """Create trade instance from dictionary"""
        # Create basic trade
        trade = cls(
            symbol=trade_dict['symbol'],
            entry_price=trade_dict['entry_price'],
            entry_time=datetime.fromisoformat(trade_dict['entry_time']),
            position_size=trade_dict['position_size'],
            stop_loss=trade_dict.get('stop_loss'),
            take_profit=trade_dict.get('take_profit'),
            trade_id=trade_dict.get('trade_id')
        )
        
        # Set other properties
        if trade_dict.get('exit_price'):
            trade.exit_price = trade_dict['exit_price']
        if trade_dict.get('exit_time'):
            trade.exit_time = datetime.fromisoformat(trade_dict['exit_time'])
        trade.status = TradeStatus[trade_dict['status']]
        trade.pnl = trade_dict.get('pnl', 0.0)
        trade.pnl_percent = trade_dict.get('pnl_percent', 0.0)
        trade.exit_reason = trade_dict.get('exit_reason')
        trade.peak_price = trade_dict.get('peak_price', trade.entry_price)
        trade.dynamic_take_profit = trade_dict.get('dynamic_take_profit')
        trade.notes = trade_dict.get('notes', [])
        
        return trade


class TopDetector:
    """Class for detecting market tops and reversals"""
    
    def __init__(
        self,
        lookback_periods: int = 20,
        smoothing_periods: int = 5,
        threshold_pct: float = 0.4,
        confirmation_periods: int = 3
    ):
        """
        Initialize the top detector
        
        Args:
            lookback_periods: Number of periods to look back for pattern recognition
            smoothing_periods: Number of periods for smoothing price data
            threshold_pct: Minimum percentage for top confirmation
            confirmation_periods: Number of periods for confirming a top
        """
        self.lookback_periods = lookback_periods
        self.smoothing_periods = smoothing_periods
        self.threshold_pct = threshold_pct
        self.confirmation_periods = confirmation_periods
        
    def detect_top(
        self,
        prices: pd.Series,
        volumes: pd.Series = None,
        indicators: Dict[str, pd.Series] = None
    ) -> Tuple[bool, float, str]:
        """
        Detect if a market top is forming
        
        Args:
            prices: Series of price data
            volumes: Series of volume data (optional)
            indicators: Dictionary of technical indicators (optional)
            
        Returns:
            Tuple containing:
            - Boolean indicating if a top is detected
            - Confidence score (0.0 to 1.0)
            - Reason for the detection
        """
        if len(prices) < self.lookback_periods:
            return False, 0.0, "Insufficient data"
            
        # Apply smoothing to reduce noise
        smoothed_prices = prices.rolling(self.smoothing_periods).mean().dropna()
        if len(smoothed_prices) < self.lookback_periods:
            return False, 0.0, "Insufficient data after smoothing"
            
        # Get recent price data for analysis
        recent_prices = smoothed_prices[-self.lookback_periods:]
        
        # Check for local maxima
        local_max_indices = argrelextrema(recent_prices.values, np.greater)[0]
        
        # If no local maxima found, not a top
        if len(local_max_indices) == 0:
            return False, 0.0, "No local maxima detected"
            
        # Check if the most recent local maximum is within the confirmation range
        latest_local_max = local_max_indices[-1]
        if len(recent_prices) - latest_local_max > self.confirmation_periods:
            # Local maximum is too far in the past
            return False, 0.0, "Local maximum is not recent"
            
        # Check if price has fallen since the local maximum
        max_price = recent_prices.iloc[latest_local_max]
        current_price = recent_prices.iloc[-1]
        
        price_fall_pct = ((current_price / max_price) - 1) * 100
        
        if price_fall_pct >= 0:  # Price hasn't fallen
            return False, 0.0, "Price has not declined from local maximum"
            
        # Calculate confidence based on price drop and pattern recognition
        confidence = min(1.0, abs(price_fall_pct) / self.threshold_pct)
        
        # Enhance detection with volume analysis if available
        if volumes is not None and len(volumes) >= len(prices):
            recent_volumes = volumes[-self.lookback_periods:]
            
            # Check for declining volume after the top
            volume_at_max = recent_volumes.iloc[latest_local_max]
            recent_volume = recent_volumes.iloc[-1]
            
            if recent_volume < volume_at_max:
                confidence *= 1.2  # Boost confidence if volume is declining
                reason = "Top detected with declining volume"
            else:
                confidence *= 0.8  # Reduce confidence if volume is not declining
                reason = "Potential top detected but volume not confirming"
        else:
            reason = "Potential top detected based on price action"
            
        # Enhance with indicators if available
        if indicators is not None:
            # Check if RSI is overbought and declining
            if 'rsi' in indicators:
                rsi = indicators['rsi'][-self.lookback_periods:]
                if len(rsi) > latest_local_max and rsi.iloc[latest_local_max] > 70:
                    if rsi.iloc[-1] < rsi.iloc[latest_local_max]:
                        confidence *= 1.3  # Boost confidence if RSI confirms
                        reason += " with RSI confirmation"
                        
            # Check if MACD is showing bearish divergence
            if 'macd' in indicators and 'macd_signal' in indicators:
                macd = indicators['macd'][-self.lookback_periods:]
                macd_signal = indicators['macd_signal'][-self.lookback_periods:]
                
                if (len(macd) > latest_local_max and 
                    macd.iloc[latest_local_max] > macd_signal.iloc[latest_local_max] and
                    macd.iloc[-1] < macd_signal.iloc[-1]):
                    confidence *= 1.25  # Boost confidence with MACD bearish crossover
                    reason += " with MACD bearish crossover"
        
        # Final decision based on confidence threshold
        is_top = confidence >= 0.7
        
        return is_top, confidence, reason


class TradingStrategy:
    """Base class for trading strategies"""
    
    def __init__(self, config: Config):
        """
        Initialize the trading strategy
        
        Args:
            config: Configuration object containing strategy parameters
        """
        self.config = config
        self.name = "BaseTradingStrategy"
        self.position = Position.FLAT
        self.current_trade = None
        self.trade_history = []
        self.feature_engineering = FeatureEngineering(config)
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from input data
        
        Args:
            data: DataFrame containing OHLCV data and indicators
            
        Returns:
            DataFrame with added signal columns
        """
        # This is a base implementation that returns neutral signals
        # Subclasses should override this method
        signals = data.copy()
        signals['entry_signal'] = 0
        signals['exit_signal'] = 0
        signals['stop_loss'] = 0.0
        signals['take_profit'] = 0.0
        
        return signals
        
    def calculate_position_size(
        self,
        capital: float,
        entry_price: float,
        stop_loss_price: float = None,
        risk_per_trade: float = None
    ) -> float:
        """
        Calculate position size based on risk management rules
        
        Args:
            capital: Available capital
            entry_price: Entry price level
            stop_loss_price: Stop loss price level (optional)
            risk_per_trade: Risk per trade as percentage of capital (optional)
            
        Returns:
            Position size in quote currency
        """
        # Default to 100% capital allocation as specified in requirements
        max_position_size = capital
        
        # If stop loss and risk per trade are provided, calculate position based on risk
        if stop_loss_price is not None and risk_per_trade is not None:
            # Calculate risk percentage
            risk_pct = abs((stop_loss_price / entry_price) - 1)
            
            if risk_pct > 0:
                # Calculate position size based on risk
                risk_based_size = (capital * risk_per_trade) / risk_pct
                
                # Cap at max position size
                position_size = min(risk_based_size, max_position_size)
                
                logger.info(
                    f"Risk-based position size: {position_size:.2f} "
                    f"(Entry: {entry_price:.2f}, Stop: {stop_loss_price:.2f}, "
                    f"Risk: {risk_pct:.2%})"
                )
            else:
                position_size = max_position_size
                logger.warning(
                    f"Invalid risk percentage: {risk_pct}. Using max position size."
                )
        else:
            # Use max position size (100% of capital)
            position_size = max_position_size
            logger.info(f"Using max position size: {position_size:.2f}")
            
        return position_size


class MomentumReversal(TradingStrategy):
    """
    Momentum and reversal hybrid strategy
    
    This strategy combines trend following during momentum phases with
    top detection and reversal anticipation to maximize compound growth.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the MomentumReversal strategy
        
        Args:
            config: Configuration object containing strategy parameters
        """
        super().__init__(config)
        self.name = "MomentumReversal"
        
        # Strategy-specific parameters
        self.momentum_threshold = config.get('trading.momentum_threshold', 0.4)
        self.top_detection_lookback = config.get('trading.top_detection_lookback', 20)
        self.stop_loss_pct = config.get('trading.stop_loss_percentage', 0.03)
        self.take_profit_pct = config.get('trading.take_profit_percentage', 0.06)
        self.risk_per_trade = config.get('trading.risk_per_trade', 0.02)
        self.trailing_stop_activation = config.get('trading.trailing_stop_activation', 0.02)
        self.max_holding_time = config.get('trading.max_holding_time', 48)
        
        # Initialize top detector
        self.top_detector = TopDetector(
            lookback_periods=self.top_detection_lookback,
            threshold_pct=self.momentum_threshold
        )
        
        # Track market regime
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_detection_window = config.get('trading.regime_detection_window', 30)
        
    def detect_market_regime(self, data: pd.DataFrame) -> MarketRegime:
        """
        Detect the current market regime
        
        Args:
            data: DataFrame containing OHLCV and indicator data
            
        Returns:
            MarketRegime enum value
        """
        if len(data) < self.regime_detection_window:
            return MarketRegime.UNKNOWN
            
        # Get recent data for regime detection
        recent_data = data.tail(self.regime_detection_window)
        
        # Calculate direction and strength
        price_change = (recent_data['close'].iloc[-1] / recent_data['close'].iloc[0]) - 1
        
        # Calculate volatility
        returns = recent_data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(len(returns))
        
        # Detect ranging market using linear regression
        y = recent_data['close'].values
        x = np.arange(len(y)).reshape(-1, 1)
        model = sm.OLS(y, sm.add_constant(x)).fit()
        r_squared = model.rsquared
        
        # Logic for regime detection
        if volatility > 0.03:  # High volatility threshold
            return MarketRegime.VOLATILE
        elif r_squared < 0.3:  # Low R-squared indicates ranging market
            return MarketRegime.RANGING
        elif price_change > 0.01:  # Positive trend
            return MarketRegime.TRENDING_UP
        elif price_change < -0.01:  # Negative trend
            return MarketRegime.TRENDING_DOWN
        else:
            return MarketRegime.RANGING
            
    def calculate_dynamic_stop_loss(
        self,
        entry_price: float,
        current_price: float,
        initial_stop_pct: float
    ) -> float:
        """
        Calculate a dynamic stop loss based on price movement
        
        Args:
            entry_price: Entry price level
            current_price: Current price level
            initial_stop_pct: Initial stop loss percentage
            
        Returns:
            Dynamic stop loss price
        """
        # Calculate price movement
        price_move_pct = (current_price / entry_price) - 1
        
        # If price has moved significantly in our favor, tighten the stop
        if price_move_pct > self.trailing_stop_activation:
            # Calculate how much to trail based on the move
            trailing_factor = min(0.7, 0.5 + price_move_pct)
            
            # Calculate trailing stop level
            trail_amount = price_move_pct * trailing_factor
            trailing_stop_level = entry_price * (1 + trail_amount - (initial_stop_pct / 2))
            
            # Ensure trailing stop is not below initial stop
            initial_stop = entry_price * (1 - initial_stop_pct)
            stop_loss = max(trailing_stop_level, initial_stop)
            
            logger.info(
                f"Dynamic stop loss: {stop_loss:.2f} "
                f"(Entry: {entry_price:.2f}, Current: {current_price:.2f}, "
                f"Move: {price_move_pct:.2%})"
            )
        else:
            # Use initial stop
            stop_loss = entry_price * (1 - initial_stop_pct)
            
        return stop_loss
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals for the MomentumReversal strategy
        
        Args:
            data: DataFrame containing OHLCV and indicator data
            
        Returns:
            DataFrame with added signal columns
        """
        if len(data) < 30:  # Minimum data required
            logger.warning("Insufficient data for signal generation")
            return data
            
        # Add technical indicators if not already present
        if 'rsi' not in data.columns:
            logger.info("Adding technical indicators for signal generation")
            data = self.feature_engineering.add_all_features(data)
            
        # Make a copy of the dataframe to add signals
        signals = data.copy()
        
        # Initialize signal columns
        signals['entry_signal'] = 0
        signals['exit_signal'] = 0
        signals['stop_loss'] = 0.0
        signals['take_profit'] = 0.0
        signals['position_size_pct'] = 1.0  # 100% by default
        signals['confidence'] = 0.0
        signals['market_regime'] = None
        
        # Detect market regime
        regime = self.detect_market_regime(data)
        signals['market_regime'] = regime.name
        
        # Store current regime
        self.current_regime = regime
        
        # Strategy-specific signal generation based on regime
        if regime == MarketRegime.TRENDING_UP:
            # In uptrend, look for momentum entry opportunities
            self._generate_momentum_signals(signals)
        elif regime == MarketRegime.VOLATILE:
            # In volatile markets, be more selective with entries
            self._generate_volatile_signals(signals)
        elif regime == MarketRegime.RANGING:
            # In ranging markets, look for breakouts
            self._generate_range_signals(signals)
        else:
            # In downtrends or unknown regimes, be conservative
            self._generate_conservative_signals(signals)
            
        # Apply top detection for exit signals
        self._apply_top_detection(signals)
        
        # Apply signal confirmation filters
        self._confirm_signals(signals)
        
        return signals
        
    def _generate_momentum_signals(self, signals: pd.DataFrame) -> None:
        """
        Generate signals for momentum regime
        
        Args:
            signals: DataFrame containing indicator data and signal columns
        """
        # Calculate momentum indicators
        ema_fast = signals['close'].ewm(span=8, adjust=False).mean()
        ema_slow = signals['close'].ewm(span=21, adjust=False).mean()
        
        # Entry conditions (long only)
        entry_condition = (
            (ema_fast > ema_slow) &  # Fast EMA above slow EMA
            (signals['rsi'] > 50) &  # RSI showing strength
            (signals['rsi'] < 75) &  # Not overbought yet
            (signals['volume'] > signals['volume'].rolling(20).mean()) &  # Above average volume
            (signals['close'] > signals['close'].rolling(10).mean())  # Price above short-term average
        )
        
        # Confidence calculation
        momentum_strength = np.clip((signals['rsi'] - 50) / 20, 0, 1)
        trend_strength = np.clip((ema_fast / ema_slow - 1) * 20, 0, 1)
        
        # Combine factors for confidence score
        signals.loc[entry_condition, 'confidence'] = (
            momentum_strength[entry_condition] * 0.6 + 
            trend_strength[entry_condition] * 0.4
        )
        
        # Generate entry signals for high confidence setups
        signals.loc[entry_condition & (signals['confidence'] > 0.6), 'entry_signal'] = 1
        
        # Calculate stop loss and take profit levels for entry signals
        for idx in signals[signals['entry_signal'] == 1].index:
            entry_price = signals.loc[idx, 'close']
            
            # Calculate stop loss level (3% below entry)
            stop_loss = entry_price * (1 - self.stop_loss_pct)
            signals.loc[idx, 'stop_loss'] = stop_loss
            
            # Calculate take profit level (6% above entry)
            take_profit = entry_price * (1 + self.take_profit_pct)
            signals.loc[idx, 'take_profit'] = take_profit
            
    def _generate_volatile_signals(self, signals: pd.DataFrame) -> None:
        """
        Generate signals for volatile regime
        
        Args:
            signals: DataFrame containing indicator data and signal columns
        """
        # In volatile markets, look for strong momentum with confirmation
        rsi_threshold = 60  # Higher RSI threshold in volatile markets
        
        # Entry conditions (long only)
        entry_condition = (
            (signals['rsi'] > rsi_threshold) &  # Strong RSI
            (signals['rsi'] < 80) &  # Not extremely overbought
            (signals['macd'] > signals['macd_signal']) &  # MACD bullish
            (signals['close'] > signals['sma_50']) &  # Above medium-term trend
            (signals['atr'] > signals['atr'].rolling(10).mean())  # Expanding volatility
        )
        
        # Configure tighter stop loss in volatile markets
        tighter_stop_pct = self.stop_loss_pct * 1.3
        
        # Confidence calculation with additional volatility factors
        vol_ratio = signals['atr'] / signals['atr'].rolling(20).mean()
        vol_factor = np.clip(1 / vol_ratio, 0.5, 1.0)  # Lower confidence with higher volatility
        
        # Combine factors for confidence score
        signals.loc[entry_condition, 'confidence'] = (
            ((signals['rsi'][entry_condition] - rsi_threshold) / (80 - rsi_threshold)) * 0.4 + 
            ((signals['close'][entry_condition] / signals['sma_50'][entry_condition] - 1) * 10) * 0.3 +
            vol_factor[entry_condition] * 0.3
        )
        
        # Generate entry signals for high confidence setups
        signals.loc[entry_condition & (signals['confidence'] > 0.7), 'entry_signal'] = 1
        
        # Calculate stop loss and take profit levels
        for idx in signals[signals['entry_signal'] == 1].index:
            entry_price = signals.loc[idx, 'close']
            
            # Higher take profit in volatile markets
            signals.loc[idx, 'stop_loss'] = entry_price * (1 - tighter_stop_pct)
            signals.loc[idx, 'take_profit'] = entry_price * (1 + self.take_profit_pct * 1.2)
            
    def _generate_range_signals(self, signals: pd.DataFrame) -> None:
        """
        Generate signals for ranging regime
        
        Args:
            signals: DataFrame containing indicator data and signal columns
        """
        # In ranging markets, look for breakouts with strong volume
        
        # Identify potential range boundaries
        upper_band = signals['close'].rolling(20).max()
        lower_band = signals['close'].rolling(20).min()
        mid_point = (upper_band + lower_band) / 2
        
        # Define range width
        range_width = (upper_band - lower_band) / mid_point
        
        # Breakout conditions
        breakout_condition = (
            (signals['close'] > upper_band.shift(1)) &  # Close above previous upper band
            (signals['volume'] > signals['volume'].rolling(20).mean() * 1.3) &  # Strong volume
            (range_width > 0.03) &  # Meaningful range width
            (signals['rsi'] > 55) &  # Showing some strength
            (signals['macd'] > signals['macd'].shift(1))  # Improving momentum
        )
        
        # Confidence based on breakout strength
        breakout_strength = (signals['close'] - upper_band.shift(1)) / upper_band.shift(1)
        vol_strength = signals['volume'] / signals['volume'].rolling(20).mean()
        
        # Combine factors for confidence
        signals.loc[breakout_condition, 'confidence'] = (
            np.clip(breakout_strength[breakout_condition] * 20, 0, 0.6) + 
            np.clip((vol_strength[breakout_condition] - 1) * 0.5, 0, 0.4)
        )
        
        # Generate entry signals for high confidence setups
        signals.loc[breakout_condition & (signals['confidence'] > 0.65), 'entry_signal'] = 1
        
        # Calculate stop loss and take profit levels
        for idx in signals[signals['entry_signal'] == 1].index:
            entry_price = signals.loc[idx, 'close']
            
            # Tighter stops for breakouts
            signals.loc[idx, 'stop_loss'] = lower_band.loc[idx]
            # Higher take profit for validated breakouts
            signals.loc[idx, 'take_profit'] = entry_price * (1 + self.take_profit_pct * 1.1)
            
    def _generate_conservative_signals(self, signals: pd.DataFrame) -> None:
        """
        Generate signals for conservative approach (downtrend or unknown regime)
        
        Args:
            signals: DataFrame containing indicator data and signal columns
        """
        # In downtrends or unknown regimes, only take high-probability setups
        # with very strong confirmation signals
        
        # Entry conditions (more stringent)
        entry_condition = (
            (signals['rsi'] > 60) &  # Strong RSI
            (signals['rsi'].shift(1) < 50) &  # RSI crossing above 50 (momentum change)
            (signals['macd'] > signals['macd_signal']) &  # MACD bullish
            (signals['macd'].shift(1) < signals['macd_signal'].shift(1)) &  # Recent MACD crossover
            (signals['close'] > signals['ema_20']) &  # Above short-term trend
            (signals['volume'] > signals['volume'].rolling(20).mean() * 1.5) &  # Very strong volume
            # Additional confirmation with Bollinger Bands
            (signals['close'] > signals['bb_upper'])  # Breaking above upper band (strong move)
        )
        
        # Calculate confidence score
        rsi_strength = np.clip((signals['rsi'] - 60) / 20, 0, 1)
        macd_strength = np.clip((signals['macd'] - signals['macd_signal']) / 
                               signals['macd'].rolling(20).std(), 0, 1)
        volume_strength = np.clip((signals['volume'] / 
                                 signals['volume'].rolling(20).mean() - 1.5) / 1.5, 0, 1)
        
        # Combine factors for confidence
        signals.loc[entry_condition, 'confidence'] = (
            rsi_strength[entry_condition] * 0.3 +
            macd_strength[entry_condition] * 0.3 +
            volume_strength[entry_condition] * 0.4
        )
        
        # Only generate signals for very high confidence setups
        signals.loc[entry_condition & (signals['confidence'] > 0.75), 'entry_signal'] = 1
        
        # Use tighter position sizing in conservative mode
        signals.loc[signals['entry_signal'] == 1, 'position_size_pct'] = 0.7  # 70% of max
        
        # Calculate stop loss and take profit levels
        for idx in signals[signals['entry_signal'] == 1].index:
            entry_price = signals.loc[idx, 'close']
            
            # Tighter stop loss in conservative mode
            signals.loc[idx, 'stop_loss'] = entry_price * (1 - self.stop_loss_pct * 0.8)
            # Standard take profit
            signals.loc[idx, 'take_profit'] = entry_price * (1 + self.take_profit_pct)
            
    def _apply_top_detection(self, signals: pd.DataFrame) -> None:
        """
        Apply top detection algorithm to generate exit signals
        
        Args:
            signals: DataFrame containing indicator data and signal columns
        """
        # Prepare data for top detection
        prices = signals['close']
        volumes = signals['volume']
        
        # Collect indicators for top detection
        indicators = {
            'rsi': signals['rsi'] if 'rsi' in signals.columns else None,
            'macd': signals['macd'] if 'macd' in signals.columns else None,
            'macd_signal': signals['macd_signal'] if 'macd_signal' in signals.columns else None
        }
        
        # Check each candle for potential tops
        for i in range(len(signals) - self.top_detection_lookback, len(signals)):
            if i < 0:
                continue
                
            # Get data subset for analysis
            price_subset = prices.iloc[max(0, i - self.top_detection_lookback):i+1]
            volume_subset = volumes.iloc[max(0, i - self.top_detection_lookback):i+1]
            
            # Prepare indicators subset
            indicators_subset = {}
            for key, indicator in indicators.items():
                if indicator is not None:
                    indicators_subset[key] = indicator.iloc[max(0, i - self.top_detection_lookback):i+1]
            
            # Detect top
            is_top, confidence, reason = self.top_detector.detect_top(
                price_subset, volume_subset, indicators_subset
            )
            
            # Generate exit signal if top detected with high confidence
            if is_top and confidence > 0.7:
                signals.loc[signals.index[i], 'exit_signal'] = 1
                logger.info(f"Top detected at {signals.index[i]} with confidence {confidence:.2f}: {reason}")
                
    def _confirm_signals(self, signals: pd.DataFrame) -> None:
        """
        Apply additional confirmation filters to the signals
        
        Args:
            signals: DataFrame containing indicator data and signal columns
        """
        # Avoid trading during very low volume periods
        low_volume_periods = signals['volume'] < signals['volume'].rolling(50).mean() * 0.5
        signals.loc[low_volume_periods, 'entry_signal'] = 0
        
        # Avoid entry signals too close to recent exit signals
        for i in range(len(signals)):
            if signals.iloc[i]['entry_signal'] == 1:
                # Check if there was an exit in the last 5 periods
                recent_exit = False
                for j in range(max(0, i-5), i):
                    if signals.iloc[j]['exit_signal'] == 1:
                        recent_exit = True
                        break
                
                # If recent exit found, cancel this entry
                if recent_exit:
                    signals.loc[signals.index[i], 'entry_signal'] = 0
                    logger.info(f"Entry signal at {signals.index[i]} cancelled due to recent exit")
        
        # Apply minimum price movement filter (>0.4% as specified)
        min_movement = 0.004  # 0.4%
        
        # For each entry signal, estimate potential return to take profit
        for i in signals[signals['entry_signal'] == 1].index:
            entry_price = signals.loc[i, 'close']
            take_profit = signals.loc[i, 'take_profit']
            
            # Calculate potential return
            potential_return = (take_profit / entry_price) - 1
            
            # Cancel signals with insufficient potential return
            if potential_return < min_movement:
                signals.loc[i, 'entry_signal'] = 0
                logger.info(
                    f"Entry signal at {i} cancelled: insufficient potential return "
                    f"{potential_return:.2%} (min: {min_movement:.2%})"
                )
                
class TradingLogic:
    """
    Main class for trading logic and decision making
    
    This class coordinates the overall trading process:
    - Selecting and configuring strategies
    - Processing market data
    - Generating and filtering signals
    - Managing trade execution and monitoring
    - Tracking performance and risk metrics
    """
    
    def __init__(self, config: Config):
        """
        Initialize the trading logic
        
        Args:
            config: Configuration object containing trading parameters
        """
        self.config = config
        
        # Initialize strategy
        strategy_name = config.get('trading.strategy', 'MomentumReversal')
        
        if strategy_name == 'MomentumReversal':
            self.strategy = MomentumReversal(config)
        else:
            logger.warning(f"Unknown strategy: {strategy_name}. Using MomentumReversal.")
            self.strategy = MomentumReversal(config)
            
        # Initialize trade tracking
        self.current_position = Position.FLAT
        self.active_trade = None
        self.trade_history = []
        
        # Risk management parameters
        self.max_drawdown_pct = config.get('trading.max_drawdown_pct', 0.15)
        self.max_consecutive_losses = config.get('trading.max_consecutive_losses', 5)
        self.max_daily_trades = config.get('trading.max_daily_trades', 5)
        
        # Performance tracking
        self.consecutive_losses = 0
        self.daily_trades_count = 0
        self.last_trade_date = None
        self.initial_capital = config.get('trading.initial_capital', 10000)
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.current_drawdown = 0.0
        
        # Trading session metrics
        self.session_start_time = datetime.now()
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_profit_pct': 0.0,
            'avg_loss_pct': 0.0,
            'largest_win_pct': 0.0,
            'largest_loss_pct': 0.0,
            'profit_factor': 0.0,
            'total_profit_pct': 0.0,
            'max_drawdown_pct': 0.0,
            'sharpe_ratio': 0.0,
            'trades_per_day': 0.0
        }
        
        logger.info(f"Trading logic initialized with {self.strategy.name} strategy")
        
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process market data and generate trading signals
        
        Args:
            data: DataFrame containing market data (OHLCV)
            
        Returns:
            DataFrame with added signal columns
        """
        # Ensure data has necessary columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Required column {col} not found in data")
                
        # Generate signals using the selected strategy
        signals = self.strategy.generate_signals(data)
        
        return signals
        
    def update(
        self,
        current_data: pd.DataFrame,
        available_capital: float,
        active_positions: Dict[str, float] = None
    ) -> Dict:
        """
        Update trading logic with new market data and generate actions
        
        Args:
            current_data: DataFrame containing latest market data
            available_capital: Available capital for trading
            active_positions: Dictionary of active positions (symbol -> size)
            
        Returns:
            Dictionary containing trading actions and updated metrics
        """
        # Update internal state
        self.current_capital = available_capital
        
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
            
        self.current_drawdown = 1 - (self.current_capital / self.peak_capital)
        
        # Reset daily trade counter if new day
        current_date = datetime.now().date()
        if self.last_trade_date != current_date:
            self.daily_trades_count = 0
            self.last_trade_date = current_date
            
        # Process data and generate signals
        signals = self.process_data(current_data)
        
        # Get the most recent signals
        latest_signals = signals.iloc[-1]
        
        # Initialize response
        response = {
            'action': 'HOLD',
            'trade_id': None,
            'symbol': current_data.iloc[-1].name if isinstance(current_data.index[-1], str) else None,
            'entry_price': None,
            'stop_loss': None,
            'take_profit': None,
            'position_size': None,
            'confidence': latest_signals.get('confidence', 0.0),
            'reason': "",
            'metrics': self.metrics
        }
        
        # Check if maximum drawdown has been reached
        if self.current_drawdown >= self.max_drawdown_pct:
            response['action'] = 'EMERGENCY_EXIT'
            response['reason'] = f"Maximum drawdown reached: {self.current_drawdown:.2%}"
            return response
            
        # Check if maximum consecutive losses has been reached
        if self.consecutive_losses >= self.max_consecutive_losses:
            response['action'] = 'PAUSE_TRADING'
            response['reason'] = f"Maximum consecutive losses reached: {self.consecutive_losses}"
            return response
            
        # Check if maximum daily trades has been reached
        if self.daily_trades_count >= self.max_daily_trades:
            response['reason'] = f"Maximum daily trades reached: {self.daily_trades_count}"
            return response
            
        # Handle active trade if exists
        if self.active_trade:
            # Check for exit signal
            exit_signal = latest_signals.get('exit_signal', 0) == 1
            
            # Get current price
            current_price = current_data.iloc[-1]['close']
            
            # Update trade with current price
            self.active_trade.update_with_new_price(current_price)
            
            # Check if stop loss hit
            stop_loss_hit = current_price <= self.active_trade.stop_loss
            
            # Check if take profit hit
            take_profit_hit = current_price >= self.active_trade.take_profit
            
            # Check if dynamic take profit hit
            dynamic_tp_hit = (
                self.active_trade.dynamic_take_profit is not None and 
                current_price <= self.active_trade.dynamic_take_profit and
                current_price > self.active_trade.entry_price
            )
            
            # Check if max holding time reached
            max_time_reached = self.active_trade.should_exit_by_time()
            
            # Determine if we should exit
            should_exit = exit_signal or stop_loss_hit or take_profit_hit or dynamic_tp_hit or max_time_reached
            
            if should_exit:
                # Determine exit reason
                if stop_loss_hit:
                    exit_reason = "Stop loss hit"
                elif take_profit_hit:
                    exit_reason = "Take profit hit"
                elif dynamic_tp_hit:
                    exit_reason = "Dynamic take profit hit"
                elif max_time_reached:
                    exit_reason = "Maximum holding time reached"
                else:
                    exit_reason = "Exit signal generated"
                    
                # Close the trade
                self.active_trade.close_trade(current_price, datetime.now(), exit_reason)
                
                # Update response
                response['action'] = 'EXIT'
                response['trade_id'] = self.active_trade.trade_id
                response['reason'] = exit_reason
                
                # Update metrics
                self._update_metrics_after_trade(self.active_trade)
                
                # Add to trade history
                self.trade_history.append(self.active_trade)
                
                # Clear active trade
                self.active_trade = None
                self.current_position = Position.FLAT
                
                return response
            
            # No exit, continue holding
            return response
            
        # No active trade, check for entry signal
        entry_signal = latest_signals.get('entry_signal', 0) == 1
        
        if entry_signal and self.daily_trades_count < self.max_daily_trades:
            # Get current price and symbol
            current_price = current_data.iloc[-1]['close']
            symbol = response['symbol'] or "BTCUSDT"  # Default if not provided
            
            # Get stop loss and take profit levels
            stop_loss = latest_signals.get('stop_loss', current_price * (1 - self.strategy.stop_loss_pct))
            take_profit = latest_signals.get('take_profit', current_price * (1 + self.strategy.take_profit_pct))
            
            # Get position size percentage
            position_size_pct = latest_signals.get('position_size_pct', 1.0)
            
            # Calculate position size
            position_size = available_capital * position_size_pct
            
            # Create new trade
            self.active_trade = Trade(
                symbol=symbol,
                entry_price=current_price,
                entry_time=datetime.now(),
                position_size=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                max_holding_time=self.strategy.max_holding_time
            )
            
            # Update state
            self.current_position = Position.LONG
            self.daily_trades_count += 1
            
            # Update response
            response['action'] = 'ENTRY'
            response['trade_id'] = self.active_trade.trade_id
            response['entry_price'] = current_price
            response['stop_loss'] = stop_loss
            response['take_profit'] = take_profit
            response['position_size'] = position_size
            response['reason'] = (f"Entry signal with {latest_signals.get('confidence', 0.0):.2f} "
                                 f"confidence in {latest_signals.get('market_regime', 'unknown')} regime")
            
            # Log the entry
            logger.info(
                f"New trade {self.active_trade.trade_id}: "
                f"Entry at {current_price:.2f}, "
                f"Stop: {stop_loss:.2f}, "
                f"Target: {take_profit:.2f}, "
                f"Size: {position_size:.2f}"
            )
            
        return response
        
    def _update_metrics_after_trade(self, trade: Trade) -> None:
        """
        Update performance metrics after a trade is closed
        
        Args:
            trade: The completed trade
        """
        # Update basic count metrics
        self.metrics['total_trades'] += 1
        
        # Update capital
        self.current_capital += trade.pnl
        
        # Update win/loss metrics
        if trade.pnl > 0:
            self.metrics['winning_trades'] += 1
            self.consecutive_losses = 0
            
            # Update largest win if applicable
            if trade.pnl_percent > self.metrics['largest_win_pct']:
                self.metrics['largest_win_pct'] = trade.pnl_percent
        else:
            self.metrics['losing_trades'] += 1
            self.consecutive_losses += 1
            
            # Update largest loss if applicable
            if trade.pnl_percent < self.metrics['largest_loss_pct']:
                self.metrics['largest_loss_pct'] = trade.pnl_percent
                
        # Recalculate win rate
        self.metrics['win_rate'] = (
            self.metrics['winning_trades'] / self.metrics['total_trades']
            if self.metrics['total_trades'] > 0 else 0.0
        )
        
        # Recalculate average profit and loss
        win_trades = [t for t in self.trade_history + [trade] if t.pnl > 0]
        loss_trades = [t for t in self.trade_history + [trade] if t.pnl <= 0]
        
        self.metrics['avg_profit_pct'] = (
            sum(t.pnl_percent for t in win_trades) / len(win_trades)
            if win_trades else 0.0
        )
        
        self.metrics['avg_loss_pct'] = (
            sum(t.pnl_percent for t in loss_trades) / len(loss_trades)
            if loss_trades else 0.0
        )
        
        # Calculate profit factor
        total_profits = sum(t.pnl for t in win_trades)
        total_losses = abs(sum(t.pnl for t in loss_trades))
        
        self.metrics['profit_factor'] = (
            total_profits / total_losses if total_losses > 0 else float('inf')
        )
        
        # Calculate total return
        self.metrics['total_profit_pct'] = (
            (self.current_capital / self.initial_capital - 1) * 100
        )
        
        # Update maximum drawdown
        self.metrics['max_drawdown_pct'] = max(
            self.metrics['max_drawdown_pct'],
            self.current_drawdown * 100
        )
        
        # Calculate trades per day
        days_active = (datetime.now() - self.session_start_time).days + 1
        self.metrics['trades_per_day'] = self.metrics['total_trades'] / max(1, days_active)
        
        # Log the metrics update
        logger.info(
            f"Metrics updated after trade {trade.trade_id}: "
            f"Win rate: {self.metrics['win_rate']:.2%}, "
            f"Profit factor: {self.metrics['profit_factor']:.2f}, "
            f"Total return: {self.metrics['total_profit_pct']:.2f}%"
        )
        
    def get_metrics(self) -> Dict:
        """Get current performance metrics"""
        return self.metrics.copy()
        
    def get_trade_history(self) -> List[Dict]:
        """Get trade history as list of dictionaries"""
        return [trade.to_dict() for trade in self.trade_history]
        
    def get_active_trade(self) -> Optional[Dict]:
        """Get active trade information if exists"""
        return self.active_trade.to_dict() if self.active_trade else None


def detect_reversal_patterns(data: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Detect candlestick reversal patterns in price data
    
    Args:
        data: DataFrame with OHLCV data
        window: Window size for pattern detection
        
    Returns:
        DataFrame with added reversal pattern columns
    """
    patterns = data.copy()
    
    # Initialize pattern columns
    patterns['doji'] = False
    patterns['hammer'] = False
    patterns['shooting_star'] = False
    patterns['engulfing_bullish'] = False
    patterns['engulfing_bearish'] = False
    patterns['evening_star'] = False
    patterns['morning_star'] = False
    
    # Calculate body size and shadows
    patterns['body_size'] = abs(patterns['close'] - patterns['open'])
    patterns['shadow_upper'] = patterns['high'] - np.maximum(patterns['close'], patterns['open'])
    patterns['shadow_lower'] = np.minimum(patterns['close'], patterns['open']) - patterns['low']
    patterns['total_range'] = patterns['high'] - patterns['low']
    
    # Avoid division by zero
    patterns['total_range'] = np.where(patterns['total_range'] == 0, 0.0001, patterns['total_range'])
    
    # Detect Doji (very small body)
    patterns['doji'] = patterns['body_size'] / patterns['total_range'] < 0.1
    
    # Detect Hammer (small body, long lower shadow, small upper shadow)
    patterns['hammer'] = (
        (patterns['body_size'] / patterns['total_range'] < 0.3) &
        (patterns['shadow_lower'] > 2 * patterns['body_size']) &
        (patterns['shadow_upper'] < patterns['body_size'])
    )
    
    # Detect Shooting Star (small body, long upper shadow, small lower shadow)
    patterns['shooting_star'] = (
        (patterns['body_size'] / patterns['total_range'] < 0.3) &
        (patterns['shadow_upper'] > 2 * patterns['body_size']) &
        (patterns['shadow_lower'] < patterns['body_size'])
    )
    
    # Detect Bullish Engulfing
    for i in range(1, len(patterns)):
        prev_body_size = patterns['body_size'].iloc[i-1]
        curr_body_size = patterns['body_size'].iloc[i]
        
        prev_bullish = patterns['close'].iloc[i-1] > patterns['open'].iloc[i-1]
        curr_bullish = patterns['close'].iloc[i] > patterns['open'].iloc[i]
        
        if (not prev_bullish and curr_bullish and
            curr_body_size > prev_body_size and
            patterns['open'].iloc[i] < patterns['close'].iloc[i-1] and
            patterns['close'].iloc[i] > patterns['open'].iloc[i-1]):
            patterns['engulfing_bullish'].iloc[i] = True
            
    # Detect Bearish Engulfing
    for i in range(1, len(patterns)):
        prev_body_size = patterns['body_size'].iloc[i-1]
        curr_body_size = patterns['body_size'].iloc[i]
        
        prev_bullish = patterns['close'].iloc[i-1] > patterns['open'].iloc[i-1]
        curr_bullish = patterns['close'].iloc[i] > patterns['open'].iloc[i]
        
        if (prev_bullish and not curr_bullish and
            curr_body_size > prev_body_size and
            patterns['open'].iloc[i] > patterns['close'].iloc[i-1] and
            patterns['close'].iloc[i] < patterns['open'].iloc[i-1]):
            patterns['engulfing_bearish'].iloc[i] = True
            
    # Detect Evening Star (needs 3 candles)
    for i in range(2, len(patterns)):
        if (patterns['close'].iloc[i-2] > patterns['open'].iloc[i-2] and  # First candle bullish
            patterns['body_size'].iloc[i-1] / patterns['total_range'].iloc[i-1] < 0.3 and  # Second candle small
            patterns['close'].iloc[i] < patterns['open'].iloc[i] and  # Third candle bearish
            patterns['close'].iloc[i] < (patterns['open'].iloc[i-2] + 
                                       (patterns['close'].iloc[i-2] - patterns['open'].iloc[i-2]) / 2)):
            patterns['evening_star'].iloc[i] = True
            
    # Detect Morning Star (needs 3 candles)
    for i in range(2, len(patterns)):
        if (patterns['close'].iloc[i-2] < patterns['open'].iloc[i-2] and  # First candle bearish
            patterns['body_size'].iloc[i-1] / patterns['total_range'].iloc[i-1] < 0.3 and  # Second candle small
            patterns['close'].iloc[i] > patterns['open'].iloc[i] and  # Third candle bullish
            patterns['close'].iloc[i] > (patterns['open'].iloc[i-2] + 
                                       (patterns['close'].iloc[i-2] - patterns['open'].iloc[i-2]) / 2)):
            patterns['morning_star'].iloc[i] = True
            
    return patterns


def calculate_profit_potential(
    data: pd.DataFrame,
    lookback_periods: int = 5,
    min_profit_pct: float = 0.4
) -> pd.DataFrame:
    """
    Calculate potential profit based on historical price movements
    
    Args:
        data: DataFrame with OHLCV data
        lookback_periods: Number of periods to look ahead for potential profit
        min_profit_pct: Minimum profit percentage threshold
        
    Returns:
        DataFrame with added profit potential column
    """
    result = data.copy()
    result['profit_potential_pct'] = 0.0
    
    # Calculate maximum price movement in next N periods
    for i in range(len(result) - lookback_periods):
        current_price = result['close'].iloc[i]
        future_prices = result['high'].iloc[i+1:i+lookback_periods+1]
        
        max_future_price = future_prices.max()
        potential_profit_pct = (max_future_price / current_price - 1) * 100
        
        result['profit_potential_pct'].iloc[i] = potential_profit_pct
        
    # Identify high potential setups
    result['high_profit_potential'] = result['profit_potential_pct'] >= min_profit_pct
    
    return result


def analyze_volume_profile(
    data: pd.DataFrame,
    price_bins: int = 50,
    lookback_periods: int = 100
) -> Dict:
    """
    Analyze volume profile to identify key price levels
    
    Args:
        data: DataFrame with OHLCV data
        price_bins: Number of price bins for volume profile
        lookback_periods: Number of periods to include in analysis
        
    Returns:
        Dictionary with volume profile analysis results
    """
    if len(data) < lookback_periods:
        return {
            'success': False,
            'error': 'Insufficient data for volume profile analysis'
        }
        
    # Get recent data
    recent_data = data.tail(lookback_periods)
    
    # Calculate price range
    price_min = recent_data['low'].min()
    price_max = recent_data['high'].max()
    
    # Create price bins
    price_range = np.linspace(price_min, price_max, price_bins + 1)
    
    # Initialize volume profile
    volume_profile = np.zeros(price_bins)
    
    # Distribute volume across price bins
    for _, row in recent_data.iterrows():
        # Calculate which bins this candle spans
        low_bin = max(0, np.digitize(row['low'], price_range) - 1)
        high_bin = min(price_bins - 1, np.digitize(row['high'], price_range) - 1)
        
        # Simple approach: distribute volume equally across price range
        if high_bin >= low_bin:
            bin_count = high_bin - low_bin + 1
            volume_per_bin = row['volume'] / bin_count
            
            for bin_idx in range(low_bin, high_bin + 1):
                volume_profile[bin_idx] += volume_per_bin
    
    # Find high volume nodes
    sorted_indices = np.argsort(volume_profile)
    high_volume_indices = sorted_indices[-int(price_bins * 0.2):]  # Top 20%
    
    # Convert indices to price levels
    high_volume_levels = [
        (price_range[idx], price_range[idx + 1], volume_profile[idx])
        for idx in high_volume_indices
    ]
    
    # Sort by price level
    high_volume_levels.sort(key=lambda x: x[0])
    
    # Calculate volume-weighted price (VWAP)
    vwap = np.sum(recent_data['close'] * recent_data['volume']) / np.sum(recent_data['volume'])
    
    return {
        'success': True,
        'price_min': price_min,
        'price_max': price_max,
        'volume_profile': list(zip(price_range[:-1], volume_profile)),
        'high_volume_levels': high_volume_levels,
        'vwap': vwap
    }


def identify_support_resistance(
    data: pd.DataFrame,
    window: int = 20,
    threshold: float = 0.03,
    min_touches: int = 2
) -> Dict[str, List[float]]:
    """
    Identify support and resistance levels
    
    Args:
        data: DataFrame with OHLCV data
        window: Window size for identifying local extrema
        threshold: Price threshold for considering levels (percentage)
        min_touches: Minimum number of times price must touch level
        
    Returns:
        Dictionary with support and resistance levels
    """
    # Get price data
    highs = data['high'].values
    lows = data['low'].values
    
    # Identify local maxima and minima (potential resistance and support)
    local_max_indices = argrelextrema(highs, np.greater, order=window)[0]
    local_min_indices = argrelextrema(lows, np.less, order=window)[0]
    
    resistance_levels = highs[local_max_indices]
    support_levels = lows[local_min_indices]
    
    # Function to group nearby levels
    def group_levels(levels, threshold_pct):
        if len(levels) == 0:
            return []
            
        # Sort levels
        sorted_levels = np.sort(levels)
        
        # Group nearby levels
        groups = []
        current_group = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            # Check if this level is within threshold of group average
            group_avg = np.mean(current_group)
            if abs(level / group_avg - 1) < threshold_pct:
                # Add to current group
                current_group.append(level)
            else:
                # Start a new group
                if len(current_group) >= min_touches:
                    groups.append(np.mean(current_group))
                current_group = [level]
                
        # Add the last group if it meets the minimum touches
        if len(current_group) >= min_touches:
            groups.append(np.mean(current_group))
            
        return groups
    
    # Group nearby levels
    resistance_groups = group_levels(resistance_levels, threshold)
    support_groups = group_levels(support_levels, threshold)
    
    return {
        'resistance': resistance_groups,
        'support': support_groups
    }


def optimize_parameters(
    data: pd.DataFrame,
    strategy_class: TradingStrategy,
    config: Config,
    param_ranges: Dict[str, List],
    metric: str = 'profit_factor',
    test_periods: int = 500
) -> Dict:
    """
    Optimize strategy parameters using grid search
    
    Args:
        data: DataFrame with OHLCV data
        strategy_class: Trading strategy class to optimize
        config: Base configuration object
        param_ranges: Dictionary of parameter ranges to test
        metric: Performance metric to optimize
        test_periods: Number of periods to use for testing
        
    Returns:
        Dictionary with optimization results
    """
    if len(data) < test_periods:
        return {
            'success': False,
            'error': 'Insufficient data for parameter optimization'
        }
        
    # Get test data
    test_data = data.tail(test_periods)
    
    # Function to evaluate parameters
    def evaluate_params(params):
        # Create a new config with these parameters
        test_config = config.copy()
        
        # Update config with test parameters
        for key, value in params.items():
            test_config.set(key, value)
            
        # Initialize strategy with this config
        strategy = strategy_class(test_config)
        
        # Initialize trading logic
        trading_logic = TradingLogic(test_config)
        trading_logic.strategy = strategy
        
        # Process data
        signals = strategy.generate_signals(test_data)
        
        # Simulate trades
        initial_capital = 10000
        current_capital = initial_capital
        position = Position.FLAT
        active_trade = None
        trades = []
        
        for i in range(1, len(signals)):
            # Get current and previous signals
            current = signals.iloc[i]
            
            # Get current price
            current_price = current['close']
            
            # Handle active trade
            if position == Position.LONG:
                # Check for exit
                exit_signal = current['exit_signal'] == 1
                stop_hit = current['low'] <= active_trade.stop_loss
                target_hit = current['high'] >= active_trade.take_profit
                
                if exit_signal or stop_hit or target_hit:
                    # Determine exit price
                    if stop_hit and not target_hit:
                        exit_price = active_trade.stop_loss
                        exit_reason = "Stop loss"
                    elif target_hit and not stop_hit:
                        exit_price = active_trade.take_profit
                        exit_reason = "Take profit"
                    else:
                        exit_price = current_price
                        exit_reason = "Exit signal"
                        
                    # Close trade
                    active_trade.close_trade(exit_price, pd.to_datetime(current.name), exit_reason)
                    trades.append(active_trade)
                    
                    # Update capital
                    current_capital += active_trade.pnl
                    
                    # Reset position
                    position = Position.FLAT
                    active_trade = None
            
            # Check for entry if no active position
            elif position == Position.FLAT and current['entry_signal'] == 1:
                # Calculate position size (100% of capital)
                position_size = current_capital
                
                # Create new trade
                active_trade = Trade(
                    symbol=test_data.iloc[i].name if isinstance(test_data.index[i], str) else "BTCUSDT",
                    entry_price=current_price,
                    entry_time=pd.to_datetime(current.name),
                    position_size=position_size,
                    stop_loss=current['stop_loss'],
                    take_profit=current['take_profit']
                )
                
                # Update position
                position = Position.LONG
        
        # Close any remaining trade at the end
        if position == Position.LONG:
            final_price = signals.iloc[-1]['close']
            active_trade.close_trade(final_price, pd.to_datetime(signals.index[-1]), "End of test")
            trades.append(active_trade)
            current_capital += active_trade.pnl
        
        # Calculate performance metrics
        if not trades:
            return {
                'params': params,
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'return_pct': 0,
                'max_drawdown': 0
            }
            
        # Calculate metrics
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        total_profit = sum(t.pnl for t in winning_trades)
        total_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        return_pct = (current_capital / initial_capital - 1) * 100
        
        # Calculate max drawdown
        equity_curve = [initial_capital]
        peak = initial_capital
        drawdowns = [0]
        
        for trade in trades:
            equity_curve.append(equity_curve[-1] + trade.pnl)
            peak = max(peak, equity_curve[-1])
            drawdown = (peak - equity_curve[-1]) / peak * 100
            drawdowns.append(drawdown)
            
        max_drawdown = max(drawdowns)
        
        return {
            'params': params,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'return_pct': return_pct,
            'max_drawdown': max_drawdown
        }
    
    # Generate all parameter combinations
    param_keys = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    
    param_combinations = []
    for combo in itertools.product(*param_values):
        param_combinations.append(dict(zip(param_keys, combo)))
    
    # Evaluate all combinations
    results = []
    for params in param_combinations:
        result = evaluate_params(params)
        results.append(result)
        
    # Sort results by metric
    sorted_results = sorted(results, key=lambda x: x.get(metric, 0), reverse=True)
    
    return {
        'success': True,
        'best_params': sorted_results[0]['params'] if sorted_results else None,
        'best_result': sorted_results[0] if sorted_results else None,
        'all_results': sorted_results
    }


# Example usage function
def example_usage():
    """Example of how to use the trading logic module"""
    # Load configuration
    config = Config('config/config.yaml')
    
    # Initialize trading logic
    trading_logic = TradingLogic(config)
    
    # Load historical data
    # This would typically be loaded from a CSV or API
    data = pd.read_csv(
        'data/processed/BTCUSDT_1h_data.csv',
        index_col=0,
        parse_dates=True
    )
    
    # Process data and generate signals
    signals = trading_logic.process_data(data)
    
    # Simulate trading with latest data
    latest_data = data.tail(100)  # Use last 100 candles
    available_capital = 10000  # Starting with $10,000
    
    # Update for each new candle
    for i in range(len(latest_data)):
        current_data = latest_data.iloc[:i+1]
        action = trading_logic.update(current_data, available_capital)
        
        # Log the action
        print(f"Timestamp: {latest_data.index[i]}")
        print(f"Action: {action['action']}")
        print(f"Reason: {action['reason']}")
        
        # Update capital if needed
        if action['action'] == 'EXIT':
            # Get the trade details and update available capital
            trade = trading_logic.trade_history[-1]
            available_capital += trade.pnl
            print(f"Trade closed: {trade.pnl_percent:.2f}% profit")
            
        print("-" * 50)
    
    # Print final metrics
    metrics = trading_logic.get_metrics()
    print("\nFinal Performance Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    example_usage()