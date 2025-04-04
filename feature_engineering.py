import numpy as np
import pandas as pd
import talib
from typing import List, Dict, Union, Optional, Tuple
import logging
from scipy import stats
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logger = logging.getLogger(__name__)

class FeatureEngineering:
    """
    Class for calculating and managing technical indicators and features for the trading model.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the FeatureEngineering class with configuration parameters.
        
        Args:
            config (Dict): Configuration dictionary containing parameters for feature engineering
        """
        self.config = config
        self.feature_config = config.get('features', {})
        self.normalize_method = self.feature_config.get('normalize_method', 'z_score')
        self.window_sizes = self.feature_config.get('window_sizes', [14, 20, 50, 100, 200])
        
        # Get list of indicators to use
        self.indicators = self.feature_config.get('indicators', [
            'sma', 'ema', 'rsi', 'macd', 'bollinger', 'atr', 'obv', 'roc', 'cci', 'stoch', 
            'williamsR', 'adx', 'mfi', 'heikin_ashi', 'fractals', 'psar', 'ichimoku'
        ])
        
        # Feature names (will be populated during processing)
        self.feature_names = []
        
        logger.info(f"Initialized FeatureEngineering with {len(self.indicators)} indicators")
        
    def _validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Validate if the dataframe has the required columns for feature engineering.
        
        Args:
            df (pd.DataFrame): Input dataframe with OHLCV data
            
        Returns:
            bool: True if valid, False otherwise
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Check if all required columns exist (case insensitive)
        df_columns_lower = [col.lower() for col in df.columns]
        missing_columns = [col for col in required_columns if col not in df_columns_lower]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
            
        # Ensure no NaN values
        if df.isnull().values.any():
            logger.warning("DataFrame contains NaN values")
            
        return True
        
    def _normalize_feature(self, series: pd.Series) -> pd.Series:
        """
        Normalize a feature series based on the configured method.
        
        Args:
            series (pd.Series): Input feature series
            
        Returns:
            pd.Series: Normalized feature series
        """
        if self.normalize_method == 'z_score':
            # Z-score normalization (mean=0, std=1)
            mean = series.rolling(window=200, min_periods=20).mean()
            std = series.rolling(window=200, min_periods=20).std()
            return (series - mean) / (std + 1e-10)  # Add small epsilon to avoid division by zero
            
        elif self.normalize_method == 'min_max':
            # Min-max normalization to [0, 1]
            min_val = series.rolling(window=200, min_periods=20).min()
            max_val = series.rolling(window=200, min_periods=20).max()
            return (series - min_val) / (max_val - min_val + 1e-10)
            
        elif self.normalize_method == 'pct_change':
            # Percentage change
            return series.pct_change(periods=1)
            
        elif self.normalize_method == 'log_return':
            # Log returns
            return np.log(series / series.shift(1))
            
        else:
            # Default: return as is
            return series
            
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all configured indicators on the input dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe with OHLCV data
            
        Returns:
            pd.DataFrame: Dataframe with all calculated indicators
        """
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Ensure column names are lowercase
        df.columns = [col.lower() for col in df.columns]
        
        # Validate dataframe
        if not self._validate_dataframe(df):
            logger.error("Invalid dataframe format for feature calculation")
            return df
            
        # Reset feature names
        self.feature_names = []
        
        # Calculate all requested indicators
        for indicator in self.indicators:
            indicator_method = getattr(self, f"_calculate_{indicator}", None)
            
            if indicator_method is None:
                logger.warning(f"No method found for indicator: {indicator}")
                continue
                
            try:
                df = indicator_method(df)
            except Exception as e:
                logger.error(f"Error calculating {indicator}: {str(e)}")
                
        # Always calculate volume features (they're fundamental)
        try:
            df = self._calculate_volume_features(df)
        except Exception as e:
            logger.error(f"Error calculating volume features: {str(e)}")
                
        logger.info(f"Calculated {len(self.feature_names)} features")
        return df
        
    def _calculate_sma(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Simple Moving Averages for different window sizes"""
        for window in self.window_sizes:
            feature_name = f'sma_{window}'
            df[feature_name] = talib.SMA(df['close'].values, timeperiod=window)
            
            # Calculate distance from price to MA (percentage)
            dist_feature = f'sma_{window}_dist'
            df[dist_feature] = (df['close'] - df[feature_name]) / df[feature_name] * 100
            
            self.feature_names.extend([feature_name, dist_feature])
            
        return df
        
    def _calculate_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Exponential Moving Averages for different window sizes"""
        for window in self.window_sizes:
            feature_name = f'ema_{window}'
            df[feature_name] = talib.EMA(df['close'].values, timeperiod=window)
            
            # Calculate distance from price to EMA (percentage)
            dist_feature = f'ema_{window}_dist'
            df[dist_feature] = (df['close'] - df[feature_name]) / df[feature_name] * 100
            
            # EMA slope (rate of change)
            slope_feature = f'ema_{window}_slope'
            df[slope_feature] = df[feature_name].pct_change(periods=5) * 100
            
            self.feature_names.extend([feature_name, dist_feature, slope_feature])
            
        return df
        
    def _calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Relative Strength Index for different window sizes"""
        for window in self.window_sizes[:3]:  # Use only first 3 window sizes for RSI
            feature_name = f'rsi_{window}'
            df[feature_name] = talib.RSI(df['close'].values, timeperiod=window)
            
            # RSI slope
            slope_feature = f'rsi_{window}_slope'
            df[slope_feature] = df[feature_name].diff(periods=3)
            
            self.feature_names.extend([feature_name, slope_feature])
            
        return df
        
    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicator"""
        # Default MACD parameters
        fast_period = self.feature_config.get('macd_fast_period', 12)
        slow_period = self.feature_config.get('macd_slow_period', 26)
        signal_period = self.feature_config.get('macd_signal_period', 9)
        
        # Calculate MACD
        macd, macd_signal, macd_hist = talib.MACD(
            df['close'].values, 
            fastperiod=fast_period, 
            slowperiod=slow_period, 
            signalperiod=signal_period
        )
        
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        df['macd_hist_slope'] = df['macd_hist'].diff(periods=3)
        
        self.feature_names.extend(['macd', 'macd_signal', 'macd_hist', 'macd_hist_slope'])
        
        return df
        
    def _calculate_bollinger(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        for window in self.window_sizes[:3]:  # Use only first 3 window sizes
            std_dev = self.feature_config.get('bb_std_dev', 2)
            
            upper, middle, lower = talib.BBANDS(
                df['close'].values,
                timeperiod=window,
                nbdevup=std_dev,
                nbdevdn=std_dev,
                matype=0  # Simple Moving Average
            )
            
            df[f'bb_{window}_upper'] = upper
            df[f'bb_{window}_middle'] = middle
            df[f'bb_{window}_lower'] = lower
            
            # Calculate %B (position within bands)
            df[f'bb_{window}_b'] = (df['close'] - lower) / (upper - lower + 1e-10)
            
            # Calculate bandwidth (volatility)
            df[f'bb_{window}_bandwidth'] = (upper - lower) / (middle + 1e-10)
            
            self.feature_names.extend([
                f'bb_{window}_upper', f'bb_{window}_middle', f'bb_{window}_lower',
                f'bb_{window}_b', f'bb_{window}_bandwidth'
            ])
            
        return df
        
    def _calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average True Range for different window sizes"""
        for window in self.window_sizes[:3]:  # Use only first 3 window sizes
            feature_name = f'atr_{window}'
            df[feature_name] = talib.ATR(
                df['high'].values, 
                df['low'].values, 
                df['close'].values, 
                timeperiod=window
            )
            
            # ATR as percentage of price
            pct_feature = f'atr_{window}_pct'
            df[pct_feature] = df[feature_name] / df['close'] * 100
            
            self.feature_names.extend([feature_name, pct_feature])
            
        return df
        
    def _calculate_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate On-Balance Volume"""
        df['obv'] = talib.OBV(df['close'].values, df['volume'].values)
        
        # Normalize OBV
        df['obv_normalized'] = self._normalize_feature(df['obv'])
        
        # OBV rate of change
        df['obv_roc'] = talib.ROC(df['obv'].values, timeperiod=10)
        
        self.feature_names.extend(['obv', 'obv_normalized', 'obv_roc'])
        
        return df
        
    def _calculate_roc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Price Rate of Change for different window sizes"""
        for window in [1, 5, 10, 20, 60]:
            feature_name = f'roc_{window}'
            df[feature_name] = talib.ROC(df['close'].values, timeperiod=window)
            
            self.feature_names.append(feature_name)
            
        return df
        
    def _calculate_cci(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Commodity Channel Index for different window sizes"""
        for window in self.window_sizes[:3]:  # Use only first 3 window sizes
            feature_name = f'cci_{window}'
            df[feature_name] = talib.CCI(
                df['high'].values,
                df['low'].values,
                df['close'].values,
                timeperiod=window
            )
            
            self.feature_names.append(feature_name)
            
        return df
        
    def _calculate_stoch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Stochastic Oscillator"""
        k_period = self.feature_config.get('stoch_k_period', 14)
        d_period = self.feature_config.get('stoch_d_period', 3)
        slowing = self.feature_config.get('stoch_slowing', 3)
        
        slowk, slowd = talib.STOCH(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            fastk_period=k_period,
            slowk_period=slowing,
            slowk_matype=0,
            slowd_period=d_period,
            slowd_matype=0
        )
        
        df['stoch_k'] = slowk
        df['stoch_d'] = slowd
        df['stoch_diff'] = slowk - slowd
        
        self.feature_names.extend(['stoch_k', 'stoch_d', 'stoch_diff'])
        
        return df
        
    def _calculate_williamsR(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Williams %R for different window sizes"""
        for window in self.window_sizes[:3]:  # Use only first 3 window sizes
            feature_name = f'williams_r_{window}'
            df[feature_name] = talib.WILLR(
                df['high'].values,
                df['low'].values,
                df['close'].values,
                timeperiod=window
            )
            
            self.feature_names.append(feature_name)
            
        return df
        
    def _calculate_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average Directional Index"""
        for window in self.window_sizes[:3]:  # Use only first 3 window sizes
            adx_feature = f'adx_{window}'
            df[adx_feature] = talib.ADX(
                df['high'].values,
                df['low'].values,
                df['close'].values,
                timeperiod=window
            )
            
            # Calculate +DI and -DI
            plus_di_feature = f'plus_di_{window}'
            minus_di_feature = f'minus_di_{window}'
            
            df[plus_di_feature] = talib.PLUS_DI(
                df['high'].values,
                df['low'].values,
                df['close'].values,
                timeperiod=window
            )
            
            df[minus_di_feature] = talib.MINUS_DI(
                df['high'].values,
                df['low'].values,
                df['close'].values,
                timeperiod=window
            )
            
            # Calculate directional difference
            df[f'di_diff_{window}'] = df[plus_di_feature] - df[minus_di_feature]
            
            self.feature_names.extend([
                adx_feature, plus_di_feature, minus_di_feature, f'di_diff_{window}'
            ])
            
        return df
        
    def _calculate_mfi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Money Flow Index for different window sizes"""
        for window in self.window_sizes[:3]:  # Use only first 3 window sizes
            feature_name = f'mfi_{window}'
            df[feature_name] = talib.MFI(
                df['high'].values,
                df['low'].values,
                df['close'].values,
                df['volume'].values,
                timeperiod=window
            )
            
            self.feature_names.append(feature_name)
            
        return df
        
    def _calculate_heikin_ashi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Heikin-Ashi candles"""
        df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        
        # Calculate ha_open (requires previous candle)
        ha_open = [(df['open'].iloc[0] + df['close'].iloc[0]) / 2]
        for i in range(1, len(df)):
            ha_open.append((ha_open[-1] + df['ha_close'].iloc[i-1]) / 2)
        df['ha_open'] = ha_open
        
        df['ha_high'] = df[['high', 'ha_open', 'ha_close']].max(axis=1)
        df['ha_low'] = df[['low', 'ha_open', 'ha_close']].min(axis=1)
        
        # Heikin-Ashi body and trend
        df['ha_body'] = abs(df['ha_close'] - df['ha_open'])
        df['ha_trend'] = np.where(df['ha_close'] >= df['ha_open'], 1, -1)
        
        self.feature_names.extend([
            'ha_open', 'ha_high', 'ha_low', 'ha_close', 'ha_body', 'ha_trend'
        ])
        
        return df
        
    def _calculate_fractals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Williams Fractals (simplified version)"""
        # Initialize fractal columns
        df['fractal_high'] = 0
        df['fractal_low'] = 0
        
        # Need at least 5 periods to calculate fractals
        if len(df) < 5:
            self.feature_names.extend(['fractal_high', 'fractal_low'])
            return df
            
        # Calculate bearish (high) fractals
        for i in range(2, len(df) - 2):
            if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                df['high'].iloc[i] > df['high'].iloc[i-2] and
                df['high'].iloc[i] > df['high'].iloc[i+1] and
                df['high'].iloc[i] > df['high'].iloc[i+2]):
                df.loc[df.index[i], 'fractal_high'] = 1
                
        # Calculate bullish (low) fractals
        for i in range(2, len(df) - 2):
            if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
                df['low'].iloc[i] < df['low'].iloc[i-2] and
                df['low'].iloc[i] < df['low'].iloc[i+1] and
                df['low'].iloc[i] < df['low'].iloc[i+2]):
                df.loc[df.index[i], 'fractal_low'] = 1
                
        self.feature_names.extend(['fractal_high', 'fractal_low'])
        
        return df
        
    def _calculate_psar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Parabolic SAR"""
        acceleration = self.feature_config.get('psar_acceleration', 0.02)
        maximum = self.feature_config.get('psar_maximum', 0.2)
        
        df['psar'] = talib.SAR(
            df['high'].values,
            df['low'].values,
            acceleration=acceleration,
            maximum=maximum
        )
        
        # Calculate if price is above or below PSAR
        df['psar_position'] = np.where(df['close'] > df['psar'], 1, -1)
        
        # Calculate distance as percentage
        df['psar_distance'] = abs(df['close'] - df['psar']) / df['close'] * 100
        
        self.feature_names.extend(['psar', 'psar_position', 'psar_distance'])
        
        return df
        
    def _calculate_ichimoku(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Ichimoku Cloud components"""
        # Default Ichimoku parameters
        tenkan_period = self.feature_config.get('ichimoku_tenkan', 9)
        kijun_period = self.feature_config.get('ichimoku_kijun', 26)
        senkou_b_period = self.feature_config.get('ichimoku_senkou_b', 52)
        
        # Calculate Tenkan-sen (Conversion Line)
        tenkan_high = df['high'].rolling(window=tenkan_period).max()
        tenkan_low = df['low'].rolling(window=tenkan_period).min()
        df['ichimoku_tenkan'] = (tenkan_high + tenkan_low) / 2
        
        # Calculate Kijun-sen (Base Line)
        kijun_high = df['high'].rolling(window=kijun_period).max()
        kijun_low = df['low'].rolling(window=kijun_period).min()
        df['ichimoku_kijun'] = (kijun_high + kijun_low) / 2
        
        # Calculate Senkou Span A (Leading Span A)
        df['ichimoku_senkou_a'] = ((df['ichimoku_tenkan'] + df['ichimoku_kijun']) / 2).shift(kijun_period)
        
        # Calculate Senkou Span B (Leading Span B)
        senkou_b_high = df['high'].rolling(window=senkou_b_period).max()
        senkou_b_low = df['low'].rolling(window=senkou_b_period).min()
        df['ichimoku_senkou_b'] = ((senkou_b_high + senkou_b_low) / 2).shift(kijun_period)
        
        # Calculate Chikou Span (Lagging Span)
        df['ichimoku_chikou'] = df['close'].shift(-kijun_period)
        
        # Cloud thickness (can indicate strength of trend)
        df['ichimoku_cloud_thickness'] = abs(df['ichimoku_senkou_a'] - df['ichimoku_senkou_b'])
        
        # Cloud direction (above/below)
        df['ichimoku_cloud_direction'] = np.where(
            df['ichimoku_senkou_a'] > df['ichimoku_senkou_b'], 1, -1
        )
        
        # Price position relative to cloud
        df['ichimoku_price_vs_cloud'] = np.where(
            df['close'] > df['ichimoku_senkou_a'], 
            np.where(df['close'] > df['ichimoku_senkou_b'], 2, 1),  # Above both or just above A
            np.where(df['close'] < df['ichimoku_senkou_b'], -2, -1)  # Below both or just below A
        )
        
        self.feature_names.extend([
            'ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_senkou_a', 
            'ichimoku_senkou_b', 'ichimoku_chikou', 'ichimoku_cloud_thickness',
            'ichimoku_cloud_direction', 'ichimoku_price_vs_cloud'
        ])
        
        return df
        
    def _calculate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional volume-based features"""
        # Volume moving averages
        for window in self.window_sizes:
            vol_sma_feature = f'volume_sma_{window}'
            df[vol_sma_feature] = talib.SMA(df['volume'].values, timeperiod=window)
            
            # Volume relative to its moving average
            df[f'volume_ratio_{window}'] = df['volume'] / df[vol_sma_feature]
            
            self.feature_names.extend([vol_sma_feature, f'volume_ratio_{window}'])
            
        # Accumulation/Distribution Line
        df['ad_line'] = talib.AD(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            df['volume'].values
        )
        
        # Chaikin Money Flow
        df['cmf_20'] = talib.ADOSC(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            df['volume'].values,
            fastperiod=3,
            slowperiod=10
        )
        
        # Ease of Movement
        high_low = df['high'] - df['low']
        move = ((df['high'] + df['low']) / 2) - ((df['high'].shift(1) + df['low'].shift(1)) / 2)
        volume = df['volume'] / 1000000  # Scale volume
        
        df['eom'] = move / (volume * high_low + 0.000001)  # Avoid division by zero
        df['eom_sma'] = df['eom'].rolling(window=14).mean()
        
        self.feature_names.extend(['ad_line', 'cmf_20', 'eom', 'eom_sma'])
        
        return df
        
    def calculate_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate custom features beyond standard technical indicators.
        
        Args:
            df (pd.DataFrame): Dataframe with price and indicator data
            
        Returns:
            pd.DataFrame: Dataframe with added custom features
        """
        if df.empty:
            logger.warning("Empty dataframe provided for custom features calculation")
            return df
            
        # Volatility features
        for window in [5, 10, 20]:
            # Historical volatility (standard deviation of returns)
            df[f'volatility_{window}'] = df['close'].pct_change().rolling(window=window).std() * 100
            
            # Range-based volatility
            df[f'range_volatility_{window}'] = ((df['high'] - df['low']) / df['close']).rolling(window=window).mean() * 100
            
            self.feature_names.extend([f'volatility_{window}', f'range_volatility_{window}'])
            
        # Momentum features
        for window in [5, 10, 20]:
            # Calculate momentum
            df[f'momentum_{window}'] = df['close'] / df['close'].shift(window) - 1
            
            # Calculate momentum acceleration
            df[f'momentum_acc_{window}'] = df[f'momentum_{window}'] - df[f'momentum_{window}'].shift(window)
            
            self.feature_names.extend([f'momentum_{window}', f'momentum_acc_{window}'])
            
        # Price patterns
        # Higher highs & higher lows (uptrend)
        df['higher_high'] = ((df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))).astype(int)
        df['higher_low'] = ((df['low'] > df['low'].shift(1)) & (df['low'].shift(1) > df['low'].shift(2))).astype(int)
        
        # Lower highs & lower lows (downtrend)
        df['lower_high'] = ((df['high'] < df['high'].shift(1)) & (df['high'].shift(1) < df['high'].shift(2))).astype(int)
        df['lower_low'] = ((df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))).astype(int)
        
        # Trend strength
        df['uptrend_strength'] = df['higher_high'].rolling(window=5).sum() + df['higher_low'].rolling(window=5).sum()
        df['downtrend_strength'] = df['lower_high'].rolling(window=5).sum() + df['lower_low'].rolling(window=5).sum()
        
        # Overall trend direction
        df['trend_direction'] = df['uptrend_strength'] - df['downtrend_strength']
        
        self.feature_names.extend([
            'higher_high', 'higher_low', 'lower_high', 'lower_low',
            'uptrend_strength', 'downtrend_strength', 'trend_direction'
        ])
        
        # Candle pattern features
        df['body_size'] = abs(df['close'] - df['open']) / df['open'] * 100
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open'] * 100
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open'] * 100
        
        # Candle color
        df['candle_color'] = np.where(df['close'] >= df['open'], 1, -1)
        
        # Doji pattern (small body)
        df['doji'] = (df['body_size'] < 0.1).astype(int)
        
        # Long legged Doji
        df['long_legged_doji'] = ((df['body_size'] < 0.1) & 
                                (df['upper_shadow'] > 0.5) & 
                                (df['lower_shadow'] > 0.5)).astype(int)
        
        # Hammer and hanging man
        df['hammer'] = ((df['body_size'] < 0.5) & 
                      (df['lower_shadow'] > 2 * df['body_size']) & 
                      (df['upper_shadow'] < 0.2)).astype(int)
        
        # Shooting star and inverted hammer
        df['shooting_star'] = ((df['body_size'] < 0.5) & 
                             (df['upper_shadow'] > 2 * df['body_size']) & 
                             (df['lower_shadow'] < 0.2)).astype(int)
        
        # Engulfing patterns
        df['bullish_engulfing'] = ((df['candle_color'] == 1) & 
                                 (df['candle_color'].shift(1) == -1) & 
                                 (df['open'] < df['close'].shift(1)) & 
                                 (df['close'] > df['open'].shift(1))).astype(int)
        
        df['bearish_engulfing'] = ((df['candle_color'] == -1) & 
                                 (df['candle_color'].shift(1) == 1) & 
                                 (df['open'] > df['close'].shift(1)) & 
                                 (df['close'] < df['open'].shift(1))).astype(int)
        
        self.feature_names.extend([
            'body_size', 'upper_shadow', 'lower_shadow', 'candle_color',
            'doji', 'long_legged_doji', 'hammer', 'shooting_star', 
            'bullish_engulfing', 'bearish_engulfing'
        ])
        
        # Calculate time-based features
        if isinstance(df.index, pd.DatetimeIndex):
            # Hour of day (for intraday data)
            df['hour'] = df.index.hour
            
            # Day of week (0=Monday, 6=Sunday)
            df['day_of_week'] = df.index.dayofweek
            
            # Week of year
            df['week_of_year'] = df.index.isocalendar().week
            
            # Month
            df['month'] = df.index.month
            
            # Is month end
            df['is_month_end'] = df.index.is_month_end.astype(int)
            
            self.feature_names.extend([
                'hour', 'day_of_week', 'week_of_year', 'month', 'is_month_end'
            ])
            
        # Cross-indicator features (interactions between indicators)
        
        # RSI divergence with price
        if 'rsi_14' in df.columns:
            df['price_rsi_divergence'] = 0
            
            # Bullish divergence: lower price low but higher RSI low
            df.loc[(df['close'] < df['close'].shift(1)) & (df['rsi_14'] > df['rsi_14'].shift(1)), 'price_rsi_divergence'] = 1
            
            # Bearish divergence: higher price high but lower RSI high
            df.loc[(df['close'] > df['close'].shift(1)) & (df['rsi_14'] < df['rsi_14'].shift(1)), 'price_rsi_divergence'] = -1
            
            self.feature_names.append('price_rsi_divergence')
        
        # Moving Average Crossovers
        if 'sma_20' in df.columns and 'sma_50' in df.columns:
            # Golden Cross / Death Cross (SMA 20 vs SMA 50)
            df['sma_20_50_cross'] = 0
            df.loc[df['sma_20'] > df['sma_50'], 'sma_20_50_cross'] = 1  # Golden Cross
            df.loc[df['sma_20'] < df['sma_50'], 'sma_20_50_cross'] = -1  # Death Cross
            
            # Track recent crossover
            df['sma_20_50_cross_change'] = df['sma_20_50_cross'].diff().fillna(0)
            
            self.feature_names.extend(['sma_20_50_cross', 'sma_20_50_cross_change'])
            
        # MACD and RSI agreement/disagreement
        if 'macd_hist' in df.columns and 'rsi_14' in df.columns:
            df['macd_rsi_agreement'] = np.where(
                (df['macd_hist'] > 0) & (df['rsi_14'] > 50), 1,  # Both bullish
                np.where((df['macd_hist'] < 0) & (df['rsi_14'] < 50), -1,  # Both bearish
                         0)  # Disagreement
            )
            
            self.feature_names.append('macd_rsi_agreement')
            
        # Advanced pattern detection
        
        # Consecutive candles in same direction
        df['consecutive_up'] = 0
        df['consecutive_down'] = 0
        
        for i in range(1, 6):  # Track up to 5 consecutive candles
            df[f'consecutive_up_{i}'] = ((df['candle_color'] == 1) & 
                                      df['candle_color'].rolling(window=i).sum().eq(i)).astype(int)
            
            df[f'consecutive_down_{i}'] = ((df['candle_color'] == -1) & 
                                        df['candle_color'].rolling(window=i).sum().eq(-i)).astype(int)
            
            # Update counters
            df.loc[df[f'consecutive_up_{i}'] == 1, 'consecutive_up'] = i
            df.loc[df[f'consecutive_down_{i}'] == 1, 'consecutive_down'] = i
            
        # Only keep the counters in the final feature list
        self.feature_names.extend(['consecutive_up', 'consecutive_down'])
            
        # Return the dataframe with all custom features
        return df
        
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process dataframe to calculate all indicators and features.
        This is the main method to call from outside.
        
        Args:
            df (pd.DataFrame): Input dataframe with OHLCV data
            
        Returns:
            pd.DataFrame: Processed dataframe with all features
        """
        # Calculate standard technical indicators
        df = self.calculate_indicators(df)
        
        # Calculate custom features
        df = self.calculate_custom_features(df)
        
        # Handle NaN values (replace with zeros or interpolate)
        if self.feature_config.get('handle_na', True):
            na_method = self.feature_config.get('na_method', 'fill')
            
            if na_method == 'fill':
                df = df.fillna(0)
            elif na_method == 'interpolate':
                df = df.interpolate(method='linear', limit_direction='both')
            elif na_method == 'ffill':
                df = df.ffill().bfill()  # Forward fill then backward fill
                
        logger.info(f"Processed dataframe with {len(df)} rows and {len(df.columns)} columns")
        return df
        
    def get_feature_importance(self, feature_importance: List[float]) -> Dict[str, float]:
        """
        Map feature importance values to feature names.
        
        Args:
            feature_importance (List[float]): List of importance values from a model
            
        Returns:
            Dict[str, float]: Dictionary mapping feature names to importance values
        """
        if len(feature_importance) != len(self.feature_names):
            logger.warning(f"Feature importance length ({len(feature_importance)}) " 
                           f"doesn't match feature names length ({len(self.feature_names)})")
            return {}
            
        # Create dictionary of feature name to importance
        importance_dict = {name: importance for name, importance in zip(self.feature_names, feature_importance)}
        
        # Sort by importance (descending)
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def prepare_for_training(self, df: pd.DataFrame, target_column: str = None, 
                            sequence_length: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare feature matrix for training by creating sequences.
        
        Args:
            df (pd.DataFrame): Processed dataframe with all features
            target_column (str): Name of the target column (e.g., 'close')
            sequence_length (int): Length of sequences to create
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: X (features) and y (targets) arrays for training
        """
        if df.empty:
            logger.error("Empty dataframe provided for training preparation")
            return np.array([]), np.array([])
            
        # Get all feature columns (exclude the target column)
        if target_column and target_column in df.columns:
            feature_columns = [col for col in df.columns if col != target_column]
        else:
            feature_columns = list(df.columns)
            if target_column:
                logger.warning(f"Target column '{target_column}' not found in dataframe")
                
        # Create sequences for X (features)
        sequences = []
        
        # Handle NaN values
        df_cleaned = df[feature_columns].fillna(0)
        
        for i in range(len(df) - sequence_length):
            sequences.append(df_cleaned.iloc[i:(i + sequence_length)].values)
            
        X = np.array(sequences)
        
        # Create target values (y)
        if target_column and target_column in df.columns:
            # Default: predict next period's value
            y = df[target_column].shift(-1).fillna(method='ffill').values[sequence_length:]
        else:
            # If no target specified, use close price
            y = np.array([])
            
        logger.info(f"Prepared {len(X)} sequences with shape {X.shape}")
        return X, y


def create_target_variables(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Create target variables for supervised learning.
    
    Args:
        df (pd.DataFrame): Input dataframe with price data
        config (Dict): Configuration dictionary
        
    Returns:
        pd.DataFrame: Dataframe with added target variables
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Get target configuration
    target_config = config.get('targets', {})
    horizons = target_config.get('horizons', [1, 5, 10, 20])
    
    # Price-based targets
    for horizon in horizons:
        # Future price
        df[f'future_close_{horizon}'] = df['close'].shift(-horizon)
        
        # Future returns (percent change)
        df[f'future_return_{horizon}'] = df['close'].pct_change(periods=-horizon) * 100
        
        # Log returns
        df[f'future_log_return_{horizon}'] = np.log(df['close'].shift(-horizon) / df['close'])
        
        # Binary direction (up/down)
        df[f'direction_{horizon}'] = np.where(df[f'future_return_{horizon}'] > 0, 1, 0)
        
    # Volatility targets
    for horizon in horizons:
        # Future volatility (standard deviation of returns)
        df[f'future_volatility_{horizon}'] = df['close'].pct_change().rolling(window=horizon).std().shift(-horizon) * 100
        
    # Significant move targets (for classification)
    threshold = target_config.get('significant_move_threshold', 2.0)  # Default: 2% move
    
    for horizon in horizons:
        # Significant move (ternary: up, neutral, down)
        df[f'significant_move_{horizon}'] = 0  # Neutral by default
        df.loc[df[f'future_return_{horizon}'] > threshold, f'significant_move_{horizon}'] = 1  # Up
        df.loc[df[f'future_return_{horizon}'] < -threshold, f'significant_move_{horizon}'] = -1  # Down
        
    # Maximum future return (for detecting optimal exit points)
    for horizon in [5, 10, 20, 50]:
        # Maximum upside in future window
        df[f'max_future_return_{horizon}'] = df['close'].rolling(window=horizon).max().shift(-horizon) / df['close'] - 1
        
        # Maximum downside in future window 
        df[f'min_future_return_{horizon}'] = df['close'].rolling(window=horizon).min().shift(-horizon) / df['close'] - 1
        
    return df