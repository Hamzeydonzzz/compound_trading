#!/usr/bin/env python3
"""
visualize_data.py

This script creates visualizations of the processed data to help analyze
features, patterns, and relationships in the cryptocurrency market data.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Union

# Add the src directory to the path
script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent / "src" if script_dir.name == "scripts" else script_dir
sys.path.insert(0, str(src_dir))

# Import project modules
from config import Config
from logger import get_logger, log_exception

# Initialize logger
logger = get_logger("visualize_data")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize cryptocurrency data and features')
    
    parser.add_argument(
        '--input_file', 
        type=str, 
        help='Input CSV file with processed data'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default=str(Path(src_dir).parents[0] / "data" / "visualizations"),
        help='Directory to save visualizations'
    )
    
    parser.add_argument(
        '--start_date', 
        type=str, 
        help='Start date for visualization (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end_date', 
        type=str, 
        help='End date for visualization (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--feature_groups',
        action='store_true',
        help='Generate feature group visualizations'
    )
    
    parser.add_argument(
        '--indicators',
        action='store_true',
        help='Generate technical indicator visualizations'
    )
    
    parser.add_argument(
        '--correlation',
        action='store_true',
        help='Generate correlation matrix visualizations'
    )
    
    parser.add_argument(
        '--target_analysis',
        action='store_true',
        help='Generate target variable analysis'
    )
    
    parser.add_argument(
        '--distributions',
        action='store_true',
        help='Generate feature distribution visualizations'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate all visualizations'
    )
    
    return parser.parse_args()

def load_data(input_file: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Load processed data from CSV file.
    
    Args:
        input_file: Path to CSV file
        start_date: Optional start date for filtering (YYYY-MM-DD)
        end_date: Optional end date for filtering (YYYY-MM-DD)
        
    Returns:
        DataFrame with processed data
    """
    logger.info(f"Loading data from {input_file}")
    
    # Check if file exists
    if not Path(input_file).exists():
        logger.error(f"Input file {input_file} does not exist")
        sys.exit(1)
    
    # Load data
    try:
        df = pd.read_csv(input_file, index_col=0, parse_dates=True)
        logger.info(f"Loaded {len(df)} rows from {df.index[0]} to {df.index[-1]}")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        log_exception(logger)
        sys.exit(1)
    
    # Filter by date range if specified
    if start_date:
        start = pd.to_datetime(start_date)
        df = df[df.index >= start]
    
    if end_date:
        end = pd.to_datetime(end_date)
        df = df[df.index <= end]
    
    if start_date or end_date:
        logger.info(f"Filtered data: {len(df)} rows from {df.index[0]} to {df.index[-1]}")
    
    return df

def plot_price_and_volume(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create price and volume chart.
    
    Args:
        df: DataFrame with OHLCV data
        output_dir: Directory to save visualizations
    """
    logger.info("Creating price and volume chart")
    
    plt.figure(figsize=(16, 10))
    
    # Plot price
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(df.index, df['close'], label='Close Price', color='blue')
    
    # Add moving averages if available
    if 'ma_50' in df.columns:
        ax1.plot(df.index, df['ma_50'], label='50-day MA', color='red', alpha=0.7)
    if 'ma_200' in df.columns:
        ax1.plot(df.index, df['ma_200'], label='200-day MA', color='green', alpha=0.7)
    
    # Add Bollinger Bands if available
    if 'upper_bb_20' in df.columns and 'lower_bb_20' in df.columns:
        ax1.plot(df.index, df['upper_bb_20'], label='Upper BB', color='gray', linestyle='--', alpha=0.5)
        ax1.plot(df.index, df['middle_bb_20'], label='Middle BB', color='gray', linestyle='-', alpha=0.5)
        ax1.plot(df.index, df['lower_bb_20'], label='Lower BB', color='gray', linestyle='--', alpha=0.5)
        ax1.fill_between(df.index, df['upper_bb_20'], df['lower_bb_20'], color='gray', alpha=0.1)
    
    ax1.set_title('Price Chart with Indicators')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot volume
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.bar(df.index, df['volume'], label='Volume', color='blue', alpha=0.5)
    
    # Add volume moving average if available
    if 'volume_sma_20' in df.columns:
        ax2.plot(df.index, df['volume_sma_20'], label='20-day Volume MA', color='red', alpha=0.7)
    
    ax2.set_title('Volume')
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Date')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'price_and_volume.png')
    plt.close()
    
    logger.info("Price and volume chart saved")

def plot_technical_indicators(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create technical indicator visualizations.
    
    Args:
        df: DataFrame with technical indicators
        output_dir: Directory to save visualizations
    """
    logger.info("Creating technical indicator visualizations")
    
    # 1. RSI Chart
    if 'rsi_14' in df.columns:
        plt.figure(figsize=(16, 6))
        plt.plot(df.index, df['rsi_14'], label='RSI(14)', color='purple')
        plt.axhline(y=70, color='red', linestyle='--', alpha=0.5)
        plt.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        plt.fill_between(df.index, df['rsi_14'], 70, where=(df['rsi_14'] >= 70), color='red', alpha=0.3)
        plt.fill_between(df.index, df['rsi_14'], 30, where=(df['rsi_14'] <= 30), color='green', alpha=0.3)
        plt.title('Relative Strength Index (RSI)')
        plt.ylabel('RSI Value')
        plt.xlabel('Date')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'rsi_indicator.png')
        plt.close()
    
    # 2. MACD Chart
    if all(col in df.columns for col in ['macd', 'macd_signal']):
        plt.figure(figsize=(16, 6))
        plt.plot(df.index, df['macd'], label='MACD', color='blue')
        plt.plot(df.index, df['macd_signal'], label='Signal Line', color='red')
        plt.bar(df.index, df['macd'] - df['macd_signal'], label='Histogram', color=['green' if val >= 0 else 'red' for val in (df['macd'] - df['macd_signal'])], alpha=0.5)
        plt.title('Moving Average Convergence Divergence (MACD)')
        plt.ylabel('MACD Value')
        plt.xlabel('Date')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'macd_indicator.png')
        plt.close()
    
    # 3. Bollinger Bands
    if all(col in df.columns for col in ['upper_bb_20', 'middle_bb_20', 'lower_bb_20']):
        plt.figure(figsize=(16, 6))
        plt.plot(df.index, df['close'], label='Close Price', color='blue')
        plt.plot(df.index, df['upper_bb_20'], label='Upper BB', color='red', linestyle='--')
        plt.plot(df.index, df['middle_bb_20'], label='Middle BB', color='purple')
        plt.plot(df.index, df['lower_bb_20'], label='Lower BB', color='green', linestyle='--')
        plt.fill_between(df.index, df['upper_bb_20'], df['lower_bb_20'], color='gray', alpha=0.1)
        plt.title('Bollinger Bands')
        plt.ylabel('Price')
        plt.xlabel('Date')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'bollinger_bands.png')
        plt.close()
    
    # 4. ATR Chart
    if 'atr_14' in df.columns:
        plt.figure(figsize=(16, 6))
        plt.plot(df.index, df['atr_14'], label='ATR(14)', color='orange')
        plt.title('Average True Range (ATR)')
        plt.ylabel('ATR Value')
        plt.xlabel('Date')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'atr_indicator.png')
        plt.close()
    
    # 5. Stochastic Oscillator
    if all(col in df.columns for col in ['stoch_14_k', 'stoch_14_d']):
        plt.figure(figsize=(16, 6))
        plt.plot(df.index, df['stoch_14_k'], label='%K Line', color='blue')
        plt.plot(df.index, df['stoch_14_d'], label='%D Line', color='red')
        plt.axhline(y=80, color='red', linestyle='--', alpha=0.5)
        plt.axhline(y=20, color='green', linestyle='--', alpha=0.5)
        plt.fill_between(df.index, df['stoch_14_k'], 80, where=(df['stoch_14_k'] >= 80), color='red', alpha=0.3)
        plt.fill_between(df.index, df['stoch_14_k'], 20, where=(df['stoch_14_k'] <= 20), color='green', alpha=0.3)
        plt.title('Stochastic Oscillator')
        plt.ylabel('Value')
        plt.xlabel('Date')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'stochastic_oscillator.png')
        plt.close()
    
    logger.info("Technical indicator visualizations saved")

def plot_correlation_matrix(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create correlation matrix visualization.
    
    Args:
        df: DataFrame with features
        output_dir: Directory to save visualizations
    """
    logger.info("Creating correlation matrix visualization")
    
    # Select price-related features
    price_cols = ['open', 'high', 'low', 'close', 'volume']
    # Add moving averages
    ma_cols = [col for col in df.columns if col.startswith('ma_') or col.startswith('ema_')]
    # Add momentum indicators
    momentum_cols = [col for col in df.columns if col.startswith('rsi_') or col.startswith('macd')]
    # Add target variables
    target_cols = [col for col in df.columns if col.startswith('target_')]
    
    # Combine feature groups
    selected_features = price_cols + ma_cols[:5] + momentum_cols[:5] + target_cols[:3]
    
    # Filter dataframe
    if len(selected_features) > 30:
        selected_features = selected_features[:30]  # Limit to 30 features for readability
    
    corr_df = df[selected_features].corr()
    
    # Create heatmap
    plt.figure(figsize=(16, 12))
    mask = np.triu(np.ones_like(corr_df))
    sns.heatmap(
        corr_df, 
        annot=True, 
        fmt=".2f", 
        cmap='coolwarm', 
        mask=mask, 
        vmin=-1, 
        vmax=1, 
        center=0,
        square=True, 
        linewidths=0.5
    )
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrix.png')
    plt.close()
    
    # Create correlation with target
    if any(col in df.columns for col in ['target_1', 'target_direction_1']):
        target_col = 'target_1' if 'target_1' in df.columns else 'target_direction_1'
        
        # Sort by correlation with target
        target_corr = df.corr()[target_col].sort_values(ascending=False)
        
        # Filter out targets and keep top 30 features
        target_corr = target_corr[~target_corr.index.str.startswith('target_')]
        target_corr = target_corr.head(30)
        
        plt.figure(figsize=(12, 10))
        target_corr.plot(kind='barh', color=['green' if x > 0 else 'red' for x in target_corr.values])
        plt.title(f'Feature Correlation with {target_col}')
        plt.xlabel('Correlation')
        plt.ylabel('Features')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f'correlation_with_{target_col}.png')
        plt.close()
    
    logger.info("Correlation matrix visualization saved")

def plot_feature_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create feature distribution visualizations.
    
    Args:
        df: DataFrame with features
        output_dir: Directory to save visualizations
    """
    logger.info("Creating feature distribution visualizations")
    
    # Create feature distribution directory
    dist_dir = output_dir / 'distributions'
    dist_dir.mkdir(exist_ok=True, parents=True)
    
    # Select important features
    feature_groups = {
        'price': ['open', 'high', 'low', 'close'],
        'volume': ['volume', 'volume_pct_change'],
        'returns': ['close_pct_change', 'log_return'],
        'rsi': [col for col in df.columns if col.startswith('rsi_')],
        'macd': [col for col in df.columns if col.startswith('macd')],
        'bollinger': [col for col in df.columns if col.startswith('bb_')],
        'targets': [col for col in df.columns if col.startswith('target_') and not col.startswith('target_direction_') and not col.startswith('target_class_')]
    }
    
    for group_name, features in feature_groups.items():
        if not features or not all(feat in df.columns for feat in features):
            continue
            
        # Limit to first 5 features in each group
        features = features[:5]
        
        plt.figure(figsize=(16, 10))
        for i, feature in enumerate(features):
            plt.subplot(len(features), 1, i+1)
            sns.histplot(df[feature].dropna(), kde=True, bins=50)
            plt.title(f'Distribution of {feature}')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(dist_dir / f'{group_name}_distributions.png')
        plt.close()
    
    # Create QQ plots for returns (check for normality)
    if 'log_return' in df.columns:
        try:
            from scipy import stats
            
            plt.figure(figsize=(12, 6))
            ax = plt.subplot(1, 1, 1)
            stats.probplot(df['log_return'].dropna(), dist="norm", plot=ax)
            plt.title('Q-Q Plot of Log Returns (Normal Distribution)')
            plt.grid(True, alpha=0.3)
            plt.savefig(dist_dir / 'qq_plot_returns.png')
            plt.close()
        except ImportError:
            logger.warning("SciPy not installed. Skipping QQ plot.")
    
    logger.info("Feature distribution visualizations saved")

def plot_target_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create target variable analysis visualizations.
    
    Args:
        df: DataFrame with features and targets
        output_dir: Directory to save visualizations
    """
    logger.info("Creating target variable analysis visualizations")
    
    # Create target analysis directory
    target_dir = output_dir / 'target_analysis'
    target_dir.mkdir(exist_ok=True, parents=True)
    
    # Get continuous target variables
    target_cols = [col for col in df.columns if col.startswith('target_') and not col.startswith('target_direction_') and not col.startswith('target_class_')]
    
    if target_cols:
        # 1. Target distributions
        plt.figure(figsize=(16, 10))
        for i, col in enumerate(target_cols[:6]):  # Limit to 6 targets
            plt.subplot(3, 2, i+1)
            sns.histplot(df[col].dropna(), kde=True, bins=50)
            plt.axvline(x=0, color='red', linestyle='--')
            plt.title(f'Distribution of {col}')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(target_dir / 'target_distributions.png')
        plt.close()
        
        # 2. Target autocorrelation
        plt.figure(figsize=(16, 10))
        for i, col in enumerate(target_cols[:6]):  # Limit to 6 targets
            plt.subplot(3, 2, i+1)
            pd.plotting.autocorrelation_plot(df[col].dropna())
            plt.title(f'Autocorrelation of {col}')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(target_dir / 'target_autocorrelation.png')
        plt.close()
    
    # Get binary target variables
    binary_targets = [col for col in df.columns if col.startswith('target_direction_')]
    
    if binary_targets:
        # 3. Binary target class distribution
        plt.figure(figsize=(12, 8))
        for i, col in enumerate(binary_targets[:6]):  # Limit to 6 targets
            plt.subplot(3, 2, i+1)
            class_counts = df[col].value_counts()
            class_perc = class_counts / class_counts.sum() * 100
            
            class_labels = ['Down', 'Up']
            plt.bar(class_labels, class_perc.values, color=['red', 'green'])
            plt.title(f'Class Distribution of {col}')
            plt.ylabel('Percentage (%)')
            plt.grid(True, alpha=0.3)
            
            # Add percentage labels
            for j, v in enumerate(class_perc.values):
                plt.text(j, v + 1, f'{v:.1f}%', ha='center')
        
        plt.tight_layout()
        plt.savefig(target_dir / 'binary_target_distribution.png')
        plt.close()
    
    # Get multiclass target variables
    multiclass_targets = [col for col in df.columns if col.startswith('target_class_')]
    
    if multiclass_targets:
        # 4. Multiclass target distribution
        plt.figure(figsize=(12, 8))
        for i, col in enumerate(multiclass_targets[:6]):  # Limit to 6 targets
            plt.subplot(3, 2, i+1)
            class_counts = df[col].value_counts().sort_index()
            class_perc = class_counts / class_counts.sum() * 100
            
            class_labels = ['Strong Down', 'Down', 'Flat', 'Up', 'Strong Up']
            plt.bar(class_labels, class_perc.values, color=['darkred', 'red', 'gray', 'green', 'darkgreen'])
            plt.xticks(rotation=45)
            plt.title(f'Class Distribution of {col}')
            plt.ylabel('Percentage (%)')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(target_dir / 'multiclass_target_distribution.png')
        plt.close()
    
    logger.info("Target variable analysis visualizations saved")

def plot_feature_groups(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create feature group visualizations.
    
    Args:
        df: DataFrame with features
        output_dir: Directory to save visualizations
    """
    logger.info("Creating feature group visualizations")
    
    # Create feature groups directory
    groups_dir = output_dir / 'feature_groups'
    groups_dir.mkdir(exist_ok=True, parents=True)
    
    # Get a subset of recent data (last 90 days) for detailed charts
    df_recent = df.iloc[-90:]
    
    # 1. Momentum indicators
    momentum_indicators = ['rsi_14']
    if all(col in df.columns for col in momentum_indicators):
        plt.figure(figsize=(16, 12))
        
        # Price subplot
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(df_recent.index, df_recent['close'], label='Close', color='blue')
        ax1.set_title('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # RSI subplot
        if 'rsi_14' in df.columns:
            ax2 = plt.subplot(3, 1, 2, sharex=ax1)
            ax2.plot(df_recent.index, df_recent['rsi_14'], label='RSI(14)', color='purple')
            ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5)
            ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5)
            ax2.set_ylim(0, 100)
            ax2.set_title('Relative Strength Index (RSI)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # MACD subplot
        if all(col in df.columns for col in ['macd', 'macd_signal']):
            ax3 = plt.subplot(3, 1, 3, sharex=ax1)
            ax3.plot(df_recent.index, df_recent['macd'], label='MACD', color='blue')
            ax3.plot(df_recent.index, df_recent['macd_signal'], label='Signal', color='red')
            ax3.bar(df_recent.index, df_recent['macd'] - df_recent['macd_signal'], label='Histogram', 
                   color=['green' if val >= 0 else 'red' for val in (df_recent['macd'] - df_recent['macd_signal'])], 
                   alpha=0.5)
            ax3.set_title('MACD')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(groups_dir / 'momentum_indicators.png')
        plt.close()
    
    # 2. Volatility indicators
    volatility_indicators = ['atr_14', 'upper_bb_20', 'lower_bb_20']
    if any(col in df.columns for col in volatility_indicators):
        plt.figure(figsize=(16, 12))
        
        # Price with Bollinger Bands
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(df_recent.index, df_recent['close'], label='Close', color='blue')
        
        if all(col in df.columns for col in ['upper_bb_20', 'middle_bb_20', 'lower_bb_20']):
            ax1.plot(df_recent.index, df_recent['upper_bb_20'], label='Upper BB', color='red', linestyle='--')
            ax1.plot(df_recent.index, df_recent['middle_bb_20'], label='Middle BB', color='purple')
            ax1.plot(df_recent.index, df_recent['lower_bb_20'], label='Lower BB', color='green', linestyle='--')
            ax1.fill_between(df_recent.index, df_recent['upper_bb_20'], df_recent['lower_bb_20'], color='gray', alpha=0.1)
        
        ax1.set_title('Price with Bollinger Bands')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ATR subplot
        if 'atr_14' in df.columns:
            ax2 = plt.subplot(2, 1, 2, sharex=ax1)
            ax2.plot(df_recent.index, df_recent['atr_14'], label='ATR(14)', color='orange')
            
            if 'atr_pct_14' in df.columns:
                ax3 = ax2.twinx()
                ax3.plot(df_recent.index, df_recent['atr_pct_14'], label='ATR % of Price', color='green', linestyle='--')
                ax3.set_ylabel('ATR % of Price')
                ax3.legend(loc='upper right')
            
            ax2.set_title('Average True Range (ATR)')
            ax2.set_ylabel('ATR Value')
            ax2.legend(loc='upper left')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(groups_dir / 'volatility_indicators.png')
        plt.close()
    
    # 3. Volume indicators
    volume_indicators = ['volume', 'volume_sma_20', 'obv']
    if any(col in df.columns for col in volume_indicators):
        plt.figure(figsize=(16, 12))
        
        # Price subplot
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(df_recent.index, df_recent['close'], label='Close', color='blue')
        ax1.set_title('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Volume subplot
        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        ax2.bar(df_recent.index, df_recent['volume'], label='Volume', color='blue', alpha=0.5)
        
        if 'volume_sma_20' in df.columns:
            ax2.plot(df_recent.index, df_recent['volume_sma_20'], label='20-day MA', color='red')
            
        ax2.set_title('Volume')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # OBV subplot
        if 'obv' in df.columns:
            ax3 = plt.subplot(3, 1, 3, sharex=ax1)
            ax3.plot(df_recent.index, df_recent['obv'], label='On-Balance Volume', color='purple')
            ax3.set_title('On-Balance Volume (OBV)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(groups_dir / 'volume_indicators.png')
        plt.close()
    
    # 4. Trend indicators
    trend_indicators = ['adx_14', 'pdi_14', 'ndi_14']
    if any(col in df.columns for col in trend_indicators):
        plt.figure(figsize=(16, 12))
        
        # Price subplot
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(df_recent.index, df_recent['close'], label='Close', color='blue')
        ax1.set_title('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ADX subplot
        if all(col in df.columns for col in ['adx_14', 'pdi_14', 'ndi_14']):
            ax2 = plt.subplot(2, 1, 2, sharex=ax1)
            ax2.plot(df_recent.index, df_recent['adx_14'], label='ADX', color='black')
            ax2.plot(df_recent.index, df_recent['pdi_14'], label='+DI', color='green')
            ax2.plot(df_recent.index, df_recent['ndi_14'], label='-DI', color='red')
            ax2.axhline(y=25, color='gray', linestyle='--', alpha=0.5)
            ax2.set_title('Average Directional Index (ADX)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(groups_dir / 'trend_indicators.png')
        plt.close()
    
    logger.info("Feature group visualizations saved")

def main():
    """Main function to create visualizations."""
    args = parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Check if input file is provided
    if not args.input_file:
        logger.error("Input file is required")
        sys.exit(1)
    
    # Load data
    df = load_data(
        input_file=args.input_file,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Create visualizations based on arguments
    # Always create price and volume chart
    plot_price_and_volume(df, output_dir)
    
    if args.all or args.indicators:
        plot_technical_indicators(df, output_dir)
    
    if args.all or args.correlation:
        plot_correlation_matrix(df, output_dir)
    
    if args.all or args.distributions:
        plot_feature_distributions(df, output_dir)
    
    if args.all or args.target_analysis:
        plot_target_analysis(df, output_dir)
    
    if args.all or args.feature_groups:
        plot_feature_groups(df, output_dir)
    
    logger.info(f"All visualizations saved to {output_dir}")

if __name__ == "__main__":
    try:
        main()
        logger.info("Visualization completed successfully")
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        log_exception(logger)
        sys.exit(1)