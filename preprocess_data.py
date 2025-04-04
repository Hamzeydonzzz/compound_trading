#!/usr/bin/env python3
"""
preprocess_data.py

This script preprocesses downloaded historical BTCUSDT data:
- Loads and combines data from CSV files
- Calculates technical indicators
- Creates feature sets for model training
- Performs data validation checks
- Saves processed data to CSV files
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime, timedelta
import glob

# Add the src directory to the path
script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent / "src" if script_dir.name == "scripts" else script_dir
sys.path.insert(0, str(src_dir))

# Import project modules
from config import Config
from utils import setup_logging
from feature_engineering import calculate_indicators, add_target_variables

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preprocess downloaded BTCUSDT data')
    
    parser.add_argument(
        '--input_dir', 
        type=str, 
        default=str(Path(src_dir).parents[0] / "data" / "historical" / "BTCUSDT"),
        help='Directory containing raw historical data CSV files'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default=str(Path(src_dir).parents[0] / "data" / "processed"),
        help='Directory to save processed data'
    )
    
    parser.add_argument(
        '--start_date', 
        type=str, 
        help='Start date for processing (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end_date', 
        type=str, 
        help='End date for processing (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--interval', 
        type=str, 
        default='15m',
        help='Candle interval of the data (default: 15m)'
    )
    
    parser.add_argument(
        '--symbol', 
        type=str, 
        default='BTCUSDT',
        help='Trading pair (default: BTCUSDT)'
    )
    
    parser.add_argument(
        '--force', 
        action='store_true',
        help='Force preprocessing even if processed files exist'
    )
    
    return parser.parse_args()

def load_data(input_dir: Path, symbol: str, interval: str, 
              start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Load historical data from CSV files in the input directory.
    
    Args:
        input_dir: Directory containing the CSV files
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Candlestick interval (e.g., '15m')
        start_date: Optional start date for filtering
        end_date: Optional end date for filtering
        
    Returns:
        DataFrame with combined historical data
    """
    logger.info(f"Loading data from {input_dir}")
    
    # Find all CSV files for the symbol and interval
    pattern = f"{symbol}_{interval}_*.csv"
    year_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    all_files = []
    for year_dir in year_dirs:
        year_files = list(year_dir.glob(pattern))
        all_files.extend(year_files)
    
    if not all_files:
        pattern = f"*{symbol}_{interval}*.csv"
        all_files = list(input_dir.glob(pattern))
        
        if not all_files:
            logger.error(f"No CSV files found for {symbol} {interval} in {input_dir}")
            sys.exit(1)
    
    logger.info(f"Found {len(all_files)} CSV files")
    
    # Load and combine all CSV files
    dfs = []
    for file in sorted(all_files):
        logger.info(f"Reading {file}")
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error reading {file}: {str(e)}")
    
    if not dfs:
        logger.error("No data could be loaded")
        sys.exit(1)
    
    # Combine all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined data: {len(combined_df)} rows")
    
    # Ensure timestamp columns are datetime
    if 'open_time' in combined_df.columns:
        combined_df['open_time'] = pd.to_datetime(combined_df['open_time'])
        combined_df.set_index('open_time', inplace=True)
    
    # Sort by timestamp
    combined_df.sort_index(inplace=True)
    
    # Remove duplicate timestamps
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
    logger.info(f"After removing duplicates: {len(combined_df)} rows")
    
    # Filter by date range if specified
    if start_date:
        start = pd.to_datetime(start_date)
        combined_df = combined_df[combined_df.index >= start]
    
    if end_date:
        end = pd.to_datetime(end_date)
        combined_df = combined_df[combined_df.index <= end]
    
    logger.info(f"Data loaded: {len(combined_df)} rows from {combined_df.index[0]} to {combined_df.index[-1]}")
    
    return combined_df

def validate_data(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    Validate and clean the data.
    
    Args:
        df: DataFrame with historical price data
        interval: Candlestick interval for checking continuity
        
    Returns:
        Validated and cleaned DataFrame
    """
    logger.info("Validating data...")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        logger.warning(f"Missing values detected:\n{missing_values[missing_values > 0]}")
        
        # Fill missing values
        logger.info("Filling missing values...")
        # Forward fill for most columns
        df.fillna(method='ffill', inplace=True)
        # For any remaining NaNs, use backward fill
        df.fillna(method='bfill', inplace=True)
    
    # Check for negative values in price and volume
    if (df[['open', 'high', 'low', 'close', 'volume']] < 0).any().any():
        logger.warning("Negative values detected in price or volume columns")
        
        # Replace negative values with NaN and then forward fill
        price_volume_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in price_volume_cols:
            neg_mask = df[col] < 0
            if neg_mask.any():
                logger.info(f"Fixing {neg_mask.sum()} negative values in {col}")
                df.loc[neg_mask, col] = np.nan
                df[col].fillna(method='ffill', inplace=True)
    
    # Check for price anomalies (e.g., high < low)
    anomalies = df[df['high'] < df['low']]
    if not anomalies.empty:
        logger.warning(f"Found {len(anomalies)} candles where high < low")
        # Fix by swapping high and low values
        df.loc[anomalies.index, ['high', 'low']] = df.loc[anomalies.index, ['low', 'high']].values
    
    # Check for continuity in timestamps
    if len(df) > 1:
        try:
            interval_minutes = int(interval[:-1]) if interval.endswith('m') else int(interval[:-1]) * 60
            expected_diff = pd.Timedelta(minutes=interval_minutes)
            actual_diff = df.index.to_series().diff().dropna()
            
            # Allow for small deviations (e.g., DST changes)
            tolerance = pd.Timedelta(minutes=5)
            continuity_issues = actual_diff[(actual_diff > expected_diff + tolerance) | 
                                          (actual_diff < expected_diff - tolerance)]
            
            if not continuity_issues.empty:
                logger.warning(f"Timestamp continuity issues detected at {len(continuity_issues)} locations")
                for idx, diff in continuity_issues.items():
                    logger.debug(f"Gap at {idx}: {diff}")
        except Exception as e:
            logger.error(f"Error checking timestamp continuity: {str(e)}")
    
    logger.info("Data validation completed")
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data for feature calculation.
    
    Args:
        df: DataFrame with historical price data
        
    Returns:
        Preprocessed DataFrame
    """
    logger.info("Preprocessing data...")
    
    # Rename columns to standard format if needed
    if 'close' not in df.columns and 'Close' in df.columns:
        df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)
    
    # Ensure we have the basic OHLCV columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        sys.exit(1)
    
    # Calculate percentage change for close price
    df['close_pct_change'] = df['close'].pct_change() * 100
    
    # Calculate log returns
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Calculate normalized volume
    df['volume_pct_change'] = df['volume'].pct_change() * 100
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_normalized'] = df['volume'] / df['volume_sma_20']
    
    # Calculate price range
    df['price_range'] = (df['high'] - df['low']) / df['close']
    df['body_size'] = abs(df['open'] - df['close']) / df['close']
    df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
    df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
    
    # Drop rows with NaN values from the calculations
    df.dropna(inplace=True)
    
    logger.info(f"Preprocessing completed: {len(df)} rows remaining")
    return df

def main():
    """Main function to preprocess historical data."""
    args = parse_args()
    
    # Setup directories
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Check if input directory exists
    if not input_dir.exists():
        logger.error(f"Input directory {input_dir} does not exist")
        sys.exit(1)
    
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Check if output file exists
    output_file = output_dir / f"{args.symbol}_{args.interval}_processed.csv"
    if output_file.exists() and not args.force:
        logger.info(f"Output file {output_file} already exists. Use --force to override.")
        sys.exit(0)
    
    # Load raw data
    df = load_data(
        input_dir=input_dir,
        symbol=args.symbol,
        interval=args.interval,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Validate data
    df = validate_data(df, args.interval)
    
    # Basic preprocessing
    df = preprocess_data(df)
    
    # Calculate technical indicators
    df = calculate_indicators(df)
    
    # Add target variables
    df = add_target_variables(df)
    
    # Save processed data
    logger.info(f"Saving processed data to {output_file}")
    df.to_csv(output_file)
    logger.info(f"Saved {len(df)} rows to {output_file}")
    
    # Generate summary statistics
    stats_file = output_dir / f"{args.symbol}_{args.interval}_stats.csv"
    stats = df.describe()
    stats.to_csv(stats_file)
    logger.info(f"Saved summary statistics to {stats_file}")
    
    # Create feature correlation matrix
    corr_file = output_dir / f"{args.symbol}_{args.interval}_correlation.csv"
    corr = df.corr()
    corr.to_csv(corr_file)
    logger.info(f"Saved correlation matrix to {corr_file}")
    
    logger.info("Data preprocessing completed successfully")

if __name__ == "__main__":
    # Setup logging
    setup_logging()
    logger = logging.getLogger("preprocess_data")
    
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        sys.exit(1)