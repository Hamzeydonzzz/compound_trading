#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to validate downloaded BTCUSDT data and perform exploratory data analysis.
This helps ensure data quality before feeding into the model training pipeline.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from utils import setup_logger

def main():
    """Main function to validate data and perform EDA."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logger
    logger = setup_logger("data_validation", log_level="INFO")
    logger.info(f"Starting data validation on {args.input_file}")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    try:
        logger.info(f"Loading data from {args.input_file}")
        df = pd.read_csv(args.input_file)
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        sys.exit(1)
    
    # Run validation and generate report
    try:
        stats = generate_validation_report(df, args.input_file, args.timeframe, args.output_dir)
        logger.info("Data validation completed successfully")
    except Exception as e:
        logger.error(f"Error during validation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Validate BTCUSDT data and perform EDA')
    
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to input CSV file (either raw or processed)')
    
    parser.add_argument('--output_dir', type=str, default='reports/data_validation',
                        help='Directory to save validation reports')
    
    parser.add_argument('--timeframe', type=str, default=None,
                        help='Timeframe of the data (for validation checks)')
    
    return parser.parse_args()

def check_missing_timestamps(df, timeframe):
    """Check for missing timestamps in time series data."""
    logger = setup_logger("validate_timestamps", log_level="INFO")
    
    # Convert timestamp column to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Determine expected frequency based on timeframe
    freq_map = {
        '1m': '1min', '3m': '3min', '5m': '5min', '15m': '15min', '30m': '30min',
        '1h': '1H', '2h': '2H', '4h': '4H', '6h': '6H', '8h': '8H', '12h': '12H',
        '1d': '1D', '3d': '3D', '1w': '1W'
    }
    
    freq = freq_map.get(timeframe, None)
    if not freq:
        logger.warning(f"Unknown timeframe: {timeframe}. Skipping timestamp continuity check.")
        return None
    
    # Create expected timestamp range
    expected_range = pd.date_range(
        start=df['timestamp'].min(),
        end=df['timestamp'].max(),
        freq=freq
    )
    
    # Find missing timestamps
    missing_timestamps = expected_range.difference(df['timestamp'])
    
    # Group missing timestamps by month for easier analysis
    if len(missing_timestamps) > 0:
        missing_by_month = missing_timestamps.to_series().dt.to_period('M').value_counts().sort_index()
        
        logger.warning(f"Found {len(missing_timestamps)} missing timestamps out of {len(expected_range)} expected")
        logger.info("Missing timestamps by month:")
        for month, count in missing_by_month.items():
            logger.info(f"  {month}: {count} missing")
    else:
        logger.info("No missing timestamps found. Data is continuous.")
    
    return missing_timestamps

def check_price_anomalies(df):
    """Check for anomalies in price data."""
    # Calculate price changes
    df['price_change_pct'] = df['close'].pct_change() * 100
    
    # Look for extreme price changes (e.g., more than 20% in one candle)
    threshold = 20.0  # 20% change
    anomalies = df[abs(df['price_change_pct']) > threshold].copy()
    
    return anomalies

def check_volume_anomalies(df):
    """Check for anomalies in volume data."""
    # Calculate rolling mean of volume (20 periods)
    df['volume_ma20'] = df['volume'].rolling(20).mean()
    
    # Look for volume spikes (e.g., more than 5x the moving average)
    threshold = 5
    df['volume_ratio'] = df['volume'] / df['volume_ma20']
    anomalies = df[df['volume_ratio'] > threshold].copy()
    
    return anomalies

def generate_validation_report(df, input_file, timeframe, output_dir):
    """Generate a comprehensive validation report."""
    logger = setup_logger("validation_report", log_level="INFO")
    logger.info("Generating validation report...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Basic statistics
    stats = {
        'filename': os.path.basename(input_file),
        'rows': len(df),
        'start_date': df['timestamp'].min(),
        'end_date': df['timestamp'].max(),
        'date_range_days': (df['timestamp'].max() - df['timestamp'].min()).days,
        'columns': len(df.columns),
        'missing_values_total': df.isna().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    # Check data types
    dtypes = pd.DataFrame({
        'column': df.dtypes.index,
        'data_type': df.dtypes.values.astype(str)
    })
    
    # Check missing values by column
    missing_values = pd.DataFrame({
        'column': df.columns,
        'missing_count': df.isna().sum().values,
        'missing_percentage': (df.isna().sum() / len(df) * 100).values
    }).sort_values('missing_count', ascending=False)
    
    # Check for missing timestamps
    if timeframe:
        missing_timestamps = check_missing_timestamps(df, timeframe)
        stats['missing_timestamps_count'] = len(missing_timestamps) if missing_timestamps is not None else "N/A"
    
    # Check for price anomalies
    price_anomalies = check_price_anomalies(df)
    stats['price_anomalies_count'] = len(price_anomalies)
    
    # Check for volume anomalies
    volume_anomalies = check_volume_anomalies(df)
    stats['volume_anomalies_count'] = len(volume_anomalies)
    
    # Generate report files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = os.path.basename(input_file).replace('.csv', '')
    
    # Save summary statistics
    with open(os.path.join(output_dir, f"{base_filename}_summary_{timestamp}.txt"), 'w') as f:
        f.write("=== Data Validation Summary ===\n\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    # Save detailed reports
    dtypes.to_csv(os.path.join(output_dir, f"{base_filename}_dtypes_{timestamp}.csv"), index=False)
    missing_values.to_csv(os.path.join(output_dir, f"{base_filename}_missing_values_{timestamp}.csv"), index=False)
    
    if len(price_anomalies) > 0:
        price_anomalies.to_csv(os.path.join(output_dir, f"{base_filename}_price_anomalies_{timestamp}.csv"), index=False)
    
    if len(volume_anomalies) > 0:
        volume_anomalies.to_csv(os.path.join(output_dir, f"{base_filename}_volume_anomalies_{timestamp}.csv"), index=False)
    
    # Generate visualizations
    generate_eda_visualizations(df, base_filename, output_dir, timestamp)
    
    logger.info(f"Validation report saved to {output_dir}")
    
    # Print summary to console
    print("\n=== Data Validation Summary ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    return stats

def generate_eda_visualizations(df, base_filename, output_dir, timestamp):
    """Generate exploratory data analysis visualizations."""
    logger = setup_logger("eda_visualization", log_level="INFO")
    logger.info("Generating EDA visualizations...")
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Set style
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Price over time
    plt.figure()
    plt.plot(df['timestamp'], df['close'])
    plt.title('BTC Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (USDT)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"{base_filename}_price_time_{timestamp}.png"))
    plt.close()
    
    # 2. Volume over time
    plt.figure()
    plt.bar(df['timestamp'], df['volume'], alpha=0.7)
    plt.title('Trading Volume Over Time')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"{base_filename}_volume_time_{timestamp}.png"))
    plt.close()
    
    # 3. Daily returns distribution
    if 'price_change_pct' not in df.columns:
        df['price_change_pct'] = df['close'].pct_change() * 100
    
    plt.figure()
    sns.histplot(df['price_change_pct'].dropna(), kde=True, bins=100)
    plt.title('Distribution of Price Changes (%)')
    plt.xlabel('Price Change (%)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"{base_filename}_returns_dist_{timestamp}.png"))
    plt.close()
    
    # 4. Correlation heatmap for OHLCV
    plt.figure()
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    if all(col in df.columns for col in ohlcv_cols):
        corr = df[ohlcv_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title('OHLCV Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f"{base_filename}_ohlcv_corr_{timestamp}.png"))
    plt.close()
    
    # 5. Box plot of OHLC prices
    plt.figure()
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        sns.boxplot(data=df[['open', 'high', 'low', 'close']])
        plt.title('OHLC Price Distribution')
        plt.ylabel('Price (USDT)')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f"{base_filename}_ohlc_boxplot_{timestamp}.png"))
    plt.close()
    
    # 6. Candlestick chart (for a sample period, e.g., last 30 data points)
    try:
        import mplfinance as mpf
        
        # Prepare data for mplfinance
        sample = df.set_index('timestamp').tail(30).copy()
        
        # Create candlestick chart
        mpf.plot(
            sample,
            type='candle',
            style='yahoo',
            title='BTC/USDT Candlestick Chart (Most Recent 30 Periods)',
            ylabel='Price (USDT)',
            volume=True,
            savefig=os.path.join(viz_dir, f"{base_filename}_candlestick_{timestamp}.png")
        )
    except ImportError:
        print("mplfinance package not installed. Skipping candlestick chart.")
    
    # 7. If we have feature columns, show distributions
    base_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                   'price_change_pct', 'volume_ma20', 'volume_ratio']
    feature_columns = [col for col in df.columns if col not in base_columns]
    
    if len(feature_columns) > 0:
        # Select up to 16 features for visualization
        selected_features = feature_columns[:16]
        
        # Create grid of histograms
        n_features = len(selected_features)
        n_cols = min(4, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        plt.figure(figsize=(16, n_rows * 4))
        for i, feature in enumerate(selected_features, 1):
            plt.subplot(n_rows, n_cols, i)
            sns.histplot(df[feature].dropna(), kde=True)
            plt.title(f'{feature} Distribution')
            plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f"{base_filename}_feature_dist_{timestamp}.png"))
        plt.close()
        