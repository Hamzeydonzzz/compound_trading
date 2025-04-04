#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to process downloaded BTCUSDT data and apply feature engineering.
"""

import os
import argparse
import pandas as pd
from datetime import datetime

# Import project components
from config import Config
from feature_engineering import FeatureEngineering
from data_handler import DataHandler
from utils import setup_logger, plot_features

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process BTCUSDT data and apply feature engineering')
    
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to input CSV file with raw data')
    
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='Directory to save processed data')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations of features')
                        
    parser.add_argument('--viz_dir', type=str, default='visualizations',
                        help='Directory to save visualizations')
    
    return parser.parse_args()

def main():
    """Main function to process data and engineer features."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logger
    logger = setup_logger("feature_engineering", log_level="INFO")
    logger.info(f"Starting feature engineering on {args.input_file}")
    
    # Initialize config
    config = Config()
    
    # Ensure output directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    if args.visualize:
        os.makedirs(args.viz_dir, exist_ok=True)
    
    # Load raw data
    logger.info(f"Loading data from {args.input_file}")
    raw_data = pd.read_csv(args.input_file)
    
    # Convert timestamp to datetime if needed
    if 'timestamp' in raw_data.columns and not pd.api.types.is_datetime64_any_dtype(raw_data['timestamp']):
        raw_data['timestamp'] = pd.to_datetime(raw_data['timestamp'])
    
    # Initialize feature engineering
    feature_eng = FeatureEngineering(config=config)
    
    # Apply feature engineering
    logger.info("Applying feature engineering...")
    processed_data = feature_eng.create_features(raw_data)
    
    # Save processed data
    filename = os.path.basename(args.input_file).replace('.csv', '_processed.csv')
    output_path = os.path.join(args.output_dir, filename)
    processed_data.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to {output_path}")
    
    # Generate feature statistics
    feature_stats = processed_data.describe().transpose()
    stats_path = os.path.join(args.output_dir, filename.replace('.csv', '_stats.csv'))
    feature_stats.to_csv(stats_path)
    logger.info(f"Feature statistics saved to {stats_path}")
    
    # Generate visualizations if requested
    if args.visualize:
        logger.info("Generating feature visualizations...")
        
        # Get list of feature columns (excluding timestamp, open, high, low, close, volume)
        base_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        feature_columns = [col for col in processed_data.columns if col not in base_columns]
        
        # Generate correlation heatmap
        corr_path = os.path.join(args.viz_dir, filename.replace('.csv', '_correlation.png'))
        plot_features.plot_correlation_matrix(processed_data[feature_columns], corr_path)
        logger.info(f"Correlation matrix saved to {corr_path}")
        
        # Generate feature distribution plots
        dist_dir = os.path.join(args.viz_dir, 'distributions')
        os.makedirs(dist_dir, exist_ok=True)
        for feature in feature_columns:
            dist_path = os.path.join(dist_dir, f"{feature}_distribution.png")
            plot_features.plot_distribution(processed_data, feature, dist_path)
        logger.info(f"Feature distributions saved to {dist_dir}")
        
        # Generate time series plots for key features
        ts_dir = os.path.join(args.viz_dir, 'timeseries')
        os.makedirs(ts_dir, exist_ok=True)
        
        # Select top features (either from config or take first 10)
        top_features = getattr(config, 'TOP_FEATURES', feature_columns[:10])
        
        for feature in top_features:
            if feature in processed_data.columns:
                ts_path = os.path.join(ts_dir, f"{feature}_timeseries.png")
                plot_features.plot_timeseries(
                    processed_data, 
                    x='timestamp', 
                    y=feature,
                    price_column='close',
                    output_path=ts_path
                )
        logger.info(f"Time series plots saved to {ts_dir}")
    
    # Print summary
    logger.info(f"Feature engineering complete. Created {len(feature_columns)} features.")
    logger.info(f"Data shape: {processed_data.shape}")
    
    # Check for NaN values
    nan_counts = processed_data.isna().sum()
    if nan_counts.sum() > 0:
        logger.warning(f"NaN values found in the following columns:")
        for col, count in nan_counts[nan_counts > 0].items():
            logger.warning(f"{col}: {count} NaN values")
        
        # Save NaN report
        nan_report = pd.DataFrame({
            'column': nan_counts.index,
            'nan_count': nan_counts.values,
            'nan_percentage': (nan_counts / len(processed_data) * 100).values
        })
        nan_report_path = os.path.join(args.output_dir, filename.replace('.csv', '_nan_report.csv'))
        nan_report.to_csv(nan_report_path, index=False)
        logger.info(f"NaN report saved to {nan_report_path}")

if __name__ == "__main__":
    main()