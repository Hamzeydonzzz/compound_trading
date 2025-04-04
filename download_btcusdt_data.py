#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to download BTCUSDT historical data using Project Compound components.
"""

import os
import argparse
from datetime import datetime, timedelta
import pandas as pd

# Import project components
from config import Config
from binance_interface import BinanceInterface
from data_handler import DataHandler
from utils import setup_logger

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Download BTCUSDT historical data')
    
    parser.add_argument('--start_date', type=str, 
                        default=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                        help='Start date in YYYY-MM-DD format')
    
    parser.add_argument('--end_date', type=str, 
                        default=datetime.now().strftime('%Y-%m-%d'),
                        help='End date in YYYY-MM-DD format')
    
    parser.add_argument('--timeframe', type=str, default='1h',
                        choices=['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w'],
                        help='Timeframe for the data')
    
    parser.add_argument('--output_dir', type=str, default='data/raw',
                        help='Directory to save downloaded data')
                        
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                        help='Trading pair symbol')
    
    return parser.parse_args()

def main():
    """Main function to download historical data."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logger
    logger = setup_logger("data_download", log_level="INFO")
    logger.info(f"Starting download of {args.symbol} data from {args.start_date} to {args.end_date}")
    
    # Initialize config
    config = Config()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize Binance interface
    binance = BinanceInterface(
        api_key=config.BINANCE_API_KEY,
        api_secret=config.BINANCE_API_SECRET,
        testnet=config.USE_TESTNET
    )
    
    # Initialize data handler
    data_handler = DataHandler(config=config)
    
    # Convert string dates to datetime objects
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # Download data
    logger.info(f"Downloading {args.symbol} {args.timeframe} data...")
    df = data_handler.fetch_historical_data(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=start_date,
        end_date=end_date,
        binance_interface=binance
    )
    
    # Save data
    filename = f"{args.symbol}_{args.timeframe}_{args.start_date}_{args.end_date}.csv"
    output_path = os.path.join(args.output_dir, filename)
    df.to_csv(output_path, index=False)
    logger.info(f"Data saved to {output_path}")
    
    # Display summary
    logger.info(f"Downloaded {len(df)} records")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Optional: Create a preview of the data
    preview_path = os.path.join(args.output_dir, f"{args.symbol}_{args.timeframe}_preview.csv")
    df.head(100).to_csv(preview_path, index=False)
    logger.info(f"Preview saved to {preview_path}")

if __name__ == "__main__":
    main()