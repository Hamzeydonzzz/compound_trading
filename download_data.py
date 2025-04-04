#!/usr/bin/env python3
"""
download_data.py

This script downloads historical BTCUSDT data from Binance with 15-minute candles
for the past 5 years. The data is saved to CSV files organized by year and month.
"""

import os
import sys
import time
import json
import logging
import argparse
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union

# Import the Binance Client and alias it as BinanceClient for type hints
from binance.client import Client as BinanceClient

from config import Config
from utils import setup_logging

# Binance API limits: 1000 candles per request and rate limits
MAX_CANDLES_PER_REQUEST = 1000
REQUEST_WEIGHT = 1
RATE_LIMIT_PER_MINUTE = 1200

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Download historical BTCUSDT data from Binance')
    parser.add_argument('--start_date', type=str, help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, help='End date in YYYY-MM-DD format')
    parser.add_argument('--interval', type=str, default='15m', help='Candle interval (default: 15m)')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair (default: BTCUSDT)')
    parser.add_argument('--output_dir', type=str, help='Output directory for data files')
    parser.add_argument('--force', action='store_true', help='Force download even if files exist')
    
    return parser.parse_args()

def get_date_range(start_date: Optional[str] = None, end_date: Optional[str] = None) -> Tuple[datetime, datetime]:
    """
    Get the date range for data download.
    If not specified, defaults to past 5 years until current date.
    """
    end = datetime.now() if end_date is None else datetime.strptime(end_date, '%Y-%m-%d')
    
    if start_date is None:
        # Default to 5 years ago
        start = end - timedelta(days=5*365)
    else:
        start = datetime.strptime(start_date, '%Y-%m-%d')
    
    logger.info(f"Date range: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    return start, end

def get_binance_client(use_test_net: bool = False) -> BinanceClient:
    """
    Initialize and return Binance client.
    Uses test_net if specified.
    """
    try:
        from binance.client import Client
        api_key = Config.BINANCE_API_KEY
        api_secret = Config.BINANCE_API_SECRET
        
        if not api_key or not api_secret:
            logger.warning("Binance API credentials not found. Using limited public endpoints.")
            return Client("", "")
        
        client = Client(api_key, api_secret, testnet=use_test_net)
        logger.info(f"Binance client initialized (testnet: {use_test_net})")
        return client
    except ImportError:
        logger.error("python-binance package not found. Install with: pip install python-binance")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to initialize Binance client: {str(e)}")
        sys.exit(1)

def get_klines(
    client: BinanceClient,
    symbol: str, 
    interval: str, 
    start_time: datetime, 
    end_time: datetime
) -> List[List]:
    """
    Get klines (candlestick data) from Binance.
    Handles rate limiting and pagination for requests.
    """
    # Convert datetimes to millisecond timestamps for Binance API
    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)
    
    all_klines = []
    current_start = start_ms
    requests_made = 0
    
    logger.info(f"Downloading {symbol} {interval} data from {start_time} to {end_time}")
    
    while current_start < end_ms:
        # Rate limiting
        if requests_made >= RATE_LIMIT_PER_MINUTE / REQUEST_WEIGHT:
            logger.info("Rate limit approaching, sleeping for 1 minute...")
            time.sleep(60)
            requests_made = 0
        
        try:
            # Get klines for the current time segment
            klines = client.get_klines(
                symbol=symbol,
                interval=interval,
                startTime=current_start,
                endTime=end_ms,
                limit=MAX_CANDLES_PER_REQUEST
            )
            
            requests_made += 1
            
            if not klines:
                logger.info(f"No more data found after {datetime.fromtimestamp(current_start/1000)}")
                break
                
            all_klines.extend(klines)
            
            # Update start time for next request
            # Use the timestamp of the last candle + 1ms for the next request
            current_start = klines[-1][0] + 1
            
            last_date = datetime.fromtimestamp(klines[-1][0]/1000)
            logger.info(f"Downloaded {len(klines)} candles, last candle at {last_date}")
            
            # Small delay to be nice to the API
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error downloading klines: {str(e)}")
            # Exponential backoff
            time.sleep(10)
    
    logger.info(f"Total downloaded: {len(all_klines)} candles")
    return all_klines

def klines_to_dataframe(klines: List[List]) -> pd.DataFrame:
    """
    Convert raw klines data to a pandas DataFrame.
    
    Binance klines format:
    [
        [
            1499040000000,      // Open time
            "0.01634790",       // Open
            "0.80000000",       // High
            "0.01575800",       // Low
            "0.01577100",       // Close
            "148976.11427815",  // Volume
            1499644799999,      // Close time
            "2434.19055334",    // Quote asset volume
            308,                // Number of trades
            "1756.87402397",    // Taker buy base asset volume
            "28.46694368",      // Taker buy quote asset volume
            "17928899.62484339" // Ignore
        ]
    ]
    """
    columns = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ]
    
    df = pd.DataFrame(klines, columns=columns)
    
    # Convert timestamp columns to datetime
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    
    # Convert string columns to numeric
    numeric_columns = [
        'open', 'high', 'low', 'close', 'volume',
        'quote_asset_volume', 'taker_buy_base_asset_volume', 
        'taker_buy_quote_asset_volume'
    ]
    
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col])
    
    # Drop unneeded column
    df.drop('ignore', axis=1, inplace=True)
    
    return df

def save_to_csv(df: pd.DataFrame, output_dir: Path, symbol: str, interval: str) -> None:
    """
    Save DataFrame to CSV files organized by year and month.
    """
    # Group by year and month
    df['year'] = df['open_time'].dt.year
    df['month'] = df['open_time'].dt.month
    
    for (year, month), group_df in df.groupby(['year', 'month']):
        # Create directory structure
        year_dir = output_dir / str(year)
        year_dir.mkdir(exist_ok=True, parents=True)
        
        # Format month with leading zero
        month_str = f"{month:02d}"
        
        # Create filename
        filename = f"{symbol}_{interval}_{year}_{month_str}.csv"
        filepath = year_dir / filename
        
        # Save to CSV
        group_df.drop(['year', 'month'], axis=1, inplace=True)
        group_df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(group_df)} rows to {filepath}")

def main():
    """Main function to download historical data."""
    args = parse_args()
    
    # Setup directories
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Default to data directory in project root
        output_dir = Path(__file__).parents[1] / "data" / "historical" / args.symbol
    
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Get date range
    start_date, end_date = get_date_range(args.start_date, args.end_date)
    
    # Initialize Binance client
    client = get_binance_client()
    
    # Get symbols exchange info
    try:
        exchange_info = client.get_exchange_info()
        symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == args.symbol), None)
        
        if symbol_info is None:
            logger.error(f"Symbol {args.symbol} not found on Binance")
            sys.exit(1)
            
        logger.info(f"Symbol info: {json.dumps(symbol_info, indent=2)}")
    except Exception as e:
        logger.error(f"Error getting exchange info: {str(e)}")
    
    # Download data in chunks
    chunk_size = timedelta(days=30)  # Process a month at a time
    current_start = start_date
    
    while current_start < end_date:
        current_end = min(current_start + chunk_size, end_date)
        
        logger.info(f"Processing chunk: {current_start} to {current_end}")
        
        # Check if files for this period already exist
        year_dir = output_dir / str(current_start.year)
        month_str = f"{current_start.month:02d}"
        filename = f"{args.symbol}_{args.interval}_{current_start.year}_{month_str}.csv"
        filepath = year_dir / filename
        
        if filepath.exists() and not args.force:
            logger.info(f"File {filepath} already exists, skipping. Use --force to override.")
            current_start = current_end
            continue
        
        # Download klines for the current chunk
        klines = get_klines(client, args.symbol, args.interval, current_start, current_end)
        
        if klines:
            # Convert to DataFrame
            df = klines_to_dataframe(klines)
            
            # Save to CSV
            save_to_csv(df, output_dir, args.symbol, args.interval)
        
        current_start = current_end

if __name__ == "__main__":
    # Setup logging
    setup_logging()
    logger = logging.getLogger("download_data")
    
    try:
        main()
        logger.info("Data download completed successfully")
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        sys.exit(1)